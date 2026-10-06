"""Web server: serves the React UI and bridges it to the voice engine.

One WebSocket (`/ws`) carries everything:

  client -> server   (text JSON)  {"type":"quit"}
  client -> server   (binary)     raw 16 kHz mono int16 mic PCM
  server -> client   (text JSON)  {"type":"state"|"user"|"agent"|"error", ...}
  server -> client   (binary)     raw float32 24 kHz mono TTS PCM

The engine (Whisper + OpenAI agent + Kokoro) runs in a worker thread so the
async event loop stays free to shuttle audio/events across the socket. A pair
of queues bridges the thread and the coroutine:
  in_q  : WS -> engine   (mic PCM bytes)
  out_q : engine -> WS   (events + TTS audio bytes); None == "sender, stop"
"""
import asyncio
import json
import logging
import os
import queue
import time
import threading
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.engine import Session
from app.io import WebSocketSource, WebSocketSink, WebSocketHandler, QUIT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("speakIelts")

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"

app = FastAPI(title="speakIelts", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    logger.info("client connected")

    # Per-connection counters for logging the mic stream (diagnostic only).
    _mic_bytes = 0
    _mic_last_log = time.monotonic()

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()

    source = WebSocketSource(in_q)
    sink = WebSocketSink(out_q)
    handler = WebSocketHandler(out_q)
    session = Session(source, sink, handler)

    loop = asyncio.get_running_loop()

    # Run the (blocking) voice engine in a worker thread.
    engine_thread = threading.Thread(target=session.loop, daemon=True)
    engine_thread.start()

    async def sender():
        # Drain engine -> WS. Running in the executor keeps the loop responsive.
        total_audio = 0
        while True:
            item = await loop.run_in_executor(None, out_q.get)
            if item is None:
                break
            if isinstance(item, (bytes, bytearray)):
                n = len(item)
                total_audio += n
                # Float32 @ 24 kHz => 4 bytes/sample => 6 bytes/ms of audio.
                ms = n // 24
                logger.info(
                    ">> streaming %d bytes of TTS audio to client (~%d ms, session total %d bytes)",
                    n,
                    ms,
                    total_audio,
                )
                await ws.send_bytes(bytes(item))
            else:
                if isinstance(item, dict):
                    etype = item.get("type")
                    if etype == "agent":
                        logger.info(">> agent reply: %s", item.get("text", "")[:80])
                    elif etype == "user":
                        logger.info(">> user said: %s", item.get("text", "")[:80])
                    elif etype == "state":
                        logger.info(">> state -> %s", item.get("state"))
                    elif etype == "error":
                        logger.info(">> ERROR event: %s", item.get("message", "")[:160])
                    else:
                        logger.info(">> event: %s", etype)
                await ws.send_json(item)

    # The very first thing the engine sends is the "listening" state. Consume
    # and send it BEFORE we start the sender task or await ws.receive(). This
    # avoids both (a) a race between two consumers on out_q and (b) the ASGI
    # layer dropping inbound binary frames because a receive was pending while
    # the initial outbound state was in flight.
    first_item = await loop.run_in_executor(None, out_q.get)
    if first_item is not None:
        if isinstance(first_item, dict) and first_item.get("type") == "state":
            logger.info(">> state -> %s", first_item.get("state"))
        await ws.send_json(first_item)

    sender_task = asyncio.create_task(sender())

    try:
        while True:
            msg = await ws.receive()
            mtype = msg.get("type")
            has_bytes = "bytes" in msg

            # This log is intentionally kept: on this machine uvicorn's WebSocket
            # receive path drops inbound binary frames unless the synchronous
            # stdout I/O gives the transport a chance to feed the next frame.
            logger.info("<< ws frame type=%r bytes=%s", mtype, has_bytes)

            if mtype == "websocket.disconnect":
                break

            if has_bytes:
                binary = msg["bytes"]
                in_q.put(binary)
                _mic_bytes += len(binary)
                now = time.monotonic()
                if now - _mic_last_log > 2.0:
                    logger.info(
                        "<< received %d bytes of mic audio from client so far", _mic_bytes
                    )
                    _mic_last_log = now

            # Text / JSON control frames (e.g. {"type":"quit"}).
            raw = msg.get("text") or msg.get("data") or msg.get("json")
            if raw is not None:
                try:
                    data = json.loads(raw) if isinstance(raw, str) else raw
                except (TypeError, json.JSONDecodeError):
                    pass
                else:
                    if isinstance(data, dict) and data.get("type") == "quit":
                        # End the session and let the finally block tear the
                        # socket down. The client reconnects into a fresh
                        # session, so a later Start works again.
                        logger.info("client asked to quit; closing session")
                        break

                    if isinstance(data, dict) and data.get("type") == "barge_in":
                        # The user talked over the reply: cancel TTS and drop
                        # any reply audio still queued for the socket.
                        logger.info("client barged in; cancelling the reply")
                        session.interrupt()
                        kept = []
                        while True:
                            try:
                                item = out_q.get_nowait()
                            except queue.Empty:
                                break
                            if not isinstance(item, (bytes, bytearray)):
                                kept.append(item)
                        for item in kept:
                            out_q.put(item)
    except WebSocketDisconnect:
        pass
    finally:
        logger.info("client disconnected")
        session.stop()
        in_q.put(QUIT)        # wake the engine thread if it's parked in listen()
        out_q.put(None)       # tell the sender coroutine to stop
        await sender_task
        try:
            await ws.close()
        except Exception:
            pass


# Serve the built React app LAST so it doesn't shadow /ws or /healthz.
# In dev, `vite` (port 5173) serves the UI and proxies /ws here instead.
if WEB_DIST.exists():
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
else:

    @app.get("/")
    async def index():
        return HTMLResponse(
            "<h1>speakIelts</h1>"
            "<p>The web UI hasn't been built yet.</p>"
            "<p>Run <code>cd web &amp;&amp; npm install &amp;&amp; npm run build</code>, "
            "then restart this server.</p>"
        )


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
