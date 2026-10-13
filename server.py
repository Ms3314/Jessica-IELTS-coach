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
import re
import threading
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.engine import Session
from app.io import WebSocketSource, WebSocketSink, WebSocketHandler, QUIT
from app import sessions as sessions_db

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("speakIelts")

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"

# LangGraph thread ids we accept from the browser (used as the SQLite key).
_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")

app = FastAPI(title="speakIelts", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


class UpdateSession(BaseModel):
    name: str | None = None
    color: str | None = None


@app.get("/sessions")
async def get_sessions():
    """List voice sessions (metadata + cached summaries), newest first."""
    return [m.__dict__ for m in sessions_db.list_sessions()]


@app.post("/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSession):
    if not _SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    sessions_db.update(session_id, name=body.name, color=body.color)
    return {"ok": True}


@app.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    if not _SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    return {"summary": sessions_db.get_summary(session_id)}


@app.post("/sessions/{session_id}/summarize")
async def summarize_session(session_id: str):
    """Generate (and cache) a one-paragraph summary of the thread."""
    if not _SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    summary = sessions_db.summarize(session_id)
    sessions_db.set_summary(session_id, summary)
    return {"summary": summary}


@app.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not _SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    sessions_db.delete(session_id)
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # The browser keeps its session id in localStorage and sends it in the
    # query string, so a reconnect (or Quit/Start) resumes the same memory.
    requested = ws.query_params.get("session", "")
    session_id = requested if _SESSION_ID.match(requested) else uuid.uuid4().hex
    logger.info("client connected (session %s)", session_id)
    sessions_db.touch(session_id)

    # Wait for the user to click the mic before starting the engine. This lets
    # us play a personalised opening question (random for new sessions, summary-
    # based for returning ones) without an engine thread idling in the background.
    started = False
    try:
        while not started:
            raw = await ws.receive_text()
            try:
                data = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if isinstance(data, dict):
                if data.get("type") == "start":
                    started = True
                elif data.get("type") == "quit":
                    return
    except WebSocketDisconnect:
        return

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()

    source = WebSocketSource(in_q)
    sink = WebSocketSink(out_q)
    handler = WebSocketHandler(out_q)
    session = Session(source, sink, handler, session_id=session_id, greet=True)

    loop = asyncio.get_running_loop()

    # Run the (blocking) voice engine in a worker thread.
    engine_thread = threading.Thread(target=session.loop, daemon=True)
    engine_thread.start()

    async def sender():
        # Drain engine -> WS. Running in the executor keeps the loop responsive.
        while True:
            item = await loop.run_in_executor(None, out_q.get)
            if item is None:
                break
            if isinstance(item, (bytes, bytearray)):
                await ws.send_bytes(bytes(item))
            else:
                if isinstance(item, dict):
                    etype = item.get("type")
                    if etype == "agent":
                        logger.info(">> agent reply: %s", item.get("text", "")[:80])
                    elif etype == "state":
                        logger.info(">> state -> %s", item.get("state"))
                    elif etype == "error":
                        logger.info(">> ERROR: %s", item.get("message", "")[:160])
                await ws.send_json(item)

    # Consume the engine's first event ("listening") BEFORE the sender task
    # starts: two consumers on out_q would race, and an in-flight outbound
    # event can make the ASGI layer drop the first inbound binary frames.
    first_item = await loop.run_in_executor(None, out_q.get)
    if first_item is not None:
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
                in_q.put(msg["bytes"])

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
                        by = "button" if data.get("by") == "button" else "voice"
                        heard = data.get("heard_idx")
                        heard = heard if isinstance(heard, int) else None
                        logger.info(
                            "client interrupted (%s, heard up to sentence %s); "
                            "cancelling the reply", by, heard,
                        )
                        session.interrupt(by, heard)
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
        # The call ended: summarise the conversation so the next visit can pick
        # up from it. Blocking LLM work goes to a worker thread.
        try:
            await loop.run_in_executor(None, sessions_db.auto_summarize, session_id)
        except Exception:
            logger.warning("auto-summary failed for %s", session_id, exc_info=True)


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
