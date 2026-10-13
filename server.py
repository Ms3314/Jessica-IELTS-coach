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
import time
import uuid
from pathlib import Path

from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app import agent as agent_module
from app import settings_store
from app import exam_report
from app.engine import Session
from app.exam_engine import ExamSession
from app import ielts
from app.ielts import EXAMS_DIR, ExamLog
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
# IELTS mock-test ids (a timestamp) and the audio files inside one.
_EXAM_ID = re.compile(r"^\d{8}-\d{6}$")
_EXAM_AUDIO = re.compile(r"^turn\d{2}(_\d)?\.wav$")

app = FastAPI(title="speakIelts", version="0.1.0")


@app.get("/healthz")
async def healthz():
    return {"ok": True}


class KeysUpdate(BaseModel):
    openai_api_key: str | None = None
    tavily_api_key: str | None = None


@app.get("/settings")
async def get_settings():
    """Which API keys are set (masked) and which model is in use. Never the keys."""
    from app.config import OPENAI_MODEL

    return {"keys": settings_store.status(), "model": OPENAI_MODEL, "provider": "openai"}


@app.post("/settings/keys")
async def save_keys(body: KeysUpdate):
    """Save API keys entered in Settings and start using them immediately."""
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    openai_key = (updates.get("openai_api_key") or "").strip()
    note = None
    if openai_key:
        # a slow network call: keep it off the event loop
        ok, note = await asyncio.get_running_loop().run_in_executor(
            None, settings_store.check_openai_key, openai_key
        )
        if not ok:
            return JSONResponse({"ok": False, "error": note}, status_code=400)
    settings_store.save(updates)
    agent_module.apply_keys(openai_key, (updates.get("tavily_api_key") or "").strip())
    return {"ok": True, "note": note, "keys": settings_store.status()}


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


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """The saved conversation of a session, as plain messages."""
    if not _SESSION_ID.match(session_id):
        return JSONResponse({"error": "invalid session id"}, status_code=400)
    return await asyncio.get_running_loop().run_in_executor(None, sessions_db.messages, session_id)


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


@app.get("/ielts/schedule")
async def ielts_schedule():
    """The mock test's timings, so the 'before you begin' screen is always accurate."""
    return ielts.schedule()


@app.get("/exams/{session_id}")
async def get_exams(session_id: str):
    """IELTS mock tests taken in a session, newest first."""
    if not _SESSION_ID.match(session_id):
        return JSONResponse({"error": "invalid session id"}, status_code=400)
    return exam_report.list_exams(session_id)


@app.get("/exams/{session_id}/{exam_id}")
async def get_exam(session_id: str, exam_id: str):
    """One mock test: turns, the analysis numbers and the report (if ready)."""
    if not (_SESSION_ID.match(session_id) and _EXAM_ID.match(exam_id)):
        return JSONResponse({"error": "invalid id"}, status_code=400)
    data = exam_report.load_exam(session_id, exam_id)
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return data


@app.get("/exams/{session_id}/{exam_id}/audio/{name}")
async def get_exam_audio(session_id: str, exam_id: str, name: str):
    """One recorded answer, so the user can listen back to it."""
    if not (_SESSION_ID.match(session_id) and _EXAM_ID.match(exam_id) and _EXAM_AUDIO.match(name)):
        return JSONResponse({"error": "invalid id"}, status_code=400)
    path = EXAMS_DIR / session_id / exam_id / name
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@app.post("/exams/{session_id}/{exam_id}/analyse")
async def reanalyse_exam(session_id: str, exam_id: str):
    """Re-run the analysis + report on a saved test (e.g. after changing models)."""
    if not (_SESSION_ID.match(session_id) and _EXAM_ID.match(exam_id)):
        return JSONResponse({"error": "invalid id"}, status_code=400)
    try:
        log = ExamLog.load(session_id, exam_id)
    except FileNotFoundError:
        return JSONResponse({"error": "not found"}, status_code=404)
    threading.Thread(
        target=exam_report.run_exam_analysis, args=(log, agent_module.model), daemon=True
    ).start()
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
    start_settings = {}
    mode = "chat"
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
                    if isinstance(data.get("settings"), dict):
                        start_settings = data["settings"]
                    if data.get("mode") == "ielts":
                        mode = "ielts"
                elif data.get("type") == "quit":
                    return
    except WebSocketDisconnect:
        return

    in_q: "queue.Queue" = queue.Queue()
    out_q: "queue.Queue" = queue.Queue()

    source = WebSocketSource(in_q)
    sink = WebSocketSink(out_q)
    handler = WebSocketHandler(out_q)
    if mode == "ielts":
        # A fixed-structure mock test instead of a free conversation. Once it ends,
        # the analysis + report run in their own thread so they outlive the socket.
        def analyse(log):
            threading.Thread(
                target=exam_report.run_exam_analysis,
                args=(log, agent_module.model, handler),
                daemon=True,
            ).start()

        sessions_db.update(session_id, name=f"IELTS mock test {time.strftime('%d %b %H:%M')}")
        session = ExamSession(
            source, sink, handler, agent_module.model, session_id=session_id, on_complete=analyse
        )
    else:
        session = Session(source, sink, handler, session_id=session_id, greet=True)

    def apply_settings(settings: dict):
        pause = settings.get("pause_s")
        if isinstance(pause, (int, float)) and not isinstance(pause, bool):
            source.set_pause(pause)
            if hasattr(session, "pause_s"):
                session.pause_s = pause   # the mock test re-applies it before every answer

    apply_settings(start_settings)

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

                    if isinstance(data, dict) and data.get("type") == "settings":
                        apply_settings(data)  # changed in the UI mid-call
                        continue

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
                        if not getattr(session, "interruptible", True):
                            continue  # the examiner can't be talked over: leave her audio alone
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
    # Local only by default: this app handles your voice and API keys. Set
    # HOST=0.0.0.0 to reach it from another device on your network.
    uvicorn.run(app, host=os.environ.get("HOST", "127.0.0.1"), port=port)
