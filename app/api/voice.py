"""The live call: one WebSocket (`/ws`) carries everything.

  client -> server   (text JSON)  {"type":"start"|"quit"|"barge_in"|"record_stop"|"settings"}
  client -> server   (binary)     raw 16 kHz mono int16 mic PCM
  server -> client   (text JSON)  {"type":"state"|"user"|"agent"|"pause"|"exam"|"error", ...}
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
import queue
import threading
import time
import uuid

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.api.ids import SESSION_ID
from app.chat import sessions as sessions_db
from app.chat.engine import Session
from app.exam import ielts, profile
from app.exam import report as exam_report
from app.exam.engine import ExamSession
from app.io import WebSocketSource, WebSocketSink, WebSocketHandler, QUIT, STOP
from app.llm import agent as agent_module

logger = logging.getLogger("speakIelts")

router = APIRouter()


@router.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()

    # The browser keeps its session id in localStorage and sends it in the
    # query string, so a reconnect (or Quit/Start) resumes the same memory.
    requested = ws.query_params.get("session", "")
    session_id = requested if SESSION_ID.match(requested) else uuid.uuid4().hex
    logger.info("client connected (session %s)", session_id)
    sessions_db.touch(session_id)

    # Wait for the user to click the mic before starting the engine. This lets
    # us play a personalised opening question (random for new sessions, summary-
    # based for returning ones) without an engine thread idling in the background.
    started = False
    start_settings = {}
    mode = "chat"
    exam_length = None
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
                        exam_length = data.get("exam_length") if data.get("exam_length") in ielts.TIMINGS else None
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
                kwargs={"store": profile.record_exam},
                daemon=True,
            ).start()

        sessions_db.update(session_id, name=f"IELTS mock test {time.strftime('%d %b %H:%M')}")
        session = ExamSession(
            source, sink, handler, agent_module.model, session_id=session_id, on_complete=analyse,
            length=exam_length,
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

                    if isinstance(data, dict) and data.get("type") == "record_stop":
                        in_q.put(STOP)  # the candidate finished their answer
                        continue

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
