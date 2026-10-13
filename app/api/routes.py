"""REST routes: everything the UI asks for outside a live call.

    /healthz
    /settings, /settings/keys          API keys (masked) and the model in use
    /sessions...                       list, rename, summarise, delete
    /profile                           mock-test scores over time
    /ielts/schedule                    the test's timings
    /exams...                          saved tests, their audio, re-analysis

The live call itself goes over the WebSocket in app/api/voice.py.
"""
import asyncio
import threading

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from app import settings_store
from app.api.ids import EXAM_AUDIO, EXAM_ID, SESSION_ID
from app.chat import sessions as sessions_db
from app.exam import ielts, profile
from app.exam import report as exam_report
from app.exam.ielts import EXAMS_DIR, ExamLog
from app.llm import agent as agent_module

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"ok": True}


class KeysUpdate(BaseModel):
    openai_api_key: str | None = None
    tavily_api_key: str | None = None


@router.get("/settings")
async def get_settings():
    """Which API keys are set (masked) and which model is in use. Never the keys."""
    from app.config import OPENAI_MODEL

    return {"keys": settings_store.status(), "model": OPENAI_MODEL, "provider": "openai"}


@router.post("/settings/keys")
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


@router.get("/sessions")
async def get_sessions():
    """List voice sessions (metadata + cached summaries), newest first."""
    return [m.__dict__ for m in sessions_db.list_sessions()]


@router.post("/sessions/{session_id}")
async def update_session(session_id: str, body: UpdateSession):
    if not SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    sessions_db.update(session_id, name=body.name, color=body.color)
    return {"ok": True}


@router.get("/sessions/{session_id}/summary")
async def get_session_summary(session_id: str):
    if not SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    return {"summary": sessions_db.get_summary(session_id)}


@router.get("/sessions/{session_id}/messages")
async def get_session_messages(session_id: str):
    """The saved conversation of a session, as plain messages."""
    if not SESSION_ID.match(session_id):
        return JSONResponse({"error": "invalid session id"}, status_code=400)
    return await asyncio.get_running_loop().run_in_executor(None, sessions_db.messages, session_id)


@router.post("/sessions/{session_id}/summarize")
async def summarize_session(session_id: str):
    """Generate (and cache) a one-paragraph summary of the thread."""
    if not SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    summary = sessions_db.summarize(session_id)
    sessions_db.set_summary(session_id, summary)
    return {"summary": summary}


@router.delete("/sessions/{session_id}")
async def delete_session(session_id: str):
    if not SESSION_ID.match(session_id):
        return {"error": "invalid session id"}, 400
    sessions_db.delete(session_id)
    return {"ok": True}


@router.get("/profile")
async def get_profile():
    """Your mock-test scores over time (kept on this computer)."""
    return await asyncio.get_running_loop().run_in_executor(None, profile.overview)


@router.get("/ielts/schedule")
async def ielts_schedule(length: str | None = None):
    """The mock test's timings for a length ("full" or "short"), so the notice is always accurate."""
    return ielts.schedule(length)


@router.get("/exams/{session_id}")
async def get_exams(session_id: str):
    """IELTS mock tests taken in a session, newest first."""
    if not SESSION_ID.match(session_id):
        return JSONResponse({"error": "invalid session id"}, status_code=400)
    return exam_report.list_exams(session_id)


@router.get("/exams/{session_id}/{exam_id}")
async def get_exam(session_id: str, exam_id: str):
    """One mock test: turns, the analysis numbers and the report (if ready)."""
    if not (SESSION_ID.match(session_id) and EXAM_ID.match(exam_id)):
        return JSONResponse({"error": "invalid id"}, status_code=400)
    data = exam_report.load_exam(session_id, exam_id)
    if data is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return data


@router.get("/exams/{session_id}/{exam_id}/audio/{name}")
async def get_exam_audio(session_id: str, exam_id: str, name: str):
    """One recorded answer, so the user can listen back to it."""
    if not (SESSION_ID.match(session_id) and EXAM_ID.match(exam_id) and EXAM_AUDIO.match(name)):
        return JSONResponse({"error": "invalid id"}, status_code=400)
    path = EXAMS_DIR / session_id / exam_id / name
    if not path.is_file():
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(path, media_type="audio/wav")


@router.post("/exams/{session_id}/{exam_id}/analyse")
async def reanalyse_exam(session_id: str, exam_id: str):
    """Re-run the analysis + report on a saved test (e.g. after changing models)."""
    if not (SESSION_ID.match(session_id) and EXAM_ID.match(exam_id)):
        return JSONResponse({"error": "invalid id"}, status_code=400)
    try:
        log = ExamLog.load(session_id, exam_id)
    except FileNotFoundError:
        return JSONResponse({"error": "not found"}, status_code=404)
    threading.Thread(
        target=exam_report.run_exam_analysis, args=(log, agent_module.model),
        kwargs={"store": profile.record_exam}, daemon=True,
    ).start()
    return {"ok": True}
