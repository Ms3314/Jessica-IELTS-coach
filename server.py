"""Web entry point: builds the FastAPI app that serves the React UI and the API.

    app/api/routes.py   REST routes (settings, sessions, profile, exams)
    app/api/voice.py    the /ws WebSocket that carries a live call
    web/dist            the built React UI, served at /

Run with `./start.sh` or `python -m fastapi dev server.py --port 8080`.
"""
import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from app.api import routes, voice

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)

WEB_DIST = Path(__file__).resolve().parent / "web" / "dist"

app = FastAPI(title="speakIelts", version="0.1.0")
app.include_router(routes.router)
app.include_router(voice.router)


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
