"""Shared configuration for TalkingAngela.

Loads .env from the project root exactly once (no matter which submodule
is imported first) and exposes the settings the other modules need.

Previously each module called load_dotenv() on its own — which worked only
because they all sat next to .env in the project root. With the package
layout, module files live in app/, so the .env path is resolved
relative to the package parent instead of the CWD (which changes if you
launch the app from elsewhere).
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# explicit path: works no matter where the app is launched from
load_dotenv(PROJECT_ROOT / ".env")

# --- agent (main brain) ------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
