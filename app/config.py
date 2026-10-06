"""Shared configuration for the app.

Loads .env from the project root exactly once and exposes the settings
other modules need. Resolving the path from this file (rather than the CWD)
means the app works no matter where it is launched from.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(PROJECT_ROOT / ".env")

# --- agent (main brain) ------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
