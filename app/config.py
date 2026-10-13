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

# --- turn-taking / noise robustness (web recorder) ---------------------------
# Each can be overridden in .env. Loudness values are RMS on a 0..1 scale.
# Speech must be this many times louder than the room's noise floor to count.
NOISE_GATE_RATIO = float(os.getenv("NOISE_GATE_RATIO", "2.5"))
# Hard safety cap on one recorded turn.
MAX_RECORD_S = float(os.getenv("MAX_RECORD_S", "15"))
