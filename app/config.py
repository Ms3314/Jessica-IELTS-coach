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

# Keys entered in the app's Settings take priority over .env.
from app import settings_store  # noqa: E402

settings_store.apply_to_environ()

# --- agent (main brain) ------------------------------------------------------
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
# Which model answers: "openai" (needs a key) or "ollama" (runs on this computer, no key).
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai").strip().lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3.5:4b")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# --- turn-taking / noise robustness (web recorder) ---------------------------
# Each can be overridden in .env. Loudness values are RMS on a 0..1 scale.
# Speech must be this many times louder than the room's noise floor to count.
NOISE_GATE_RATIO = float(os.getenv("NOISE_GATE_RATIO", "2.5"))
# Hard safety cap on one recorded turn.
MAX_RECORD_S = float(os.getenv("MAX_RECORD_S", "15"))
# How long you must stay quiet before Jessica decides you are done. The user can
# change this live in the Settings panel; this is the default for new visits.
FINAL_SILENCE_S = float(os.getenv("FINAL_SILENCE_S", "1.2"))
# --- IELTS mock test timing ---------------------------------------------------------
# Parts 1 and 3 run against the clock (the real test is 4-5 minutes each). Lower
# these in .env to try a short test while developing.
IELTS_PART1_S = float(os.getenv("IELTS_PART1_S", "300"))
IELTS_PART3_S = float(os.getenv("IELTS_PART3_S", "300"))
