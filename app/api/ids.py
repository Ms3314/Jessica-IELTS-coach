"""Ids that arrive from the browser, and the patterns they must match.

They end up in SQLite keys and file paths, so every route checks them first.
"""
import re

# LangGraph thread ids we accept from the browser (used as the SQLite key).
SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
# IELTS mock-test ids (a timestamp) and the audio files inside one.
EXAM_ID = re.compile(r"^\d{8}-\d{6}$")
EXAM_AUDIO = re.compile(r"^turn\d{2}(_\d)?\.wav$")
