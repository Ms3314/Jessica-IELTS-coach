"""API keys entered in the app's Settings, kept on this computer.

The keys go in a small JSON file (data/settings.json, readable only by you) and
are loaded into the environment at startup, taking priority over .env. They are
never sent back to the browser: the UI only ever learns whether a key is set
and its last four characters.
"""
import json
import os
import tempfile
from pathlib import Path
from typing import Optional

DEFAULT_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"

# settings field -> environment variable
KEYS = {
    "openai_api_key": "OPENAI_API_KEY",
    "tavily_api_key": "TAVILY_API_KEY",
}


def load(path: Path = DEFAULT_PATH) -> dict:
    try:
        data = json.loads(Path(path).read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def apply_to_environ(path: Path = DEFAULT_PATH) -> None:
    """Make saved keys the ones the rest of the app sees (they beat .env)."""
    saved = load(path)
    for field, env in KEYS.items():
        value = saved.get(field)
        if isinstance(value, str) and value.strip():
            os.environ[env] = value.strip()


def save(updates: dict, path: Path = DEFAULT_PATH) -> None:
    """Merge `updates` into the file. An empty string removes that key.

    Written atomically with owner-only permissions, so a crash can't leave a
    half-written file and other users on the machine can't read it.
    """
    path = Path(path)
    data = load(path)
    for field, value in updates.items():
        if field not in KEYS or value is None:
            continue
        value = str(value).strip()
        if value:
            data[field] = value
        else:
            data.pop(field, None)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".settings-")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(data, f, indent=2)
        os.chmod(tmp, 0o600)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    apply_to_environ(path)
    # a removed key must also leave the running process's environment
    for field in updates:
        if field in KEYS and not data.get(field):
            os.environ.pop(KEYS[field], None)


def _hint(value: str) -> str:
    return "••••" + value[-4:] if len(value) >= 8 else "••••"


def status(path: Path = DEFAULT_PATH) -> dict:
    """For each key: is it set, where from, and a masked hint. Never the key itself."""
    saved = load(path)
    out = {}
    for field, env in KEYS.items():
        from_file = isinstance(saved.get(field), str) and bool(saved[field].strip())
        value = os.environ.get(env, "").strip()
        out[field] = {
            "set": bool(value),
            "source": "settings" if from_file else ("environment" if value else None),
            "hint": _hint(value) if value else None,
        }
    return out


def check_openai_key(key: str) -> tuple[bool, Optional[str]]:
    """Ask OpenAI whether a key is accepted. (True, None) when it is.

    Only a rejected key counts as a failure; if OpenAI can't be reached we can't
    tell, so the key is accepted and the caller says it wasn't verified.
    """
    import openai

    try:
        openai.OpenAI(api_key=key, timeout=10, max_retries=0).models.list()
        return True, None
    except openai.AuthenticationError:
        return False, "OpenAI rejected that key. Check it and try again."
    except openai.PermissionDeniedError:
        return False, "That key doesn't have permission to use the API."
    except Exception as e:  # offline, timeout, OpenAI outage...
        return True, f"Couldn't reach OpenAI to verify it ({type(e).__name__}); the key was saved."
