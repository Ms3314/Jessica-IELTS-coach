"""API key storage tests — temp files only, no network.

    python tests/test_settings_store.py
"""
import json
import os
import stat
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import settings_store as ss


def _tmp():
    d = tempfile.TemporaryDirectory()
    return d, Path(d.name) / "data" / "settings.json"


def _clean_env():
    for env in ss.KEYS.values():
        os.environ.pop(env, None)


def test_a_saved_key_is_stored_and_put_in_the_environment():
    _clean_env()
    d, path = _tmp()
    ss.save({"openai_api_key": "  sk-test-1234567890  "}, path)
    assert json.loads(path.read_text()) == {"openai_api_key": "sk-test-1234567890"}
    assert os.environ["OPENAI_API_KEY"] == "sk-test-1234567890"
    _clean_env(); d.cleanup()


def test_the_file_is_private_to_the_owner():
    _clean_env()
    d, path = _tmp()
    ss.save({"openai_api_key": "sk-test-1234567890"}, path)
    mode = stat.S_IMODE(path.stat().st_mode)
    assert mode == 0o600, oct(mode)
    _clean_env(); d.cleanup()


def test_saving_one_key_keeps_the_other():
    _clean_env()
    d, path = _tmp()
    ss.save({"openai_api_key": "sk-aaaaaaaa1111"}, path)
    ss.save({"tavily_api_key": "tvly-bbbbbbbb2222"}, path)
    assert ss.load(path) == {"openai_api_key": "sk-aaaaaaaa1111", "tavily_api_key": "tvly-bbbbbbbb2222"}
    _clean_env(); d.cleanup()


def test_an_empty_value_removes_the_key_everywhere():
    _clean_env()
    d, path = _tmp()
    ss.save({"tavily_api_key": "tvly-bbbbbbbb2222"}, path)
    ss.save({"tavily_api_key": ""}, path)
    assert "tavily_api_key" not in ss.load(path)
    assert "TAVILY_API_KEY" not in os.environ, "a removed key must leave the running process too"
    d.cleanup()


def test_unknown_fields_are_ignored():
    _clean_env()
    d, path = _tmp()
    ss.save({"evil": "x", "openai_api_key": "sk-test-1234567890"}, path)
    assert "evil" not in ss.load(path)
    _clean_env(); d.cleanup()


def test_status_never_reveals_the_key():
    _clean_env()
    d, path = _tmp()
    ss.save({"openai_api_key": "sk-secret-abcdefgh9876"}, path)
    st = ss.status(path)
    text = json.dumps(st)
    assert "secret" not in text and "sk-" not in text
    assert st["openai_api_key"] == {"set": True, "source": "settings", "hint": "••••9876"}
    assert st["tavily_api_key"]["set"] is False
    _clean_env(); d.cleanup()


def test_a_key_from_the_environment_is_reported_as_such():
    _clean_env()
    d, path = _tmp()
    os.environ["OPENAI_API_KEY"] = "sk-from-dotenv-0000"
    assert ss.status(path)["openai_api_key"]["source"] == "environment"
    _clean_env(); d.cleanup()


def test_saved_keys_beat_the_environment_at_startup():
    _clean_env()
    d, path = _tmp()
    ss.save({"openai_api_key": "sk-saved-in-settings"}, path)
    os.environ["OPENAI_API_KEY"] = "sk-from-dotenv-0000"
    ss.apply_to_environ(path)
    assert os.environ["OPENAI_API_KEY"] == "sk-saved-in-settings"
    _clean_env(); d.cleanup()


def test_a_missing_or_corrupt_file_is_just_empty():
    d, path = _tmp()
    assert ss.load(path) == {}
    path.parent.mkdir(parents=True)
    path.write_text("{not json")
    assert ss.load(path) == {}
    d.cleanup()


def test_a_rejected_key_fails_the_check_and_an_unreachable_api_does_not():
    import openai

    class Auth:
        def __init__(self, **_k):
            pass

        @property
        def models(self):
            class M:
                def list(_s):
                    raise openai.AuthenticationError("bad", response=_resp(401), body=None)
            return M()

    def _resp(code):
        import httpx
        return httpx.Response(code, request=httpx.Request("GET", "https://api.openai.com/v1/models"))

    real = openai.OpenAI
    try:
        openai.OpenAI = Auth
        ok, msg = ss.check_openai_key("sk-bad")
        assert ok is False and "rejected" in msg

        class Offline(Auth):
            @property
            def models(self):
                class M:
                    def list(_s):
                        raise ConnectionError("no network")
                return M()
        openai.OpenAI = Offline
        ok, msg = ss.check_openai_key("sk-whatever")
        assert ok is True and "verify" in msg
    finally:
        openai.OpenAI = real


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print(f"  ok: {t.__name__}")
    print(f"{len(tests)} settings store tests passed")
