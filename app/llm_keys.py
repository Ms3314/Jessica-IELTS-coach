"""Swap a new API key into already-built clients, so no restart is needed."""
from pydantic import SecretStr


def swap_openai_credentials(model, fresh) -> None:
    """Make `model` (a ChatOpenAI wired into the agent) talk to OpenAI with the
    credentials of `fresh`, a ChatOpenAI built from the new key."""
    for attr in ("client", "async_client", "root_client", "root_async_client", "openai_api_key"):
        if hasattr(fresh, attr):
            setattr(model, attr, getattr(fresh, attr))


def swap_tavily_key(search, key: str) -> bool:
    """Point an existing TavilySearch tool at a new key. False if it can't be done."""
    wrapper = getattr(search, "api_wrapper", None)
    if wrapper is None or not hasattr(wrapper, "tavily_api_key"):
        return False
    wrapper.tavily_api_key = SecretStr(key)
    return True
