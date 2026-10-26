
from __future__ import annotations

from nlpcc import models


def test_configured_local_api_is_preferred(monkeypatch) -> None:
    api = models.Provider("openai-compatible", "local-api-model")
    an = models.Provider("anthropic", "claude-opus-5-5")
    ol = models.Provider("ollama", "local-b")

    for name, found in (("_openai_compatible", api), ("_anthropic", an),
                        ("_ollama", ol)):
        monkeypatch.setattr(models, name, lambda found=found: found)
    assert models.available() is api

    monkeypatch.setattr(models, "_openai_compatible", lambda: None)
    assert models.available() is an

    monkeypatch.setattr(models, "_anthropic", lambda: None)
    assert models.available() is ol

    assert api.local and ol.local
    assert not an.local


def test_a_provider_can_be_pinned(monkeypatch) -> None:
    api = models.Provider("openai-compatible", "local-a")
    monkeypatch.setattr(models, "_openai_compatible", lambda: api)
    assert models.available(prefer="openai-compatible") is api
    assert models.available(prefer="lmstudio") is api
    assert models.available(prefer="nonsense") is None


def test_local_api_probe_uses_bearer_key(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("NLPCC_LOCAL_API_KEY", "test-key-not-used")
    importlib.reload(models)
    observed = {}

    def get_json(url, timeout=models.PROBE_TIMEOUT, headers=None):
        observed.update(url=url, headers=headers)
        return {"data": [{"id": "my-model"}]}

    monkeypatch.setattr(models, "_get_json", get_json)
    provider = models._openai_compatible()

    assert observed["headers"] == {"Authorization": "Bearer test-key-not-used"}
    assert observed["url"] == f"{models.LOCAL_API_URL}/models"
    assert provider is not None and provider.model == "my-model"

    def post_json(url, payload, timeout=models.REQUEST_TIMEOUT, headers=None):
        observed.update(chat_url=url, chat_headers=headers)
        return {"choices": [{"message": {"content": "answer"}}]}

    monkeypatch.setattr(models, "_post_json", post_json)
    assert models._local_chat(provider, "system", "prompt") == "answer"
    assert observed["chat_headers"] == observed["headers"]
    assert observed["chat_url"] == f"{models.LOCAL_API_URL}/chat/completions"


def test_a_local_model_can_be_pinned(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("NLPCC_LOCAL_MODEL", "pinned-model")
    importlib.reload(models)
    monkeypatch.setattr(models, "_get_json", lambda url, timeout=0, headers=None:
                        {"data": [{"id": "first"}, {"id": "pinned-model"}]})
    provider = models._openai_compatible()
    assert provider is not None and provider.model == "pinned-model"


def test_describe_explains_what_to_do_when_there_is_no_model() -> None:
    message = models.describe()
    assert "NLPCC_LOCAL_URL" in message or "Ollama" in message


def test_env_file_does_not_override_the_environment(monkeypatch) -> None:
    import importlib

    monkeypatch.setenv("NLPCC_LOCAL_API_KEY", "from-the-shell")
    importlib.reload(models)
    assert models._local_api_headers() == {
        "Authorization": "Bearer from-the-shell"}


def test_a_keyless_anthropic_client_is_not_a_provider(monkeypatch) -> None:

    import importlib

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    real = importlib.reload(models)._anthropic
    assert real() is None


def test_json_is_recovered_from_a_messy_reply() -> None:
    assert models._extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert models._extract_json("<think>hmm</think> here: {\"a\": 2} ok") == {"a": 2}
    assert models._extract_json('{"a": "brace } inside"}') == {"a": "brace } inside"}
    assert models._extract_json("no json here") is None
    assert models._extract_json('{"truncated": "val') is None


def test_schema_mismatch_is_treated_as_no_answer() -> None:
    from pydantic import BaseModel

    class Shape(BaseModel):
        name: str
        count: int

    original = models._local_chat
    models._local_chat = lambda *a, **k: '{"name": "x", "count": "not a number"}'
    try:
        got = models.complete("sys", "prompt", Shape,
                              provider=models.Provider("openai-compatible", "m"))
    finally:
        models._local_chat = original
    assert got is None
