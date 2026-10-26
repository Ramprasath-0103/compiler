
from __future__ import annotations

from nlpcc import models

# conftest replaces every finder before each test; keep the real one.
_REAL_GEMINI = models._gemini


def test_gemini_is_preferred_then_the_local_api(monkeypatch) -> None:
    ge = models.Provider("gemini", "gemini-flash-latest")
    api = models.Provider("openai-compatible", "local-api-model")
    an = models.Provider("anthropic", "claude-opus-5-5")
    ol = models.Provider("ollama", "local-b")

    for name, found in (("_gemini", ge), ("_openai_compatible", api),
                        ("_anthropic", an), ("_ollama", ol)):
        monkeypatch.setattr(models, name, lambda found=found: found)
    assert models.available() is ge

    monkeypatch.setattr(models, "_gemini", lambda: None)
    assert models.available() is api

    monkeypatch.setattr(models, "_openai_compatible", lambda: None)
    assert models.available() is an

    monkeypatch.setattr(models, "_anthropic", lambda: None)
    assert models.available() is ol

    assert api.local and ol.local
    assert not an.local and not ge.local


def test_a_provider_can_be_pinned(monkeypatch) -> None:
    api = models.Provider("openai-compatible", "local-a")
    ge = models.Provider("gemini", "g")
    monkeypatch.setattr(models, "_openai_compatible", lambda: api)
    monkeypatch.setattr(models, "_gemini", lambda: ge)
    assert models.available(prefer="openai-compatible") is api
    assert models.available(prefer="lmstudio") is api
    assert models.available(prefer="gemini") is ge
    assert models.available(prefer="nonsense") is None


def test_gemini_is_a_provider_only_with_a_key(monkeypatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    assert _REAL_GEMINI() is None

    monkeypatch.setenv("GOOGLE_API_KEY", "google-key")
    assert _REAL_GEMINI() is not None

    monkeypatch.setenv("GEMINI_API_KEY", "gemini-key")
    assert models._gemini_key() == "gemini-key"
    provider = _REAL_GEMINI()
    assert provider is not None and provider.kind == "gemini"


def test_gemini_request_and_reply(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key-not-used")
    observed = {}

    def post_json(url, payload, timeout=models.REQUEST_TIMEOUT, headers=None):
        observed.update(url=url, payload=payload, headers=headers)
        return {"candidates": [{"content": {"parts": [
            {"text": "thinking", "thought": True},
            {"text": '{"a": '}, {"text": "1}"}]}}]}

    monkeypatch.setattr(models, "_post_json", post_json)
    provider = models.Provider("gemini", "gemini-test")
    text = models._chat(provider, "system", "prompt", 100, json_mode=True)

    assert text == '{"a": 1}'
    assert observed["url"] == (
        f"{models.GEMINI_URL}/models/gemini-test:generateContent")
    assert observed["headers"] == {"x-goog-api-key": "test-key-not-used"}
    payload = observed["payload"]
    assert payload["systemInstruction"] == {"parts": [{"text": "system"}]}
    assert payload["contents"] == [{"role": "user", "parts": [{"text": "prompt"}]}]
    assert payload["generationConfig"]["responseMimeType"] == "application/json"
    assert payload["generationConfig"]["maxOutputTokens"] > 100


def test_a_gemini_error_is_reported_and_not_retried(monkeypatch) -> None:
    from pydantic import BaseModel

    class Shape(BaseModel):
        name: str

    calls = []

    def post_json(url, payload, timeout=models.REQUEST_TIMEOUT, headers=None):
        calls.append(url)
        return {"error": {"code": 400, "status": "INVALID_ARGUMENT",
                          "message": "API key not valid."}}

    monkeypatch.setattr(models, "_post_json", post_json)
    provider = models.Provider("gemini", "g")

    reasons: list[str] = []
    assert models.complete("sys", "prompt", Shape, provider=provider,
                           reasons=reasons) is None
    assert reasons == ["the Gemini API refused the request: API key not valid."]
    assert len(calls) == 1

    reasons = []
    assert models.complete_text("sys", "prompt", provider=provider,
                                reasons=reasons) == ""
    assert reasons == ["the Gemini API refused the request: API key not valid."]


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
    assert "GEMINI_API_KEY" in message


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
