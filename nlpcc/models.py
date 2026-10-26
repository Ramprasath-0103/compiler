"""Where the model comes from.

Google Gemini is tried first, whenever GEMINI_API_KEY (or GOOGLE_API_KEY) is
set. Without a key the configured OpenAI-compatible local API is next,
followed by Anthropic and Ollama. The local API can be LM Studio or another
server that implements the OpenAI chat completions API.

The project still works with no API key at all: the local providers need no
account, and everything outside this module (the compiler, explanations,
scoring and suggestions) needs no model whatsoever. Pin a provider with
NLPCC_PROVIDER.

Every provider here is reached with urllib from the standard library, so none
of this requires an extra Python package.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Type, TypeVar

def _load_env_file() -> None:
    """Read KEY=value lines from a .env beside the project, if there is one.

    So a key set once keeps working after a restart without anyone having to
    remember to export it. Values already in the environment win, and .env is
    in .gitignore -- a key belongs on the machine, never in the repository.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    try:
        lines = path.read_text().splitlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, _, value = line.partition("=")
        name, value = name.strip(), value.strip().strip("'\"")
        if name and name not in os.environ:
            os.environ[name] = value


_load_env_file()

LOCAL_API_URL = os.environ.get(
    "NLPCC_LOCAL_URL", "http://127.0.0.1:8317/v1").rstrip("/")
# Pin the model ID; left empty, the first one the server lists is used.
LOCAL_MODEL = os.environ.get("NLPCC_LOCAL_MODEL", "")
OLLAMA_URL = os.environ.get("NLPCC_OLLAMA_URL", "http://127.0.0.1:11434")
ANTHROPIC_MODEL = "claude-opus-5-5"

GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta"
# An alias Google moves to its current Flash model, so a retired model ID
# never breaks the project. Pin a specific one with NLPCC_GEMINI_MODEL.
GEMINI_MODEL = os.environ.get("NLPCC_GEMINI_MODEL", "gemini-flash-latest")
# Gemini thinks before it answers and charges the thinking to
# maxOutputTokens, so a budget sized for the answer alone can be spent
# entirely on thinking and come back with no text.
GEMINI_THINKING_HEADROOM = 8192

PROBE_TIMEOUT = 1.5
REQUEST_TIMEOUT = 90.0       # a small local model on CPU is not fast

# A token budget is a budget, not a limit a small model stops short of: given
# 8192 it will cheerfully produce 8192 tokens of rambling notes, which on a
# local CPU model is several minutes for one answer. These ceilings are sized
# for what the task actually needs.
MAX_TOKENS_LOCAL = 1536

T = TypeVar("T")


@dataclass
class Provider:
    kind: str           # "gemini" | "openai-compatible" | "ollama" | "anthropic"
    model: str
    detail: str = ""

    @property
    def local(self) -> bool:
        return self.kind in ("openai-compatible", "ollama")

    def __str__(self) -> str:
        where = "local" if self.local else "hosted"
        return f"{self.model} ({self.kind}, {where})"


class ModelError(Exception):
    """A hosted API answered with an error instead of a reply.

    Retrying will not help -- a bad key, an unknown model or a spent quota
    is the same a second later -- so callers stop and report the message.
    """


def _gemini_key() -> str:
    return (os.environ.get("GEMINI_API_KEY")
            or os.environ.get("GOOGLE_API_KEY", ""))


def _local_api_headers() -> dict[str, str]:
    key = os.environ.get("NLPCC_LOCAL_API_KEY", "")
    return {"Authorization": f"Bearer {key}"} if key else {}


def _get_json(url: str, timeout: float = PROBE_TIMEOUT,
              headers: dict[str, str] | None = None):
    request = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _post_json(url: str, payload: dict, timeout: float = REQUEST_TIMEOUT,
               headers: dict | None = None):
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        try:
            return json.loads(error.read() or b"{}")     # keep the API's message
        except ValueError:
            return None
    except (urllib.error.URLError, OSError, ValueError, TimeoutError):
        return None


def _gemini() -> Provider | None:
    # No probe request: available() runs on every model-backed call, and a
    # round trip to Google each time would cost more than it proves. A bad
    # key surfaces as the API's own message on the first real request.
    if not _gemini_key():
        return None
    return Provider("gemini", GEMINI_MODEL, "API key found")


def _openai_compatible() -> Provider | None:
    data = _get_json(f"{LOCAL_API_URL}/models", headers=_local_api_headers())
    models = (data or {}).get("data") or []
    if not models:
        return None
    model = LOCAL_MODEL or models[0].get("id", "local-model")
    return Provider("openai-compatible", model,
                    f"{len(models)} model(s) available")


def _ollama() -> Provider | None:
    data = _get_json(f"{OLLAMA_URL}/api/tags")
    models = (data or {}).get("models") or []
    if not models:
        return None
    return Provider("ollama", models[0].get("name", "local-model"),
                    f"{len(models)} model(s) available")


def _anthropic() -> Provider | None:
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
    except Exception:                            # noqa: BLE001
        return None
    # Constructing a client proves nothing: the SDK defers credential
    # resolution, so a keyless client builds fine and only fails at request
    # time. Check that something actually resolved, or we would pick this
    # provider over a working local model and then fail every call.
    if not (client.api_key or getattr(client, "auth_token", None)):
        return None
    return Provider("anthropic", ANTHROPIC_MODEL, "API credentials found")


def available(prefer: str = "") -> Provider | None:
    """The provider to use, or None when there is no model anywhere.

    ``prefer`` pins one ("gemini", "openai-compatible", "ollama",
    "anthropic"); NLPCC_PROVIDER does the same from the environment.
    """
    prefer = prefer or os.environ.get("NLPCC_PROVIDER", "")
    finders = {"gemini": _gemini, "google": _gemini,
               "openai-compatible": _openai_compatible,
               "lmstudio": _openai_compatible,
               "anthropic": _anthropic, "ollama": _ollama}
    if prefer:
        finder = finders.get(prefer)
        return finder() if finder else None
    for finder in (_gemini, _openai_compatible, _anthropic, _ollama):
        found = finder()
        if found is not None:
            return found
    return None


def unreachable(task: str) -> str:
    """One sentence for when ``task`` needs a model and there is none."""
    return (f"{task} needs a model, and none is configured -- set "
            "GEMINI_API_KEY in .env.")


def describe() -> str:
    """One line for the CLI and the web UI."""
    provider = available()
    if provider is None:
        return ("no model available -- set GEMINI_API_KEY in .env, or "
                "configure a local API with NLPCC_LOCAL_URL")
    return str(provider)


def complete_text(system: str, prompt: str, provider: Provider | None = None,
                  max_tokens: int = 4096,
                  reasons: list[str] | None = None) -> str:
    """Ask for plain text back, with no JSON wrapper.

    Code generation uses this rather than `complete`. Wrapping a C program in
    a JSON string means escaping every newline, quote and backslash in it, and
    a small local model gets that wrong far more often than it gets the C
    wrong -- measured at zero usable replies out of five, every one of them
    failing on the escaping rather than the program. Asking for the code
    directly removes the problem instead of retrying through it.
    """
    provider = provider or available()
    if provider is None:
        if reasons is not None:
            reasons.append("no model available")
        return ""

    if provider.kind == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.create(
                model=provider.model, max_tokens=max_tokens, system=system,
                messages=[{"role": "user", "content": prompt}])
            return "".join(block.text for block in response.content
                           if getattr(block, "type", "") == "text")
        except Exception as exc:                 # noqa: BLE001
            if reasons is not None:
                reasons.append(f"the API call failed ({exc.__class__.__name__})")
            return ""

    try:
        text = _chat(provider, system, prompt, max_tokens)
    except ModelError as exc:
        if reasons is not None:
            reasons.append(str(exc))
        return ""
    if not text and reasons is not None and not reasons:
        reasons.append("the model returned nothing")
    return text


_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> dict | None:
    """Pull a JSON object out of a reply, however the model wrapped it.

    Small local models add prose, fences and occasionally a <think> block
    around the answer, so finding the object matters more than insisting the
    whole reply is valid JSON.
    """
    if not text:
        return None
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S)

    fenced = _FENCE.search(text)
    if fenced:
        text = fenced.group(1)

    start = text.find("{")
    if start == -1:
        return None
    # Walk to the matching brace, ignoring braces inside strings.
    depth, in_string, escaped = 0, False, False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:index + 1])
                except ValueError:
                    return None
    return None


def _local_chat(provider: Provider, system: str, prompt: str,
                max_tokens: int = 4096) -> str:
    """One chat round trip to a local server.

    max_tokens has to be sent explicitly: the server's own default is small
    enough to cut a generated program off mid-string, which arrives as a
    truncated JSON object and fails to parse with no obvious cause.
    """
    if provider.kind == "ollama":
        data = _post_json(f"{OLLAMA_URL}/api/chat", {
            "model": provider.model, "stream": False,
            "options": {"num_predict": max_tokens, "temperature": 0.2},
            "messages": [{"role": "system", "content": system},
                         {"role": "user", "content": prompt}],
        })
        return ((data or {}).get("message") or {}).get("content", "")

    data = _post_json(f"{LOCAL_API_URL}/chat/completions", {
        "model": provider.model,
        "temperature": 0.2,
        "max_tokens": max_tokens,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": prompt}],
    }, headers=_local_api_headers())
    choices = (data or {}).get("choices") or []
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "")


def _gemini_chat(provider: Provider, system: str, prompt: str,
                 max_tokens: int = 4096, json_mode: bool = False) -> str:
    """One generateContent round trip to the Gemini API.

    Temperature is left at Google's default: current Gemini models are tuned
    for it, and lowering it is documented to cause looping replies.
    """
    config: dict = {"maxOutputTokens": max_tokens + GEMINI_THINKING_HEADROOM}
    if json_mode:
        config["responseMimeType"] = "application/json"
    data = _post_json(f"{GEMINI_URL}/models/{provider.model}:generateContent", {
        "systemInstruction": {"parts": [{"text": system}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": config,
    }, headers={"x-goog-api-key": _gemini_key()})
    data = data or {}

    error = data.get("error")
    if error:
        message = str(error.get("message") or error.get("status") or error)
        raise ModelError(f"the Gemini API refused the request: {message[:200]}")
    blocked = (data.get("promptFeedback") or {}).get("blockReason")
    if blocked:
        raise ModelError(f"Gemini blocked the prompt ({blocked})")

    candidates = data.get("candidates") or []
    if not candidates:
        return ""
    parts = (candidates[0].get("content") or {}).get("parts") or []
    return "".join(part.get("text", "") for part in parts
                   if not part.get("thought"))


def _chat(provider: Provider, system: str, prompt: str, max_tokens: int,
          json_mode: bool = False) -> str:
    if provider.kind == "gemini":
        return _gemini_chat(provider, system, prompt, max_tokens, json_mode)
    return _local_chat(provider, system, prompt,
                       max_tokens=min(max_tokens, MAX_TOKENS_LOCAL))


# Small local models produce a malformed reply often enough that one attempt
# is not a fair test of them -- usually a bad escape inside a string, which
# json.loads rejects. Anthropic uses its structured-output API instead.
LOCAL_ATTEMPTS = 3


def complete(system: str, prompt: str, schema: Type[T],
             provider: Provider | None = None,
             max_tokens: int = 4096,
             reasons: list[str] | None = None) -> T | None:
    """Ask for one object matching ``schema`` (a pydantic model).

    Returns None rather than raising for every failure -- no provider, a
    refused connection, a reply that is not JSON, a reply that does not match
    the schema. Callers treat None as "the model had nothing to add".

    Pass ``reasons`` to collect why each attempt failed. Without it a caller
    can only say "no code", which is indistinguishable from the model being
    slow, unreachable, or producing malformed JSON -- three problems with
    three different answers.
    """
    def note(text: str) -> None:
        if reasons is not None:
            reasons.append(text)

    provider = provider or available()
    if provider is None:
        note("no model available")
        return None

    if provider.kind == "anthropic":
        try:
            import anthropic
            client = anthropic.Anthropic()
            response = client.messages.parse(
                model=provider.model, max_tokens=max_tokens, system=system,
                output_config={"effort": "low"},
                messages=[{"role": "user", "content": prompt}],
                output_format=schema)
            return getattr(response, "parsed_output", None)
        except Exception as exc:                 # noqa: BLE001
            note(f"the API call failed ({exc.__class__.__name__})")
            return None

    # Everything else gets the schema in the prompt and is validated on the
    # way back.
    try:
        shape = json.dumps(schema.model_json_schema(), indent=None)
    except Exception:                            # noqa: BLE001
        return None

    instructions = (
        f"{system}\n\n"
        "Reply with one JSON object and nothing else. No prose, no code fence.\n"
        "Escape every newline and quote inside a JSON string correctly.\n"
        f"It must match this JSON schema:\n{shape}"
    )
    deadline = time.monotonic() + REQUEST_TIMEOUT

    for attempt in range(LOCAL_ATTEMPTS):
        if time.monotonic() > deadline:
            note(f"ran out of time after {attempt} attempt(s) "
                 f"({REQUEST_TIMEOUT:.0f}s budget)")
            return None
        try:
            raw = _chat(provider, instructions, prompt, max_tokens,
                        json_mode=True)
        except ModelError as exc:
            note(str(exc))
            return None
        if not raw:
            note("the model server returned nothing")
            continue
        data = _extract_json(raw)
        if data is None:
            note(f"the reply was not valid JSON ({len(raw)} chars)")
            continue
        try:
            return schema.model_validate(data)
        except Exception:                        # noqa: BLE001 - shape mismatch
            note("the reply did not match the expected shape")
            continue
    return None
