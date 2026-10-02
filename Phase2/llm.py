"""Minimal multi-provider chat client for the validation harness.

Two providers, one interface. Anthropic goes through its SDK; anything
OpenAI-compatible (DeepSeek, and most hosted endpoints) goes through a plain
urllib POST, so switching provider costs no new dependency.

The point of the abstraction is not portability for its own sake -- it is that
series labeling is a bulk classification job where a cheap model plausibly
matches an expensive one, and the only honest way to find out is to run both
and measure agreement. That requires swapping the model without touching the
prompt.
"""

import json
import os
import time
import urllib.error
import urllib.request

def load_dotenv(path: str | None = None) -> None:
    """Populate os.environ from a KEY=VALUE file, without overriding a value
    already exported in the shell.  Deliberately minimal -- no interpolation,
    no multi-line values, no dependency.  Silently does nothing if absent, so
    the shell-only workflow stays first-class.  Defaults to `.env` at the repo
    root so it works regardless of the directory a script is run from."""
    if path is None:
        path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    try:
        with open(path) as fh:
            lines = fh.readlines()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


load_dotenv()

MAX_RETRIES = 5
BACKOFF_BASE = 2

PROVIDERS = {
    "anthropic": {
        "base_url": "https://api.anthropic.com",
        "key_env": "ANTHROPIC_API_KEY",
        "default_model": "claude-sonnet-5",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "key_env": "DEEPSEEK_API_KEY",
        # The first-party API accepts ONLY the unversioned aliases -- dated
        # ids like deepseek-v4-flash-0731 are a third-party (OpenRouter,
        # Together) convention and 400 here. The alias floats, though, and the
        # response echoes the alias rather than the snapshot that served it,
        # so provenance cannot be recovered from the wire. Use --label-as to
        # record the snapshot you know is live; see commands/label.py.
        "default_model": "deepseek-v4-flash",
    },
    "openai": {
        "base_url": "https://api.openai.com/v1",
        "key_env": "OPENAI_API_KEY",
        "default_model": "gpt-5.6-luna",
    },
    "zai": {
        # OpenAI-compatible surface, so it needs no code of its own. Included
        # for labeller diversity rather than price: an agreement statistic
        # across models from one family measures shared architecture, not
        # correctness. glm-4.5-flash is free but rate-limited, so throughput
        # rather than cost is what bounds it on a 930k-item corpus.
        "base_url": "https://api.z.ai/api/paas/v4",
        "key_env": "ZAI_API_KEY",
        "default_model": "glm-4.5-flash",
    },
}


class ProviderError(RuntimeError):
    """Carries status_code so callers can tell fatal from transient."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


def resolve(provider: str, model: str | None, base_url: str | None) -> tuple:
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown provider {provider!r}; "
                         f"expected one of {', '.join(PROVIDERS)}")
    spec = PROVIDERS[provider]
    return (model or spec["default_model"],
            base_url or spec["base_url"],
            spec["key_env"])


# Providers disagree on the output-cap parameter and on whether reasoning is
# billable by default. Both are per-provider facts, not caller concerns.
PROVIDER_DEFAULTS = {
    # gpt-5.6-* reject `max_tokens` outright, and reason by default: measured
    # 55-70 reasoning tokens per triage request, ~85% of all output. On a
    # bulk classification job that buys nothing -- on a 2,000-event A/B the
    # `none` setting found every item the default did, for 77% fewer output
    # tokens. `minimal` is not a supported value on this model.
    "openai": {"token_field": "max_completion_tokens",
               "extra": {"reasoning_effort": "none"}},
    # GLM reasons by default and z.ai SILENTLY IGNORES unknown parameters --
    # `reasoning_effort` is accepted and discarded, measuring identical to a
    # deliberately bogus key. The knob that works is `thinking`. Disabling it
    # took a one-line classification from 400 output tokens in 12.2s (capped
    # mid-reasoning, content field EMPTY) to 2 tokens in 3.7s, same answer.
    # The empty-content case matters beyond cost: a chunk whose reasoning
    # overruns max_tokens returns no labels at all.
    "zai": {"extra": {"thinking": {"type": "disabled"}}},
}


def complete(provider: str, model: str, base_url: str, key_env: str,
             system: str, user: str, max_tokens: int = 8000,
             extra: dict | None = None) -> tuple:
    """Send one system+user turn. Returns (text, {input, output} token counts).

    Token counts come back so a bulk run can report its own burn rate -- a
    long job that only reveals its cost when the balance runs dry is a job
    that wastes both money and wall-clock."""
    api_key = os.environ.get(key_env)
    if not api_key:
        raise ProviderError(f"{key_env} is not set", status_code=401)

    if provider == "anthropic":
        return _anthropic(model, system, user, max_tokens)
    spec = PROVIDER_DEFAULTS.get(provider, {})
    body = dict(spec.get("extra") or {})
    body.update(extra or {})
    return _openai_compatible(model, base_url, api_key, system, user,
                              max_tokens,
                              spec.get("token_field", "max_tokens"), body)


def _anthropic(model: str, system: str, user: str, max_tokens: int) -> str:
    try:
        import anthropic
    except ImportError as exc:
        raise RuntimeError(
            "anthropic SDK not installed: pip install anthropic "
            "(provider 'anthropic' is not reproducible from requirements.txt)"
        ) from exc
    try:
        response = anthropic.Anthropic().messages.create(
            model=model, max_tokens=max_tokens, system=system,
            messages=[{"role": "user", "content": user}],
        )
    except Exception as exc:
        raise ProviderError(str(exc),
                            getattr(exc, "status_code", None)) from exc

    parts = [b.text for b in response.content
             if getattr(b, "type", None) == "text"]
    if not parts:
        raise ProviderError("response contained no text block")
    usage = {"input": response.usage.input_tokens,
             "output": response.usage.output_tokens}
    return "\n".join(parts), usage


def _openai_compatible(model: str, base_url: str, api_key: str,
                       system: str, user: str, max_tokens: int,
                       token_field: str = "max_tokens",
                       extra: dict | None = None) -> str:
    payload = json.dumps({
        "model": model,
        token_field: max_tokens,
        **(extra or {}),
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        # Native JSON mode -- removes the fenced-``` stripping guesswork the
        # Anthropic path needs, though parse_response still handles fences.
        "response_format": {"type": "json_object"},
    }).encode()

    request = urllib.request.Request(
        f"{base_url.rstrip('/')}/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"},
    )

    last: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            with urllib.request.urlopen(request, timeout=180) as resp:
                body = json.loads(resp.read().decode())
            usage = body.get("usage") or {}
            return (body["choices"][0]["message"]["content"],
                    {"input": usage.get("prompt_tokens", 0),
                     "output": usage.get("completion_tokens", 0)})
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:400]
            last = ProviderError(f"HTTP {exc.code}: {detail}", exc.code)
            # Newer OpenAI models reject `max_tokens` and require
            # `max_completion_tokens`; older ones reject the reverse. Which is
            # current changes under us, so swap once on the 400 that says so
            # rather than pinning a guess.
            if (exc.code == 400 and "max_completion_tokens" in detail
                    and token_field == "max_tokens"):
                return _openai_compatible(model, base_url, api_key, system,
                                          user, max_tokens,
                                          "max_completion_tokens", extra)
            # Symmetric case: a provider that only knows `max_tokens`.
            if (exc.code == 400 and "max_tokens" in detail
                    and token_field == "max_completion_tokens"):
                return _openai_compatible(model, base_url, api_key, system,
                                          user, max_tokens, "max_tokens", extra)
            # An unsupported knob should not lose the whole run: drop the
            # optional extras and retry once before giving up.
            if exc.code == 400 and extra and "Unsupported" in detail:
                return _openai_compatible(model, base_url, api_key, system,
                                          user, max_tokens, token_field, None)
            # 402/401/403 are account-level and identical next time.
            if exc.code in (400, 401, 402, 403, 404):
                raise last from exc
            time.sleep(BACKOFF_BASE ** attempt)
        except KeyboardInterrupt:
            raise
        except Exception as exc:
            last = exc
            time.sleep(BACKOFF_BASE ** attempt)

    raise ProviderError(f"failed after {MAX_RETRIES} attempts; last: {last}")
