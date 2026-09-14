"""
Thin wrapper around an LLM API so the rest of the codebase never touches any
particular SDK directly. Swap providers here without touching
classify.py / draft_reply.py / escalation.py / eval/judge.py.

Supports:
  - Anthropic (default): set ANTHROPIC_API_KEY
  - Google Gemini: set GEMINI_API_KEY and LLM_PROVIDER=gemini
  - Groq (free tier, high rate limits): set GROQ_API_KEY and
    LLM_PROVIDER=groq (get a free key at https://console.groq.com)
"""
import os
import json
import time

from dotenv import load_dotenv

load_dotenv()

PROVIDER = os.environ.get("LLM_PROVIDER", "anthropic").lower()
_DEFAULT_MODELS = {
    "anthropic": "claude-sonnet-4-6",
    "gemini": "gemini-2.0-flash",
    "groq": "llama-3.3-70b-versatile",
}
MODEL = os.environ.get("LLM_MODEL", _DEFAULT_MODELS.get(PROVIDER, "claude-sonnet-4-6"))

_GEMINI_RPM = float(os.environ.get("GEMINI_REQUESTS_PER_MINUTE", "5"))
_MIN_INTERVAL = 60.0 / _GEMINI_RPM if _GEMINI_RPM > 0 else 0.0
_last_call_at = 0.0

_client = None
_temperature_supported = True


def _throttle_if_needed():
    global _last_call_at
    if PROVIDER != "gemini" or _MIN_INTERVAL <= 0:
        return
    elapsed = time.monotonic() - _last_call_at
    wait = _MIN_INTERVAL - elapsed
    if wait > 0:
        time.sleep(wait)
    _last_call_at = time.monotonic()


def _extract_retry_delay_seconds(err: Exception):
    import re
    match = re.search(r"retry_delay\s*\{\s*seconds:\s*(\d+)", str(err))
    if match:
        return float(match.group(1))
    match = re.search(r"retry in ([\d.]+)s", str(err))
    if match:
        return float(match.group(1))
    return None


def _supports_temperature() -> bool:
    return _temperature_supported


def _disable_temperature():
    global _temperature_supported
    if _temperature_supported:
        print("[llm_client] Note: installed SDK rejected the 'temperature' "
              "argument -- disabling it for all future calls.")
    _temperature_supported = False


def _get_anthropic_client():
    global _client
    if _client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError(
                "ANTHROPIC_API_KEY not set. Copy .env.example to .env and fill it in."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _call_anthropic(system: str, user: str, max_tokens: int, temperature: float) -> str:
    client = _get_anthropic_client()
    kwargs = dict(
        model=MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    if _supports_temperature():
        kwargs["temperature"] = temperature
    try:
        resp = client.messages.create(**kwargs)
    except TypeError as e:
        if "temperature" in str(e):
            _disable_temperature()
            kwargs.pop("temperature", None)
            resp = client.messages.create(**kwargs)
        else:
            raise
    return "".join(block.text for block in resp.content if block.type == "text")


def _get_gemini_client():
    global _client
    if _client is None:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Get a free key at "
                "https://aistudio.google.com/apikey and add it to .env, "
                "along with LLM_PROVIDER=gemini."
            )
        genai.configure(api_key=api_key)
        _client = genai
    return _client


def _call_gemini(system: str, user: str, max_tokens: int, temperature: float) -> str:
    genai = _get_gemini_client()
    model = genai.GenerativeModel(model_name=MODEL, system_instruction=system)
    resp = model.generate_content(
        user,
        generation_config={"max_output_tokens": max_tokens, "temperature": temperature},
    )
    return resp.text


def _get_groq_client():
    global _client
    if _client is None:
        from groq import Groq
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GROQ_API_KEY not set. Get a free key at "
                "https://console.groq.com and add it to .env, along with "
                "LLM_PROVIDER=groq."
            )
        _client = Groq(api_key=api_key)
    return _client


def _call_groq(system: str, user: str, max_tokens: int, temperature: float) -> str:
    client = _get_groq_client()
    resp = client.chat.completions.create(
        model=MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    )
    return resp.choices[0].message.content


_PROVIDER_CALLERS = {
    "anthropic": _call_anthropic,
    "gemini": _call_gemini,
    "groq": _call_groq,
}


def call_llm(system: str, user: str, max_tokens: int = 500, temperature: float = 0.0,
             retries: int = 6) -> str:
    caller = _PROVIDER_CALLERS.get(PROVIDER, _call_anthropic)
    last_err = None
    for attempt in range(retries):
        try:
            _throttle_if_needed()
            return caller(system, user, max_tokens, temperature)
        except Exception as e:  # noqa: BLE001
            last_err = e
            is_rate_limit = "429" in str(e) or "quota" in str(e).lower() or "rate limit" in str(e).lower()
            if is_rate_limit:
                delay = _extract_retry_delay_seconds(e) or 20.0
                print(f"[llm_client] Rate limited, waiting {delay:.0f}s before retry "
                      f"(attempt {attempt + 1}/{retries})...")
                time.sleep(delay + 2.0)
            else:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"LLM call failed after {retries} attempts: {last_err}")


def call_llm_json(system: str, user: str, max_tokens: int = 500,
                   temperature: float = 0.0) -> dict:
    raw = call_llm(system, user, max_tokens=max_tokens, temperature=temperature)
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in LLM output: {raw[:300]}")
    return json.loads(text[start:end + 1])