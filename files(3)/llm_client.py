"""
LLM client — Ollama only (fully local). Every other module calls through
here, so this is the one place that knows how to talk to Ollama.

Every function either returns a usable result or None; nothing here
raises. Callers treat None as "no verdict available" and degrade to their
own non-LLM fallback (TF-IDF, needs_human_review, etc.).

Models are read from config.py / env vars. They must be pulled first:
    ollama pull llama3.1:8b        # text (judging, key parsing fallback)
    ollama pull llava              # vision (OCR escalation)
    ollama pull nomic-embed-text   # embeddings
Run `python3 llm_client.py` to see which of these are actually present.
"""

import re
import io
import json
import time
import base64

try:
    import requests
except ImportError:
    requests = None

import config

_status_cache = {"at": 0.0, "tags": None}


def _installed_models():
    """Returns the list of pulled model names, or None if Ollama is
    unreachable. Cached briefly so a loop of 50 judge calls doesn't do 50
    health-check GETs."""
    if requests is None or config.DISABLE_OLLAMA:
        return None
    now = time.time()
    if now - _status_cache["at"] < config.OLLAMA_STATUS_CACHE_SECONDS:
        return _status_cache["tags"]
    tags = None
    try:
        r = requests.get(f"{config.OLLAMA_BASE_URL}/api/tags", timeout=config.LLM_HEALTHCHECK_TIMEOUT_SECONDS)
        if r.status_code == 200:
            tags = [m.get("name", "") for m in r.json().get("models", [])]
    except Exception:
        tags = None
    _status_cache.update(at=now, tags=tags)
    return tags


def _ollama_available() -> bool:
    return _installed_models() is not None


def _has_model(model: str) -> bool:
    tags = _installed_models() or []
    return any(t == model or t.split(":")[0] == model.split(":")[0] and ":" not in model for t in tags)


def _strip_json_fences(raw: str) -> str:
    return re.sub(r"^```(json)?|```$", "", raw.strip(), flags=re.MULTILINE).strip()


def generate_text(prompt: str, image=None, temperature: float = 0.0, timeout: int = None):
    """image: optional PIL.Image -> multimodal call using the vision model.
    Returns the response text, or None if Ollama or the needed model is
    unavailable / the call failed."""
    if not _ollama_available():
        return None
    model = config.OLLAMA_VISION_MODEL if image is not None else config.OLLAMA_TEXT_MODEL
    if not _has_model(model):
        return None
    if timeout is None:
        timeout = config.OLLAMA_VISION_TIMEOUT_SECONDS if image is not None else config.LLM_GENERATE_TIMEOUT_SECONDS
    try:
        payload = {"model": model, "prompt": prompt, "stream": False, "options": {"temperature": temperature}}
        if image is not None:
            buf = io.BytesIO()
            image.convert("RGB").save(buf, format="PNG")
            payload["images"] = [base64.b64encode(buf.getvalue()).decode("utf-8")]
        resp = requests.post(f"{config.OLLAMA_BASE_URL}/api/generate", json=payload, timeout=timeout)
        resp.raise_for_status()
        text = resp.json().get("response", "").strip()
        return text or None
    except Exception:
        return None


def generate_json(prompt: str, image=None, temperature: float = 0.0):
    """Like generate_text but parses JSON. None if generation failed OR the
    output wasn't valid JSON — callers must treat both as 'no verdict'."""
    raw = generate_text(prompt, image=image, temperature=temperature)
    if raw is None:
        return None
    try:
        return json.loads(_strip_json_fences(raw))
    except Exception:
        return None


def embed_texts(texts: list):
    """One vector per input string, same order — or None if no embedding
    backend is available (caller falls back to TF-IDF)."""
    if not texts:
        return []
    if not _ollama_available() or not _has_model(config.OLLAMA_EMBED_MODEL):
        return None
    try:
        # Newer Ollama: batched /api/embed
        resp = requests.post(
            f"{config.OLLAMA_BASE_URL}/api/embed",
            json={"model": config.OLLAMA_EMBED_MODEL, "input": texts},
            timeout=config.LLM_GENERATE_TIMEOUT_SECONDS,
        )
        if resp.status_code == 200:
            vectors = resp.json().get("embeddings")
            if vectors and len(vectors) == len(texts):
                return vectors
        # Older Ollama: one call per text via /api/embeddings
        vectors = []
        for t in texts:
            r = requests.post(
                f"{config.OLLAMA_BASE_URL}/api/embeddings",
                json={"model": config.OLLAMA_EMBED_MODEL, "prompt": t},
                timeout=config.LLM_GENERATE_TIMEOUT_SECONDS,
            )
            r.raise_for_status()
            vectors.append(r.json()["embedding"])
        return vectors
    except Exception:
        return None


def backend_status() -> dict:
    """Diagnostic: is Ollama up, and are the three models this pipeline
    needs actually pulled? Reports what's missing instead of leaving you to
    guess why a tier silently didn't run."""
    reachable = _ollama_available()
    needed = {
        "text": config.OLLAMA_TEXT_MODEL,
        "vision": config.OLLAMA_VISION_MODEL,
        "embedding": config.OLLAMA_EMBED_MODEL,
    }
    return {
        "ollama_reachable": reachable,
        "models": {role: (_has_model(m) if reachable else False) for role, m in needed.items()},
        "missing_models": [m for role, m in needed.items() if reachable and not _has_model(m)],
    }


if __name__ == "__main__":
    print(json.dumps(backend_status(), indent=2))
