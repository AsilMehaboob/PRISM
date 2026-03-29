import json
import logging
import math
import os
import time
from functools import lru_cache
from typing import Any, Dict

from cachetools import TTLCache, cached

logger = logging.getLogger(__name__)

TierScores = Dict[str, float]

EMBED_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models"
EMBED_MODEL = "gemini-embedding-001"
EMBED_RETRIES = 3

CENTROIDS_CACHE_FILE = os.path.join(os.path.dirname(__file__), "centroids.json")

_TIER_ORDER = ("SCRATCH", "SESSION", "LONGTERM")

_PROTOTYPES: Dict[str, list[str]] = {
    "SCRATCH": [
        "Loading step 3 retry attempt debug traceback",
        "stderr timeout error cache miss log line",
        "temporary status update heartbeat ping",
    ],
    "SESSION": [
        "Remind me to follow up on this task this week",
        "Current sprint ticket assigned to me and next steps",
        "Short-term context for this ongoing conversation",
    ],
    "LONGTERM": [
        "My name is Alex and I always prefer Python",
        "Permanent user preference and stable personal fact",
        "Long-term profile information and enduring habits",
    ],
}

def _fallback_scores(content: str, source: str) -> TierScores:
    text = content.lower()
    scratch = 1.0 if any(x in text for x in ("debug", "traceback", "retry", "loading", "stderr")) else 0.2
    session = 1.0 if any(x in text for x in ("remind", "follow up", "ticket", "task", "this week")) else 0.3
    longterm = 1.0 if any(x in text for x in ("my name", "i always", "i prefer", "permanent", "usually")) else 0.4
    if source in {"system", "tool"}:
        scratch += 0.4
    return _softmax({"SCRATCH": scratch, "SESSION": session, "LONGTERM": longterm})

def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))

def _norm(a: list[float]) -> float:
    return math.sqrt(sum(x * x for x in a))

def _cosine(a: list[float], b: list[float]) -> float:
    na = _norm(a)
    nb = _norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return _dot(a, b) / (na * nb)

def _softmax(scores: Dict[str, float], temperature: float = 0.12) -> Dict[str, float]:
    vals = list(scores.values())
    max_val = max(vals)
    exp_vals = {k: math.exp((v - max_val) / temperature) for k, v in scores.items()}
    total = sum(exp_vals.values())
    return {k: exp_vals[k] / total for k in _TIER_ORDER}


_EMBED_CACHE = TTLCache(maxsize=1024, ttl=600)  # 10 minute expiry for PII security

@cached(_EMBED_CACHE)
def _embed(text: str) -> list[float]:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")

    try:
        import httpx
    except ImportError as exc:
        raise RuntimeError("httpx is not installed") from exc

    url = f"{EMBED_BASE_URL}/{EMBED_MODEL}:embedContent"
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    body = {
        "model": f"models/{EMBED_MODEL}",
        "content": {"parts": [{"text": text}]},
    }

    last_exc: Exception | None = None
    for attempt in range(EMBED_RETRIES):
        try:
            with httpx.Client(timeout=30.0) as client:
                resp = client.post(url, json=body, headers=headers)
                resp.raise_for_status()
            data = resp.json()
            values = data["embedding"]["values"]
            if not isinstance(values, list) or not values:
                raise ValueError("Invalid embedding response")
            return [float(v) for v in values]
        except Exception as exc:  
            last_exc = exc
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status in (429, 500, 502, 503, 504) and attempt < EMBED_RETRIES - 1:
                wait = 2 ** (attempt + 1)
                logger.warning("Stage 2 | embed retry in %ss due to %s", wait, exc)
                time.sleep(wait)
                continue
            break

    raise RuntimeError(f"Embedding call failed: {last_exc}")

@lru_cache(maxsize=1)
def _prototype_centroids() -> Dict[str, list[float]]:
    if os.path.exists(CENTROIDS_CACHE_FILE):
        try:
            with open(CENTROIDS_CACHE_FILE, "r") as f:
                data = json.load(f)
                if all(tier in data for tier in _TIER_ORDER):
                    logger.debug("Stage 2 | loaded centroids from disk cache")
                    return data
        except Exception as exc:
            logger.warning("Stage 2 | failed to load centroids disk cache: %s", exc)

    centroids: Dict[str, list[float]] = {}
    logger.info("Stage 2 | computing prototype centroids (this will hit the API)...")
    for tier in _TIER_ORDER:
        vectors = [_embed(text) for text in _PROTOTYPES[tier]]
        dim = len(vectors[0])
        centroid = [0.0] * dim
        for vec in vectors:
            for i, value in enumerate(vec):
                centroid[i] += value
        centroids[tier] = [value / len(vectors) for value in centroid]

    try:
        with open(CENTROIDS_CACHE_FILE, "w") as f:
            json.dump(centroids, f)
        logger.info("Stage 2 | saved centroids to disk cache: %s", CENTROIDS_CACHE_FILE)
    except Exception as exc:
        logger.warning("Stage 2 | failed to save centroids disk cache: %s", exc)

    return centroids

def classify(content: str, metadata: Dict[str, Any]) -> TierScores:
    """
    Classify text using real Gemini embeddings + prototype similarity.

    Returns probabilities for SCRATCH, SESSION, LONGTERM.
    Falls back to lexical scoring if embedding calls are unavailable.
    """
    source = str(metadata.get("source", "user")).lower()
    try:
        content_vec = _embed(content)
        centroids = _prototype_centroids()
        sims = {tier: _cosine(content_vec, centroids[tier]) for tier in _TIER_ORDER}
        if source in {"system", "tool"}:
            sims["SCRATCH"] += 0.02
        probs = _softmax(sims)
        logger.debug(
            "Stage 2 | backend=gemini-embed model=%s sims=%s probs=%s",
            EMBED_MODEL,
            {k: f"{v:.4f}" for k, v in sims.items()},
            {k: f"{v:.3f}" for k, v in probs.items()},
        )
        return probs
    except Exception as exc:  
        logger.warning("Stage 2 | using fallback scoring due to: %s", exc)
        return _fallback_scores(content, source)
