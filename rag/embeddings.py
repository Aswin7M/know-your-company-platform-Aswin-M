"""Local embeddings.

Primary backend : sentence-transformers (default BAAI/bge-small-en-v1.5, 384-dim, ~130 MB).
Fallback backend: deterministic hashing embedder (no download, lexical only).

The fallback exists so the app degrades gracefully instead of crashing when the
model cannot be downloaded/loaded - and so unit tests run without any model.
It is clearly flagged (`is_fallback`) and the UI shows a warning when it is active.
Vector dimensions are always read from the active embedder - never hard-coded.
"""
from __future__ import annotations

import hashlib
import logging
import math
import re
import threading
from collections import Counter
from typing import Optional, Protocol

from config import load_settings

log = logging.getLogger(__name__)

HASH_DIM = 2048
_STOPWORDS = frozenset(
    "a an and are as at be by for from has have in is it its of on or that the this to was were will with".split()
)
_BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder(Protocol):
    id: str
    dim: int
    is_fallback: bool
    warning: Optional[str]

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]: ...


class HashEmbedder:
    """Feature-hashing bag of (stemmed) words + bigrams. Deterministic across processes.

    Unsigned buckets with a large dimension keep collisions rare, so text that shares no
    words with the query scores ~0. That makes a small `min_score` meaningful for this backend.
    """

    default_min_score = 0.02

    def __init__(self, dim: int = HASH_DIM, warning: Optional[str] = None):
        self.dim = dim
        self.id = f"hash-{dim}"
        self.is_fallback = True
        self.warning = warning or "Using the built-in lightweight hashing embedder (keyword-style matching)."

    @staticmethod
    def _stem(t: str) -> str:
        """Very small suffix stripper so raise/raised/raises/raising share one feature."""
        if len(t) <= 3 or t.isdigit():
            return t
        for suffix, repl in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
            if t.endswith(suffix) and len(t) - len(suffix) >= 3 and not (suffix == "s" and t.endswith("ss")):
                t = t[: len(t) - len(suffix)] + repl
                break
        return t.rstrip("e") if len(t) > 4 else t

    @classmethod
    def _tokens(cls, text: str) -> list[str]:
        return [cls._stem(t) for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in _STOPWORDS]

    def _vector(self, text: str) -> list[float]:
        tokens = self._tokens(text)
        counts = Counter(tokens + [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])])
        vec = [0.0] * self.dim
        for feat, n in counts.items():
            h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "big")
            vec[h % self.dim] += 1.0 + math.log(n)           # sub-linear term frequency
        norm = math.sqrt(sum(v * v for v in vec))
        if norm == 0:
            vec[0] = 1.0
            return vec
        return [v / norm for v in vec]

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        return [self._vector(t) for t in texts]


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str):
        from sentence_transformers import SentenceTransformer  # heavy import, keep lazy

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device="cpu")
        try:                                   # keep CPU memory/latency modest
            self._model.max_seq_length = min(int(self._model.max_seq_length or 512), 512)
        except Exception:                      # pragma: no cover
            pass
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.id = f"st-{model_name}"
        self.is_fallback = False
        self.warning = None
        self._bge = "bge" in model_name.lower() and "m3" not in model_name.lower()

    def encode(self, texts: list[str], is_query: bool = False) -> list[list[float]]:
        if is_query and self._bge:
            texts = [_BGE_QUERY_PREFIX + t for t in texts]
        arr = self._model.encode(texts, batch_size=16, normalize_embeddings=True, show_progress_bar=False)
        return arr.tolist()


_CACHE: dict[tuple[str, str], Embedder] = {}
_LOCK = threading.Lock()


def reset_embedder_cache() -> None:
    with _LOCK:
        _CACHE.clear()


def get_embedder(model: Optional[str] = None, backend: Optional[str] = None) -> Embedder:
    """Return a cached embedder. Loading happens once per (model, backend)."""
    s = load_settings()
    model = model or s.embedding_model
    backend = (backend or s.embedding_backend or "auto").lower()
    key = (backend, model)
    with _LOCK:
        if key in _CACHE:
            return _CACHE[key]
        if backend == "hash":
            emb: Embedder = HashEmbedder(warning="Embedding backend is set to 'hash' (keyword-style matching).")
        else:
            try:
                emb = SentenceTransformerEmbedder(model)
            except Exception as exc:           # ImportError, download/network error, bad model name...
                msg = (f"Could not load embedding model '{model}' ({type(exc).__name__}: {str(exc)[:160]}). "
                       "Falling back to the lightweight hashing embedder - retrieval quality will be lower. "
                       "Check your internet connection (first run downloads the model) or EMBEDDING_MODEL.")
                log.warning(msg)
                emb = HashEmbedder(warning=msg)
        _CACHE[key] = emb
        return emb


def embed(texts: list[str]) -> list[list[float]]:
    return get_embedder().encode(list(texts), is_query=False) if texts else []


def embed_query(text: str) -> list[float]:
    return get_embedder().encode([text], is_query=True)[0]


def embedding_dimension() -> int:
    return get_embedder().dim
