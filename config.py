"""Central configuration for Know Your Company.

Precedence (lowest -> highest):
    built-in defaults  <  environment / .env  <  data/settings.json (Settings page)

Nothing here requires a paid API key. Everything runs locally.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")

# Keys the Settings page is allowed to persist to data/settings.json.
UI_EDITABLE = (
    "ollama_base_url",
    "ollama_model",
    "embedding_model",
    "top_k",
    "chunk_size",
    "chunk_overlap",
)


def _get(name: str, default, cast=str):
    raw = os.getenv(name)
    if raw is None or not raw.strip():
        return default
    try:
        return cast(raw.strip())
    except ValueError:
        return default


@dataclass
class Settings:
    # --- LLM (Ollama, local) ---
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "qwen3:1.7b"
    ollama_timeout: int = 300          # CPU inference is slow; be generous
    ollama_num_ctx: int = 4096         # keep small for 8 GB RAM
    ollama_think: str = "auto"         # auto | off  (qwen3 "thinking" is disabled by default)

    # --- Embeddings (local) ---
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_backend: str = "auto"    # auto | sentence-transformers | hash

    # --- RAG ---
    top_k: int = 5
    chunk_size: int = 500              # approx. tokens
    chunk_overlap: int = 60            # approx. tokens
    min_score: float = 0.10            # below this a chunk is treated as irrelevant
    max_context_chars: int = 6000      # evidence budget sent to the LLM (~1.7k tokens)

    # --- Web research limits (kept small on purpose) ---
    max_website_pages: int = 8
    max_search_queries: int = 10
    max_results_per_query: int = 5
    max_source_pages: int = 12         # search-result pages actually downloaded
    request_timeout: int = 12

    data_dir: str = str(PROJECT_ROOT / "data")

    # ---- derived paths ----
    @property
    def companies_dir(self) -> Path:
        return Path(self.data_dir) / "companies"

    @property
    def vector_dir(self) -> Path:
        return Path(self.data_dir) / "vector_store"

    @property
    def settings_file(self) -> Path:
        return Path(self.data_dir) / "settings.json"

    def validate(self) -> "Settings":
        """Clamp obviously bad values instead of crashing the app."""
        self.top_k = max(1, min(int(self.top_k), 20))
        self.chunk_size = max(100, min(int(self.chunk_size), 2000))
        self.chunk_overlap = max(0, min(int(self.chunk_overlap), self.chunk_size // 2))
        self.ollama_base_url = (self.ollama_base_url or "http://localhost:11434").rstrip("/")
        return self


def load_settings() -> Settings:
    """Build Settings from defaults, .env / environment, then data/settings.json."""
    s = Settings(
        ollama_base_url=_get("OLLAMA_BASE_URL", Settings.ollama_base_url),
        ollama_model=_get("OLLAMA_MODEL", Settings.ollama_model),
        ollama_timeout=_get("OLLAMA_TIMEOUT", Settings.ollama_timeout, int),
        ollama_num_ctx=_get("OLLAMA_NUM_CTX", Settings.ollama_num_ctx, int),
        ollama_think=_get("OLLAMA_THINK", Settings.ollama_think).lower(),
        embedding_model=_get("EMBEDDING_MODEL", Settings.embedding_model),
        embedding_backend=_get("EMBEDDING_BACKEND", Settings.embedding_backend).lower(),
        top_k=_get("TOP_K", Settings.top_k, int),
        chunk_size=_get("CHUNK_SIZE", Settings.chunk_size, int),
        chunk_overlap=_get("CHUNK_OVERLAP", Settings.chunk_overlap, int),
        min_score=_get("MIN_SCORE", Settings.min_score, float),
        max_context_chars=_get("MAX_CONTEXT_CHARS", Settings.max_context_chars, int),
        max_website_pages=_get("MAX_WEBSITE_PAGES", Settings.max_website_pages, int),
        max_search_queries=_get("MAX_SEARCH_QUERIES", Settings.max_search_queries, int),
        max_results_per_query=_get("MAX_RESULTS_PER_QUERY", Settings.max_results_per_query, int),
        max_source_pages=_get("MAX_SOURCE_PAGES", Settings.max_source_pages, int),
        request_timeout=_get("REQUEST_TIMEOUT", Settings.request_timeout, int),
        data_dir=_get("DATA_DIR", str(PROJECT_ROOT / "data")),
    )
    f = s.settings_file
    if f.exists():
        try:
            overrides = json.loads(f.read_text(encoding="utf-8"))
            for key in UI_EDITABLE:
                if key in overrides and overrides[key] not in (None, ""):
                    caster = type(getattr(s, key))
                    setattr(s, key, caster(overrides[key]))
        except (OSError, ValueError, TypeError):
            pass  # a corrupt settings file must never stop the app
    return s.validate()


def save_settings(s: Settings) -> None:
    """Persist only the UI-editable subset (never secrets - there are none)."""
    s.validate()
    s.settings_file.parent.mkdir(parents=True, exist_ok=True)
    payload = {k: getattr(s, k) for k in UI_EDITABLE}
    s.settings_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
