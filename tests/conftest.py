import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def isolated_env(tmp_path, monkeypatch):
    """Every test gets its own data dir, the offline hash embedder and no API keys."""
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("EMBEDDING_BACKEND", "hash")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://127.0.0.1:9")   # nothing listens here
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GROQ_API_KEY", "GEMINI_API_KEY",
                "TAVILY_API_KEY", "SERPAPI_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    from rag import embeddings, vector_store
    from research import website_parser
    embeddings.reset_embedder_cache()
    vector_store.reset_default_stores()
    website_parser.clear_robots_cache()
    yield
    embeddings.reset_embedder_cache()
    vector_store.reset_default_stores()
