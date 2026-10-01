"""RAG retriever: question -> query embedding -> company-filtered Qdrant search."""
from __future__ import annotations

from typing import Optional

from config import Settings, load_settings
from rag.embeddings import Embedder, get_embedder
from rag.vector_store import SearchHit, VectorStore, get_default_store
from storage.json_store import slugify


class Retriever:
    def __init__(self, store: Optional[VectorStore] = None, embedder: Optional[Embedder] = None,
                 settings: Optional[Settings] = None):
        self._store = store
        self._embedder = embedder
        self.settings = settings or load_settings()

    @property
    def store(self) -> VectorStore:
        return self._store or get_default_store()

    @property
    def embedder(self) -> Embedder:
        return self._embedder or get_embedder()

    def search(self, company: str, query: str, top_k: Optional[int] = None,
               min_score: Optional[float] = None, max_per_source: int = 2) -> list[SearchHit]:
        """Top-k evidence chunks for one company.

        `company` may be a name or slug. Results are capped per source so a single long
        page cannot crowd out the rest of the evidence.
        """
        top_k = top_k or self.settings.top_k
        if min_score is None:
            emb = self.embedder
            min_score = getattr(emb, "default_min_score", None) if getattr(emb, "is_fallback", False) else None
            min_score = self.settings.min_score if min_score is None else min_score
        if not query.strip():
            return []
        vector = self.embedder.encode([query], is_query=True)[0]
        raw = self.store.search(vector, company_slug=slugify(company), top_k=top_k * 3, min_score=min_score)
        hits: list[SearchHit] = []
        per_source: dict[str, int] = {}
        seen_text: set[str] = set()
        for h in raw:
            key = " ".join(h.text.split())[:160]
            if key in seen_text:
                continue
            if per_source.get(h.source_id, 0) >= max_per_source:
                continue
            seen_text.add(key)
            per_source[h.source_id] = per_source.get(h.source_id, 0) + 1
            hits.append(h)
            if len(hits) >= top_k:
                break
        return hits
