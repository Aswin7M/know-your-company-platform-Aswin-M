"""Local Qdrant (embedded mode - no server, no Docker, no cloud).

Design notes (fixing problems seen in the reference project):
* Vector size is taken from the active embedder - never hard-coded.
* The collection name includes the embedder id, so switching embedding models can
  never mix incompatible vectors.
* Point ids are deterministic (company + source + chunk), so re-indexing is idempotent
  and one company can never overwrite another.
* Every point carries `company_slug`; all queries filter on it.
"""
from __future__ import annotations

import atexit
import re
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance, FieldCondition, Filter, FilterSelector, MatchValue, PointStruct, VectorParams,
)

from config import load_settings
from rag.chunker import Chunk


class VectorStoreError(RuntimeError):
    """Raised with a message that is safe to show to the user."""


@dataclass
class SearchHit:
    text: str
    score: float
    company_slug: str
    company: str
    source_id: str
    url: str
    title: str
    source_type: str
    chunk_index: int


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")[:60] or "default"


def _company_filter(company_slug: str) -> Filter:
    return Filter(must=[FieldCondition(key="company_slug", match=MatchValue(value=company_slug))])


class VectorStore:
    def __init__(self, path: str | Path = ":memory:", collection_name: str = "kyc_default", dim: int = 384):
        self.path = str(path)
        self.collection_name = collection_name
        self.dim = dim
        try:
            self.client = QdrantClient(":memory:") if self.path == ":memory:" else QdrantClient(path=self.path)
        except Exception as exc:
            raise VectorStoreError(
                f"Could not open the local vector store at '{self.path}': {exc}. "
                "If another copy of the app is running, close it and try again."
            ) from exc

    # ---- collection management -------------------------------------------------
    def create_collection(self, dim: Optional[int] = None) -> None:
        dim = dim or self.dim
        try:
            existing = {c.name for c in self.client.get_collections().collections}
            if self.collection_name in existing:
                size = self.client.get_collection(self.collection_name).config.params.vectors.size
                if size != dim:
                    raise VectorStoreError(
                        f"Collection '{self.collection_name}' has {size}-dim vectors but the active "
                        f"embedding model produces {dim}-dim vectors. Re-index the company."
                    )
                return
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
            )
        except VectorStoreError:
            raise
        except Exception as exc:
            raise VectorStoreError(f"Vector store error while creating collection: {exc}") from exc

    def _exists(self) -> bool:
        return self.collection_name in {c.name for c in self.client.get_collections().collections}

    # ---- write -------------------------------------------------------------------
    def add_documents(self, chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors must have the same length")
        if not chunks:
            return 0
        self.create_collection(dim=len(vectors[0]))
        points = [
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{c.company_slug}|{c.source_id}|{c.chunk_index}")),
                vector=list(v),
                payload={
                    "text": c.text, "company_slug": c.company_slug, "company": c.company,
                    "source_id": c.source_id, "url": c.url, "title": c.title,
                    "source_type": c.source_type, "chunk_index": c.chunk_index,
                },
            )
            for c, v in zip(chunks, vectors)
        ]
        try:
            for i in range(0, len(points), 64):          # small batches keep memory flat
                self.client.upsert(collection_name=self.collection_name, points=points[i:i + 64])
        except Exception as exc:
            raise VectorStoreError(f"Vector store error while indexing: {exc}") from exc
        return len(points)

    def delete_company(self, company_slug: str) -> None:
        if not self._exists():
            return
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=FilterSelector(filter=_company_filter(company_slug)),
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector store error while deleting: {exc}") from exc

    # ---- read --------------------------------------------------------------------
    def company_exists(self, company_slug: str) -> bool:
        return self.count(company_slug) > 0

    def count(self, company_slug: Optional[str] = None) -> int:
        if not self._exists():
            return 0
        try:
            res = self.client.count(
                collection_name=self.collection_name,
                count_filter=_company_filter(company_slug) if company_slug else None,
                exact=True,
            )
            return int(res.count)
        except Exception as exc:
            raise VectorStoreError(f"Vector store error while counting: {exc}") from exc

    def search(self, query_vector: Sequence[float], company_slug: Optional[str] = None,
               top_k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
        if not self._exists():
            return []
        try:
            res = self.client.query_points(
                collection_name=self.collection_name,
                query=list(query_vector),
                query_filter=_company_filter(company_slug) if company_slug else None,
                limit=top_k,
                with_payload=True,
            )
        except Exception as exc:
            raise VectorStoreError(f"Vector store error while searching: {exc}") from exc
        hits = []
        for p in res.points:
            if p.score < min_score:
                continue
            pl = p.payload or {}
            hits.append(SearchHit(
                text=pl.get("text", ""), score=float(p.score),
                company_slug=pl.get("company_slug", ""), company=pl.get("company", ""),
                source_id=pl.get("source_id", ""), url=pl.get("url", ""), title=pl.get("title", ""),
                source_type=pl.get("source_type", "other"), chunk_index=int(pl.get("chunk_index", 0)),
            ))
        return hits

    def close(self) -> None:
        try:
            self.client.close()
        except Exception:                                 # pragma: no cover
            pass


# ---- module-level convenience API (mirrors the reference project's simple style) -------
_STORES: dict[tuple[str, str], VectorStore] = {}
_LOCK = threading.Lock()


def get_default_store() -> VectorStore:
    """Process-wide store for the active settings + embedder (created lazily)."""
    from rag.embeddings import get_embedder

    s = load_settings()
    emb = get_embedder()
    name = f"kyc_{_slug(emb.id)}"
    path = str(s.vector_dir)
    key = (path, name)
    with _LOCK:
        if key not in _STORES:
            Path(path).mkdir(parents=True, exist_ok=True)
            _STORES[key] = VectorStore(path=path, collection_name=name, dim=emb.dim)
        return _STORES[key]


def reset_default_stores() -> None:
    with _LOCK:
        for store in _STORES.values():
            store.close()
        _STORES.clear()


def create_collection() -> None:
    get_default_store().create_collection()


def add_documents(chunks: Sequence[Chunk], vectors: Sequence[Sequence[float]]) -> int:
    return get_default_store().add_documents(chunks, vectors)


def search(query_vector: Sequence[float], company_slug: Optional[str] = None,
           top_k: int = 5, min_score: float = 0.0) -> list[SearchHit]:
    return get_default_store().search(query_vector, company_slug, top_k, min_score)


def delete_company(company_slug: str) -> None:
    get_default_store().delete_company(company_slug)


def company_exists(company_slug: str) -> bool:
    return get_default_store().company_exists(company_slug)


atexit.register(reset_default_stores)   # close embedded Qdrant cleanly on interpreter exit
