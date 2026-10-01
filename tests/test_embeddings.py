import math

import pytest

from rag import embeddings
from rag.embeddings import HASH_DIM, HashEmbedder, embed, embed_query, embedding_dimension, get_embedder


def cos(a, b):
    return sum(x * y for x, y in zip(a, b))


def test_embedding_dimension_is_valid_and_consistent():
    vecs = embed(["healthcare claims API", "eligibility verification"])
    dim = embedding_dimension()
    assert dim == HASH_DIM and all(len(v) == dim for v in vecs)
    assert len(embed_query("what does it do")) == dim


def test_vectors_are_unit_length_and_deterministic():
    a = embed(["Northwind provides APIs"])[0]
    b = embed(["Northwind provides APIs"])[0]
    assert a == b
    assert math.isclose(math.sqrt(sum(x * x for x in a)), 1.0, rel_tol=1e-6)


def test_empty_text_does_not_break():
    v = embed([""])[0]
    assert len(v) == HASH_DIM and all(math.isfinite(x) for x in v)
    assert embed([]) == []


def test_related_text_scores_higher_than_unrelated():
    e = HashEmbedder()
    q = e.encode(["insurance eligibility verification api"], is_query=True)[0]
    related, unrelated = e.encode(["Real-time insurance eligibility verification API for providers",
                                   "Our office dog loves long walks in the park"])
    assert cos(q, related) > cos(q, unrelated) + 0.1
    assert cos(q, unrelated) < 0.05                      # no shared words -> ~0


def test_embedder_is_cached():
    assert get_embedder() is get_embedder()


def test_model_failure_falls_back_gracefully(monkeypatch):
    def boom(name):
        raise RuntimeError("no internet")
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", boom)
    embeddings.reset_embedder_cache()
    emb = get_embedder(model="some/model", backend="auto")
    assert emb.is_fallback and "Falling back" in emb.warning and "some/model" in emb.warning
    assert len(emb.encode(["x"])[0]) == emb.dim
    assert get_embedder(model="some/model", backend="auto") is emb       # failure is cached, not retried


def test_successful_model_is_used_when_available(monkeypatch):
    class Stub:
        id, dim, is_fallback, warning = "st-stub", 7, False, None
        def __init__(self, name): pass
        def encode(self, texts, is_query=False): return [[0.0] * 7 for _ in texts]
    monkeypatch.setattr(embeddings, "SentenceTransformerEmbedder", Stub)
    embeddings.reset_embedder_cache()
    emb = get_embedder(model="stub/model", backend="sentence-transformers")
    assert not emb.is_fallback and embedding_dimension() in (7, HASH_DIM)
