import pytest

from config import load_settings
from models.source import Evidence
from rag.chunker import chunk_all
from rag.embeddings import HashEmbedder
from rag.retriever import Retriever
from rag.vector_store import VectorStore


@pytest.fixture
def setup():
    emb = HashEmbedder()
    store = VectorStore(":memory:", "t", emb.dim)
    docs = [
        ("Products", "Acme Eligibility API returns real-time insurance eligibility for patients. " * 3),
        ("Funding", "Acme announced a Series B funding round led by Example Ventures. " * 3),
        ("Leadership", "Jane Example is the CEO of Acme and leads the leadership team. " * 3),
    ]
    ev = [Evidence(source_id=f"SRC-{i + 1:03d}", title=t, url=f"https://acme.example/{i}", source_type="official_website", content=c)
          for i, (t, c) in enumerate(docs)]
    chunks = chunk_all(ev, "Acme", "acme", 500, 50)
    store.add_documents(chunks, emb.encode([c.text for c in chunks]))
    yield Retriever(store=store, embedder=emb, settings=load_settings()), store
    store.close()


def test_search_returns_the_most_relevant_chunk_with_metadata(setup):
    retriever, _ = setup
    hits = retriever.search(company="acme", query="What products does Acme offer for insurance eligibility?", top_k=3)
    assert hits[0].title == "Products"
    assert hits[0].source_id == "SRC-001" and hits[0].url.startswith("https://acme.example")
    assert len(hits) <= 3


def test_search_accepts_company_name_and_filters_by_company(setup):
    retriever, _ = setup
    assert retriever.search("Acme", "funding round investors")[0].title == "Funding"
    assert retriever.search("Some Other Co", "funding round investors") == []


def test_blank_query_returns_nothing(setup):
    assert setup[0].search("acme", "   ") == []


def test_unrelated_query_scores_far_below_relevant_query(setup):
    retriever, _ = setup
    good = retriever.search("acme", "insurance eligibility api", min_score=0.0)[0].score
    bad = retriever.search("acme", "zebra giraffe volcano", min_score=0.0)
    # (a hashing embedder can produce a weak accidental collision, so assert a large margin instead of "nothing")
    assert not bad or bad[0].score < good * 0.5


def test_per_source_cap_and_dedupe():
    emb = HashEmbedder()
    store = VectorStore(":memory:", "t", emb.dim)
    long = "\n".join(f"eligibility api line {i} " + "word " * 40 for i in range(40))
    ev = [Evidence(source_id="SRC-001", title="Big", url="https://a.example", source_type="other", content=long),
          Evidence(source_id="SRC-002", title="Small", url="https://b.example", source_type="other", content="eligibility api overview")]
    chunks = chunk_all(ev, "Acme", "acme", 60, 5)
    store.add_documents(chunks, emb.encode([c.text for c in chunks]))
    hits = Retriever(store=store, embedder=emb, settings=load_settings()).search("acme", "eligibility api", top_k=5, max_per_source=2)
    assert sum(1 for h in hits if h.source_id == "SRC-001") <= 2
    assert any(h.source_id == "SRC-002" for h in hits)
    store.close()
