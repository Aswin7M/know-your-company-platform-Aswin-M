import pytest

from models.source import Evidence
from rag.chunker import chunk_all
from rag.embeddings import HashEmbedder
from rag.vector_store import VectorStore, VectorStoreError, add_documents, company_exists, create_collection, delete_company
from rag.vector_store import search as module_search


def _index(store, emb, company, slug, docs):
    ev = [Evidence(source_id=f"SRC-{i + 1:03d}", title=t, url=f"https://{slug}.example/{i}", source_type="official_website", content=c)
          for i, (t, c) in enumerate(docs)]
    chunks = chunk_all(ev, company, slug, 200, 20)
    store.add_documents(chunks, emb.encode([c.text for c in chunks]))
    return chunks


@pytest.fixture
def store():
    emb = HashEmbedder()
    s = VectorStore(":memory:", "test_collection", emb.dim)
    yield s
    s.close()


def test_documents_can_be_inserted_and_retrieved(store):
    emb = HashEmbedder()
    _index(store, emb, "Acme", "acme", [("Products", "Acme eligibility API verifies patient insurance coverage in real time."),
                                        ("Funding", "Acme raised a Series A round from Example Ventures.")])
    hits = store.search(emb.encode(["insurance eligibility"], is_query=True)[0], "acme", top_k=2)
    assert hits and hits[0].title == "Products" and hits[0].source_id == "SRC-001"
    assert hits[0].url == "https://acme.example/0" and hits[0].company_slug == "acme"
    assert hits[0].score >= hits[-1].score


def test_company_filtering_isolates_companies(store):
    emb = HashEmbedder()
    _index(store, emb, "Acme", "acme", [("A", "eligibility verification platform for billing companies")])
    _index(store, emb, "Beta", "beta", [("B", "eligibility verification platform for hospitals")])
    q = emb.encode(["eligibility verification"], is_query=True)[0]
    assert {h.company_slug for h in store.search(q, "acme", 5)} == {"acme"}
    assert {h.company_slug for h in store.search(q, "beta", 5)} == {"beta"}
    assert {h.company_slug for h in store.search(q, None, 5)} == {"acme", "beta"}
    assert store.search(q, "ghost", 5) == []


def test_company_exists_and_delete(store):
    emb = HashEmbedder()
    assert not store.company_exists("acme")                       # nothing created yet
    _index(store, emb, "Acme", "acme", [("A", "some text about acme products and services")])
    _index(store, emb, "Beta", "beta", [("B", "some text about beta products and services")])
    assert store.company_exists("acme") and store.company_exists("beta")
    store.delete_company("acme")
    assert not store.company_exists("acme") and store.company_exists("beta")
    store.delete_company("never-existed")                          # no error


def test_reindexing_is_idempotent_and_never_overwrites_other_companies(store):
    emb = HashEmbedder()
    _index(store, emb, "Acme", "acme", [("A", "acme text one two three four")])
    _index(store, emb, "Beta", "beta", [("B", "beta text one two three four")])
    before = store.count()
    _index(store, emb, "Acme", "acme", [("A", "acme text one two three four")])
    assert store.count() == before == 2
    assert store.count("beta") == 1


def test_min_score_filters(store):
    emb = HashEmbedder()
    _index(store, emb, "Acme", "acme", [("A", "eligibility verification api")])
    q = emb.encode(["zebra giraffe"], is_query=True)[0]
    assert store.search(q, "acme", 5, min_score=0.2) == []


def test_dimension_mismatch_is_reported_clearly(store):
    store.create_collection(dim=HashEmbedder().dim)
    with pytest.raises(VectorStoreError, match="dim"):
        store.create_collection(dim=12)


def test_length_mismatch_rejected(store):
    with pytest.raises(VectorStoreError):
        store.add_documents([object()], [])


def test_module_level_api_uses_default_store():
    emb = HashEmbedder()
    chunks = chunk_all([Evidence(source_id="SRC-001", title="T", url="https://a.example", source_type="other",
                                 content="quantum widgets are made here")], "Acme", "acme", 200, 20)
    create_collection()
    add_documents(chunks, emb.encode([c.text for c in chunks]))
    assert company_exists("acme")
    assert module_search(emb.encode(["quantum widgets"], is_query=True)[0], "acme", 3)
    delete_company("acme")
    assert not company_exists("acme")
