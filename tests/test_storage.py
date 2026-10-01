from models import Company, CompanyIntelligence, Evidence, Product, Source
from storage.json_store import CompanyStore, slugify


def _intel(name="Acme Health", slug="acme-health", when="2026-01-01T00:00:00+00:00"):
    return CompanyIntelligence(slug=slug, company=Company(name=name, official_website="https://acme.example"),
                               products=[Product(product_name="API", source_ids=["SRC-001"])], researched_at=when)


def _evidence():
    return ([Evidence(source_id="SRC-001", title="T", url="https://acme.example", source_type="official_website", content="hello")],
            [Source(source_id="SRC-001", title="T", url="https://acme.example", source_type="official_website", snippet="hello")])


def test_save_and_load_roundtrip(tmp_path):
    store = CompanyStore(tmp_path)
    ev, src = _evidence()
    d = store.save(_intel(), ev, src, "# Report")
    assert {p.name for p in d.iterdir()} == {"company.json", "evidence.json", "sources.json", "report.md"}
    loaded = store.load("acme-health")
    assert loaded.company.name == "Acme Health" and loaded.products[0].product_name == "API"
    assert store.load_evidence("acme-health")[0].content == "hello"
    assert store.load_sources("acme-health")[0].source_id == "SRC-001"
    assert store.load_report("acme-health") == "# Report"
    assert store.exists("acme-health") and not store.exists("nope")


def test_missing_and_corrupt_data_return_empty_not_errors(tmp_path):
    store = CompanyStore(tmp_path)
    assert store.load("ghost") is None and store.load_evidence("ghost") == [] and store.load_report("ghost") is None
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "company.json").write_text("{not json", encoding="utf-8")
    assert store.load("bad") is None and store.list_companies() == []


def test_list_companies_newest_first_and_delete(tmp_path):
    store = CompanyStore(tmp_path)
    ev, src = _evidence()
    store.save(_intel("Old Co", "old-co", "2025-01-01T00:00:00+00:00"), ev, src)
    store.save(_intel("New Co", "new-co", "2026-06-01T00:00:00+00:00"), ev, src)
    assert [c["slug"] for c in store.list_companies()] == ["new-co", "old-co"]
    assert store.delete("old-co") and not store.delete("old-co")
    assert [c["slug"] for c in store.list_companies()] == ["new-co"]


def test_slugify_is_safe():
    assert slugify("Stedi") == "stedi" and slugify("  Acme Health, Inc.  ") == "acme-health-inc"
    assert slugify("../../etc/passwd") == "etc-passwd" and slugify("Ünïcode Ço") == "unicode-co"
    assert slugify("!!!") == ""


def test_path_traversal_cannot_escape_root(tmp_path):
    store = CompanyStore(tmp_path / "companies")
    assert store.company_dir("../../evil").parent == tmp_path / "companies"
    assert not store.exists("")
