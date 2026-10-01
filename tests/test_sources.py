from models.source import Evidence, Source
from rag.sources import SourceRegistry, extract_cited_ids, normalize_url, render_sources_markdown, validate_source_ids


def test_ids_are_sequential_and_urls_deduplicated():
    reg = SourceRegistry()
    a = reg.add(title="A", url="https://www.acme.com/about/?utm_source=x#top", source_type="official_website")
    b = reg.add(title="B", url="https://acme.com/about", source_type="official_website")
    c = reg.add(title="C", url="https://acme.com/products", source_type="official_website")
    assert (a.source_id, c.source_id) == ("SRC-001", "SRC-002")
    assert b is a and len(reg) == 2


def test_normalize_url():
    assert normalize_url("HTTPS://WWW.Acme.com/A/?b=1&utm_campaign=z&fbclid=q") == "https://acme.com/A?b=1"
    assert normalize_url("http://acme.com") == "http://acme.com/"


def test_source_has_required_fields():
    s = SourceRegistry().add(title="T", url="https://a.example", source_type="news", snippet="x " * 400)
    assert s.source_id and s.title and s.url and s.source_type and s.retrieved_at and len(s.snippet) <= 280


def test_validate_source_ids_flags_unknown_ids():
    valid, invalid = validate_source_ids(["SRC-001", "src-002", "SRC-099", "SRC-001"], {"SRC-001", "SRC-002"})
    assert valid == ["SRC-001", "SRC-002"] and invalid == ["SRC-099"]
    assert validate_source_ids([], {"SRC-001"}) == ([], []) and validate_source_ids(None, set()) == ([], [])


def test_extract_cited_ids():
    assert extract_cited_ids("See SRC-002 and SRC-001, also SRC-002 again; not SRC-1 or XSRC-003") == ["SRC-002", "SRC-001"]


def test_registry_restores_from_saved_sources_without_reusing_ids():
    reg = SourceRegistry([Source(source_id="SRC-001", url="https://a.example", title="A"),
                          Source(source_id="SRC-005", url="https://b.example", title="B")])
    assert reg.add(title="C", url="https://c.example").source_id == "SRC-003"
    assert reg.find_by_url("https://b.example/").source_id == "SRC-005"


def test_add_evidence_and_markdown_rendering():
    reg = SourceRegistry()
    reg.add_evidence(Evidence(source_id="SRC-007", title="Doc", url="https://d.example", source_type="documentation", content="c"))
    md = render_sources_markdown(["SRC-007", "SRC-404"], {s.source_id: s for s in reg.to_list()})
    assert "SRC-007" in md and "https://d.example" in md and "SRC-404" not in md and md.startswith("[1]")
