"""End-to-end: sample evidence -> chunking -> embedding -> vector store -> retrieval -> (mock) Ollama -> response.

The fake LLM deliberately fabricates items; the assertions prove they never reach the saved intelligence.
"""
import re

import pytest
from fakes import FakeLLM

from ai.analyzer import answer_question, run_quick_action
from config import load_settings
from models.source import Evidence
from rag.retriever import Retriever
from rag.vector_store import get_default_store
from research.pipeline import STEPS, load_sample_company, reindex_company, run_research
from research.website_parser import FetchResult
from storage.json_store import CompanyStore


@pytest.fixture
def settings():
    s = load_settings()
    s.top_k, s.max_context_chars = 12, 60000          # let the tiny sample fit entirely in one prompt
    return s


@pytest.fixture
def sample(settings):
    events = []
    result = load_sample_company(settings=settings, llm=FakeLLM(), progress=lambda k, s, d: events.append((k, s, d)))
    assert result.ok, result.error
    return result, events, settings


def referenced_ids(intel):
    ids = set()
    ids |= set(intel.company.source_ids)
    for group in (intel.products, intel.technologies, intel.competitors, intel.funding.events, intel.signals, intel.people):
        for item in group:
            ids |= set(item.source_ids)
    ids |= set(intel.funding.total_funding_source_ids)
    if intel.gtm:
        ids |= set(intel.gtm.source_ids) | {s for c in intel.gtm.verified_evidence for s in c.source_ids}
    return ids


def test_pipeline_saves_everything(sample):
    result, _, settings = sample
    store = CompanyStore(settings.companies_dir)
    d = store.company_dir(result.slug)
    assert {p.name for p in d.iterdir()} == {"company.json", "evidence.json", "sources.json", "report.md"}
    assert len(store.load_sources(result.slug)) == 9 == len(store.load_evidence(result.slug))
    assert get_default_store().company_exists(result.slug)


def test_every_referenced_source_id_exists(sample):
    result, _, _ = sample
    known = {s.source_id for s in result.sources}
    refs = referenced_ids(result.intel)
    assert refs and refs <= known, f"dangling citations: {refs - known}"


def test_hallucinations_are_dropped(sample):
    i = sample[0].intel
    assert {p.product_name for p in i.products} == {"Northwind Eligibility API", "Northwind Claims Gateway"}
    assert all(p.pricing is None for p in i.products)                         # "$99/month" was never stated
    assert {t.name for t in i.technologies} >= {"AWS", "FHIR", "Kubernetes"} and "Snowflake" not in {t.name for t in i.technologies}
    assert "Globex Medical" not in {c.name for c in i.competitors}
    assert [(e.round_type, e.amount) for e in i.funding.events] == [("Series B", "$20 million")]   # Series C / $500M dropped
    assert "Imaginary Capital" not in i.funding.events[0].investors and i.funding.events[0].date == "2025"
    assert i.funding.total_funding == "$32 million"
    assert all("MedCo" not in s.description for s in i.signals) and len(i.signals) == 1
    assert {p.name for p in i.people} == {"Jane Example", "Raj Sample"}       # John Doe + Maria's wrong role dropped
    assert not any("five hundred" in c.statement for c in i.gtm.verified_evidence)


def test_bogus_citations_are_repaired_from_the_evidence(sample):
    i = sample[0].intel
    gateway = next(p for p in i.products if p.product_name == "Northwind Claims Gateway")
    assert gateway.source_ids and "SRC-099" not in gateway.source_ids


def test_verified_vs_inferred_technology(sample):
    tech = {t.name: t for t in sample[0].intel.technologies}
    assert tech["AWS"].confidence == "Verified" and tech["FHIR"].confidence == "Verified"     # official documentation
    assert tech["Kubernetes"].confidence == "Inferred" and tech["Kubernetes"].reason            # only a job posting
    assert any(t.confidence == "Verified" for t in tech.values())


def test_competitor_labels(sample):
    comps = {c.name: c for c in sample[0].intel.competitors}
    assert comps["Contoso Claims Exchange"].confidence == "Verified"         # explicit 'alternative to' statement
    assert "Globex Medical" not in comps
    # (the partner-only case is tested deterministically in test_modules.py, independent of retrieval ranking)
    assert all(c.confidence == "Inferred" for n, c in comps.items() if n != "Contoso Claims Exchange")


def test_gtm_confidence_is_derived_not_trusted(sample):
    g = sample[0].intel.gtm
    assert g.confidence != "High" and len(g.verified_evidence) == 2           # the model claimed "High"
    assert g.icp and g.why_this_company and g.source_ids


def test_progress_is_real_and_ordered(sample):
    _, events, _ = sample
    finished = [(k, s) for k, s, _ in events if s != "running"]
    assert [k for k, _ in finished] == [k for k, _ in STEPS]
    assert dict(finished)["save"] == "done" and dict(finished)["index"] in ("done", "warning")
    order = [k for k, s, _ in events if s == "running"]
    assert order.index("index") < order.index("products")                    # evidence is indexed BEFORE analysis


def test_chat_over_indexed_sample(sample):
    result, _, settings = sample
    store = CompanyStore(settings.companies_dir)
    sources = {s.source_id: s for s in store.load_sources(result.slug)}
    llm = FakeLLM()
    ans = answer_question("How much funding did the company raise in its Series B round?", company_slug=result.slug, company_name=result.intel.company.name,
                          sources=sources, retriever=Retriever(settings=settings), llm=llm, settings=settings)
    assert ans.hits and "SRC-007" in [h.source_id for h in ans.hits[:2]]      # the press release ranks near the top
    assert ans.cited_ids and set(ans.cited_ids) <= set(sources)
    assert "northwind-health.example/press/series-b" in ans.text and "totally-made-up" not in ans.text
    assert "EVIDENCE" in llm.calls[0] and "[SRC-007]" in llm.calls[0]
    quick = run_quick_action("products", company_slug=result.slug, company_name="N", sources=sources,
                             retriever=Retriever(settings=settings), llm=FakeLLM(), settings=settings)
    assert quick.hits


def test_report_has_all_sections_and_marks_inference(sample):
    md = sample[0].report_md
    for n, title in enumerate(["Executive Summary", "Company Overview", "Products & Services", "Technology", "Target Customers",
                               "Competitors", "Funding", "Growth Signals", "People", "GTM Analysis", "Why This Company?", "Sources"], 1):
        assert f"## {n}. {title}" in md
    assert "AI inference" in md and "Inferred" in md and "Verified" in md
    assert "Maria" not in md and "John Doe" not in md and "QuantumBilling" not in md
    assert md.count("SRC-") > 10


def test_no_llm_run_still_collects_indexes_and_saves(settings):
    llm = FakeLLM()
    res = load_sample_company(settings=settings, llm=llm, run_analysis=False)
    assert res.ok and llm.calls == []
    assert all(s.status == "skipped" for k, s in res.intel.module_status.items())
    assert len(res.sources) == 9 and "AI analysis skipped" in " ".join(res.intel.warnings)
    assert "Not verified" in res.report_md and get_default_store().company_exists(res.slug)


def test_ollama_down_skips_analysis_but_completes(settings):
    res = load_sample_company(settings=settings, llm=FakeLLM(ready=False))
    assert res.ok and res.intel.products == [] and res.intel.gtm is None
    assert any("Cannot reach Ollama" in w for w in res.intel.warnings)


def test_model_returning_garbage_is_handled(settings):
    res = load_sample_company(settings=settings, llm=FakeLLM(fail_json=True))
    assert res.ok
    assert all(res.intel.module_status[k].status == "failed" for k in ("products", "competitors", "funding", "people", "gtm"))
    assert res.intel.products == [] and res.intel.funding.total_funding is None


def test_one_failing_module_does_not_stop_the_rest(settings):
    class Flaky(FakeLLM):
        def chat(self, messages, **kw):
            if "explicitly described" in messages[-1]["content"] or "products or services" in messages[-1]["content"]:
                raise RuntimeError("model crashed")
            return super().chat(messages, **kw)
    res = load_sample_company(settings=settings, llm=Flaky())
    assert res.ok and res.intel.module_status["products"].status == "failed"
    assert res.intel.people and res.intel.funding.events                       # later modules still ran


def test_indexing_failure_is_reported_and_evidence_still_saved(settings):
    class BadEmbedder:
        id, dim, is_fallback, warning = "bad", 4, False, None
        def encode(self, texts, is_query=False): raise RuntimeError("embedding exploded")
    res = load_sample_company(settings=settings, llm=FakeLLM(), embedder=BadEmbedder())
    assert res.ok and res.intel.module_status["products"].status == "skipped"
    assert any("Indexing failed" in w for w in res.intel.warnings)
    assert len(CompanyStore(settings.companies_dir).load_evidence(res.slug)) == 9


def test_reindex_company_rebuilds_vectors(settings):
    res = load_sample_company(settings=settings, llm=FakeLLM(), run_analysis=False)
    get_default_store().delete_company(res.slug)
    assert not get_default_store().company_exists(res.slug)
    assert reindex_company(res.slug, settings) == 9 and get_default_store().company_exists(res.slug)


def test_empty_company_name_is_rejected(settings):
    res = run_research("   ", settings=settings, llm=FakeLLM())
    assert not res.ok and "company name" in res.error


# ---- simulated web research (no real network) ---------------------------------------
HOME_HTML = """<html><head><title>Acme Health</title><script src="https://cdn.segment.com/a.js"></script></head><body><main>
<h1>Acme Health</h1><p>Acme Health builds claims connectivity APIs for healthcare payers and providers across the United States, helping them exchange data.</p>
<p>Our customers include billing companies, physician practices and digital health startups that want modern integrations instead of batch files.</p>
<a href="/products">Products</a></main></body></html>"""
PRODUCTS_HTML = "<body><main><h1>Acme Health products</h1><p>" + "The Acme Health Eligibility API verifies patient insurance coverage in real time for providers. " * 4 + "</p></main></body>"
NEWS_HTML = "<body><article><h1>Acme Health raises funding</h1><p>" + "Acme Health announced a new funding round to expand its healthcare API platform this year. " * 4 + "</p></article></body>"


def test_full_pipeline_with_simulated_web(settings):
    pages = {"https://acme.com": HOME_HTML, "https://acme.com/products": PRODUCTS_HTML, "https://news.example/acme": NEWS_HTML}

    def fetch(url, timeout=12, **kw):
        html = pages.get(url.rstrip("/") if url != "https://acme.com" else url)
        if html is None:
            return FetchResult(url, status=404, error="HTTP 404")
        return FetchResult(url, url, 200, html, {"server": "cloudflare"})

    def search(query, n):
        return [{"title": "Acme Health news", "href": "https://news.example/acme", "body": "Acme Health funding announcement and more details."},
                {"title": "Unrelated", "href": "https://elsewhere.example/x", "body": "Nothing about the company at all, just filler text here."},
                {"title": "Acme on LinkedIn", "href": "https://www.linkedin.com/company/acme-health", "body": "Acme Health | LinkedIn - healthcare API company with many employees."}]

    events = []
    res = run_research("Acme Health", "acme.com", "focus on funding", settings=settings, llm=FakeLLM(ready=False),
                       search_fn=search, fetch_fn=fetch, progress=lambda k, s, d: events.append((k, s)))
    assert res.ok, res.error
    urls = {s.url for s in res.sources}
    assert {"https://acme.com", "https://acme.com/products", "https://news.example/acme"} <= urls
    assert not any("elsewhere.example" in u for u in urls)                      # unrelated result filtered out
    linkedin = next(s for s in res.sources if "linkedin.com" in s.url)
    assert next(e for e in res.evidence if e.source_id == linkedin.source_id).fetched_full_page is False   # snippet only
    assert any(t.name == "Segment" and t.confidence == "Verified" for t in res.intel.technologies)        # observed in HTML
    assert ("identify", "done") in events and res.intel.company.website_confirmed
    assert {s.source_id for s in res.sources} == {e.source_id for e in res.evidence}


def test_web_research_with_everything_failing_reports_cleanly(settings):
    def dead_fetch(url, timeout=12, **kw):
        return FetchResult(url, error="ConnectionError")

    def dead_search(q, n):
        raise RuntimeError("DuckDuckGo unreachable")
    res = run_research("Ghost Corp", "ghost.example", settings=settings, llm=FakeLLM(), search_fn=dead_search, fetch_fn=dead_fetch)
    assert not res.ok and "No usable evidence" in res.error


def test_website_is_discovered_when_not_provided(settings):
    def search(q, n):
        return [{"title": "Acme Health", "href": "https://www.acmehealth.com/about", "body": "Acme Health official site"}]
    pages = {"https://www.acmehealth.com": HOME_HTML.replace("Acme Health", "Acme Health")}

    def fetch(url, timeout=12, **kw):
        return FetchResult(url, url, 200, pages[url], {}) if url in pages else FetchResult(url, status=404, error="HTTP 404")
    res = run_research("Acme Health", "", settings=settings, llm=FakeLLM(ready=False), search_fn=search, fetch_fn=fetch)
    assert res.ok and res.intel.company.official_website == "https://www.acmehealth.com"
    assert any("chosen automatically" in w for w in res.intel.warnings)
