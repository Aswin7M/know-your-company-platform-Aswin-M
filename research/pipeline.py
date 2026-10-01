"""Research pipeline orchestrator.

    identify -> website -> web search -> index evidence -> (LLM) overview, products,
    technology, competitors, funding, signals, people, GTM -> save + report

Rules:
* Python does the research (search/fetch/clean). The LLM only analyses stored evidence.
* A step is reported complete only after it really succeeded; failures are reported
  as warnings and the pipeline continues wherever possible.
* All external dependencies are injectable so the whole flow runs offline in tests.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ai.ollama import OllamaClient
from config import PROJECT_ROOT, Settings, load_settings
from models.company import Company, CompanyIntelligence, ModuleStatus
from models.source import Evidence, Source
from models.technology import Technology
from rag.chunker import chunk_all
from rag.embeddings import Embedder, get_embedder
from rag.retriever import Retriever
from rag.sources import SourceRegistry, now_iso
from rag.vector_store import VectorStore, get_default_store
from research import (competitors, funding, gtm, people, products, signals, technology)
from research.common import ResearchContext, mentions, norm
from research.company import extract_company, identify_company
from research.web_search import (can_fetch_domain, classify_source_type, run_company_searches)
from research.website_parser import (MIN_USEFUL_CHARS, clean_html, dedupe_paragraphs, fetch_page, redact_pii,
                                     registered_domain, research_website)
from storage.json_store import CompanyStore, slugify
from storage.report import build_report

log = logging.getLogger(__name__)
SAMPLE_DIR = PROJECT_ROOT / "examples" / "sample_company"

# (key, label) in the order they really run.
STEPS = [
    ("identify", "Company identification"),
    ("website", "Official website"),
    ("search", "Web search & public sources"),
    ("index", "Evidence indexing"),
    ("company", "Company overview"),
    ("products", "Products"),
    ("technology", "Technology"),
    ("competitors", "Competitors"),
    ("funding", "Funding"),
    ("signals", "Growth signals"),
    ("people", "People"),
    ("gtm", "GTM analysis"),
    ("save", "Saving report"),
]
ANALYSIS_KEYS = ["company", "products", "technology", "competitors", "funding", "signals", "people", "gtm"]
Progress = Callable[[str, str, str], None]        # (step_key, status, detail)
# status: running | done | warning | failed | skipped


@dataclass
class ResearchResult:
    ok: bool
    slug: str = ""
    intel: Optional[CompanyIntelligence] = None
    evidence: list[Evidence] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)
    report_md: str = ""
    error: Optional[str] = None


def _relevance(name: str, title: str, text: str, official: bool) -> str:
    if official:
        return "high"
    n = norm(name)
    mentions_count = norm(text).count(n) if n else 0
    if mentions(name, norm(title)) and mentions_count >= 3:
        return "high"
    return "medium" if mentions_count >= 1 else "low"


class _Collector:
    """Turns raw pages into cleaned, de-duplicated Evidence with stable source ids."""

    def __init__(self, company_name: str):
        self.name = company_name
        self.registry = SourceRegistry()
        self.evidence: list[Evidence] = []
        self._seen: set[str] = set()

    def add(self, *, title: str, url: str, source_type: str, text: str, official: bool = False,
            full_page: bool = True, min_chars: int = 80) -> Optional[Evidence]:
        if self.registry.find_by_url(url):
            return None
        text = dedupe_paragraphs(redact_pii(text), self._seen).strip()
        if len(text) < min_chars:
            return None
        src = self.registry.add(title=title or url, url=url, source_type=source_type, snippet=text)
        ev = Evidence(source_id=src.source_id, title=src.title, url=url, source_type=source_type,  # type: ignore[arg-type]
                      content=text, retrieved_at=src.retrieved_at,
                      relevance=_relevance(self.name, title, text, official),  # type: ignore[arg-type]
                      fetched_full_page=full_page)
        self.evidence.append(ev)
        return ev


def index_evidence(evidence: list[Evidence], company: str, slug: str, settings: Settings,
                   store: Optional[VectorStore] = None, embedder: Optional[Embedder] = None) -> tuple[int, Embedder]:
    """Chunk -> embed -> store (replacing any previous vectors for this company)."""
    embedder = embedder or get_embedder()
    store = store or get_default_store()
    chunks = chunk_all(evidence, company, slug, settings.chunk_size, settings.chunk_overlap)
    if not chunks:
        raise ValueError("No text chunks were produced from the evidence.")
    store.delete_company(slug)
    vectors: list[list[float]] = []
    for i in range(0, len(chunks), 32):
        vectors.extend(embedder.encode([c.text for c in chunks[i:i + 32]], is_query=False))
    store.add_documents(chunks, vectors)
    return len(chunks), embedder


def load_sample_evidence() -> tuple[str, str, list[dict]]:
    data = json.loads((SAMPLE_DIR / "evidence.json").read_text(encoding="utf-8"))
    return data["company"], data["website"], data["documents"]


def run_research(company_name: str, website: str = "", instructions: str = "", *,
                 settings: Optional[Settings] = None, run_analysis: bool = True,
                 progress: Optional[Progress] = None,
                 search_fn: Optional[Callable] = None, fetch_fn: Callable = fetch_page,
                 llm: Optional[OllamaClient] = None, vector_store: Optional[VectorStore] = None,
                 embedder: Optional[Embedder] = None, company_store: Optional[CompanyStore] = None,
                 evidence_override: Optional[list[dict]] = None) -> ResearchResult:
    settings = settings or load_settings()
    name = (company_name or "").strip()
    slug = slugify(name)
    if not slug:
        return ResearchResult(ok=False, error="Please enter a company name.")

    def emit(key: str, status: str, detail: str = "") -> None:
        if progress:
            try:
                progress(key, status, detail)
            except Exception:                      # a UI callback must never break research
                log.exception("progress callback failed")

    store_files = company_store or CompanyStore(settings.companies_dir)
    collector = _Collector(name)
    warnings: list[str] = []
    statuses: dict[str, ModuleStatus] = {}
    observed_tech: list[Technology] = []
    website_url: Optional[str] = None
    website_confirmed = False

    # ------------------------------------------------------------------ 1-3 collect evidence
    if evidence_override is not None:
        for key in ("identify", "website", "search"):
            emit(key, "skipped", "Using bundled sample evidence (no web access).")
        website_url = (website or "").strip() or None
        for doc in evidence_override:
            collector.add(title=doc["title"], url=doc["url"], source_type=doc.get("source_type", "other"),
                          text=doc["content"], official=doc.get("source_type") == "official_website")
    else:
        emit("identify", "running")
        ident = identify_company(name, website, fetch_fn=fetch_fn, search_fn=search_fn, timeout=settings.request_timeout)
        website_url, website_confirmed = ident.website, ident.confirmed
        warnings += ident.notes
        emit("identify", "done" if ident.confirmed else "warning",
             ident.notes[-1] if ident.notes else f"Website confirmed: {ident.website}")

        official_domain = registered_domain(website_url.split("//")[-1].split("/")[0]) if website_url else None
        emit("website", "running")
        if website_url:
            site = research_website(website_url, settings.max_website_pages, settings.request_timeout, fetch=fetch_fn)
            for page in site.pages:
                collector.add(title=page.title, url=page.url, source_type=page.source_type_hint,
                              text=page.text, official=True, min_chars=MIN_USEFUL_CHARS)
            if site.tech and site.tech_page_url:
                lines = [f"Observed technology: {t.name} ({t.category}) - {t.reason}" for t in site.tech]
                ev = collector.add(title=f"Technology signals observed on {site.tech_page_url}",
                                   url=site.tech_page_url + ("&" if "?" in site.tech_page_url else "?") + "kyc=tech-signals",
                                   source_type="official_website", text="\n".join(lines), official=True, min_chars=20)
                if ev:
                    observed_tech = [Technology(category=t.category, name=t.name, confidence="Verified",
                                                reason=t.reason, source_ids=[ev.source_id]) for t in site.tech]
            n_site = len(site.pages)
            for err in site.errors[:5]:
                warnings.append(f"Website: {err}")
            emit("website", "done" if n_site else "warning",
                 f"{n_site} page(s) read from {website_url}" if n_site else "No readable pages on the official website.")
        else:
            emit("website", "skipped", "No official website available.")

        emit("search", "running")
        results, search_errors = run_company_searches(
            name, max_queries=settings.max_search_queries, per_query=settings.max_results_per_query,
            search_fn=search_fn)
        results = ident.seed_results + results if ident.seed_results else results
        warnings += [f"Search: {e}" for e in search_errors[:3]]
        fetched = 0
        for r in results:
            if collector.registry.find_by_url(r.url):
                continue
            if not mentions(name, norm(f"{r.title} {r.snippet} {r.url}")):
                continue                              # unrelated result (name ambiguity)
            stype = classify_source_type(r.url, official_domain)
            text, full = "", False
            if can_fetch_domain(r.url) and fetched < settings.max_source_pages:
                fetched += 1
                page = fetch_fn(r.url, timeout=settings.request_timeout)
                if page.ok:
                    cleaned = clean_html(page.html)
                    if not cleaned.low_content and mentions(name, norm(cleaned.title + " " + cleaned.text)):
                        text, full = f"{cleaned.title}\n{cleaned.text}", True
            if not text and len(r.snippet) >= 60:     # fall back to the search snippet itself
                text = f"{r.title}\n{r.snippet}"
            if text:
                collector.add(title=r.title, url=r.url, source_type=stype, text=text, full_page=full,
                              official=stype in ("official_website", "documentation") and bool(official_domain))
        emit("search", "done" if results and not search_errors else "warning",
             f"{len(results)} result(s) found, {len(collector.evidence)} source(s) kept in total"
             + (f" ({len(search_errors)} search error(s))" if search_errors else ""))

    if not collector.evidence:
        msg = ("No usable evidence could be collected. Check your internet connection and the company name/website "
               "(web search may also be temporarily rate-limited - wait a minute and retry).")
        emit("index", "failed", msg)
        return ResearchResult(ok=False, slug=slug, error=msg)

    # ------------------------------------------------------------------ 4 index
    index_ok, embedder_id = False, ""
    retriever: Optional[Retriever] = None
    emit("index", "running")
    try:
        emb = embedder or get_embedder()
        vs = vector_store or get_default_store()
        n_chunks, emb = index_evidence(collector.evidence, name, slug, settings, vs, emb)
        retriever = Retriever(store=vs, embedder=emb, settings=settings)
        index_ok, embedder_id = True, emb.id
        if getattr(emb, "is_fallback", False) and emb.warning:
            warnings.append(emb.warning)
        emit("index", "warning" if getattr(emb, "is_fallback", False) else "done",
             f"{n_chunks} chunks from {len(collector.evidence)} sources indexed"
             + (" (fallback embedder)" if getattr(emb, "is_fallback", False) else ""))
    except Exception as exc:
        warnings.append(f"Indexing failed: {exc}")
        emit("index", "failed", str(exc))

    # ------------------------------------------------------------------ 5 analysis (LLM)
    llm = llm or OllamaClient(settings=settings)
    intel = CompanyIntelligence(
        slug=slug, company=Company(name=name, official_website=website_url, website_confirmed=website_confirmed),
        technologies=list(observed_tech), instructions=instructions.strip(), researched_at=now_iso(),
        ollama_model=llm.model, embedding_id=embedder_id,
    )
    skip_reason = ""
    if not run_analysis:
        skip_reason = "AI analysis was turned off for this run."
    elif not index_ok:
        skip_reason = "Evidence indexing failed, so AI analysis could not run."
    else:
        st = llm.status()
        if not st.ready:
            skip_reason = st.error or "Ollama is not ready."
    if skip_reason:
        warnings.append(f"AI analysis skipped: {skip_reason}")
        for key in ANALYSIS_KEYS:
            statuses[key] = ModuleStatus(status="skipped", detail=skip_reason)
            emit(key, "skipped", skip_reason)
        if observed_tech:
            statuses["technology"] = ModuleStatus(status="partial", items=len(observed_tech),
                                                  detail=f"{len(observed_tech)} observed in page HTML; AI analysis skipped.")
    else:
        ctx = ResearchContext(company_name=name, slug=slug, settings=settings, retriever=retriever,  # type: ignore[arg-type]
                              llm=llm, evidence={e.source_id: e for e in collector.evidence},
                              instructions=instructions, website=website_url)

        def run_module(key: str, fn: Callable, apply: Callable) -> None:
            emit(key, "running")
            try:
                res = fn()
            except Exception as exc:               # OllamaError, VectorStoreError, anything unexpected
                statuses[key] = ModuleStatus(status="failed", detail=str(exc)[:300])
                warnings.append(f"{dict(STEPS)[key]} failed: {str(exc)[:200]}")
                emit(key, "failed", str(exc)[:300])
                return
            statuses[key] = res.status
            if res.items is not None:
                apply(res.items)
            ui = {"ok": "done", "partial": "warning", "failed": "failed", "skipped": "skipped"}[res.status.status]
            if res.status.status == "failed":
                warnings.append(f"{dict(STEPS)[key]} failed: {res.status.detail}")
            emit(key, ui, res.status.detail)

        def set_company(c: Company) -> None:
            c.website_confirmed = website_confirmed
            intel.company = c

        run_module("company", lambda: extract_company(ctx), set_company)
        run_module("products", lambda: products.extract_products(ctx), lambda v: setattr(intel, "products", v))
        run_module("technology", lambda: technology.extract_technology(ctx, observed_tech),
                   lambda v: setattr(intel, "technologies", v))
        run_module("competitors", lambda: competitors.extract_competitors(ctx), lambda v: setattr(intel, "competitors", v))
        run_module("funding", lambda: funding.extract_funding(ctx), lambda v: setattr(intel, "funding", v))
        run_module("signals", lambda: signals.extract_signals(ctx), lambda v: setattr(intel, "signals", v))
        run_module("people", lambda: people.extract_people(ctx), lambda v: setattr(intel, "people", v))
        run_module("gtm", lambda: gtm.extract_gtm(ctx, intel), lambda v: setattr(intel, "gtm", v))

    # ------------------------------------------------------------------ 6 save
    emit("save", "running")
    intel.module_status = statuses
    intel.warnings = warnings
    sources = collector.registry.to_list()
    report = build_report(intel, sources)
    try:
        store_files.save(intel, collector.evidence, sources, report)
    except OSError as exc:
        emit("save", "failed", str(exc))
        return ResearchResult(ok=False, slug=slug, intel=intel, evidence=collector.evidence, sources=sources,
                              report_md=report, error=f"Could not save results: {exc}")
    emit("save", "done", f"Saved to data/companies/{slug}/")
    return ResearchResult(ok=True, slug=slug, intel=intel, evidence=collector.evidence, sources=sources, report_md=report)


def load_sample_company(**kwargs) -> ResearchResult:
    """Index + analyse the bundled fictional sample company (no web access needed)."""
    name, site, docs = load_sample_evidence()
    return run_research(name, site, evidence_override=docs, **kwargs)


def reindex_company(slug: str, settings: Optional[Settings] = None, company_store: Optional[CompanyStore] = None,
                    vector_store: Optional[VectorStore] = None, embedder: Optional[Embedder] = None) -> int:
    """Rebuild a company's vectors from its saved evidence (e.g. after changing the embedding model)."""
    settings = settings or load_settings()
    store_files = company_store or CompanyStore(settings.companies_dir)
    intel = store_files.load(slug)
    evidence = store_files.load_evidence(slug)
    if not intel or not evidence:
        raise ValueError("No saved evidence for this company.")
    n, emb = index_evidence(evidence, intel.company.name, slug, settings, vector_store, embedder)
    intel.embedding_id = emb.id
    store_files.save(intel, evidence, store_files.load_sources(slug), store_files.load_report(slug))
    return n
