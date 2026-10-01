"""Company identification (website confirmation) and overview extraction."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import Company, ModuleStatus
from research.common import ModuleResult, ResearchContext, failed, hit_ids, mentions, norm
from research.web_search import SearchResult, pick_official_website, search
from research.website_parser import clean_html, fetch_page, normalize_website


@dataclass
class Identification:
    website: Optional[str] = None
    confirmed: bool = False
    name_found_on_page: bool = False
    notes: list[str] = field(default_factory=list)
    seed_results: list[SearchResult] = field(default_factory=list)


def identify_company(name: str, website: str = "", *, fetch_fn: Callable = fetch_page,
                     search_fn: Optional[Callable] = None, timeout: int = 12) -> Identification:
    """Find/confirm the official website. Never raises; unknowns are reported in notes."""
    ident = Identification()
    candidate = normalize_website(website)
    if website and not candidate:
        ident.notes.append(f"'{website}' is not a valid URL; searching for the official website instead.")

    if not candidate:
        try:
            results = search(f'"{name}" official website', 6, search_fn)
            ident.seed_results = results
            candidate = pick_official_website(name, results) or ""
            if candidate:
                ident.notes.append(f"Official website chosen automatically: {candidate} (please verify).")
            else:
                ident.notes.append("Could not determine the official website. Continuing with web search only.")
        except Exception as exc:
            ident.notes.append(f"Website search failed ({exc}). Continuing without an official website.")

    if candidate:
        ident.website = candidate
        page = fetch_fn(candidate, timeout=timeout)
        if page.ok:
            ident.confirmed = True
            cleaned = clean_html(page.html)
            ident.name_found_on_page = mentions(name, norm(cleaned.title + " " + cleaned.text))
            if not ident.name_found_on_page:
                ident.notes.append(f"Website reachable, but '{name}' was not found on the home page - double-check it.")
        else:
            ident.notes.append(f"{candidate} could not be fetched ({page.error}).")
    return ident


def _year_ok(value: Optional[str], text_raw: str) -> bool:
    years = re.findall(r"\b(?:19|20)\d{2}\b", value or "")
    return bool(years) and all(y in text_raw for y in years)


def extract_company(ctx: ResearchContext) -> ModuleResult:
    hits = ctx.retrieve("company overview about us what we do mission customers founded headquarters")
    base = Company(name=ctx.company_name, official_website=ctx.website, website_confirmed=False)
    if not hits:
        return ModuleResult(base, ModuleStatus(status="partial", detail="No evidence retrieved for the overview."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="company", evidence=ctx.block(hits), schema_model=Company)
    if data is None:
        return failed("The model did not return valid JSON for the company overview.")
    data.pop("name", None)
    try:
        c = Company.model_validate({**data, "name": ctx.company_name})
    except ValidationError as exc:
        return failed(f"Company overview could not be parsed: {exc.errors()[0]['msg']}")

    pool = hit_ids(hits)
    cited = ctx.valid_ids(c.source_ids) or pool
    text_norm, text_raw = ctx.joined_norm(cited + pool), " ".join(ctx.text_raw(s) for s in cited + pool)
    dropped = []

    if c.founded and not _year_ok(c.founded, text_raw):
        dropped.append("founded"); c.founded = None
    if c.headquarters and not mentions(c.headquarters.split(",")[0], text_norm):
        dropped.append("headquarters"); c.headquarters = None
    if c.description and ctx.overlap(c.description, text_norm) < 0.35:
        dropped.append("description"); c.description = None
    c.target_customers = c.target_customers[:6]
    c.markets = c.markets[:6]
    c.source_ids = cited[:6]
    c.official_website = ctx.website
    detail = "Overview extracted from cited sources."
    if dropped:
        detail += f" Unsupported field(s) discarded: {', '.join(dropped)}."
    return ModuleResult(c, ModuleStatus(status="partial" if dropped else "ok", items=1, detail=detail))
