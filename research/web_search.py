"""Free web search (DuckDuckGo via the `ddgs` package) + result normalisation.

No API key. If DuckDuckGo rate-limits or is unreachable we return what we have and
report the error - the pipeline keeps going with the official website.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass
from typing import Callable, Optional
from urllib.parse import urlsplit

from rag.sources import normalize_url
from research.website_parser import registered_domain

log = logging.getLogger(__name__)

# Ordered by importance; Settings.max_search_queries decides how many run (default 10).
QUERY_TEMPLATES = [
    '"{c}" official website',
    '"{c}" products',
    '"{c}" healthcare customers',
    '"{c}" competitors alternatives',
    '"{c}" funding investors',
    '"{c}" partnerships',
    '"{c}" technology API',
    '"{c}" leadership team CEO',
    '"{c}" hiring jobs',
    '"{c}" acquisition news',
    '"{c}" healthcare',
    '"{c}" news',
]

# Never downloaded (only their search snippet may be used as evidence).
NO_FETCH_DOMAINS = {
    "linkedin.com", "facebook.com", "twitter.com", "x.com", "instagram.com", "tiktok.com",
    "youtube.com", "reddit.com", "pinterest.com", "quora.com",
}
# Aggregators / encyclopedias that are never the company's own site.
NOT_OFFICIAL_DOMAINS = NO_FETCH_DOMAINS | {
    "crunchbase.com", "pitchbook.com", "tracxn.com", "cbinsights.com", "dealroom.co", "zoominfo.com",
    "wikipedia.org", "glassdoor.com", "indeed.com", "g2.com", "capterra.com", "owler.com", "craft.co",
    "github.com", "medium.com", "bloomberg.com", "forbes.com", "techcrunch.com", "apollo.io",
}
FUNDING_DB_DOMAINS = {"crunchbase.com", "pitchbook.com", "tracxn.com", "cbinsights.com", "dealroom.co",
                      "craft.co", "owler.com", "growjo.com", "rocketreach.co"}
JOB_DOMAINS = {"greenhouse.io", "lever.co", "ashbyhq.com", "workable.com", "indeed.com", "glassdoor.com",
               "wellfound.com", "smartrecruiters.com", "bamboohr.com", "jobs.lever.co"}
PRESS_DOMAINS = {"prnewswire.com", "businesswire.com", "globenewswire.com", "einpresswire.com", "accesswire.com"}
NEWS_DOMAINS = {
    "techcrunch.com", "fiercehealthcare.com", "healthcareitnews.com", "beckershospitalreview.com",
    "reuters.com", "bloomberg.com", "forbes.com", "businessinsider.com", "statnews.com", "axios.com",
    "modernhealthcare.com", "medcitynews.com", "mobihealthnews.com", "healthcarefinancenews.com",
    "finsmes.com", "venturebeat.com", "wsj.com", "cnbc.com", "healthleadersmedia.com",
}


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    query: str = ""
    rank: int = 0


class SearchError(RuntimeError):
    pass


def _load_ddgs():
    try:
        from ddgs import DDGS          # current package name
        return DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS   # legacy name
            return DDGS
        except ImportError as exc:
            raise SearchError("Install the search package: pip install ddgs") from exc


def _ddg_text(query: str, max_results: int) -> list[dict]:
    DDGS = _load_ddgs()
    return list(DDGS().text(query, max_results=max_results) or [])


def normalize_results(raw: list[dict], query: str = "") -> list[SearchResult]:
    """Accept the different key names search libraries use and drop unusable rows."""
    out = []
    for i, r in enumerate(raw or []):
        url = (r.get("href") or r.get("url") or r.get("link") or "").strip()
        title = (r.get("title") or "").strip()
        snippet = (r.get("body") or r.get("snippet") or r.get("description") or "").strip()
        if not url.lower().startswith(("http://", "https://")):
            continue
        out.append(SearchResult(title=title or url, url=url, snippet=snippet, query=query, rank=i))
    return out


def search(query: str, max_results: int = 5,
           search_fn: Optional[Callable[[str, int], list[dict]]] = None) -> list[SearchResult]:
    """One normalised search. Raises SearchError on failure."""
    fn = search_fn or _ddg_text
    try:
        return normalize_results(fn(query, max_results), query)
    except SearchError:
        raise
    except Exception as exc:
        raise SearchError(f"Search failed for {query!r}: {type(exc).__name__}: {str(exc)[:120]}") from exc


def run_company_searches(company: str, *, max_queries: int = 10, per_query: int = 5,
                         search_fn: Optional[Callable[[str, int], list[dict]]] = None,
                         delay: float = 0.8) -> tuple[list[SearchResult], list[str]]:
    """Run a bounded set of queries; returns (deduplicated results, error messages)."""
    results: list[SearchResult] = []
    errors: list[str] = []
    seen: set[str] = set()
    consecutive_failures = 0
    for i, template in enumerate(QUERY_TEMPLATES[:max(1, max_queries)]):
        query = template.format(c=company)
        try:
            found = search(query, per_query, search_fn)
            consecutive_failures = 0
        except SearchError as exc:
            errors.append(str(exc))
            consecutive_failures += 1
            if consecutive_failures >= 3:          # probably rate-limited; stop hammering
                errors.append("Stopped searching after 3 consecutive failures.")
                break
            continue
        for r in found:
            key = normalize_url(r.url)
            if key not in seen:
                seen.add(key)
                results.append(r)
        if delay and search_fn is None and i < max_queries - 1:
            time.sleep(delay)                      # be polite to DuckDuckGo
    return results, errors


# ---- helpers used by the pipeline ---------------------------------------------------
def host_of(url: str) -> str:
    return urlsplit(url).netloc.lower().split(":")[0]


def domain_in(url: str, domains: set[str]) -> bool:
    host = host_of(url)
    return any(host == d or host.endswith("." + d) for d in domains)


def can_fetch_domain(url: str) -> bool:
    return not domain_in(url, NO_FETCH_DOMAINS)


def classify_source_type(url: str, official_domain: Optional[str] = None) -> str:
    host, path = host_of(url), urlsplit(url).path.lower()
    if official_domain and registered_domain(host) == official_domain:
        if re.search(r"/(docs?|developers?|api|documentation)(/|$)", path) or host.startswith(("docs.", "developer.", "developers.")):
            return "documentation"
        if re.search(r"/(careers?|jobs?)(/|$)", path):
            return "job_posting"
        if re.search(r"/(press|newsroom|press-releases?|news)(/|$)", path):
            return "press_release"
        if "/blog" in path or host.startswith("blog."):
            return "company_blog"
        return "official_website"
    if domain_in(url, FUNDING_DB_DOMAINS):
        return "funding_database"
    if domain_in(url, JOB_DOMAINS) or re.search(r"/(jobs?|careers?)/", path):
        return "job_posting"
    if domain_in(url, PRESS_DOMAINS):
        return "press_release"
    if domain_in(url, NEWS_DOMAINS) or "/news/" in path:
        return "news"
    return "search_result"


def pick_official_website(company: str, results: list[SearchResult]) -> Optional[str]:
    """Pick the most plausible homepage: its domain label must resemble the company name."""
    target = re.sub(r"[^a-z0-9]", "", company.lower())
    if not target:
        return None
    best, best_rank = None, 10 ** 6
    for r in results:
        if domain_in(r.url, NOT_OFFICIAL_DOMAINS):
            continue
        label = re.sub(r"[^a-z0-9]", "", registered_domain(host_of(r.url)).split(".")[0])
        if label and (label == target or target in label or (len(label) >= 4 and label in target)):
            if r.rank < best_rank:
                parts = urlsplit(r.url)
                best, best_rank = f"{parts.scheme}://{parts.netloc}", r.rank
    return best
