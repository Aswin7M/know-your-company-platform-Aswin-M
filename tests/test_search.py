import pytest

from research.web_search import (QUERY_TEMPLATES, SearchError, SearchResult, classify_source_type,
                                 normalize_results, pick_official_website, run_company_searches, search)


def test_search_returns_normalized_results():
    raw = [
        {"title": "Stedi", "href": "https://www.stedi.com", "body": "Healthcare APIs"},
        {"title": "Other key names", "url": "https://x.example/a", "snippet": "snippet text"},
        {"title": "No url here", "body": "dropped"},
        {"title": "Bad scheme", "href": "ftp://x.example"},
    ]
    results = search("stedi", 5, search_fn=lambda q, n: raw)
    assert [r.url for r in results] == ["https://www.stedi.com", "https://x.example/a"]
    assert results[0].snippet == "Healthcare APIs" and results[1].snippet == "snippet text"
    assert all(isinstance(r, SearchResult) and r.query == "stedi" for r in results)


def test_search_wraps_backend_errors():
    def boom(q, n):
        raise RuntimeError("rate limited")
    with pytest.raises(SearchError):
        search("x", 3, search_fn=boom)


def test_normalize_handles_empty():
    assert normalize_results([]) == [] and normalize_results(None) == []


def test_company_searches_dedupe_and_limit():
    calls = []

    def fake(q, n):
        calls.append(q)
        return [{"title": "Same", "href": "https://a.example/page?utm_source=x", "body": "b"},
                {"title": q, "href": f"https://b.example/{len(calls)}", "body": "b"}]

    results, errors = run_company_searches("Acme", max_queries=3, per_query=5, search_fn=fake)
    assert len(calls) == 3 and errors == []
    assert sum(1 for r in results if "a.example" in r.url) == 1          # de-duplicated across queries
    assert all("Acme" in c for c in calls)


def test_company_searches_survive_failures_and_stop_after_three():
    def always_fail(q, n):
        raise RuntimeError("blocked")
    results, errors = run_company_searches("Acme", max_queries=10, search_fn=always_fail)
    assert results == [] and any("Stopped searching" in e for e in errors)


def test_company_searches_partial_failure_keeps_results():
    state = {"n": 0}

    def flaky(q, n):
        state["n"] += 1
        if state["n"] == 2:
            raise RuntimeError("blip")
        return [{"title": q, "href": f"https://s.example/{state['n']}", "body": "x"}]
    results, errors = run_company_searches("Acme", max_queries=4, search_fn=flaky)
    assert len(results) == 3 and len(errors) == 1


def test_query_templates_are_reasonable():
    assert 8 <= len(QUERY_TEMPLATES) <= 14
    assert all("{c}" in t for t in QUERY_TEMPLATES)


@pytest.mark.parametrize("url,expected", [
    ("https://www.acme.com/blog/post", "company_blog"),
    ("https://docs.acme.com/start", "documentation"),
    ("https://www.acme.com/careers/engineer", "job_posting"),
    ("https://www.acme.com/press/news", "press_release"),
    ("https://www.acme.com/about", "official_website"),
    ("https://www.crunchbase.com/organization/acme", "funding_database"),
    ("https://boards.greenhouse.io/acme/jobs/1", "job_posting"),
    ("https://www.prnewswire.com/news-releases/acme", "press_release"),
    ("https://www.fiercehealthcare.com/x", "news"),
    ("https://random.example/x", "search_result"),
])
def test_classify_source_type(url, expected):
    assert classify_source_type(url, official_domain="acme.com") == expected


def test_pick_official_website():
    results = [SearchResult("LinkedIn", "https://www.linkedin.com/company/acme", "", rank=0),
               SearchResult("Crunchbase", "https://www.crunchbase.com/acme", "", rank=1),
               SearchResult("Acme", "https://www.acme.com/about", "", rank=2)]
    assert pick_official_website("Acme", results) == "https://www.acme.com"
    assert pick_official_website("Totally Different", results) is None
