import requests

from research import website_parser as wp
from research.website_parser import (FetchResult, clean_html, dedupe_paragraphs, discover_links, fetch_page,
                                     fingerprint_technologies, normalize_website, redact_pii, registered_domain,
                                     research_website)

HOME = """<html><head><title>Acme Health</title><script src="https://cdn.segment.com/analytics.js"></script>
<script src="https://js.hs-scripts.com/123.js"></script></head><body>
<nav><a href="/products">Products</a><a href="/about">About</a></nav>
<div class="cookie-banner">We use cookies. Accept all cookies</div>
<main><h1>Acme Health</h1>
<p>Acme builds claims connectivity APIs for healthcare payers and providers across the United States.</p>
<ul><li>Real-time eligibility checks for providers and payers</li><li>Claims status tracking and remittance</li></ul>
<p>Email press@acme.example or call 555-123-4567 to learn more about the enterprise platform we offer today.</p>
<a href="/products">Products</a><a href="/about">About us</a><a href="https://other.example/x">x</a><a href="mailto:a@b.co">m</a>
</main><form><input name="email"></form><footer>Copyright Acme 2026 All rights reserved worldwide</footer></body></html>"""


def test_clean_html_keeps_content_and_drops_boilerplate():
    page = clean_html(HOME)
    assert page.title == "Acme Health"
    assert "# Acme Health" in page.text and "claims connectivity APIs" in page.text
    assert "- Real-time eligibility checks" in page.text
    for junk in ("cookies", "Copyright", "segment.com", "Accept all"):
        assert junk not in page.text


def test_clean_html_deduplicates_lines_and_handles_garbage():
    html = "<body><p>Repeated marketing sentence for testing.</p><p>Repeated marketing sentence for testing.</p></body>"
    assert clean_html(html).text.count("Repeated marketing") == 1
    assert clean_html("").text == "" and clean_html("<<<not html>>>").low_content


def test_pii_is_redacted():
    out = redact_pii("Reach jane.doe@acme.example or (614) 555-0199 / 614-555-0199 / +1 614 555 0199.")
    assert "@" not in out and "555" not in out and "[email removed]" in out and "[phone removed]" in out
    assert redact_pii("Raised $20 million in 2025") == "Raised $20 million in 2025"     # numbers untouched


def test_dedupe_paragraphs_across_pages():
    seen = set()
    first = dedupe_paragraphs("A long repeated boilerplate sentence that appears on every page of the site.\nUnique A", seen)
    second = dedupe_paragraphs("A long repeated boilerplate sentence that appears on every page of the site.\nUnique B", seen)
    assert "boilerplate" in first and "boilerplate" not in second and "Unique B" in second


def test_technology_fingerprints():
    names = {t.name for t in fingerprint_technologies(HOME, {"Server": "cloudflare", "CF-Ray": "abc"})}
    assert {"Segment", "HubSpot", "Cloudflare (CDN/proxy)"} <= names
    assert fingerprint_technologies("<html>nothing special</html>", {}) == []


def test_url_helpers():
    assert normalize_website("stedi.com") == "https://stedi.com"
    assert normalize_website("HTTP://Www.Acme.com/path/?q=1#x") == "http://www.acme.com/path"
    assert normalize_website("not a url") == "" and normalize_website("") == "" and normalize_website("localhost") == ""
    assert registered_domain("docs.api.acme.com") == "acme.com" and registered_domain("shop.acme.co.uk") == "acme.co.uk"


def test_discover_links_same_site_and_prioritised():
    links = discover_links("https://acme.com", HOME, 10)
    assert links == ["https://acme.com/products", "https://acme.com/about"]


# ---- fetching: errors are data, never exceptions -----------------------------------
class Resp:
    def __init__(self, status=200, body=b"<html>ok</html>", ctype="text/html; charset=utf-8", url="https://a.example/"):
        self.status_code, self._body, self.headers, self.url, self.encoding = status, body, {"content-type": ctype}, url, "utf-8"

    def iter_content(self, n): yield self._body
    def __enter__(self): return self
    def __exit__(self, *a): return False


def _patch_get(monkeypatch, handler):
    monkeypatch.setattr(wp.requests, "get", handler)


def test_fetch_success(monkeypatch):
    _patch_get(monkeypatch, lambda url, **kw: Resp(404) if url.endswith("robots.txt") else Resp())
    res = fetch_page("https://a.example/")
    assert res.ok and "ok" in res.html


def test_fetch_404_timeout_connection_and_bad_content_type(monkeypatch):
    def handler(url, **kw):
        if url.endswith("robots.txt"):
            return Resp(404)
        if "missing" in url: return Resp(404)
        if "slow" in url: raise requests.Timeout()
        if "down" in url: raise requests.ConnectionError("refused")
        if "pdf" in url: return Resp(ctype="application/pdf")
        return Resp()
    _patch_get(monkeypatch, handler)
    assert fetch_page("https://a.example/missing").error == "HTTP 404"
    assert fetch_page("https://a.example/slow").error == "timeout"
    assert "ConnectionError" in fetch_page("https://a.example/down").error
    assert "unsupported content type" in fetch_page("https://a.example/pdf").error


def test_robots_txt_is_respected(monkeypatch):
    def handler(url, **kw):
        if url.endswith("robots.txt"):
            r = Resp(ctype="text/plain")
            r._body = b""
            r.text = "User-agent: *\nDisallow: /private\n"
            return r
        raise AssertionError("page must not be requested when robots.txt forbids it")
    _patch_get(monkeypatch, handler)
    res = fetch_page("https://a.example/private/page")
    assert res.error == "blocked by robots.txt" and not res.ok


def test_research_website_survives_404s_js_pages_and_failures():
    pages = {
        "https://acme.com": FetchResult("https://acme.com", "https://acme.com", 200, HOME, {"server": "cloudflare"}),
        "https://acme.com/products": FetchResult("https://acme.com/products", status=200,
                                                 html="<body><main><h1>Products</h1><p>" + "The Acme Eligibility API verifies coverage in real time for providers. " * 6 + "</p></main></body>"),
        "https://acme.com/about": FetchResult("https://acme.com/about", status=200, html="<body><div id='root'></div><script>app()</script></body>"),
        "https://acme.com/team": FetchResult("https://acme.com/team", error="timeout"),
    }

    def fake_fetch(url, timeout=12, **kw):
        return pages.get(url, FetchResult(url, status=404, error="HTTP 404"))

    result = research_website("acme.com", max_pages=5, fetch=fake_fetch)
    assert result.home_ok and {p.url for p in result.pages} == {"https://acme.com", "https://acme.com/products"}
    assert any("team" in e and "timeout" in e for e in result.errors)
    assert any("Cloudflare" in t.name for t in result.tech)


def test_research_website_unreachable_site_returns_empty_result():
    result = research_website("https://dead.example", fetch=lambda url, timeout=12, **kw: FetchResult(url, error="ConnectionError"))
    assert not result.home_ok and result.pages == [] and result.errors


def test_research_website_invalid_url():
    result = research_website("not a url")
    assert result.pages == [] and result.errors
