"""Official-website research: polite fetching, cleaning and technology fingerprints.

Never raises on a bad page: every failure is returned as data so one broken URL
cannot take down the whole research run.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass, field
from typing import Callable, Optional
from urllib.parse import urljoin, urlsplit, urlunsplit
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

USER_AGENT = "KnowYourCompany/1.0 (local research tool; honours robots.txt)"
MAX_PAGE_BYTES = 1_500_000
MAX_TEXT_CHARS = 20_000
MIN_USEFUL_CHARS = 200

CANDIDATE_PATHS = [
    "/", "/about", "/about-us", "/products", "/solutions", "/platform", "/services", "/customers",
    "/company", "/leadership", "/team", "/pricing", "/integrations", "/partners", "/developers",
    "/docs", "/blog", "/news", "/press", "/careers",
]
# keyword -> priority (lower = fetched earlier) used to pick the best links found on the homepage
LINK_PRIORITY = [
    "product", "platform", "solution", "service", "about", "company", "customer", "leadership", "team",
    "integration", "partner", "developer", "docs", "pricing", "news", "press", "blog",
]

_MULTI_TLDS = {"co.uk", "org.uk", "com.au", "co.in", "co.jp", "com.br", "co.nz", "co.za", "com.sg"}


# ---- domains & URLs ---------------------------------------------------------------
def normalize_website(url: str) -> str:
    """'stedi.com' -> 'https://stedi.com'. Returns '' for unusable input."""
    url = (url or "").strip()
    if not url:
        return ""
    if not re.match(r"^https?://", url, re.I):
        url = "https://" + url
    parts = urlsplit(url)
    if not parts.netloc or " " in parts.netloc or "." not in parts.netloc:
        return ""
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), parts.path.rstrip("/"), "", ""))


def registered_domain(host: str) -> str:
    host = host.lower().split(":")[0]
    labels = host.split(".")
    if len(labels) <= 2:
        return host
    if ".".join(labels[-2:]) in _MULTI_TLDS:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def same_site(url_a: str, url_b: str) -> bool:
    return registered_domain(urlsplit(url_a).netloc) == registered_domain(urlsplit(url_b).netloc)


# ---- robots.txt ----------------------------------------------------------------------
_ROBOTS: dict[str, Optional[RobotFileParser]] = {}
_ROBOTS_LOCK = threading.Lock()


def clear_robots_cache() -> None:
    with _ROBOTS_LOCK:
        _ROBOTS.clear()


def robots_allowed(url: str, timeout: int = 6) -> bool:
    parts = urlsplit(url)
    origin = f"{parts.scheme}://{parts.netloc}"
    with _ROBOTS_LOCK:
        if origin not in _ROBOTS:
            rp: Optional[RobotFileParser] = None
            try:
                r = requests.get(origin + "/robots.txt", headers={"User-Agent": USER_AGENT}, timeout=timeout)
                if r.status_code == 200 and "html" not in r.headers.get("content-type", "").lower():
                    rp = RobotFileParser()
                    rp.parse(r.text.splitlines())
            except requests.RequestException:
                rp = None                       # no robots.txt reachable -> allowed
            _ROBOTS[origin] = rp
        rp = _ROBOTS[origin]
    return True if rp is None else rp.can_fetch(USER_AGENT, url)


# ---- fetching ------------------------------------------------------------------------
@dataclass
class FetchResult:
    url: str
    final_url: str = ""
    status: int = 0
    html: str = ""
    headers: dict = field(default_factory=dict)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and bool(self.html) and self.error is None


def fetch_page(url: str, timeout: int = 12, respect_robots: bool = True) -> FetchResult:
    res = FetchResult(url=url, final_url=url)
    try:
        if respect_robots and not robots_allowed(url):
            res.error = "blocked by robots.txt"
            return res
        with requests.get(url, headers={"User-Agent": USER_AGENT, "Accept": "text/html,*/*;q=0.5"},
                          timeout=timeout, stream=True, allow_redirects=True) as r:
            res.status, res.final_url = r.status_code, r.url
            res.headers = {k.lower(): v for k, v in r.headers.items()}
            if r.status_code >= 400:
                res.error = f"HTTP {r.status_code}"
                return res
            ctype = res.headers.get("content-type", "").lower()
            if ctype and "html" not in ctype and "xml" not in ctype and "text" not in ctype:
                res.error = f"unsupported content type ({ctype.split(';')[0]})"
                return res
            body, size = [], 0
            for piece in r.iter_content(65536):
                body.append(piece)
                size += len(piece)
                if size >= MAX_PAGE_BYTES:
                    break
            res.html = b"".join(body).decode(r.encoding or "utf-8", errors="replace")
    except requests.Timeout:
        res.error = "timeout"
    except requests.RequestException as exc:
        res.error = f"{type(exc).__name__}: {str(exc)[:100]}"
    except Exception as exc:                    # never let a parser/decoder bug escape
        res.error = f"unexpected error: {type(exc).__name__}"
    return res


# ---- cleaning ------------------------------------------------------------------------
_REMOVE_TAGS = ["script", "style", "noscript", "svg", "iframe", "form", "nav", "footer", "header", "aside",
                "button", "template", "canvas", "input", "select", "textarea", "dialog"]
_BOILERPLATE_RE = re.compile(
    r"cookie|consent|gdpr|onetrust|cookiebot|cc-window|popup|modal|newsletter|subscribe|breadcrumb|"
    r"skip-link|social|share-|navbar|toast|chat-widget", re.I)
_INLINE_TAGS = ["a", "span", "strong", "em", "b", "i", "small", "u", "mark", "sup", "sub", "code", "abbr", "label"]
_BOILERPLATE_LINES = {
    "learn more", "read more", "sign in", "log in", "login", "get started", "contact sales", "book a demo",
    "request a demo", "contact us", "accept all cookies", "accept all", "reject all", "privacy policy",
    "terms of service", "skip to content", "menu", "close", "back to top", "see all", "view all", "subscribe",
    "sign up", "try it free", "get a demo",
}
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_PHONE_RE = re.compile(r"(?<![\w.])(?:\+?\d{1,3}[\s.\-]?)?(?:\(\d{3}\)|\d{3})[\s.\-]\d{3}[\s.\-]\d{4}(?![\w])")


def redact_pii(text: str) -> str:
    """Privacy: never store private email addresses or phone numbers."""
    return _PHONE_RE.sub("[phone removed]", _EMAIL_RE.sub("[email removed]", text or ""))


@dataclass
class CleanedPage:
    title: str
    text: str

    @property
    def low_content(self) -> bool:
        return len(self.text) < MIN_USEFUL_CHARS


def clean_html(html: str, max_chars: int = MAX_TEXT_CHARS) -> CleanedPage:
    soup = BeautifulSoup(html or "", "html.parser")
    title = ""
    if soup.title and soup.title.string:
        title = " ".join(soup.title.string.split())[:200]

    for tag in soup(_REMOVE_TAGS):
        tag.decompose()
    doomed = soup.find_all(class_=_BOILERPLATE_RE) + soup.find_all(id=_BOILERPLATE_RE)
    for el in doomed:
        if not el.decomposed:
            el.decompose()

    root = soup.find("main") or soup.find("article") or soup.body or soup
    if len(root.get_text(" ", strip=True)) < 300:          # <main> too thin -> use the whole body
        root = soup.body or soup

    for level in range(1, 7):                              # keep headings as markdown-style markers
        for h in root.find_all(f"h{level}"):
            h.insert(0, "#" * min(level, 3) + " ")
    for li in root.find_all("li"):
        li.insert(0, "- ")
    for t in root.find_all(_INLINE_TAGS):
        t.unwrap()
    root.smooth()

    lines, seen = [], set()
    for raw in root.get_text("\n", strip=True).splitlines():
        line = " ".join(raw.split())
        if len(line) < 3 or line.lower().strip("-# ") in _BOILERPLATE_LINES:
            continue
        if not line.startswith("#") and len(line.split()) < 3:
            continue
        key = line.lower()
        if key in seen:
            continue
        seen.add(key)
        lines.append(line)
    return CleanedPage(title=title, text="\n".join(lines)[:max_chars])


def dedupe_paragraphs(text: str, seen: set[str]) -> str:
    """Drop long lines already collected from another page (repeated blurbs/boilerplate)."""
    kept = []
    for line in text.splitlines():
        key = re.sub(r"\W+", " ", line.lower()).strip()
        if len(key) > 40:
            if key in seen:
                continue
            seen.add(key)
        kept.append(line)
    return "\n".join(kept)


# ---- technology fingerprints (verified by direct observation of page HTML / headers) ----
# (category, name, where, pattern)
_TECH_SIGNATURES: list[tuple[str, str, str, str]] = [
    ("Analytics", "Google Analytics", "html", r"googletagmanager\.com/gtag|google-analytics\.com/(?:analytics|ga)\.js"),
    ("Analytics", "Google Tag Manager", "html", r"googletagmanager\.com/gtm\.js"),
    ("Analytics", "Segment", "html", r"cdn\.segment\.com"),
    ("Analytics", "Mixpanel", "html", r"cdn\.mxpnl\.com|mixpanel\.com/libs"),
    ("Analytics", "Amplitude", "html", r"cdn\.amplitude\.com"),
    ("Analytics", "Hotjar", "html", r"static\.hotjar\.com"),
    ("Analytics", "FullStory", "html", r"fullstory\.com/s/fs\.js"),
    ("Analytics", "Heap", "html", r"heap-?analytics|cdn\.heapanalytics\.com"),
    ("CRM", "HubSpot", "html", r"js\.hs-scripts\.com|js\.hsforms\.net|hs-analytics\.net"),
    ("CRM", "Salesforce (Pardot)", "html", r"pi\.pardot\.com"),
    ("Marketing", "Marketo", "html", r"munchkin\.marketo\.net"),
    ("Marketing", "LinkedIn Insight Tag", "html", r"snap\.licdn\.com"),
    ("Marketing", "Meta Pixel", "html", r"connect\.facebook\.net/[a-z_]+/fbevents\.js"),
    ("Marketing", "Intercom", "html", r"widget\.intercom\.io"),
    ("Marketing", "Drift", "html", r"js\.driftt\.com"),
    ("Frontend", "Next.js", "html", r"/_next/static|__NEXT_DATA__"),
    ("Frontend", "Nuxt", "html", r"/_nuxt/"),
    ("Frontend", "Gatsby", "html", r"___gatsby"),
    ("Frontend", "Webflow", "html", r"data-wf-site|assets\.website-files\.com"),
    ("Frontend", "WordPress", "html", r"/wp-content/|/wp-includes/"),
    ("Frontend", "Framer", "html", r"framerusercontent\.com"),
    ("Frontend", "Squarespace", "html", r"static1\.squarespace\.com"),
    ("Frontend", "Wix", "html", r"static\.parastorage\.com|wixstatic\.com"),
    ("Frontend", "Shopify", "html", r"cdn\.shopify\.com"),
    ("APIs", "Stripe.js", "html", r"js\.stripe\.com"),
    ("Security", "Google reCAPTCHA", "html", r"google\.com/recaptcha"),
    ("Security", "Cloudflare Turnstile", "html", r"challenges\.cloudflare\.com/turnstile"),
    ("Security", "OneTrust", "html", r"cdn\.cookielaw\.org|onetrust"),
    ("Infrastructure", "Cloudflare (CDN/proxy)", "header", r"^server:\s*cloudflare|^cf-ray:"),
    ("Infrastructure", "Vercel", "header", r"^server:\s*vercel|^x-vercel-id:"),
    ("Infrastructure", "Netlify", "header", r"^server:\s*netlify|^x-nf-request-id:"),
    ("Cloud", "Amazon CloudFront (CDN)", "header", r"^via:.*cloudfront|^x-amz-cf-id:"),
    ("Cloud", "Fastly (CDN)", "header", r"^x-served-by:.*cache-|^x-fastly-request-id:"),
]


@dataclass
class TechSignal:
    category: str
    name: str
    reason: str


def fingerprint_technologies(html: str, headers: Optional[dict] = None) -> list[TechSignal]:
    """Technologies *directly observed* in a page's markup or response headers."""
    found: dict[str, TechSignal] = {}
    header_blob = "\n".join(f"{k}: {v}" for k, v in (headers or {}).items()).lower()
    for category, name, where, pattern in _TECH_SIGNATURES:
        if name in found:
            continue
        if where == "html":
            hit = re.search(pattern, html or "", re.I)
            reason = "Observed in the page HTML (script/asset reference)."
        else:
            hit = re.search(pattern, header_blob, re.I | re.M)
            reason = "Observed in HTTP response headers."
        if hit:
            found[name] = TechSignal(category, name, reason)
    return list(found.values())


# ---- site crawl ----------------------------------------------------------------------
@dataclass
class PageDoc:
    url: str
    title: str
    text: str
    source_type_hint: str = "official_website"


@dataclass
class WebsiteResult:
    base_url: str
    pages: list[PageDoc] = field(default_factory=list)
    tech: list[TechSignal] = field(default_factory=list)
    tech_page_url: Optional[str] = None
    errors: list[str] = field(default_factory=list)
    home_ok: bool = False


def discover_links(base_url: str, html: str, limit: int) -> list[str]:
    """Same-site links from the homepage whose URL looks like an informative page."""
    soup = BeautifulSoup(html or "", "html.parser")
    scored: dict[str, int] = {}
    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        if href.startswith(("mailto:", "tel:", "javascript:", "#")):
            continue
        full = urljoin(base_url + "/", href).split("#")[0].split("?")[0].rstrip("/")
        if not full.startswith(("http://", "https://")) or not same_site(full, base_url):
            continue
        path = urlsplit(full).path.lower()
        if not path or path == "/" or re.search(r"\.(pdf|png|jpe?g|gif|svg|zip|mp4|css|js)$", path):
            continue
        if path.count("/") > 3:
            continue
        for rank, kw in enumerate(LINK_PRIORITY):
            if kw in path:
                scored[full] = min(scored.get(full, 99), rank)
                break
    ordered = sorted(scored, key=lambda u: (scored[u], len(u)))
    return ordered[:limit]


def _page_type(url: str) -> str:
    path = urlsplit(url).path.lower()
    host = urlsplit(url).netloc.lower()
    if re.search(r"/(docs?|developers?|api|documentation)(/|$)", path) or host.startswith(("docs.", "developer.", "developers.")):
        return "documentation"
    if re.search(r"/(press|newsroom|press-releases?|news)(/|$)", path):
        return "press_release"
    if "/blog" in path:
        return "company_blog"
    if re.search(r"/(careers?|jobs?)(/|$)", path):
        return "job_posting"
    return "official_website"


def research_website(base_url: str, max_pages: int = 8, timeout: int = 12,
                     fetch: Callable[..., FetchResult] = fetch_page) -> WebsiteResult:
    """Fetch the homepage + the most informative internal pages. Never raises."""
    base_url = normalize_website(base_url)
    result = WebsiteResult(base_url=base_url)
    if not base_url:
        result.errors.append("No valid website URL.")
        return result

    home = fetch(base_url, timeout=timeout)
    queue: list[str] = []
    if home.ok:
        result.home_ok = True
        cleaned = clean_html(home.html)
        if cleaned.low_content:
            result.errors.append(f"{base_url}: very little readable text (JavaScript-heavy page?)")
        else:
            result.pages.append(PageDoc(home.final_url or base_url, cleaned.title or "Home", cleaned.text, "official_website"))
        result.tech = fingerprint_technologies(home.html, home.headers)
        result.tech_page_url = home.final_url or base_url
        queue = discover_links(base_url, home.html, limit=max_pages * 2)
    else:
        result.errors.append(f"{base_url}: {home.error or 'unreachable'}")

    # fall back to conventional paths only after the links the site itself advertises
    for path in CANDIDATE_PATHS[1:]:
        candidate = base_url + path
        if candidate not in queue:
            queue.append(candidate)
    if not home.ok and not queue:
        return result

    seen_urls = {base_url, (home.final_url or base_url).rstrip("/")}
    failures = 0
    for url in queue:
        if len(result.pages) >= max_pages or (not home.ok and failures >= 4):
            break
        if url in seen_urls:
            continue
        seen_urls.add(url)
        page = fetch(url, timeout=timeout)
        if not page.ok:
            failures += 1
            if page.error and "404" not in page.error:
                result.errors.append(f"{url}: {page.error}")
            continue
        if not home.ok and not result.home_ok:
            result.home_ok = True                   # a sub-page worked even though '/' did not
        cleaned = clean_html(page.html)
        if cleaned.low_content:
            continue
        result.pages.append(PageDoc(page.final_url or url, cleaned.title or url, cleaned.text, _page_type(url)))
        if not result.tech:
            result.tech = fingerprint_technologies(page.html, page.headers)
            result.tech_page_url = page.final_url or url
    return result
