"""Source registry: stable SRC-### ids, URL de-duplication and citation helpers."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Iterable, Optional
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from models.source import Evidence, Source

SOURCE_ID_RE = re.compile(r"\bSRC-\d{3,}\b")
_TRACKING_PARAMS = {"fbclid", "gclid", "mc_cid", "mc_eid", "ref", "ref_src", "igshid"}


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalize_url(url: str) -> str:
    """Canonical form used for de-duplication (not for fetching)."""
    try:
        parts = urlsplit(url.strip())
    except ValueError:
        return url.strip()
    host = parts.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    path = parts.path.rstrip("/") or "/"
    query = [(k, v) for k, v in parse_qsl(parts.query, keep_blank_values=True)
             if not k.lower().startswith("utm_") and k.lower() not in _TRACKING_PARAMS]
    return urlunsplit((parts.scheme.lower() or "https", host, path, urlencode(query), ""))


class SourceRegistry:
    """Hands out SRC-001, SRC-002 ... and remembers what each id points to."""

    def __init__(self, sources: Optional[Iterable[Source]] = None):
        self._by_id: dict[str, Source] = {}
        self._by_url: dict[str, str] = {}
        for s in sources or []:
            self._by_id[s.source_id] = s
            self._by_url[normalize_url(s.url)] = s.source_id

    def next_id(self) -> str:
        n = len(self._by_id) + 1
        while f"SRC-{n:03d}" in self._by_id:
            n += 1
        return f"SRC-{n:03d}"

    def find_by_url(self, url: str) -> Optional[Source]:
        sid = self._by_url.get(normalize_url(url))
        return self._by_id.get(sid) if sid else None

    def add(self, *, title: str, url: str, source_type: str = "other",
            snippet: str = "", retrieved_at: Optional[str] = None) -> Source:
        existing = self.find_by_url(url)
        if existing:
            return existing
        src = Source(
            source_id=self.next_id(), title=(title or "Untitled").strip()[:200], url=url,
            source_type=source_type, retrieved_at=retrieved_at or now_iso(),
            snippet=" ".join(snippet.split())[:280],
        )
        self._by_id[src.source_id] = src
        self._by_url[normalize_url(url)] = src.source_id
        return src

    def add_evidence(self, evidence: Evidence) -> None:
        """Register a pre-built Evidence item (used when loading sample/stored data)."""
        if evidence.source_id not in self._by_id:
            src = evidence.to_source()
            self._by_id[src.source_id] = src
            self._by_url[normalize_url(src.url)] = src.source_id

    def get(self, source_id: str) -> Optional[Source]:
        return self._by_id.get(source_id)

    def ids(self) -> set[str]:
        return set(self._by_id)

    def to_list(self) -> list[Source]:
        return sorted(self._by_id.values(), key=lambda s: s.source_id)

    def __len__(self) -> int:
        return len(self._by_id)


def validate_source_ids(ids: Iterable[str], known_ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """Split ids into (valid, invalid). Order is preserved and duplicates removed."""
    known = set(known_ids)
    valid: list[str] = []
    invalid: list[str] = []
    for raw in ids or []:
        sid = str(raw).strip().upper()
        bucket = valid if sid in known else invalid
        if sid not in bucket:
            bucket.append(sid)
    return valid, invalid


def extract_cited_ids(text: str) -> list[str]:
    """SRC ids that literally appear in a piece of text, in order of first use."""
    seen: list[str] = []
    for m in SOURCE_ID_RE.findall(text or ""):
        if m not in seen:
            seen.append(m)
    return seen


def source_link(source: Source) -> str:
    return f"[{source.title}]({source.url})"


def render_sources_markdown(ids: Iterable[str], registry_like: dict[str, Source]) -> str:
    """'[1] SRC-001 - Title (link)' lines for the ids that actually exist."""
    lines = []
    for i, sid in enumerate([x for x in ids if x in registry_like], start=1):
        s = registry_like[sid]
        lines.append(f"[{i}] **{sid}** - {source_link(s)} _({s.source_type})_")
    return "\n".join(lines)
