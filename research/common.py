"""Shared context + *grounding* helpers used by every extraction module.

Architectural rule: the LLM never creates evidence. It only proposes structured
facts; Python then verifies each proposal against the stored evidence text and
drops or downgrades anything it cannot find there. Small local models are good at
formatting but prone to inventing details, so these checks are not optional.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from ai.analyzer import build_evidence_block
from ai.ollama import OllamaClient
from config import Settings
from models.company import ModuleStatus
from models.source import Evidence
from rag.retriever import Retriever
from rag.vector_store import SearchHit

_STOP = frozenset(
    "the and for with that this from are was were has have had its their they them our you your who which what "
    "will can into over also more most such than then only other any all not but use used using based provides "
    "provide offers offer including include includes across within through help helps".split()
)
_WORD_RE = re.compile(r"[a-z0-9]+")


def norm(text: str) -> str:
    """Lower-case, punctuation -> single spaces. Used for tolerant substring checks."""
    return " ".join(_WORD_RE.findall((text or "").lower()))


def mentions(key: str, text_norm: str) -> bool:
    """True if the phrase `key` occurs as whole words in already-normalised text.

    A trailing plural ("API" -> "APIs") is tolerated; prefixes and partial words are not.
    """
    k = norm(key)
    if not k or not text_norm:
        return False
    return re.search(rf"(?<![a-z0-9]){re.escape(k)}(?:s|es)?(?![a-z0-9])", text_norm) is not None


def overlap_ratio(statement: str, text_norm: str, ignore: Iterable[str] = ()) -> float:
    """Share of the statement's meaningful words that occur in the evidence text.

    `ignore` (typically the company name) is excluded: those words appear everywhere
    and would make any sentence about the company look supported.
    """
    skip = set(ignore)
    words = {w for w in _WORD_RE.findall((statement or "").lower()) if len(w) > 3 and w not in _STOP and w not in skip}
    if not words:
        return 0.0
    vocab = set(text_norm.split())
    present = sum(1 for w in words if w in vocab or w.rstrip("s") in vocab or f"{w}s" in vocab)
    return present / len(words)


def cue_near(key: str, text_norm: str, cue_re: re.Pattern, window: int = 220) -> bool:
    """Is one of the cue words within `window` characters of any occurrence of `key`?"""
    k = norm(key)
    if not k:
        return False
    for m in re.finditer(rf"(?<![a-z0-9]){re.escape(k)}(?![a-z0-9])", text_norm):
        lo, hi = max(0, m.start() - window), min(len(text_norm), m.end() + window)
        if cue_re.search(text_norm[lo:hi]):
            return True
    return False


def clean_str_list(values: Any, limit: int = 6) -> list[str]:
    out: list[str] = []
    for v in values or []:
        s = " ".join(str(v).split())
        if s and s.lower() not in {x.lower() for x in out}:
            out.append(s[:300])
        if len(out) >= limit:
            break
    return out


@dataclass
class ResearchContext:
    company_name: str
    slug: str
    settings: Settings
    retriever: Retriever
    llm: OllamaClient
    evidence: dict[str, Evidence]
    instructions: str = ""
    website: Optional[str] = None
    _norm_cache: dict[str, str] = field(default_factory=dict)

    @property
    def name_tokens(self) -> list[str]:
        return norm(self.company_name).split()

    def overlap(self, statement: str, text_norm: str) -> float:
        return overlap_ratio(statement, text_norm, ignore=self.name_tokens)

    # ---- evidence access ----
    def text_norm(self, source_id: str) -> str:
        if source_id not in self._norm_cache:
            ev = self.evidence.get(source_id)
            self._norm_cache[source_id] = norm(ev.content) if ev else ""
        return self._norm_cache[source_id]

    def text_raw(self, source_id: str) -> str:
        ev = self.evidence.get(source_id)
        return ev.content.lower() if ev else ""

    def joined_norm(self, source_ids: Iterable[str]) -> str:
        return " ".join(self.text_norm(s) for s in source_ids)

    def source_types(self, source_ids: Iterable[str]) -> set[str]:
        return {self.evidence[s].source_type for s in source_ids if s in self.evidence}

    # ---- retrieval ----
    def retrieve(self, query: str, top_k: Optional[int] = None) -> list[SearchHit]:
        return self.retriever.search(self.slug, query, top_k=top_k or self.settings.top_k)

    def block(self, hits: list[SearchHit]) -> str:
        return build_evidence_block(hits, self.settings.max_context_chars)

    # ---- grounding ----
    def valid_ids(self, ids: Iterable[str]) -> list[str]:
        out: list[str] = []
        for raw in ids or []:
            sid = str(raw).strip().upper()
            if sid in self.evidence and sid not in out:
                out.append(sid)
        return out

    def supporting_sources(self, key: str, cited: Iterable[str], pool: Iterable[str]) -> list[str]:
        """Sources that genuinely contain `key`. Prefers the model's citations, then the retrieved pool."""
        cited_valid = self.valid_ids(cited)
        good = [s for s in cited_valid if mentions(key, self.text_norm(s))]
        if good:
            return good
        return [s for s in dict.fromkeys(pool) if s in self.evidence and mentions(key, self.text_norm(s))][:3]


@dataclass
class ModuleResult:
    items: Any
    status: ModuleStatus
    warnings: list[str] = field(default_factory=list)


def hit_ids(hits: list[SearchHit]) -> list[str]:
    return list(dict.fromkeys(h.source_id for h in hits))


def finish(label: str, items: list, dropped: int, extra: str = "") -> ModuleStatus:
    """Consistent status line. 'partial' = something was found but some proposals were rejected."""
    note = f" {dropped} unsupported item(s) discarded." if dropped else ""
    if items:
        return ModuleStatus(status="partial" if dropped else "ok", items=len(items),
                            detail=f"{len(items)} {label} verified against sources.{note}{extra}")
    return ModuleStatus(status="partial", items=0,
                        detail=f"No {label} could be verified from the available sources.{note}{extra}")


def failed(detail: str) -> ModuleResult:
    return ModuleResult(items=None, status=ModuleStatus(status="failed", detail=detail))
