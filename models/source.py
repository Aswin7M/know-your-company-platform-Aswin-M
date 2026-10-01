"""Source + Evidence models (the backbone of every citation)."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .base import LenientModel

SourceType = Literal[
    "official_website", "company_blog", "press_release", "news", "funding_database",
    "documentation", "job_posting", "search_result", "other",
]
Relevance = Literal["high", "medium", "low"]
Confidence = Literal["Verified", "Inferred", "Not verified"]

# Source types written by the company itself (used to decide what may be "Verified").
OFFICIAL_SOURCE_TYPES = {"official_website", "company_blog", "press_release", "documentation"}


class Source(LenientModel):
    source_id: str
    title: str = "Untitled"
    url: str
    source_type: SourceType = "other"
    retrieved_at: str = ""
    snippet: str = ""


class Evidence(LenientModel):
    source_id: str
    title: str = "Untitled"
    url: str
    source_type: SourceType = "other"
    content: str = ""
    retrieved_at: str = ""
    relevance: Relevance = "medium"
    fetched_full_page: bool = True   # False => only a search-result snippet was available
    extra: dict = Field(default_factory=dict)

    def to_source(self, snippet_chars: int = 280) -> Source:
        snippet = " ".join(self.content.split())[:snippet_chars]
        return Source(
            source_id=self.source_id, title=self.title, url=self.url,
            source_type=self.source_type, retrieved_at=self.retrieved_at, snippet=snippet,
        )


class Claim(LenientModel):
    """A statement plus the sources that support it."""
    statement: str
    source_ids: list[str] = Field(default_factory=list)


def optional_str() -> Optional[str]:  # pragma: no cover - tiny helper for readability
    return None
