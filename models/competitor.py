from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from .base import LenientModel
from .source import Confidence

COMPETITOR_CATEGORIES = ["Direct", "Indirect", "Adjacent", "Similar"]


class Competitor(LenientModel):
    name: str
    website: Optional[str] = None
    category: str = "Similar"
    reason: Optional[str] = None
    overlap: Optional[str] = None
    confidence: Confidence = "Inferred"
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v):
        if isinstance(v, str):
            for cat in COMPETITOR_CATEGORIES:
                if v.strip().lower() == cat.lower():
                    return cat
        return "Similar"

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        if isinstance(v, str) and v.strip().lower().startswith("verif"):
            return "Verified"
        return "Inferred"


class CompetitorList(LenientModel):
    competitors: list[Competitor] = Field(default_factory=list)
