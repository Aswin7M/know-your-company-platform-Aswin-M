from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from .base import LenientModel
from .source import Confidence

TECH_CATEGORIES = [
    "Frontend", "Backend", "Cloud", "Infrastructure", "Database", "Analytics", "CRM",
    "Marketing", "AI/ML", "APIs", "Integrations", "Developer Tools", "Security", "Other",
]


def normalize_category(value: Optional[str]) -> str:
    if not value:
        return "Other"
    v = value.strip().lower().replace("_", " ")
    for cat in TECH_CATEGORIES:
        if v == cat.lower():
            return cat
    aliases = {
        "ai": "AI/ML", "ml": "AI/ML", "machine learning": "AI/ML", "api": "APIs",
        "integration": "Integrations", "devtools": "Developer Tools", "dev tools": "Developer Tools",
        "developer tooling": "Developer Tools", "front end": "Frontend", "back end": "Backend",
        "infra": "Infrastructure", "data": "Database", "marketing automation": "Marketing",
    }
    return aliases.get(v, "Other")


class Technology(LenientModel):
    category: str = "Other"
    name: str
    confidence: Confidence = "Inferred"    # Verified = observed directly; Inferred = deduced
    reason: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("category", mode="before")
    @classmethod
    def _cat(cls, v):
        return normalize_category(v if isinstance(v, str) else None)

    @field_validator("confidence", mode="before")
    @classmethod
    def _conf(cls, v):
        if isinstance(v, str):
            low = v.strip().lower()
            if low.startswith("verif"):
                return "Verified"
            if low.startswith("infer"):
                return "Inferred"
        return "Inferred"   # never upgrade silently


class TechnologyList(LenientModel):
    technologies: list[Technology] = Field(default_factory=list)
