from __future__ import annotations

from typing import Optional

from pydantic import Field, field_validator

from .base import LenientModel

SIGNAL_TYPES = [
    "Funding", "Product launch", "Partnership", "Acquisition", "Hiring",
    "Geographic expansion", "Leadership change", "Customer announcement",
    "Integration", "New market", "Other",
]
IMPORTANCE = ["High", "Medium", "Low"]   # qualitative labels only - no invented numeric scores


class GrowthSignal(LenientModel):
    signal_type: str = "Other"
    date: Optional[str] = None
    description: str
    importance: str = "Medium"
    source_ids: list[str] = Field(default_factory=list)

    @field_validator("signal_type", mode="before")
    @classmethod
    def _type(cls, v):
        if isinstance(v, str):
            low = v.strip().lower()
            for t in SIGNAL_TYPES:
                if low == t.lower():
                    return t
            for t in SIGNAL_TYPES:          # e.g. "new product launch" -> "Product launch"
                if t.lower() in low:
                    return t
        return "Other"

    @field_validator("importance", mode="before")
    @classmethod
    def _imp(cls, v):
        if isinstance(v, str):
            for label in IMPORTANCE:
                if v.strip().lower() == label.lower():
                    return label
        return "Medium"


class SignalList(LenientModel):
    signals: list[GrowthSignal] = Field(default_factory=list)
