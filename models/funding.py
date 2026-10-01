from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import LenientModel


class FundingEvent(LenientModel):
    date: Optional[str] = None
    round_type: Optional[str] = None
    amount: Optional[str] = None
    investors: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)


class FundingSummary(LenientModel):
    events: list[FundingEvent] = Field(default_factory=list)
    # Only set when a cited source states a total explicitly. Never computed by us.
    total_funding: Optional[str] = None
    total_funding_source_ids: list[str] = Field(default_factory=list)
