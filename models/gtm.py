"""GTM analysis. Only `verified_evidence` is factual; everything else is AI inference."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .base import LenientModel
from .source import Claim


class GTMAnalysis(LenientModel):
    verified_evidence: list[Claim] = Field(default_factory=list)   # grounded in sources
    # ---- AI inference below ----
    icp: list[str] = Field(default_factory=list)                   # who the company appears to sell to
    target_customers: list[str] = Field(default_factory=list)
    business_needs: list[str] = Field(default_factory=list)
    pain_points: list[str] = Field(default_factory=list)
    relevant_functions: list[str] = Field(default_factory=list)
    opportunities: list[str] = Field(default_factory=list)
    why_this_company: Optional[str] = None
    outreach_angles: list[str] = Field(default_factory=list)
    confidence: Literal["Low", "Medium", "High"] = "Low"
    source_ids: list[str] = Field(default_factory=list)
