"""Company profile + the aggregate record saved for each researched company."""
from __future__ import annotations

from typing import Literal, Optional

from pydantic import Field

from .base import LenientModel
from .competitor import Competitor
from .funding import FundingSummary
from .gtm import GTMAnalysis
from .person import Person
from .product import Product
from .signal import GrowthSignal
from .technology import Technology

NOT_VERIFIED = "Not verified"


class Company(LenientModel):
    name: str
    official_website: Optional[str] = None
    website_confirmed: bool = False
    industry: Optional[str] = None
    subindustry: Optional[str] = None
    company_type: Optional[str] = None
    headquarters: Optional[str] = None
    founded: Optional[str] = None
    markets: list[str] = Field(default_factory=list)
    business_model: Optional[str] = None
    target_customers: list[str] = Field(default_factory=list)
    description: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)

    def display(self, field: str) -> str:
        """Human-friendly value; unknowns are shown as 'Not verified', never guessed."""
        value = getattr(self, field, None)
        if value in (None, "", []):
            return NOT_VERIFIED
        if isinstance(value, list):
            return ", ".join(value)
        return str(value)

    @property
    def headline(self) -> str:
        parts = [p for p in (self.industry, self.subindustry) if p]
        return " / ".join(parts) if parts else "Industry not verified"


class ModuleStatus(LenientModel):
    status: Literal["ok", "partial", "failed", "skipped"] = "skipped"
    detail: str = ""
    items: int = 0


class CompanyIntelligence(LenientModel):
    """Everything we know about one researched company (saved as company.json)."""
    slug: str
    company: Company
    products: list[Product] = Field(default_factory=list)
    technologies: list[Technology] = Field(default_factory=list)
    competitors: list[Competitor] = Field(default_factory=list)
    funding: FundingSummary = Field(default_factory=FundingSummary)
    signals: list[GrowthSignal] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    gtm: Optional[GTMAnalysis] = None
    module_status: dict[str, ModuleStatus] = Field(default_factory=dict)
    instructions: str = ""
    researched_at: str = ""
    ollama_model: str = ""
    embedding_id: str = ""
    warnings: list[str] = Field(default_factory=list)
