from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import LenientModel


class Product(LenientModel):
    product_name: str
    description: Optional[str] = None
    features: list[str] = Field(default_factory=list)
    healthcare_use_case: Optional[str] = None
    target_users: list[str] = Field(default_factory=list)
    integrations: list[str] = Field(default_factory=list)
    pricing: Optional[str] = None       # only kept when publicly verified in the evidence
    source_ids: list[str] = Field(default_factory=list)


class ProductList(LenientModel):
    products: list[Product] = Field(default_factory=list)
