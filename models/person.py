"""Public professional information only. No emails, phones or personal details."""
from __future__ import annotations

from typing import Optional

from pydantic import Field

from .base import LenientModel

FUNCTIONS = [
    "CEO", "Founder", "Revenue", "Sales", "Marketing", "Growth",
    "Partnerships", "Product", "Technology", "Operations", "Other",
]


class Person(LenientModel):
    name: str
    role: str
    company: Optional[str] = None
    relevant_function: Optional[str] = None
    source_ids: list[str] = Field(default_factory=list)


class PersonList(LenientModel):
    people: list[Person] = Field(default_factory=list)
