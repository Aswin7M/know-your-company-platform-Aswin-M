"""Shared base for LLM-facing Pydantic models.

Small local models produce sloppy JSON ("null" strings, a string where a list is
expected, numbers where text is expected). LenientModel absorbs those harmless
quirks so one bad field doesn't discard an otherwise valid extraction.
It never *invents* values: unknowns become None / [].
"""
from __future__ import annotations

from typing import Any, get_origin

from pydantic import BaseModel, ConfigDict, model_validator

_PLACEHOLDERS = {
    "", "null", "none", "n/a", "na", "unknown", "not verified", "not specified",
    "not available", "unspecified", "not mentioned", "-",
}


class LenientModel(BaseModel):
    model_config = ConfigDict(
        extra="ignore",
        str_strip_whitespace=True,
        coerce_numbers_to_str=True,
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_llm_quirks(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        out = dict(data)
        for name, field in cls.model_fields.items():
            if name not in out:
                continue
            value = out[name]
            if get_origin(field.annotation) is list:
                if value is None or (isinstance(value, str) and not value.strip()):
                    out[name] = []
                elif isinstance(value, str):
                    out[name] = [value]
            elif field.default is None and isinstance(value, str):
                if value.strip().lower() in _PLACEHOLDERS:
                    out[name] = None
        return out
