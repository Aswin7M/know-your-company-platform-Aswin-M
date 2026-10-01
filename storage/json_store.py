"""Plain-JSON persistence. No database of any kind.

data/companies/<slug>/
    company.json    structured intelligence (CompanyIntelligence)
    evidence.json   cleaned evidence documents
    sources.json    source registry (SRC ids, titles, urls)
    report.md       generated Markdown report
"""
from __future__ import annotations

import json
import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Optional

from models.company import CompanyIntelligence
from models.source import Evidence, Source


def slugify(name: str) -> str:
    """Filesystem- and URL-safe slug. Also blocks path traversal ('../x' -> 'x')."""
    text = unicodedata.normalize("NFKD", name or "").encode("ascii", "ignore").decode("ascii")
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:80]


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    os.replace(tmp, path)


class CompanyStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)

    def company_dir(self, slug: str) -> Path:
        slug = slugify(slug)
        if not slug:
            raise ValueError("Company name produces an empty slug")
        return self.root / slug

    # ---- write ----
    def save(self, intel: CompanyIntelligence, evidence: list[Evidence], sources: list[Source],
             report_md: Optional[str] = None) -> Path:
        d = self.company_dir(intel.slug)
        _atomic_write(d / "company.json", intel.model_dump_json(indent=2))
        _atomic_write(d / "evidence.json", json.dumps([e.model_dump() for e in evidence], indent=2, ensure_ascii=False))
        _atomic_write(d / "sources.json", json.dumps([s.model_dump() for s in sources], indent=2, ensure_ascii=False))
        if report_md is not None:
            _atomic_write(d / "report.md", report_md)
        return d

    def save_report(self, slug: str, report_md: str) -> Path:
        path = self.company_dir(slug) / "report.md"
        _atomic_write(path, report_md)
        return path

    # ---- read ----
    def exists(self, slug: str) -> bool:
        try:
            return (self.company_dir(slug) / "company.json").exists()
        except ValueError:
            return False

    def load(self, slug: str) -> Optional[CompanyIntelligence]:
        path = self.company_dir(slug) / "company.json"
        if not path.exists():
            return None
        try:
            return CompanyIntelligence.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    def load_evidence(self, slug: str) -> list[Evidence]:
        path = self.company_dir(slug) / "evidence.json"
        try:
            return [Evidence.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            return []

    def load_sources(self, slug: str) -> list[Source]:
        path = self.company_dir(slug) / "sources.json"
        try:
            return [Source.model_validate(x) for x in json.loads(path.read_text(encoding="utf-8"))]
        except (OSError, ValueError):
            return []

    def load_report(self, slug: str) -> Optional[str]:
        path = self.company_dir(slug) / "report.md"
        try:
            return path.read_text(encoding="utf-8")
        except OSError:
            return None

    def list_companies(self) -> list[dict]:
        """Lightweight listing (newest first) without loading evidence."""
        items = []
        if not self.root.exists():
            return items
        for d in sorted(p for p in self.root.iterdir() if p.is_dir()):
            intel = self.load(d.name)
            if intel:
                items.append({
                    "slug": intel.slug, "name": intel.company.name,
                    "headline": intel.company.headline, "researched_at": intel.researched_at,
                    "website": intel.company.official_website,
                    "has_report": (d / "report.md").exists(),
                })
        return sorted(items, key=lambda x: x["researched_at"], reverse=True)

    def delete(self, slug: str) -> bool:
        d = self.company_dir(slug)
        if d.exists():
            shutil.rmtree(d)
            return True
        return False
