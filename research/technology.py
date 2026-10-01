from __future__ import annotations

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import ModuleStatus
from models.source import OFFICIAL_SOURCE_TYPES
from models.technology import Technology, TechnologyList
from research.common import ModuleResult, ResearchContext, failed, finish, hit_ids


def extract_technology(ctx: ResearchContext, observed: list[Technology]) -> ModuleResult:
    """`observed` = technologies seen directly in page HTML/headers (always 'Verified')."""
    items: list[Technology] = list(observed)
    known = {t.name.lower() for t in items}
    hits = ctx.retrieve("technology platform API integrations cloud infrastructure security developers standards")
    if not hits:
        return ModuleResult(items, ModuleStatus(status="partial", items=len(items),
                            detail=f"{len(items)} observed in page HTML; no text evidence retrieved."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="technology", evidence=ctx.block(hits), schema_model=TechnologyList)
    if data is None:
        status = ModuleStatus(status="failed" if not items else "partial", items=len(items),
                              detail="The model did not return valid JSON; showing only technologies observed in page HTML.")
        return ModuleResult(items, status)

    pool, dropped = hit_ids(hits), 0
    for raw in (data.get("technologies") or data.get("items") or [])[:12]:
        try:
            t = Technology.model_validate(raw)
        except ValidationError:
            dropped += 1
            continue
        if t.name.lower() in known:
            continue
        sids = ctx.supporting_sources(t.name, t.source_ids, pool)
        if not sids:
            dropped += 1
            continue
        t.source_ids = sids
        # Only the company's own publications can make a technology "Verified".
        if t.confidence == "Verified" and not (ctx.source_types(sids) & OFFICIAL_SOURCE_TYPES):
            t.confidence = "Inferred"
        if t.confidence == "Inferred" and not t.reason:
            t.reason = "Mentioned in third-party or indirect evidence; not confirmed by the company."
        known.add(t.name.lower())
        items.append(t)
    return ModuleResult(items, finish("technology signal(s)", items, dropped))
