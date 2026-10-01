from __future__ import annotations

import re

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import ModuleStatus
from models.competitor import Competitor, CompetitorList
from research.common import ModuleResult, ResearchContext, cue_near, failed, finish, hit_ids, norm

_CUES = re.compile(r"\b(competitors?|competes? with|competing|alternatives?|versus|vs|compared (?:to|with)|rivals?|instead of|similar to)\b")


def extract_competitors(ctx: ResearchContext) -> ModuleResult:
    hits = ctx.retrieve("competitors alternatives compared versus rivals similar companies market landscape")
    if not hits:
        return ModuleResult([], ModuleStatus(status="partial", detail="No evidence retrieved for competitors."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="competitors", evidence=ctx.block(hits), schema_model=CompetitorList)
    if data is None:
        return failed("The model did not return valid JSON for competitors.")
    pool, items, dropped, seen = hit_ids(hits), [], 0, {norm(ctx.company_name)}
    for raw in (data.get("competitors") or data.get("items") or [])[:10]:
        try:
            c = Competitor.model_validate(raw)
        except ValidationError:
            dropped += 1
            continue
        key = norm(c.name)
        if not key or key in seen:
            continue
        sids = ctx.supporting_sources(c.name, c.source_ids, pool)
        if not sids:                               # never invent competitors
            dropped += 1
            continue
        seen.add(key)
        c.source_ids = sids
        explicit = any(cue_near(c.name, ctx.text_norm(s), _CUES) for s in sids)
        c.confidence = "Verified" if explicit else "Inferred"
        if not explicit:
            c.reason = (c.reason + " " if c.reason else "") + "(Named in the evidence, but no explicit competitor statement - inference.)"
        items.append(c)
    return ModuleResult(items, finish("competitor(s)", items, dropped))
