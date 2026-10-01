from __future__ import annotations

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import ModuleStatus
from models.signal import GrowthSignal, SignalList
from research.common import ModuleResult, ResearchContext, failed, finish, hit_ids, norm
from research.funding import _years_ok


def extract_signals(ctx: ResearchContext) -> ModuleResult:
    hits = ctx.retrieve("announces launch partnership expansion hiring acquisition growth new customers integration")
    if not hits:
        return ModuleResult([], ModuleStatus(status="partial", detail="No evidence retrieved for growth signals."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="signals", evidence=ctx.block(hits), schema_model=SignalList)
    if data is None:
        return failed("The model did not return valid JSON for growth signals.")
    pool, items, dropped, seen = hit_ids(hits), [], 0, set()
    for raw in (data.get("signals") or data.get("items") or [])[:10]:
        try:
            s = GrowthSignal.model_validate(raw)
        except ValidationError:
            dropped += 1
            continue
        cited = ctx.valid_ids(s.source_ids)
        # A signal is only kept if its description is actually supported by text in a source.
        best = [sid for sid in (cited or pool) if ctx.overlap(s.description, ctx.text_norm(sid)) >= 0.4]
        if not best and cited:
            best = [sid for sid in pool if ctx.overlap(s.description, ctx.text_norm(sid)) >= 0.4]
        key = norm(s.description)[:80]
        if not best or key in seen:
            dropped += 0 if key in seen else 1
            continue
        seen.add(key)
        s.source_ids = best[:3]
        if s.date and not _years_ok(s.date, " ".join(ctx.text_raw(x) for x in best)):
            s.date = None
        items.append(s)
    return ModuleResult(items, finish("growth signal(s)", items, dropped))
