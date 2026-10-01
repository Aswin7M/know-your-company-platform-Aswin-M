from __future__ import annotations

import re

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import ModuleStatus
from models.funding import FundingEvent, FundingSummary
from research.common import ModuleResult, ResearchContext, failed, hit_ids, mentions, norm

_TOTAL_CUE = re.compile(r"\b(total|to date|in total|overall|altogether|so far)\b")


def amount_grounded(amount: str, text_raw: str) -> bool:
    """Every figure in the amount must literally occur in the evidence."""
    nums = [n.replace(",", "") for n in re.findall(r"\d+(?:[.,]\d+)*", amount or "")]
    if not nums:
        return bool(norm(amount)) and norm(amount) in norm(text_raw)
    text = re.sub(r"(?<=\d),(?=\d)", "", text_raw)
    return all(re.search(rf"(?<![\d.]){re.escape(n)}(?![\d])", text) for n in nums)


def _years_ok(value: str, text_raw: str) -> bool:
    years = re.findall(r"\b(?:19|20)\d{2}\b", value or "")
    return bool(years) and all(y in text_raw for y in years)


def extract_funding(ctx: ResearchContext) -> ModuleResult:
    hits = ctx.retrieve("funding raised series round investors led by valuation total funding")
    if not hits:
        return ModuleResult(FundingSummary(), ModuleStatus(status="partial", detail="No evidence retrieved for funding."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="funding", evidence=ctx.block(hits), schema_model=FundingSummary)
    if data is None:
        return failed("The model did not return valid JSON for funding.")
    pool = hit_ids(hits)
    events, dropped = [], 0
    for raw in (data.get("events") or data.get("items") or [])[:8]:
        try:
            ev = FundingEvent.model_validate(raw)
        except ValidationError:
            dropped += 1
            continue
        cited = ctx.valid_ids(ev.source_ids) or pool
        text_raw = " ".join(ctx.text_raw(s) for s in cited)
        text_norm = ctx.joined_norm(cited)
        amount_ok = bool(ev.amount) and amount_grounded(ev.amount, text_raw)
        round_ok = bool(ev.round_type) and mentions(ev.round_type, text_norm)
        if not (amount_ok or round_ok):            # nothing verifiable -> discard the whole event
            dropped += 1
            continue
        ev.amount = ev.amount if amount_ok else None
        ev.round_type = ev.round_type if round_ok else None
        ev.investors = [i for i in ev.investors if mentions(i, text_norm)][:8]
        if ev.date and not _years_ok(ev.date, text_raw):
            ev.date = None
        ev.source_ids = [s for s in cited if s in ctx.evidence][:4]
        events.append(ev)

    summary = FundingSummary(events=events)
    total = (data.get("total_funding") or "").strip() if isinstance(data.get("total_funding"), str) else ""
    if total and total.lower() not in {"null", "none", "unknown", "not verified"}:
        for sid in dict.fromkeys(ctx.valid_ids(data.get("total_funding_source_ids") or []) + pool):
            if amount_grounded(total, ctx.text_raw(sid)) and _TOTAL_CUE.search(ctx.text_norm(sid)):
                summary.total_funding, summary.total_funding_source_ids = total, [sid]
                break
        else:
            dropped += 1                             # stated total not supported -> "Not verified"
    note = f" {dropped} unsupported item(s) discarded." if dropped else ""
    if events or summary.total_funding:
        status = ModuleStatus(status="partial" if dropped else "ok", items=len(events),
                              detail=f"{len(events)} funding event(s) verified.{note}")
    else:
        status = ModuleStatus(status="partial", detail=f"No funding information could be verified.{note}")
    return ModuleResult(summary, status)
