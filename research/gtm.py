"""GTM analysis: verified evidence (grounded) kept strictly apart from AI inference."""
from __future__ import annotations

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import CompanyIntelligence, ModuleStatus
from models.gtm import GTMAnalysis
from models.source import Claim
from research.common import ModuleResult, ResearchContext, clean_str_list, failed, hit_ids


def _facts_summary(intel: CompanyIntelligence) -> str:
    """Compact, already-verified facts (with their SRC ids) handed to the GTM prompt."""
    def ids(x) -> str:
        return f" ({', '.join(x.source_ids)})" if getattr(x, "source_ids", None) else ""
    lines = []
    if intel.company.description:
        lines.append(f"Description: {intel.company.description}{ids(intel.company)}")
    if intel.products:
        lines.append("Products: " + "; ".join(f"{p.product_name}{ids(p)}" for p in intel.products[:6]))
    if intel.technologies:
        lines.append("Technology: " + "; ".join(f"{t.name} [{t.confidence}]{ids(t)}" for t in intel.technologies[:8]))
    if intel.competitors:
        lines.append("Competitors: " + "; ".join(c.name for c in intel.competitors[:6]))
    if intel.signals:
        lines.append("Signals: " + "; ".join(f"{s.signal_type}: {s.description[:90]}{ids(s)}" for s in intel.signals[:5]))
    return "\n".join(lines)


def derive_confidence(claims: list[Claim]) -> str:
    """Coarse, transparent rule (not a score): depends only on how much verified evidence we hold."""
    distinct = {s for c in claims for s in c.source_ids}
    if len(claims) >= 3 and len(distinct) >= 2:
        return "High" if len(claims) >= 5 and len(distinct) >= 3 else "Medium"
    return "Medium" if len(claims) == 2 else "Low"


def extract_gtm(ctx: ResearchContext, intel: CompanyIntelligence) -> ModuleResult:
    hits = ctx.retrieve("customers target market needs challenges partners growth sales healthcare", top_k=ctx.settings.top_k)
    if not hits:
        return ModuleResult(None, ModuleStatus(status="partial", detail="No evidence retrieved for GTM analysis."))
    focus = f"User research focus: {ctx.instructions.strip()[:400]}" if ctx.instructions.strip() else ""
    evidence = _facts_summary(intel)
    block = ctx.block(hits)
    text = (f"ALREADY VERIFIED FACTS:\n{evidence}\n\n" if evidence else "") + block
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="gtm", evidence=text, schema_model=GTMAnalysis, extra_focus=focus)
    if data is None:
        return failed("The model did not return valid JSON for the GTM analysis.")
    try:
        raw = GTMAnalysis.model_validate(data)
    except ValidationError as exc:
        return failed(f"GTM analysis could not be parsed: {exc.errors()[0]['msg']}")

    pool, claims, dropped = hit_ids(hits), [], 0
    for c in raw.verified_evidence[:8]:
        cited = ctx.valid_ids(c.source_ids)
        ok = [s for s in (cited or pool) if ctx.overlap(c.statement, ctx.text_norm(s)) >= 0.35]
        if not ok and cited:
            ok = [s for s in pool if ctx.overlap(c.statement, ctx.text_norm(s)) >= 0.35]
        if ok:
            claims.append(Claim(statement=c.statement, source_ids=ok[:3]))
        else:
            dropped += 1                              # claimed as fact but not found in any source

    gtm = GTMAnalysis(
        verified_evidence=claims,
        icp=clean_str_list(raw.icp, 4), target_customers=clean_str_list(raw.target_customers, 4),
        business_needs=clean_str_list(raw.business_needs, 4), pain_points=clean_str_list(raw.pain_points, 4),
        relevant_functions=clean_str_list(raw.relevant_functions, 5), opportunities=clean_str_list(raw.opportunities, 4),
        why_this_company=(raw.why_this_company or "").strip() or None,
        outreach_angles=clean_str_list(raw.outreach_angles, 4),
        confidence=derive_confidence(claims),
        source_ids=sorted({s for c in claims for s in c.source_ids}),
    )
    note = f" {dropped} unsupported 'verified' claim(s) discarded." if dropped else ""
    status = ModuleStatus(status="partial" if dropped or not claims else "ok", items=len(claims),
                          detail=f"{len(claims)} evidence-backed claim(s); all other sections are AI inference.{note}")
    return ModuleResult(gtm, status)
