"""Deterministic Markdown report (no LLM involved, so it cannot add new claims)."""
from __future__ import annotations

from typing import Iterable

from ai.prompts import REPORT_INFERENCE_NOTE
from models.company import NOT_VERIFIED, CompanyIntelligence
from models.source import Source

BADGE = {"Verified": "✅ Verified", "Inferred": "🔶 Inferred (AI inference)", "Not verified": "❔ Not verified"}


def _refs(ids: Iterable[str]) -> str:
    ids = list(ids or [])
    return f" _[{', '.join(ids)}]_" if ids else ""


def _bullets(items: Iterable[str]) -> str:
    items = [i for i in items if i]
    return "\n".join(f"- {i}" for i in items) if items else f"- {NOT_VERIFIED}"


def build_report(intel: CompanyIntelligence, sources: list[Source]) -> str:
    c = intel.company
    out: list[str] = [f"# {c.name} - Company Intelligence Report", ""]
    out.append(f"_Generated: {intel.researched_at or 'n/a'} · Model: {intel.ollama_model or 'n/a'} · "
               f"Sources: {len(sources)}_")
    out.append("")
    out.append("> Every factual statement is tied to source IDs. Items marked **AI inference** are model reasoning, "
               "not verified facts. Anything that could not be verified is shown as “Not verified”.")
    out.append("")

    # 1 Executive summary
    out += ["## 1. Executive Summary", ""]
    out.append(c.description + _refs(c.source_ids) if c.description else f"Company description: {NOT_VERIFIED}.")
    out.append("")
    out.append(f"Collected evidence from {len(sources)} source(s): {len(intel.products)} product(s), "
               f"{len(intel.technologies)} technology signal(s), {len(intel.competitors)} competitor(s), "
               f"{len(intel.funding.events)} funding event(s), {len(intel.signals)} growth signal(s), "
               f"{len(intel.people)} named people.")
    out.append("")

    # 2 Overview
    out += ["## 2. Company Overview", "", "| Field | Value |", "|---|---|"]
    for label, fld in [("Website", "official_website"), ("Industry", "industry"), ("Sub-industry", "subindustry"),
                       ("Company type", "company_type"), ("Headquarters", "headquarters"), ("Founded", "founded"),
                       ("Markets", "markets"), ("Business model", "business_model"),
                       ("Target customers", "target_customers")]:
        out.append(f"| {label} | {c.display(fld)} |")
    if c.website_confirmed is False and c.official_website:
        out.append("")
        out.append("_The official website could not be confirmed automatically._")
    out.append("")

    # 3 Products
    out += ["## 3. Products & Services", ""]
    if not intel.products:
        out.append(f"- {NOT_VERIFIED}")
    for p in intel.products:
        out.append(f"### {p.product_name}{_refs(p.source_ids)}")
        if p.description:
            out.append(p.description)
        if p.healthcare_use_case:
            out.append(f"- **Healthcare use case:** {p.healthcare_use_case}")
        if p.features:
            out.append(f"- **Features:** {', '.join(p.features)}")
        if p.target_users:
            out.append(f"- **Target users:** {', '.join(p.target_users)}")
        if p.integrations:
            out.append(f"- **Integrations:** {', '.join(p.integrations)}")
        out.append(f"- **Pricing:** {p.pricing or 'Not publicly verified'}")
        out.append("")

    # 4 Technology
    out += ["## 4. Technology", ""]
    if not intel.technologies:
        out.append(f"- {NOT_VERIFIED}")
    for t in intel.technologies:
        line = f"- **{t.category}: {t.name}** - {BADGE[t.confidence]}{_refs(t.source_ids)}"
        if t.reason:
            line += f"\n  - Reason: {t.reason}"
        out.append(line)
    out.append("")

    # 5 Target customers
    out += ["## 5. Target Customers", "", _bullets(c.target_customers), ""]

    # 6 Competitors
    out += ["## 6. Competitors", ""]
    if not intel.competitors:
        out.append(f"- {NOT_VERIFIED}")
    for comp in intel.competitors:
        line = f"- **{comp.name}** ({comp.category}) - {BADGE[comp.confidence]}{_refs(comp.source_ids)}"
        if comp.reason:
            line += f"\n  - {comp.reason}"
        if comp.overlap:
            line += f"\n  - Overlap: {comp.overlap}"
        out.append(line)
    out.append("")

    # 7 Funding
    out += ["## 7. Funding", ""]
    f = intel.funding
    out.append(f"**Total funding:** {f.total_funding + _refs(f.total_funding_source_ids) if f.total_funding else NOT_VERIFIED}")
    out.append("")
    if not f.events:
        out.append(f"- Funding rounds: {NOT_VERIFIED}")
    for e in f.events:
        out.append(f"- {e.date or 'Date not verified'} · {e.round_type or 'Round not verified'} · "
                   f"{e.amount or 'Amount not verified'}"
                   f"{' · Investors: ' + ', '.join(e.investors) if e.investors else ''}{_refs(e.source_ids)}")
    out.append("")

    # 8 Signals
    out += ["## 8. Growth Signals", ""]
    if not intel.signals:
        out.append(f"- {NOT_VERIFIED}")
    for s in intel.signals:
        out.append(f"- **{s.signal_type}** ({s.importance} importance, {s.date or 'date not verified'}): "
                   f"{s.description}{_refs(s.source_ids)}")
    out.append("")

    # 9 People
    out += ["## 9. People", "", "_Public professional information only._", ""]
    if not intel.people:
        out.append(f"- {NOT_VERIFIED}")
    for p in intel.people:
        out.append(f"- **{p.name}** - {p.role} ({p.relevant_function or 'Other'}){_refs(p.source_ids)}")
    out.append("")

    # 10 GTM
    out += ["## 10. GTM Analysis", ""]
    g = intel.gtm
    if not g:
        out.append(f"- {NOT_VERIFIED}")
    else:
        out += ["### Verified evidence", ""]
        out.append("\n".join(f"- {cl.statement}{_refs(cl.source_ids)}" for cl in g.verified_evidence)
                   or f"- {NOT_VERIFIED}")
        out += ["", REPORT_INFERENCE_NOTE, ""]
        for title, items in [("Ideal customer profile", g.icp), ("Target customers", g.target_customers),
                             ("Business needs", g.business_needs), ("Potential pain points", g.pain_points),
                             ("Relevant functions", g.relevant_functions), ("Potential opportunities", g.opportunities),
                             ("Potential outreach angles", g.outreach_angles)]:
            out += [f"**{title}** _(AI inference)_", _bullets(items), ""]
        out.append(f"**Confidence in this analysis:** {g.confidence} (based on the amount of verified evidence)")
    out.append("")

    # 11 Why this company
    out += ["## 11. Why This Company?", "", REPORT_INFERENCE_NOTE, ""]
    out.append(g.why_this_company if g and g.why_this_company else f"{NOT_VERIFIED} - no AI reasoning was generated.")
    out.append("")

    # 12 Sources
    out += ["## 12. Sources", ""]
    for s in sources:
        out.append(f"- **{s.source_id}** - [{s.title}]({s.url}) · {s.source_type} · retrieved {s.retrieved_at[:10]}")
    if intel.warnings:
        out += ["", "## Research notes", ""] + [f"- {w}" for w in intel.warnings]
    return "\n".join(out).rstrip() + "\n"
