"""Deterministic tests of each extraction module with a stub retriever (independent of ranking quality)."""
import json

import pytest
from fakes import FakeLLM

from config import load_settings
from models import Company, CompanyIntelligence, Technology
from models.source import Evidence
from rag.vector_store import SearchHit
from research import competitors, funding, gtm, people, products, signals, technology
from research.common import ResearchContext
from research.company import extract_company
from research.pipeline import load_sample_evidence


class StubRetriever:
    """Always returns one chunk per evidence document - so module logic is tested in isolation."""
    def __init__(self, evidence):
        self.evidence = evidence

    def search(self, slug, query, top_k=None, **kw):
        return [SearchHit(text=e.content, score=0.9, company_slug=slug, company="N", source_id=e.source_id, url=e.url,
                          title=e.title, source_type=e.source_type, chunk_index=0) for e in self.evidence]


@pytest.fixture
def ctx():
    _, _, docs = load_sample_evidence()
    ev = [Evidence(source_id=f"SRC-{i + 1:03d}", title=d["title"], url=d["url"], source_type=d["source_type"], content=d["content"])
          for i, d in enumerate(docs)]
    s = load_settings()
    s.max_context_chars = 60000
    return ResearchContext(company_name="Northwind Health Clearinghouse (Sample)", slug="northwind", settings=s,
                           retriever=StubRetriever(ev), llm=FakeLLM(), evidence={e.source_id: e for e in ev})


def scripted(ctx, payload):
    """Make the fake model return exactly `payload` for any extraction."""
    ctx.llm.chat = lambda messages, **kw: json.dumps(payload)


def test_competitor_verified_only_with_explicit_competitor_statement(ctx):
    res = competitors.extract_competitors(ctx)
    by = {c.name: c for c in res.items}
    assert by["Contoso Claims Exchange"].confidence == "Verified"
    assert by["Fabrikam Health Network"].confidence == "Inferred"          # merely a partner in the evidence
    assert "inference" in by["Fabrikam Health Network"].reason.lower()
    assert "Globex Medical" not in by and res.status.status == "partial"   # hallucination discarded, reported as such


def test_competitor_never_lists_the_company_itself(ctx):
    scripted(ctx, {"competitors": [{"name": "Northwind Health Clearinghouse (Sample)", "source_ids": ["SRC-001"]}]})
    assert competitors.extract_competitors(ctx).items == []


def test_technology_official_statement_is_verified_but_job_posting_is_inferred(ctx):
    scripted(ctx, {"technologies": [
        {"name": "AWS", "category": "Cloud", "confidence": "Verified", "source_ids": ["SRC-005"]},
        {"name": "PostgreSQL", "category": "Database", "confidence": "Verified", "source_ids": ["SRC-008"]},
        {"name": "Terraform", "category": "Infrastructure", "confidence": "Inferred", "source_ids": ["SRC-008"]}]})
    by = {t.name: t for t in technology.extract_technology(ctx, []).items}
    assert by["AWS"].confidence == "Verified"
    assert by["PostgreSQL"].confidence == "Inferred" and by["PostgreSQL"].reason       # downgraded: job posting isn't official
    assert by["Terraform"].confidence == "Inferred" and by["Terraform"].reason


def test_observed_technologies_are_kept_verified_and_not_duplicated(ctx):
    observed = [Technology(category="Analytics", name="Segment", confidence="Verified", reason="Observed in HTML", source_ids=["SRC-001"])]
    scripted(ctx, {"technologies": [{"name": "Segment", "confidence": "Inferred", "source_ids": ["SRC-001"]},
                                    {"name": "AWS", "confidence": "Verified", "source_ids": ["SRC-005"]}]})
    items = technology.extract_technology(ctx, observed).items
    assert [t.name for t in items] == ["Segment", "AWS"] and items[0].confidence == "Verified"


def test_technology_model_failure_keeps_observed_signals(ctx):
    ctx.llm = FakeLLM(fail_json=True)
    observed = [Technology(name="Segment", confidence="Verified", source_ids=["SRC-001"])]
    res = technology.extract_technology(ctx, observed)
    assert res.items == observed and res.status.status == "partial"


def test_funding_total_requires_an_explicit_total_in_the_source(ctx):
    scripted(ctx, {"events": [], "total_funding": "$52 million", "total_funding_source_ids": ["SRC-007"]})
    assert funding.extract_funding(ctx).items.total_funding is None        # "$52 million" isn't in the text
    scripted(ctx, {"events": [], "total_funding": "$32 million", "total_funding_source_ids": ["SRC-007"]})
    assert funding.extract_funding(ctx).items.total_funding == "$32 million"


def test_funding_total_is_never_summed_from_events(ctx):
    scripted(ctx, {"events": [{"round_type": "Series B", "amount": "$20 million", "source_ids": ["SRC-007"]}], "total_funding": None})
    res = funding.extract_funding(ctx)
    assert res.items.total_funding is None and len(res.items.events) == 1


def test_funding_date_must_be_supported(ctx):
    scripted(ctx, {"events": [{"round_type": "Series B", "amount": "$20 million", "date": "March 2019", "source_ids": ["SRC-007"]}]})
    assert funding.extract_funding(ctx).items.events[0].date is None


def test_products_pricing_only_when_stated(ctx):
    scripted(ctx, {"products": [
        {"product_name": "Northwind Eligibility API", "pricing": "$99/month", "source_ids": ["SRC-002"]},
        {"product_name": "Northwind Claims Gateway", "pricing": "Available on request", "source_ids": ["SRC-002"]}]})
    items = products.extract_products(ctx).items
    assert all(p.pricing is None for p in items)          # the page only says "Pricing is available on request"


def test_products_duplicates_and_invalid_items(ctx):
    scripted(ctx, {"products": [{"product_name": "Northwind Eligibility API", "source_ids": ["SRC-002"]},
                                {"product_name": "northwind eligibility api", "source_ids": ["SRC-002"]},
                                {"description": "no name"}, "garbage", None]})
    res = products.extract_products(ctx)
    assert [p.product_name for p in res.items] == ["Northwind Eligibility API"]


def test_people_roles_are_verified_against_the_source(ctx):
    scripted(ctx, {"people": [
        {"name": "Jane Example", "role": "CEO and Co-founder", "source_ids": ["SRC-004"]},
        {"name": "Raj Sample", "role": "CEO", "source_ids": ["SRC-004"]},                       # wrong role
        {"name": "Maria Placeholder", "role": "VP of Partnerships", "relevant_function": "Nonsense", "source_ids": ["SRC-001"]},  # wrong source
        {"name": "jane@example.com", "role": "CEO"}, {"name": "X", "role": "CEO"}, {"name": "Only Name", "role": ""}]})
    items = {p.name: p for p in people.extract_people(ctx).items}
    assert set(items) == {"Jane Example", "Maria Placeholder"}                                    # source repaired from the pool
    assert items["Maria Placeholder"].relevant_function == "Partnerships" and items["Maria Placeholder"].source_ids == ["SRC-004"]
    assert items["Jane Example"].company == "Northwind Health Clearinghouse (Sample)"


def test_signals_must_be_supported_by_the_cited_text(ctx):
    scripted(ctx, {"signals": [
        {"signal_type": "Partnership", "description": "Partnership with Fabrikam Health Network to bring eligibility checks to member providers", "source_ids": ["SRC-006"], "date": "2031"},
        {"signal_type": "Hiring", "description": "Hiring a thousand astronauts for the moon base", "source_ids": ["SRC-008"]}]})
    items = signals.extract_signals(ctx).items
    assert len(items) == 1 and items[0].signal_type == "Partnership" and items[0].date is None


def test_company_overview_drops_unsupported_fields(ctx):
    scripted(ctx, {"industry": "Healthcare Technology", "headquarters": "Reykjavik, Iceland", "founded": "1999",
                   "description": "A sentient cloud that paints penguins and sells unicorn insurance",
                   "target_customers": ["Billing companies"], "source_ids": ["SRC-003"]})
    c = extract_company(ctx).items
    assert c.industry == "Healthcare Technology" and c.target_customers == ["Billing companies"]
    assert c.headquarters is None and c.founded is None and c.description is None


def test_company_overview_keeps_supported_fields(ctx):
    scripted(ctx, {"headquarters": "Columbus, Ohio", "founded": "2017", "source_ids": ["SRC-003"],
                   "description": "Northwind provides healthcare connectivity APIs for eligibility claims and remittance data"})
    c = extract_company(ctx).items
    assert (c.headquarters, c.founded) == ("Columbus, Ohio", "2017") and c.description


def test_gtm_claims_need_support_and_confidence_is_derived(ctx):
    scripted(ctx, {"verified_evidence": [
        {"statement": "Serves medical billing companies and digital health companies", "source_ids": ["SRC-003"]},
        {"statement": "Raised a Series B round led by Fabrikam Ventures", "source_ids": ["SRC-999"]},
        {"statement": "Has a secret underwater data centre", "source_ids": ["SRC-003"]}],
        "icp": ["a", "a", "b"], "why_this_company": "because", "confidence": "High"})
    intel = CompanyIntelligence(slug="northwind", company=Company(name="Northwind"))
    g = gtm.extract_gtm(ctx, intel).items
    assert len(g.verified_evidence) == 2 and g.confidence == "Medium"
    assert g.icp == ["a", "b"] and g.verified_evidence[1].source_ids[0] == "SRC-007"            # bogus SRC-999 repaired


def test_derive_confidence_rule():
    from models import Claim
    c = lambda *ids: Claim(statement="s", source_ids=list(ids))
    assert gtm.derive_confidence([]) == "Low" and gtm.derive_confidence([c("SRC-001")]) == "Low"
    assert gtm.derive_confidence([c("SRC-001"), c("SRC-002")]) == "Medium"
    assert gtm.derive_confidence([c("SRC-001"), c("SRC-001"), c("SRC-001")]) == "Low"      # 3 claims but a single source
    assert gtm.derive_confidence([c("SRC-001"), c("SRC-002"), c("SRC-001")]) == "Medium"   # 3 claims, 2 sources
    many = [c("SRC-001"), c("SRC-002"), c("SRC-003"), c("SRC-004"), c("SRC-005")]
    assert gtm.derive_confidence(many) == "High"


def test_no_evidence_retrieved_is_handled_by_every_module(ctx):
    ctx.retriever = StubRetriever([])
    for res in (products.extract_products(ctx), competitors.extract_competitors(ctx), funding.extract_funding(ctx),
                signals.extract_signals(ctx), people.extract_people(ctx), technology.extract_technology(ctx, []),
                extract_company(ctx), gtm.extract_gtm(ctx, CompanyIntelligence(slug="n", company=Company(name="N")))):
        assert res.status.status == "partial"
        assert "evidence" in res.status.detail.lower()
