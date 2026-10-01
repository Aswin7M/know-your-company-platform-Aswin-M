import pytest
from pydantic import ValidationError

from models import (Claim, Company, CompanyIntelligence, Competitor, Evidence, FundingEvent, FundingSummary,
                    GrowthSignal, GTMAnalysis, Person, Product, Source, Technology)
from models.company import NOT_VERIFIED, ModuleStatus


def test_valid_complete_company():
    c = Company(name="Acme", official_website="https://acme.example", industry="Healthcare Technology",
                subindustry="Clearinghouse", company_type="Private", headquarters="Columbus, Ohio", founded="2017",
                markets=["US"], business_model="API", target_customers=["Billing companies"], description="Does things.")
    assert c.name == "Acme" and c.headline == "Healthcare Technology / Clearinghouse"


def test_incomplete_company_is_valid_and_unknowns_are_not_verified():
    c = Company(name="Acme")
    assert c.industry is None and c.markets == [] and c.description is None
    assert c.display("headquarters") == NOT_VERIFIED and c.display("markets") == NOT_VERIFIED
    assert c.headline == "Industry not verified"


def test_name_is_required():
    with pytest.raises(ValidationError):
        Company()


def test_llm_quirks_are_absorbed_without_inventing_data():
    c = Company.model_validate({"name": "Acme", "founded": 2017, "industry": "null", "headquarters": "Unknown",
                                "markets": "US", "target_customers": None, "extra_field": "ignored"})
    assert c.founded == "2017"                      # number -> text
    assert c.industry is None and c.headquarters is None   # placeholders -> None
    assert c.markets == ["US"] and c.target_customers == []


def test_technology_confidence_defaults_to_inferred_and_is_normalised():
    assert Technology(name="AWS").confidence == "Inferred"
    assert Technology(name="AWS", confidence="verified").confidence == "Verified"
    assert Technology(name="AWS", confidence="definitely!").confidence == "Inferred"   # never upgraded silently
    assert Technology(name="X", category="infra").category == "Infrastructure"
    assert Technology(name="X", category="weird").category == "Other"


def test_competitor_category_and_confidence_normalisation():
    assert Competitor(name="A", category="direct").category == "Direct"
    assert Competitor(name="A", category="rival").category == "Similar"
    assert Competitor(name="A").confidence == "Inferred"


def test_growth_signal_type_and_importance():
    assert GrowthSignal(description="d", signal_type="new product launch").signal_type == "Product launch"
    assert GrowthSignal(description="d", signal_type="???").signal_type == "Other"
    assert GrowthSignal(description="d", importance="urgent!!").importance == "Medium"
    assert not hasattr(GrowthSignal(description="d"), "score")          # no invented numeric scores


def test_product_funding_person_models():
    p = Product(product_name="API", source_ids="SRC-001")
    assert p.source_ids == ["SRC-001"] and p.pricing is None
    assert FundingSummary().total_funding is None and FundingEvent(amount="$1M").investors == []
    person = Person(name="Jane Example", role="CEO")
    assert person.company is None and not hasattr(person, "email") and not hasattr(person, "phone")


def test_gtm_defaults_and_claims():
    g = GTMAnalysis(verified_evidence=[Claim(statement="x", source_ids=["SRC-001"])])
    assert g.confidence == "Low" and g.icp == [] and g.verified_evidence[0].source_ids == ["SRC-001"]


def test_source_and_evidence_models():
    e = Evidence(source_id="SRC-001", title="T", url="https://a.example", source_type="official_website", content="word " * 200)
    s = e.to_source()
    assert isinstance(s, Source) and s.source_id == "SRC-001" and len(s.snippet) <= 280
    with pytest.raises(ValidationError):
        Evidence(source_id="SRC-001", url="https://a.example", source_type="made_up_type")


def test_company_intelligence_roundtrip():
    intel = CompanyIntelligence(slug="acme", company=Company(name="Acme"),
                                module_status={"products": ModuleStatus(status="ok", items=2)})
    again = CompanyIntelligence.model_validate_json(intel.model_dump_json())
    assert again == intel and again.funding.total_funding is None
