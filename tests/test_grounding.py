"""The anti-hallucination helpers: claims must be found in the evidence or they are dropped."""
import re

import pytest

from research.common import clean_str_list, cue_near, mentions, norm, overlap_ratio
from research.funding import amount_grounded
from research.people import infer_function, role_alternatives, role_supported
from research.products import _price_grounded

TEXT = norm("Northwind raised a $20 million Series B round led by Fabrikam Ventures in 2025. "
            "Acme Claims APIs are used by billing companies.")


def test_mentions_uses_whole_words():
    assert mentions("Series B", TEXT) and mentions("Fabrikam Ventures", TEXT)
    assert mentions("Acme Claims API", TEXT)             # plural tolerated
    assert not mentions("Series C", TEXT)                # regression: 'c' must not match 'claims'/'companies'
    assert not mentions("Fabrik", TEXT) and not mentions("", TEXT) and not mentions("x", "")
    assert not mentions("Ventures Fabrikam", TEXT)       # order matters


def test_overlap_ignores_company_name_and_common_words():
    supported = "Acme Claims APIs serve billing companies"
    invented = "Acme employs five hundred engineers worldwide"
    assert overlap_ratio(supported, TEXT, ignore=["acme"]) > 0.6
    assert overlap_ratio(invented, TEXT, ignore=["acme"]) < 0.2
    assert overlap_ratio("", TEXT) == 0.0 and overlap_ratio("the and for", TEXT) == 0.0


def test_cue_near_requires_proximity():
    cue = re.compile(r"\b(competitor|alternative)\b")
    near = norm("Northwind is an alternative to Contoso Claims Exchange in the market")
    far = norm("Contoso Claims Exchange is a company. " + "filler " * 80 + "alternative")
    assert cue_near("Contoso Claims Exchange", near, cue)
    assert not cue_near("Contoso Claims Exchange", far, cue, window=100)


@pytest.mark.parametrize("amount,expected", [
    ("$20 million", True), ("$20M", True), ("$500 million", False), ("$2 million", False),
    ("$1,000", False), ("20.5 million", False),
])
def test_amount_grounded(amount, expected):
    assert amount_grounded(amount, "raised $20 million in a series b round") is expected


def test_amount_with_thousand_separators():
    assert amount_grounded("$1,250,000", "a seed round of $1,250,000 was announced")
    assert not amount_grounded("undisclosed", "raised $20 million")
    assert amount_grounded("undisclosed", "the round size was undisclosed")


def test_price_grounding():
    assert _price_grounded("$99/month", "plans start at $99 per month")
    assert not _price_grounded("$99/month", "pricing is available on request")
    assert not _price_grounded("Contact sales", "anything")            # no figures -> not a verifiable price


LEADERS = ("Jane Example, CEO and Co-founder. Jane previously led product.\n"
           "Raj Sample, Chief Technology Officer. Raj oversees engineering.\n"
           "Maria Placeholder, VP of Partnerships. Maria leads integrations.\n"
           "Pat Vertical\nHead of Growth\nSam Next\nChief Financial Officer")


@pytest.mark.parametrize("name,role,expected", [
    ("Jane Example", "CEO", True), ("Jane Example", "Chief Executive Officer", True), ("Jane Example", "Co-founder", True),
    ("Raj Sample", "CTO", True), ("Raj Sample", "Chief Technology Officer", True),
    ("Raj Sample", "CEO", False),                                   # neighbour's title must not count
    ("Maria Placeholder", "Chief Medical Officer", False),          # invented role
    ("Maria Placeholder", "Vice President of Partnerships", True),
    ("Pat Vertical", "Head of Growth", True),                       # name and title on separate lines
    ("Pat Vertical", "Chief Financial Officer", False),             # next person's title must not leak backwards
    ("Sam Next", "CFO", True), ("John Doe", "CEO", False),
])
def test_role_supported(name, role, expected):
    assert role_supported(name, role, LEADERS) is expected


def test_role_alternatives_and_function_inference():
    assert ["ceo"] in role_alternatives("Chief Executive Officer") and ["founder"] in role_alternatives("Co-Founder")
    assert role_alternatives("") == []
    assert infer_function("VP of Partnerships") == "Partnerships" and infer_function("Chief Technology Officer") == "Technology"
    assert infer_function("Janitor") == "Other"


def test_clean_str_list():
    assert clean_str_list(["a", " a ", "B", "", None, "b"], limit=5) == ["a", "B", "None"]
    assert clean_str_list(None) == [] and len(clean_str_list(list("abcdefgh"), limit=3)) == 3
