"""UI tests using Streamlit's headless AppTest harness (no browser, no Ollama, no network)."""
import json
from pathlib import Path

import pytest
import streamlit as st
from fakes import FakeLLM
from streamlit.testing.v1 import AppTest

from ai.ollama import OllamaClient
from config import load_settings

APP = str(Path(__file__).resolve().parent.parent / "app.py")
SLUG = "northwind-health-clearinghouse-sample"
PAGES = ["🏠 Home", "🔍 Research Company", "🏢 Companies", "💬 AI Chat", "📄 Reports", "⚙️ Settings"]


@pytest.fixture(autouse=True)
def fresh_cache():
    st.cache_data.clear()                   # the Ollama status lookup is cached for a few seconds
    yield
    st.cache_data.clear()


@pytest.fixture
def ollama_up(monkeypatch):
    """Pretend Ollama is running, answering with the (hallucinating) FakeLLM."""
    fake = FakeLLM()
    monkeypatch.setenv("TOP_K", "12")
    monkeypatch.setenv("MAX_CONTEXT_CHARS", "60000")
    monkeypatch.setattr(OllamaClient, "status", lambda self, timeout=2.0: fake.status())
    monkeypatch.setattr(OllamaClient, "chat", lambda self, messages, **kw: fake.chat(messages, **kw))
    return fake


def launch() -> AppTest:
    return AppTest.from_file(APP, default_timeout=90).run()


def goto(at: AppTest, page: str) -> AppTest:
    return at.sidebar.radio[0].set_value(page).run()


def md(at: AppTest) -> str:
    """Everything a user can read: markdown, metrics and captions."""
    return "\n".join([m.value for m in at.markdown] + [f"{m.label} {m.value}" for m in at.metric] + [c.value for c in at.caption])


def button(at: AppTest, contains: str):
    matches = [b for b in at.button if contains in b.label]
    assert matches, f"no button containing {contains!r}; have {[b.label for b in at.button]}"
    return matches[0]


def load_sample(at: AppTest) -> AppTest:
    return at.button(key="load_sample").click().run()


def test_app_starts_and_every_page_renders_without_ollama():
    at = launch()
    assert not at.exception
    assert "Know Your Company" in md(at) and "Ollama Not Available" in md(at)
    assert [r for r in at.sidebar.radio[0].options] == PAGES
    for page in PAGES:
        goto(at, page)
        assert not at.exception, f"{page}: {[e.value for e in at.exception]}"
    assert any("Ollama Not Available" in e.value for e in at.error)              # settings page red status


def test_home_page_shows_the_research_form_from_the_spec():
    at = launch()
    labels = [t.label for t in at.text_input] + [t.label for t in at.text_area]
    assert labels == ["Company Name", "Company Website (optional)", "Research Instructions (optional)"]
    assert any("Research Company" in b.label for b in at.button)


def test_empty_company_name_is_rejected_politely():
    at = launch()
    button(at, "Research Company").click().run()
    assert not at.exception and any("enter a company name" in w.value for w in at.warning)


def test_sample_company_with_ollama_down_still_produces_a_browsable_workspace():
    at = load_sample(launch())
    assert not at.exception
    assert "Northwind Health Clearinghouse (Sample)" in md(at)
    assert len(at.tabs) == 10 and [t.label for t in at.tabs] == ["Overview", "Products", "Technology", "Competitors", "Funding", "Signals", "People", "GTM", "Sources", "AI Chat"]
    assert any("did not fully succeed" in w.value for w in at.warning)            # warnings are surfaced, not hidden
    assert "Not verified" in md(at)                                               # unknowns are shown honestly
    # a quick action degrades to a clear message instead of crashing
    button(at, "Company Summary").click().run()
    assert not at.exception and "Cannot reach Ollama" in md(at)


def test_sample_company_with_working_model_shows_grounded_results(ollama_up):
    at = load_sample(launch())
    assert not at.exception
    text = md(at)
    for expected in ("Northwind Eligibility API", "Jane Example", "Contoso Claims Exchange", "$32 million", "AI inference"):
        assert expected in text, expected
    for forbidden in ("QuantumBilling", "John Doe", "Globex Medical", "$500 million", "Snowflake"):
        assert forbidden not in text, forbidden
    assert "🟢" in md(at) and "Ollama Connected" in md(at)

    button(at, "Funding").click().run()                                           # quick action
    assert not at.exception and "**Sources**" in md(at) and "totally-made-up" not in md(at)

    at.chat_input[0].set_value("How much funding did the company raise in its Series B round?").run()   # chat
    assert not at.exception and len(at.chat_message) >= 2
    assert "SRC-" in md(at) and "Retrieved evidence" in " ".join(e.label for e in at.expander)


def test_companies_reports_and_settings_pages_with_a_saved_company(ollama_up, tmp_path):
    at = load_sample(launch())
    data = Path(load_settings().data_dir) / "companies" / SLUG
    assert (data / "company.json").exists()

    goto(at, "🏢 Companies")
    assert not at.exception and "Northwind Health Clearinghouse (Sample)" in md(at)

    goto(at, "💬 AI Chat")
    assert not at.exception and "Ask anything about Northwind" in md(at)

    goto(at, "📄 Reports")
    button(at, "Generate Markdown Report").click().run()
    assert not at.exception and any("report.md" in s.value for s in at.success)
    report = (data / "report.md").read_text(encoding="utf-8")
    assert "## 12. Sources" in report and "## 10. GTM Analysis" in report

    goto(at, "⚙️ Settings")
    assert any("Ollama Connected" in s.value for s in at.success)
    at.text_input[1].set_value("qwen3:4b")
    at.number_input[0].set_value(7)
    button(at, "Save settings").click().run()
    saved = json.loads((Path(load_settings().data_dir) / "settings.json").read_text(encoding="utf-8"))
    assert saved["ollama_model"] == "qwen3:4b" and saved["top_k"] == 7
    assert load_settings().ollama_model == "qwen3:4b"                              # persisted and picked up again


def test_delete_company_requires_confirmation_and_removes_files():
    at = load_sample(launch())
    goto(at, "🏢 Companies")
    at.button(key=f"del_{SLUG}").click().run()
    assert any("cannot be undone" in w.value for w in at.warning)
    assert (Path(load_settings().data_dir) / "companies" / SLUG).exists()          # not deleted yet
    at.button(key=f"yes_{SLUG}").click().run()
    assert not at.exception and not (Path(load_settings().data_dir) / "companies" / SLUG).exists()
    assert any("No companies yet" in i.value for i in at.info)
