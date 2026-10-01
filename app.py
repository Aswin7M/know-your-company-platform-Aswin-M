"""Know Your Company - Streamlit UI.

Research a healthcare company -> build a local, source-grounded knowledge base -> chat with it.
Runs entirely locally (Ollama + sentence-transformers + embedded Qdrant). No API keys.

Nothing heavy happens at import time: the embedding model, vector store and Ollama are
only touched when the user researches, asks a question or opens the relevant Settings check.
"""
from __future__ import annotations

import re
from typing import Optional

import streamlit as st

from ai.analyzer import answer_question, run_quick_action
from ai.ollama import OllamaClient, OllamaStatus
from ai.prompts import QUICK_ACTIONS
from config import UI_EDITABLE, load_settings, save_settings
from models.company import NOT_VERIFIED, CompanyIntelligence
from models.source import Source
from rag.retriever import Retriever
from research.pipeline import STEPS, load_sample_company, reindex_company, run_research
from storage.json_store import CompanyStore
from storage.report import build_report

st.set_page_config(page_title="Know Your Company", page_icon="🏥", layout="wide")

PAGES = ["🏠 Home", "🔍 Research Company", "🏢 Companies", "💬 AI Chat", "📄 Reports", "⚙️ Settings"]
CONF_BADGE = {"Verified": "✅ Verified", "Inferred": "🔶 Inferred", "Not verified": "❔ Not verified"}
STEP_ICON = {"pending": "○", "running": "⏳", "done": "✓", "warning": "⚠", "failed": "✗", "skipped": "⏭"}

st.markdown("""
<style>
.kyc-hero {padding: 1.6rem 1.8rem; border-radius: 14px; margin-bottom: 1rem;
  background: linear-gradient(120deg, #0f766e 0%, #155e75 60%, #1e3a8a 100%); color: white;}
.kyc-hero h1 {margin: 0; font-size: 2.1rem; color: white;}
.kyc-hero p {margin: .3rem 0 0 0; opacity: .92; font-size: 1.05rem;}
.kyc-pill {display:inline-block; padding: .15rem .6rem; border-radius: 999px; font-size: .8rem; background:#e2e8f0; color:#0f172a;}
.kyc-note {font-size: .85rem; opacity: .75;}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------- helpers
def settings():
    return load_settings()


def company_store() -> CompanyStore:
    return CompanyStore(settings().companies_dir)


@st.cache_data(ttl=8, show_spinner=False)
def _cached_status(base_url: str, model: str) -> OllamaStatus:
    return OllamaClient(base_url=base_url, model=model).status(timeout=1.5)


def ollama_status() -> OllamaStatus:
    s = settings()
    return _cached_status(s.ollama_base_url, s.ollama_model)


def src_links(ids: list[str], sources: dict[str, Source]) -> str:
    parts = []
    for sid in ids or []:
        src = sources.get(sid)
        parts.append(f"[{sid}]({src.url})" if src else f"{sid}")
    return ", ".join(parts) if parts else "—"


def conf(label: str) -> str:
    return CONF_BADGE.get(label, label)


def _slug_label(items: list[dict]) -> dict[str, str]:
    return {i["slug"]: f"{i['name']}" for i in items}


# --------------------------------------------------------------------------- research with real progress
def run_with_progress(name: str, website: str, instructions: str, run_ai: bool, sample: bool = False) -> Optional[str]:
    """Run the pipeline and show only what actually happened. Returns the company slug on success."""
    state = {key: ("pending", "") for key, _ in STEPS}
    labels = dict(STEPS)
    box = st.status(f"Researching {name}...", expanded=True)
    with box:
        area = st.empty()

        def render() -> None:
            lines = []
            for key, label in STEPS:
                status, detail = state[key]
                extra = f" - _{detail}_" if detail and status != "pending" else ""
                lines.append(f"{STEP_ICON[status]} **{label}**{extra}" if status != "pending" else f"{STEP_ICON[status]} {label}")
            area.markdown("\n\n".join(lines))

        def on_progress(key: str, status: str, detail: str = "") -> None:
            state[key] = (status, detail)
            render()

        render()
        kwargs = dict(progress=on_progress, run_analysis=run_ai)
        result = load_sample_company(**kwargs) if sample else run_research(name, website, instructions, **kwargs)

    if not result.ok:
        box.update(label="Research could not be completed", state="error", expanded=True)
        st.error(result.error or "Research failed.")
        return None
    problems = [k for k, (st_, _) in state.items() if st_ in ("warning", "failed")]
    if problems:
        box.update(label=f"Research complete with {len(problems)} warning(s)", state="complete", expanded=False)
        st.warning("Some steps did not fully succeed: " + ", ".join(labels[k] for k in problems) +
                   ". Details are listed on the Overview tab.")
    else:
        box.update(label="Research complete.", state="complete", expanded=False)
    return result.slug


def research_form(prefix: str, advanced: bool) -> None:
    with st.form(f"research_{prefix}"):
        name = st.text_input("Company Name", placeholder="e.g. Stedi")
        website = st.text_input("Company Website (optional)", placeholder="https://www.stedi.com")
        instructions = st.text_area(
            "Research Instructions (optional)", height=90,
            placeholder="Research products, technology, GTM, competitors, people and growth signals.")
        run_ai = True
        if advanced:
            run_ai = st.checkbox("Run AI analysis (slower; needs Ollama). Untick to only collect and index evidence.", value=True)
        submitted = st.form_submit_button("🔍 Research Company", type="primary")
    if not submitted:
        return
    if not name.strip():
        st.warning("Please enter a company name.")
        return
    if run_ai and not ollama_status().ready:
        st.warning("Ollama is not ready - evidence will still be collected and indexed, but AI analysis will be skipped. "
                   "See Settings / OLLAMA_SETUP.md.")
    slug = run_with_progress(name.strip(), website.strip(), instructions.strip(), run_ai)
    if slug:
        st.session_state["_open"] = slug
        st.session_state["_just_researched"] = slug


# --------------------------------------------------------------------------- chat + quick actions
def _answer_kwargs(slug: str, intel: CompanyIntelligence, sources: dict[str, Source]) -> dict:
    s = settings()
    return dict(company_slug=slug, company_name=intel.company.name, sources=sources,
                retriever=Retriever(settings=s), llm=OllamaClient(settings=s), settings=s)


def _add_answer(slug: str, question: str, ans) -> None:
    st.session_state.setdefault(f"chat_{slug}", [])
    st.session_state[f"chat_{slug}"] += [
        {"role": "user", "content": question},
        {"role": "assistant", "content": ans.text,
         "hits": [{"id": h.source_id, "title": h.title, "url": h.url, "score": h.score, "text": h.text} for h in ans.hits]},
    ]


def render_hits(hits: list[dict]) -> None:
    if not hits:
        return
    with st.expander(f"Retrieved evidence ({len(hits)} chunks, with similarity scores)"):
        for h in hits:
            st.markdown(f"**{h['id']}** · [{h['title']}]({h['url']}) · score `{h['score']:.3f}`")
            st.caption(" ".join(h["text"].split())[:400] + ("…" if len(h["text"]) > 400 else ""))


def render_quick_actions(slug: str, intel: CompanyIntelligence, sources: dict[str, Source], prefix: str) -> None:
    if not ollama_status().ready:
        st.caption("⚠ Quick actions need Ollama (see Settings). Stored evidence and sources are still browsable.")
    keys = list(QUICK_ACTIONS)
    cols = st.columns(5)
    clicked = None
    for i, key in enumerate(keys):
        if cols[i % 5].button(QUICK_ACTIONS[key][0], key=f"{prefix}_qa_{key}", use_container_width=True):
            clicked = key
    if clicked:
        with st.spinner("Retrieving evidence and asking the local model (CPU models can take a minute)..."):
            ans = run_quick_action(clicked, **_answer_kwargs(slug, intel, sources))
        _add_answer(slug, QUICK_ACTIONS[clicked][0].split(" ", 1)[1], ans)
        st.session_state[f"last_quick_{slug}"] = st.session_state[f"chat_{slug}"][-1]
    last = st.session_state.get(f"last_quick_{slug}")
    if last:
        with st.container(border=True):
            st.markdown(last["content"])
            render_hits(last.get("hits", []))


def render_chat(slug: str, intel: CompanyIntelligence, sources: dict[str, Source], prefix: str) -> None:
    st.markdown(f"#### Ask anything about {intel.company.name}")
    st.caption("Answers come from the collected evidence. Facts cite source IDs; reasoning beyond the evidence is labelled “AI inference”.")
    history = st.session_state.setdefault(f"chat_{slug}", [])
    if st.button("🧹 New chat", key=f"{prefix}_clear"):
        st.session_state[f"chat_{slug}"] = []
        st.session_state.pop(f"last_quick_{slug}", None)
        st.rerun()
    for msg in history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            render_hits(msg.get("hits", []))
    question = st.chat_input(f"Ask about {intel.company.name}", key=f"{prefix}_input")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching evidence and thinking..."):
                ans = answer_question(question, history=history, **_answer_kwargs(slug, intel, sources))
            st.markdown(ans.text)
            render_hits([{"id": h.source_id, "title": h.title, "url": h.url, "score": h.score, "text": h.text} for h in ans.hits])
        history += [{"role": "user", "content": question},
                    {"role": "assistant", "content": ans.text,
                     "hits": [{"id": h.source_id, "title": h.title, "url": h.url, "score": h.score, "text": h.text} for h in ans.hits]}]


# --------------------------------------------------------------------------- workspace tabs
def tab_overview(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    c = intel.company
    if c.description:
        st.markdown(f"> {c.description}")
        st.caption(f"AI summary of cited sources: {src_links(c.source_ids, sources)}")
    else:
        st.info(f"Company description: {NOT_VERIFIED}.")
    left, right = st.columns(2)
    facts = [("Website", "official_website"), ("Industry", "industry"), ("Sub-industry", "subindustry"),
             ("Company type", "company_type"), ("Headquarters", "headquarters"), ("Founded", "founded"),
             ("Markets", "markets"), ("Business model", "business_model"), ("Target customers", "target_customers")]
    for i, (label, fld) in enumerate(facts):
        (left if i % 2 == 0 else right).markdown(f"**{label}:** {c.display(fld)}")
    if c.official_website and not c.website_confirmed:
        st.warning("The official website could not be confirmed automatically - please check it.")
    st.markdown("##### Research status")
    icons = {"ok": "✅", "partial": "⚠️", "failed": "❌", "skipped": "⏭️"}
    for key, label in STEPS:
        ms = intel.module_status.get(key)
        if ms:
            st.markdown(f"{icons[ms.status]} **{label}** - {ms.detail}")
    if intel.warnings:
        with st.expander(f"Research notes ({len(intel.warnings)})"):
            for w in intel.warnings:
                st.markdown(f"- {w}")


def tab_products(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    if not intel.products:
        st.info(f"Products: {NOT_VERIFIED} from the available sources.")
    for p in intel.products:
        with st.container(border=True):
            st.markdown(f"**{p.product_name}**")
            if p.description:
                st.write(p.description)
            for label, val in [("Healthcare use case", p.healthcare_use_case), ("Features", ", ".join(p.features)),
                               ("Target users", ", ".join(p.target_users)), ("Integrations", ", ".join(p.integrations))]:
                if val:
                    st.markdown(f"- **{label}:** {val}")
            st.markdown(f"- **Pricing:** {p.pricing or 'Not publicly verified'}")
            st.caption(f"Sources: {src_links(p.source_ids, sources)}")


def tab_technology(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    st.caption("✅ Verified = observed directly (page markup/headers) or stated by the company. 🔶 Inferred = deduced from indirect evidence.")
    if not intel.technologies:
        st.info(f"Technology: {NOT_VERIFIED} from the available sources.")
    for t in intel.technologies:
        with st.container(border=True):
            st.markdown(f"**{t.category}: {t.name}** · {conf(t.confidence)}")
            if t.reason:
                st.caption(f"Reason: {t.reason}")
            st.caption(f"Sources: {src_links(t.source_ids, sources)}")


def tab_competitors(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    if not intel.competitors:
        st.info(f"Competitors: {NOT_VERIFIED} from the available sources.")
    for c in intel.competitors:
        with st.container(border=True):
            st.markdown(f"**{c.name}** · {c.category} · {conf(c.confidence)}")
            if c.reason:
                st.write(c.reason)
            if c.overlap:
                st.caption(f"Overlap: {c.overlap}")
            st.caption(f"Sources: {src_links(c.source_ids, sources)}")


def tab_funding(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    f = intel.funding
    if f.total_funding:
        st.metric("Total funding (as stated by a source)", f.total_funding)
        st.caption(f"Source: {src_links(f.total_funding_source_ids, sources)}")
    else:
        st.info(f"Total funding: {NOT_VERIFIED}. (Totals are never calculated from partial data.)")
    if not f.events:
        st.info(f"Funding rounds: {NOT_VERIFIED} from the available sources.")
    for e in f.events:
        with st.container(border=True):
            st.markdown(f"**{e.round_type or 'Round not verified'}** · {e.amount or 'Amount not verified'} · {e.date or 'Date not verified'}")
            if e.investors:
                st.write("Investors: " + ", ".join(e.investors))
            st.caption(f"Sources: {src_links(e.source_ids, sources)}")


def tab_signals(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    if not intel.signals:
        st.info(f"Growth signals: {NOT_VERIFIED} from the available sources.")
    for s in intel.signals:
        with st.container(border=True):
            st.markdown(f"**{s.signal_type}** · {s.importance} importance · {s.date or 'date not verified'}")
            st.write(s.description)
            st.caption(f"Sources: {src_links(s.source_ids, sources)}")


def tab_people(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    st.caption("Public professional information only. Contact details are never collected.")
    if not intel.people:
        st.info(f"People: {NOT_VERIFIED} from the available sources.")
    for p in intel.people:
        with st.container(border=True):
            st.markdown(f"**{p.name}** - {p.role} · _{p.relevant_function or 'Other'}_")
            st.caption(f"Sources: {src_links(p.source_ids, sources)}")


def tab_gtm(intel: CompanyIntelligence, sources: dict[str, Source]) -> None:
    g = intel.gtm
    if not g:
        st.info(f"GTM analysis: {NOT_VERIFIED} (AI analysis did not run or found nothing verifiable).")
        return
    st.markdown("##### ✅ Verified evidence")
    if g.verified_evidence:
        for cl in g.verified_evidence:
            st.markdown(f"- {cl.statement}  \n  <span class='kyc-note'>Sources: {src_links(cl.source_ids, sources)}</span>", unsafe_allow_html=True)
    else:
        st.write(NOT_VERIFIED)
    st.markdown("##### 🔶 AI inference")
    st.warning("Everything below is reasoning by a local language model over the evidence - an interpretation, not a fact.")
    sections = [("Ideal customer profile", g.icp), ("Target customers", g.target_customers), ("Business needs", g.business_needs),
                ("Potential pain points", g.pain_points), ("Relevant functions", g.relevant_functions),
                ("Potential opportunities", g.opportunities), ("Potential outreach angles", g.outreach_angles)]
    for title, items in sections:
        if items:
            st.markdown(f"**{title}**")
            for it in items:
                st.markdown(f"- {it}")
    if g.why_this_company:
        st.markdown("**Why this company?**")
        st.write(g.why_this_company)
    st.caption(f"Confidence: **{g.confidence}** - derived from how much verified evidence was found (not from the model's opinion).")


def tab_sources(slug: str, sources: dict[str, Source]) -> None:
    store = company_store()
    evidence = {e.source_id: e for e in store.load_evidence(slug)}
    if not sources:
        st.info("No sources stored.")
        return
    types = sorted({s.source_type for s in sources.values()})
    chosen = st.multiselect("Filter by source type", types, default=types, key=f"srcfilter_{slug}")
    st.caption(f"{len(sources)} sources · click a title to open the original page.")
    for sid, s in sorted(sources.items()):
        if s.source_type not in chosen:
            continue
        ev = evidence.get(sid)
        with st.expander(f"{sid} · {s.title}"):
            st.markdown(f"[{s.url}]({s.url})")
            st.markdown(f"**Type:** {s.source_type} · **Retrieved:** {s.retrieved_at[:19]} · **Relevance:** {ev.relevance if ev else 'n/a'}"
                        + ("" if not ev or ev.fetched_full_page else " · _search snippet only_"))
            st.caption(s.snippet)
            if ev:
                st.text_area("Stored evidence text", ev.content[:3000], height=160, disabled=True, key=f"ev_{slug}_{sid}")


def render_workspace(slug: str) -> None:
    store = company_store()
    intel = store.load(slug)
    if not intel:
        st.warning("That company could not be loaded.")
        return
    sources = {s.source_id: s for s in store.load_sources(slug)}
    st.markdown(f"## {intel.company.name}")
    st.caption(intel.company.headline + (f" · [{intel.company.official_website}]({intel.company.official_website})" if intel.company.official_website else ""))
    m1, m2, m3 = st.columns(3)
    m1.metric("Sources", len(sources))
    m2.metric("Researched", (intel.researched_at or "n/a")[:10])
    m3.metric("Model", intel.ollama_model or "n/a")
    if intel.embedding_id.startswith("hash"):
        st.warning("This company was indexed with the lightweight fallback embedder (keyword-style matching). "
                   "Retrieval quality is lower. Fix the embedding model in Settings, then Companies → Re-index.")
    st.markdown("##### Quick actions")
    render_quick_actions(slug, intel, sources, prefix=f"ws_{slug}")
    tabs = st.tabs(["Overview", "Products", "Technology", "Competitors", "Funding", "Signals", "People", "GTM", "Sources", "AI Chat"])
    with tabs[0]: tab_overview(intel, sources)
    with tabs[1]: tab_products(intel, sources)
    with tabs[2]: tab_technology(intel, sources)
    with tabs[3]: tab_competitors(intel, sources)
    with tabs[4]: tab_funding(intel, sources)
    with tabs[5]: tab_signals(intel, sources)
    with tabs[6]: tab_people(intel, sources)
    with tabs[7]: tab_gtm(intel, sources)
    with tabs[8]: tab_sources(slug, sources)
    with tabs[9]: render_chat(slug, intel, sources, prefix=f"wschat_{slug}")


# --------------------------------------------------------------------------- pages
def page_home() -> None:
    st.markdown("""<div class="kyc-hero"><h1>🏥 Know Your Company</h1>
    <p>AI-powered healthcare company intelligence. Research a company, build a local evidence base, and ask questions grounded in cited sources.</p></div>""",
                unsafe_allow_html=True)
    research_form("home", advanced=False)
    with st.expander("No internet or just want to try it? Load the fictional sample company"):
        st.caption("Indexes bundled demo evidence about a made-up company. No web access needed.")
        if st.button("Load sample company", key="load_sample"):
            slug = run_with_progress("Northwind Health Clearinghouse (Sample)", "", "", run_ai=True, sample=True)
            if slug:
                st.session_state["_open"] = slug
                st.session_state["_just_researched"] = slug
    shown = st.session_state.pop("_just_researched", None) or st.session_state.get("active_company")
    if shown:
        st.divider()
        render_workspace(shown)


def page_research() -> None:
    st.markdown("## 🔍 Research Company")
    st.caption("Runs a bounded set of public searches plus the official website, indexes the evidence locally, then analyses it with your local model.")
    research_form("research", advanced=True)
    shown = st.session_state.pop("_just_researched", None) or st.session_state.get("active_company")
    if shown:
        st.divider()
        render_workspace(shown)


def page_companies() -> None:
    st.markdown("## 🏢 Companies")
    store = company_store()
    items = store.list_companies()
    if not items:
        st.info("No companies yet. Research one on the Home page.")
        return
    for it in items:
        with st.container(border=True):
            c1, c2, c3, c4 = st.columns([4, 1, 1, 1])
            c1.markdown(f"**{it['name']}**  \n<span class='kyc-note'>{it['headline']} · researched {it['researched_at'][:10]}</span>", unsafe_allow_html=True)
            if c2.button("Open", key=f"open_{it['slug']}"):
                st.session_state["_open"] = it["slug"]
                st.rerun()
            if c3.button("Re-index", key=f"reindex_{it['slug']}", help="Rebuild vectors from the saved evidence (use after changing the embedding model)."):
                try:
                    with st.spinner("Re-indexing..."):
                        n = reindex_company(it["slug"])
                    st.success(f"Re-indexed {n} chunks.")
                except Exception as exc:
                    st.error(f"Re-index failed: {exc}")
            if c4.button("Delete", key=f"del_{it['slug']}"):
                st.session_state["_confirm_delete"] = it["slug"]
            if st.session_state.get("_confirm_delete") == it["slug"]:
                st.warning(f"Delete {it['name']} and all its saved evidence? This cannot be undone.")
                d1, d2 = st.columns(2)
                if d1.button("Yes, delete", key=f"yes_{it['slug']}"):
                    try:
                        from rag.vector_store import get_default_store
                        get_default_store().delete_company(it["slug"])
                    except Exception:
                        pass                       # vectors are rebuilt on demand; files are what matter
                    store.delete(it["slug"])
                    for k in (f"chat_{it['slug']}", f"last_quick_{it['slug']}"):
                        st.session_state.pop(k, None)
                    st.session_state.pop("_confirm_delete", None)
                    if st.session_state.get("active_company") == it["slug"]:
                        st.session_state["_open"] = ""
                    st.session_state["_just_researched"] = None
                    st.rerun()
                if d2.button("Cancel", key=f"no_{it['slug']}"):
                    st.session_state.pop("_confirm_delete", None)
                    st.rerun()
    active = st.session_state.get("active_company")
    if active and store.exists(active):
        st.divider()
        render_workspace(active)


def _pick_company(key: str) -> Optional[str]:
    items = company_store().list_companies()
    if not items:
        st.info("No companies yet. Research one on the Home page.")
        return None
    labels = _slug_label(items)
    slugs = list(labels)
    default = st.session_state.get("active_company")
    idx = slugs.index(default) if default in slugs else 0
    return st.selectbox("Company", slugs, index=idx, format_func=lambda s: labels[s], key=key)


def page_chat() -> None:
    st.markdown("## 💬 AI Chat")
    slug = _pick_company("chat_company")
    if not slug:
        return
    store = company_store()
    intel = store.load(slug)
    if not intel:
        return
    sources = {s.source_id: s for s in store.load_sources(slug)}
    render_quick_actions(slug, intel, sources, prefix=f"chatpage_{slug}")
    st.divider()
    render_chat(slug, intel, sources, prefix=f"chatpage_{slug}")


def page_reports() -> None:
    st.markdown("## 📄 Reports")
    slug = _pick_company("report_company")
    if not slug:
        return
    store = company_store()
    intel = store.load(slug)
    if not intel:
        return
    if st.button("📝 Generate Markdown Report", type="primary"):
        report = build_report(intel, store.load_sources(slug))
        store.save_report(slug, report)
        st.success(f"Saved to data/companies/{slug}/report.md")
    report = store.load_report(slug)
    if report:
        st.download_button("⬇️ Download report (.md)", report, file_name=f"{slug}-report.md", mime="text/markdown")
        with st.container(border=True):
            st.markdown(report)
    else:
        st.info("No report yet. Click “Generate Markdown Report”.")


def page_settings() -> None:
    st.markdown("## ⚙️ Settings")
    s = settings()
    status = ollama_status()
    if status.ready:
        st.success(f"🟢 Ollama Connected - model **{status.model}** is installed.")
    elif status.connected:
        st.warning(f"🟡 Ollama is running, but model **{status.model}** is not installed. Run: `ollama pull {status.model}`")
        if status.installed_models:
            st.caption("Installed models: " + ", ".join(status.installed_models))
    else:
        st.error("🔴 Ollama Not Available")
        st.caption(status.error or "")
    with st.form("settings_form"):
        url = st.text_input("Ollama Base URL", s.ollama_base_url)
        model = st.text_input("Ollama Model", s.ollama_model, help="Pick a small model for 8 GB RAM, e.g. qwen3:1.7b")
        emb = st.text_input("Embedding Model", s.embedding_model, help="A sentence-transformers model id. Changing it requires re-indexing companies.")
        c1, c2, c3 = st.columns(3)
        top_k = c1.number_input("Top K", 1, 20, int(s.top_k))
        chunk = c2.number_input("Chunk Size (≈tokens)", 100, 2000, int(s.chunk_size), step=50)
        overlap = c3.number_input("Chunk Overlap (≈tokens)", 0, 1000, int(s.chunk_overlap), step=10)
        saved = st.form_submit_button("💾 Save settings", type="primary")
    if saved:
        changed_embed = emb.strip() != s.embedding_model
        s.ollama_base_url, s.ollama_model, s.embedding_model = url.strip(), model.strip(), emb.strip()
        s.top_k, s.chunk_size, s.chunk_overlap = int(top_k), int(chunk), int(overlap)
        save_settings(s)
        _cached_status.clear()
        from rag.embeddings import reset_embedder_cache
        reset_embedder_cache()
        st.success("Settings saved.")
        if changed_embed:
            st.info("Embedding model changed: open **Companies** and click **Re-index** for each saved company.")
        st.rerun()
    st.markdown("##### Embedding model check")
    st.caption("Loads the embedding model (first use may download ~130 MB). Not run automatically.")
    if st.button("Check embedding model"):
        from rag.embeddings import get_embedder
        with st.spinner("Loading embedding model..."):
            e = get_embedder()
        if e.is_fallback:
            st.warning(f"Fallback embedder active (`{e.id}`). {e.warning}")
        else:
            st.success(f"Loaded `{e.id}` ({e.dim} dimensions).")
    st.markdown("##### Storage")
    st.code(f"Data folder: {s.data_dir}\nVector store: {s.vector_dir}", language="text")
    st.caption("Everything stays on this computer. No API keys are used or required.")


# --------------------------------------------------------------------------- main
def main() -> None:
    # Programmatic navigation/selection must be applied before the widgets are created.
    if "_goto" in st.session_state:
        st.session_state["page"] = st.session_state.pop("_goto")
    if "_open" in st.session_state:
        st.session_state["active_company"] = st.session_state.pop("_open")

    store = company_store()
    items = store.list_companies()
    labels = _slug_label(items)
    if st.session_state.get("active_company") not in ([""] + list(labels)):
        st.session_state["active_company"] = ""          # e.g. the company was deleted
    with st.sidebar:
        st.markdown("### 🏥 Know Your Company")
        page = st.radio("Navigate", PAGES, key="page", label_visibility="collapsed")
        st.divider()
        status = ollama_status()
        if status.ready:
            st.markdown(f"🟢 **Ollama Connected**  \n<span class='kyc-note'>{status.model}</span>", unsafe_allow_html=True)
        elif status.connected:
            st.markdown(f"🟡 **Model missing**  \n<span class='kyc-note'>ollama pull {status.model}</span>", unsafe_allow_html=True)
        else:
            st.markdown("🔴 **Ollama Not Available**  \n<span class='kyc-note'>See OLLAMA_SETUP.md</span>", unsafe_allow_html=True)
        if labels:
            options = [""] + list(labels)
            st.selectbox("Active company", options, format_func=lambda s: labels.get(s, "— none —"), key="active_company")
    {
        PAGES[0]: page_home, PAGES[1]: page_research, PAGES[2]: page_companies,
        PAGES[3]: page_chat, PAGES[4]: page_reports, PAGES[5]: page_settings,
    }[page]()


main()
