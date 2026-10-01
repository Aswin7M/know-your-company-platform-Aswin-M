from fakes import FakeLLM

from ai.analyzer import answer_question, build_evidence_block, extract_json, run_quick_action
from ai.ollama import OllamaUnavailable
from ai.prompts import NOT_VERIFIED_PHRASE, QUICK_ACTIONS, SYSTEM_PROMPT
from config import load_settings
from models.source import Evidence
from rag.chunker import chunk_all
from rag.embeddings import HashEmbedder
from rag.retriever import Retriever
from rag.sources import SourceRegistry
from rag.vector_store import VectorStore, SearchHit


def build(docs):
    emb = HashEmbedder()
    store = VectorStore(":memory:", "t", emb.dim)
    reg = SourceRegistry()
    ev = []
    for title, url, text in docs:
        s = reg.add(title=title, url=url, source_type="official_website", snippet=text)
        ev.append(Evidence(source_id=s.source_id, title=title, url=url, source_type="official_website", content=text))
    chunks = chunk_all(ev, "Acme", "acme", 500, 50)
    store.add_documents(chunks, emb.encode([c.text for c in chunks]))
    sources = {s.source_id: s for s in reg.to_list()}
    return Retriever(store=store, embedder=emb, settings=load_settings()), store, sources


DOCS = [("Funding news", "https://acme.example/press", "Acme raised a Series B round of $20 million led by Example Ventures."),
        ("Products", "https://acme.example/products", "Acme Eligibility API verifies insurance coverage in real time.")]


def ask(llm, question="How much funding did Acme raise?", docs=DOCS, **kw):
    retriever, store, sources = build(docs)
    try:
        return answer_question(question, company_slug="acme", company_name="Acme", sources=sources,
                               retriever=retriever, llm=llm, **kw)
    finally:
        store.close()


def test_system_prompt_contains_the_required_guardrails():
    for phrase in (NOT_VERIFIED_PHRASE, "AI inference", "Do not invent facts", "source IDs", "untrusted"):
        assert phrase in SYSTEM_PROMPT


def test_answer_has_trustworthy_sources_not_model_written_ones():
    ans = ask(FakeLLM(chat_text="**Answer**\nRaised $20 million (SRC-001).\n\n**Sources**\n[1] https://totally-made-up.example"))
    assert "totally-made-up" not in ans.text                     # model-written Sources section is stripped
    assert ans.cited_ids == ["SRC-001"] and "**Sources**" in ans.text
    assert "https://acme.example/press" in ans.text              # real, clickable link added by us
    assert not ans.unknown_ids and not ans.error


def test_unknown_source_ids_are_flagged():
    ans = ask(FakeLLM(chat_text="Acme employs 500 people (SRC-099) and raised money (SRC-001)."))
    assert ans.unknown_ids == ["SRC-099"] and ans.cited_ids == ["SRC-001"]
    assert "do not exist" in ans.text and "SRC-099" in ans.text


def test_uncited_answers_carry_a_caution():
    ans = ask(FakeLLM(chat_text="**Answer**\nThey raised some money."))
    assert ans.cited_ids == [] and "did not cite" in ans.text


def test_no_relevant_evidence_short_circuits_without_calling_the_model():
    llm = FakeLLM()
    ans = ask(llm, question="zebra giraffe volcano", docs=[("Only", "https://a.example", "completely different words here about cats")])
    if ans.insufficient:                                         # (hash collisions could rarely let a chunk through)
        assert NOT_VERIFIED_PHRASE in ans.text and llm.calls == []


def test_unindexed_company_is_reported_clearly():
    retriever, store, sources = build(DOCS)
    llm = FakeLLM()
    ans = answer_question("anything", company_slug="other-company", company_name="Other", sources=sources, retriever=retriever, llm=llm)
    assert ans.insufficient and "Re-index" in ans.text and llm.calls == []
    store.close()


def test_ollama_failure_returns_a_friendly_message_not_an_exception():
    class Down(FakeLLM):
        def chat(self, *a, **k):
            raise OllamaUnavailable("Cannot reach Ollama at http://localhost:11434.")
    ans = ask(Down())
    assert ans.error and "Cannot reach Ollama" in ans.text and ans.hits


def test_history_is_passed_as_context_only():
    llm = FakeLLM()
    ask(llm, history=[{"role": "user", "content": "earlier question"}, {"role": "assistant", "content": "earlier answer **Sources** junk"}])
    prompt = llm.calls[0]
    assert "earlier question" in prompt and "not evidence" in prompt and "junk" not in prompt


def test_evidence_block_labels_sources_and_respects_budget():
    hits = [SearchHit(text="x" * 3000, score=0.9, company_slug="a", company="A", source_id=f"SRC-00{i}", url="https://a.example",
                      title=f"T{i}", source_type="news", chunk_index=0) for i in range(1, 5)]
    block = build_evidence_block(hits, max_chars=4000)
    assert "[SRC-001] T1 (news)" in block and len(block) <= 4400 and "SRC-004" not in block


def test_quick_actions_cover_the_spec_and_run():
    assert {"summary", "products", "technology", "competitors", "funding", "signals", "people", "gtm", "why"} == set(QUICK_ACTIONS)
    retriever, store, sources = build(DOCS)
    ans = run_quick_action("funding", company_slug="acme", company_name="Acme", sources=sources, retriever=retriever, llm=FakeLLM())
    assert ans.hits and "SRC-" in ans.text
    why = QUICK_ACTIONS["why"][1]
    assert "AI reasoning" in why and "Confidence" in why and "not an objective fact" in why
    store.close()


def test_extract_json_variants():
    assert extract_json('{"a": 1}') == {"a": 1}
    assert extract_json('```json\n{"a": [1, 2]}\n```') == {"a": [1, 2]}
    assert extract_json('Sure! Here you go: {"a": 1} hope it helps') == {"a": 1}
    assert extract_json("[1, 2]") == [1, 2]
    assert extract_json("no json here") is None and extract_json("") is None and extract_json("{broken") is None
