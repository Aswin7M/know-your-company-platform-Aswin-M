"""Turns retrieved evidence + a question into a grounded, cited answer."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from ai import prompts
from ai.ollama import OllamaClient, OllamaError, simplify_schema
from config import Settings, load_settings
from models.source import Source
from rag.retriever import Retriever
from rag.sources import extract_cited_ids, render_sources_markdown
from rag.vector_store import SearchHit, VectorStoreError


@dataclass
class ChatAnswer:
    text: str                                  # final markdown shown to the user
    raw: str = ""
    hits: list[SearchHit] = field(default_factory=list)
    cited_ids: list[str] = field(default_factory=list)
    unknown_ids: list[str] = field(default_factory=list)
    insufficient: bool = False
    error: Optional[str] = None


# ---- evidence formatting ------------------------------------------------------------
def build_evidence_block(hits: list[SearchHit], max_chars: int = 6000) -> str:
    """Label every chunk with its SRC id; stop before the context budget is exceeded."""
    parts, used = [], 0
    for h in hits:
        header = f"[{h.source_id}] {h.title} ({h.source_type}) {h.url}"
        body = h.text.strip()
        room = max_chars - used - len(header) - 4
        if room < 200 and parts:
            break
        body = body[: max(room, 200)]
        block = f"{header}\n{body}"
        parts.append(block)
        used += len(block) + 2
    return "\n\n".join(parts)


def _history_block(history: Optional[list[dict]], max_turns: int = 4) -> str:
    if not history:
        return ""
    lines = []
    for m in history[-max_turns * 2:]:
        role = "User" if m.get("role") == "user" else "Assistant"
        text = " ".join(str(m.get("content", "")).split())
        # drop the previous answer's Sources section to save context
        text = re.split(r"\*\*Sources\*\*", text)[0][:600]
        lines.append(f"{role}: {text}")
    return "Earlier conversation (for context only - not evidence):\n" + "\n".join(lines) + "\n\n"


def _clean_model_text(text: str) -> str:
    """Remove any 'Sources' section the model wrote itself - we add a trustworthy one."""
    return re.split(r"\n\s*#{0,4}\s*\**Sources\**\s*:?\s*\n", "\n" + text, maxsplit=1, flags=re.I)[0].strip()


# ---- chat / quick actions -----------------------------------------------------------
def answer_question(question: str, *, company_slug: str, company_name: str,
                    sources: dict[str, Source], retriever: Retriever, llm: OllamaClient,
                    history: Optional[list[dict]] = None, retrieval_query: Optional[str] = None,
                    settings: Optional[Settings] = None) -> ChatAnswer:
    settings = settings or load_settings()
    try:
        hits = retriever.search(company_slug, retrieval_query or question, top_k=settings.top_k)
    except VectorStoreError as exc:
        return ChatAnswer(text=f"⚠️ {exc}", error=str(exc))
    except Exception as exc:                                   # embedding failure etc.
        msg = f"Could not search the company evidence: {exc}"
        return ChatAnswer(text=f"⚠️ {msg}", error=msg)

    if not hits:
        try:
            unindexed = retriever.store.count(company_slug) == 0
        except Exception:
            unindexed = False
        if unindexed:
            msg = ("This company has no searchable index for the current embedding model. "
                   "Open **Companies** and click **Re-index** (or research it again).")
            return ChatAnswer(text=f"⚠️ {msg}", error=msg, insufficient=True)
        return ChatAnswer(
            text=f"**Answer**\n{prompts.NOT_VERIFIED_PHRASE}\n\nNo relevant evidence was found for this question "
                 f"in the collected sources for {company_name}.",
            insufficient=True,
        )

    user = prompts.CHAT_USER_TEMPLATE.format(
        company=company_name,
        evidence=build_evidence_block(hits, settings.max_context_chars),
        history=_history_block(history),
        question=question,
        format=prompts.CHAT_FORMAT,
    )
    try:
        raw = llm.chat(
            [{"role": "system", "content": prompts.SYSTEM_PROMPT}, {"role": "user", "content": user}],
            temperature=0.2, num_predict=700,
        )
    except OllamaError as exc:
        return ChatAnswer(text=f"⚠️ {exc}", hits=hits, error=str(exc))

    body = _clean_model_text(raw)
    known = set(sources) | {h.source_id for h in hits}
    cited = [c for c in extract_cited_ids(body) if c in known]
    unknown = [c for c in extract_cited_ids(body) if c not in known]

    text = body
    if unknown:
        text += ("\n\n> ⚠️ The model cited source IDs that do not exist (" + ", ".join(unknown) +
                 "). Ignore those citations.")
    src_lookup = {sid: sources[sid] for sid in known if sid in sources}
    if cited:
        text += "\n\n**Sources**\n" + render_sources_markdown(cited, src_lookup)
    else:
        text += "\n\n> ℹ️ The answer did not cite specific source IDs; treat it with extra caution."
    retrieved = [h.source_id for h in hits if h.source_id not in cited and h.source_id in src_lookup]
    if retrieved:
        text += "\n\n_Also retrieved (not cited):_ " + ", ".join(dict.fromkeys(retrieved))
    return ChatAnswer(text=text, raw=raw, hits=hits, cited_ids=cited, unknown_ids=unknown)


def run_quick_action(action_key: str, **kwargs: Any) -> ChatAnswer:
    _label, question, retrieval_query = prompts.QUICK_ACTIONS[action_key]
    return answer_question(question, retrieval_query=retrieval_query, **kwargs)


# ---- structured extraction ----------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def extract_json(text: str) -> Optional[Any]:
    """Best-effort JSON parse of a model reply (handles code fences and chatter)."""
    if not text:
        return None
    cleaned = _FENCE_RE.sub("", text.strip()).strip()
    try:
        return json.loads(cleaned)
    except ValueError:
        pass
    start = min([i for i in (cleaned.find("{"), cleaned.find("[")) if i != -1], default=-1)
    if start == -1:
        return None
    opener = cleaned[start]
    closer = "}" if opener == "{" else "]"
    end = cleaned.rfind(closer)
    if end <= start:
        return None
    try:
        return json.loads(cleaned[start:end + 1])
    except ValueError:
        return None


def run_extraction(llm: OllamaClient, *, company: str, kind: str, evidence: str,
                   schema_model: Optional[type] = None, extra_focus: str = "",
                   retries: int = 1) -> Optional[dict]:
    """One JSON extraction call. Returns a dict, or None if the model never produced valid JSON."""
    task, shape = prompts.TASKS[kind]
    task = task.replace("{focus}", extra_focus)
    user = prompts.EXTRACTION_USER_TEMPLATE.format(company=company, evidence=evidence, task=task, shape=shape)
    schema = simplify_schema(schema_model.model_json_schema()) if schema_model else None
    messages = [{"role": "system", "content": prompts.EXTRACTION_SYSTEM}, {"role": "user", "content": user}]
    for attempt in range(retries + 1):
        raw = llm.chat(messages, json_schema=schema, json_mode=schema is None,
                       temperature=0.1 if attempt == 0 else 0.3, num_predict=900)
        data = extract_json(raw)
        if isinstance(data, dict):
            return data
        if isinstance(data, list):
            return {"items": data}
    return None
