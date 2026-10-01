"""Public professional information only - no contact details, no private data."""
from __future__ import annotations

import re

from pydantic import ValidationError

from ai.analyzer import run_extraction
from models.company import ModuleStatus
from models.person import FUNCTIONS, Person, PersonList
from research.common import ModuleResult, ResearchContext, cue_near, failed, finish, hit_ids, norm

_NAME_RE = re.compile(r"^[^\W\d_][^\W\d_'’.\-]*(?:[.'’\-][^\W\d_]+)?(?: [^\W\d_][^\W\d_'’.\-]*(?:[.'’\-][^\W\d_]+)?){1,3}$", re.UNICODE)
_ROLE_STOP = {"of", "the", "and", "for", "at", "to", "a"}
_ABBR = {
    "ceo": ["chief", "executive"], "cto": ["chief", "technology"], "cfo": ["chief", "financial"],
    "coo": ["chief", "operating"], "cmo": ["chief", "marketing"], "cro": ["chief", "revenue"],
    "cpo": ["chief", "product"], "vp": ["vice", "president"],
}
_FUNCTION_KEYWORDS = [
    ("CEO", r"\bceo\b|chief executive"), ("Founder", r"founder|founded"), ("Revenue", r"revenue|\bcro\b|\bcfo\b|finance|billing"),
    ("Sales", r"sales|business development|account exec"), ("Marketing", r"marketing|\bcmo\b|brand"),
    ("Growth", r"growth"), ("Partnerships", r"partner|alliances"),
    ("Product", r"product|\bcpo\b"), ("Technology", r"\bcto\b|technology|engineering|architect|data|software"),
    ("Operations", r"operations|\bcoo\b|delivery|customer success"),
]


def role_alternatives(role: str) -> list[list[str]]:
    """Acceptable word-sets for a role, e.g. 'CEO and Co-founder' -> [[ceo], [chief, executive], [founder], ...]."""
    alts: list[list[str]] = []
    for part in re.split(r"\band\b|&|/|,|;|\|", role.lower()):
        toks = [t for t in norm(part).split() if t not in _ROLE_STOP]
        if not toks:
            continue
        alts.append(toks)
        if len(toks) == 1 and toks[0] in _ABBR:
            alts.append(_ABBR[toks[0]])
        for abbr, phrase in _ABBR.items():
            if abbr != "vp" and all(w in toks for w in phrase) and "officer" in toks:
                alts.append([abbr])
        if "vice" in toks and "president" in toks:
            alts.append(["vp"])
        if "founder" in toks:
            alts.append(["founder"])
    return alts


def role_supported(name: str, role: str, content: str) -> bool:
    """Is `name` tied to `role` *on the same line* (or the line right after a short name line)?

    Requires ALL words of at least one acceptable role phrase. Neighbouring people's titles
    therefore cannot vouch for someone else's role.
    """
    alts = role_alternatives(role)
    if not alts:
        return False
    name_re = re.compile(r"\W+".join(re.escape(t) for t in name.split()), re.IGNORECASE)
    lines = content.splitlines()
    for i, line in enumerate(lines):
        m = name_re.search(line)
        if not m:
            continue
        segment = line[max(0, m.start() - 60): m.end() + 160]
        if len(line.strip()) < 80 and i + 1 < len(lines):      # vertical layout: name on one line, title on the next
            segment += " " + lines[i + 1][:100]
        words = set(norm(segment).split())
        if any(all(w in words for w in alt) for alt in alts):
            return True
    return False


def infer_function(role: str) -> str:
    low = role.lower()
    for label, pattern in _FUNCTION_KEYWORDS:
        if re.search(pattern, low):
            return label
    return "Other"


def extract_people(ctx: ResearchContext) -> ModuleResult:
    hits = ctx.retrieve("leadership team CEO founder chief officer head of vice president management")
    if not hits:
        return ModuleResult([], ModuleStatus(status="partial", detail="No evidence retrieved for people."))
    data = run_extraction(ctx.llm, company=ctx.company_name, kind="people", evidence=ctx.block(hits), schema_model=PersonList)
    if data is None:
        return failed("The model did not return valid JSON for people.")
    pool, items, dropped, seen = hit_ids(hits), [], 0, set()
    for raw in (data.get("people") or data.get("items") or [])[:12]:
        try:
            p = Person.model_validate(raw)
        except ValidationError:
            dropped += 1
            continue
        if not _NAME_RE.match(p.name) or "@" in p.name or not p.role:
            dropped += 1
            continue
        key = norm(p.name)
        if key in seen:
            continue
        sids = ctx.supporting_sources(p.name, p.source_ids, pool)
        # Person must be named in a source AND that source must tie the name to this exact role.
        sids = [sid for sid in sids if role_supported(p.name, p.role, ctx.evidence[sid].content)]
        if not sids:
            dropped += 1
            continue
        seen.add(key)
        p.source_ids = sids
        p.company = p.company or ctx.company_name
        if p.relevant_function not in FUNCTIONS:
            p.relevant_function = infer_function(p.role)
        items.append(p)
    return ModuleResult(items, finish("person(s)", items, dropped))
