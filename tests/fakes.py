"""Test doubles: no Ollama, no network."""
from __future__ import annotations

import json
import re

import requests

from ai import prompts
from ai.ollama import OllamaStatus


def ids_for(prompt: str, *keywords: str) -> list[str]:
    """SRC ids in the evidence block whose header line mentions any keyword."""
    out = []
    for line in prompt.splitlines():
        m = re.match(r"\[(SRC-\d+)\](.*)", line)
        if m and any(k.lower() in m.group(2).lower() for k in keywords):
            out.append(m.group(1))
    return out


class FakeLLM:
    """Mimics a small local model. Answers are mostly faithful but include deliberate
    fabrications (marked HALLUCINATION) that the grounding layer must discard."""

    model = "fake-model:1b"

    def __init__(self, ready: bool = True, chat_text: str | None = None, fail_json: bool = False):
        self.ready, self.chat_text, self.fail_json = ready, chat_text, fail_json
        self.calls: list[str] = []

    def status(self, timeout: float = 2.0) -> OllamaStatus:
        if self.ready:
            return OllamaStatus(connected=True, model=self.model, model_installed=True, installed_models=[self.model])
        return OllamaStatus(connected=False, model=self.model, error="Cannot reach Ollama (fake).")

    def _kind(self, user: str) -> str | None:
        for kind, (task, _shape) in prompts.TASKS.items():
            if task.split("{focus}")[0][:45] in user:
                return kind
        return None

    def chat(self, messages, *, json_schema=None, json_mode=False, temperature=0.2, num_predict=None) -> str:
        user = messages[-1]["content"]
        self.calls.append(user)
        kind = self._kind(user)
        if kind is None:                                   # normal chat / quick action
            if self.chat_text is not None:
                return self.chat_text
            src = (ids_for(user, "press", "series") or ids_for(user, "") or ["SRC-001"])[0]
            return (f"**Answer**\nThe company raised a Series B of $20 million ({src}).\n\n**Evidence**\n- {src} : funding press release\n\n"
                    f"**AI inference**\nNone.\n\n**Sources**\n[1] totally-made-up.example")
        if self.fail_json:
            return "I'm sorry, I cannot produce JSON right now."
        return json.dumps(getattr(self, f"_{kind}")(user))

    # ---- canned structured outputs -------------------------------------------------
    def _company(self, u):
        return {"industry": "Healthcare Technology", "subindustry": "Healthcare clearinghouse",
                "company_type": "Private", "headquarters": "Columbus, Ohio", "founded": "2017",
                "markets": ["United States"], "business_model": "API platform",
                "target_customers": ["Medical billing companies", "Digital health companies"],
                "description": "Northwind Health Clearinghouse provides healthcare connectivity APIs for eligibility, "
                               "claims and remittance data.", "source_ids": ids_for(u, "Northwind Health Clearinghouse -", "About")}

    def _products(self, u):
        return {"products": [
            {"product_name": "Northwind Eligibility API", "description": "Returns real-time insurance eligibility and "
             "benefits information for patients.", "pricing": "$99/month", "features": ["Sandbox"],
             "source_ids": ids_for(u, "Products")},
            {"product_name": "Northwind Claims Gateway", "description": "Submit claims electronically and track claim status.",
             "source_ids": ["SRC-099"]},                     # bogus citation -> must be repaired from the evidence
            {"product_name": "QuantumBilling AI", "description": "An AI biller.", "source_ids": ids_for(u, "Products")},  # HALLUCINATION
        ]}

    def _technology(self, u):
        return {"technologies": [
            {"category": "Cloud", "name": "AWS", "confidence": "Verified", "source_ids": ids_for(u, "documentation")},
            {"category": "Infrastructure", "name": "Kubernetes", "confidence": "Verified", "source_ids": ids_for(u, "careers")},
            {"category": "APIs", "name": "FHIR", "confidence": "Verified", "source_ids": ids_for(u, "documentation")},
            {"category": "Database", "name": "Snowflake", "confidence": "Verified", "source_ids": ids_for(u, "documentation")},  # HALLUCINATION
        ]}

    def _competitors(self, u):
        return {"competitors": [
            {"name": "Contoso Claims Exchange", "category": "Direct", "reason": "Incumbent", "source_ids": ids_for(u, "market")},
            {"name": "Fabrikam Health Network", "category": "Adjacent", "source_ids": ids_for(u, "launches")},
            {"name": "Globex Medical", "category": "Direct", "source_ids": ids_for(u, "market")},          # HALLUCINATION
        ]}

    def _funding(self, u):
        pr = ids_for(u, "raises")
        return {"events": [
            {"date": "2025", "round_type": "Series B", "amount": "$20 million",
             "investors": ["Fabrikam Ventures", "Adventure Works Capital", "Imaginary Capital"], "source_ids": pr},
            {"date": "2026", "round_type": "Series C", "amount": "$500 million", "investors": ["Phantom Fund"], "source_ids": pr},  # HALLUCINATION
        ], "total_funding": "$32 million", "total_funding_source_ids": pr}

    def _signals(self, u):
        return {"signals": [
            {"signal_type": "Product launch", "date": None, "description": "Launched Northwind Eligibility API v2 with "
             "faster responses and expanded payer coverage", "importance": "High", "source_ids": ids_for(u, "launches")},
            {"signal_type": "Acquisition", "description": "Acquired MedCo for one billion dollars in a blockbuster deal",
             "importance": "High", "source_ids": ids_for(u, "launches")},                                   # HALLUCINATION
        ]}

    def _people(self, u):
        lead = ids_for(u, "Leadership")
        return {"people": [
            {"name": "Jane Example", "role": "CEO", "source_ids": lead},
            {"name": "Raj Sample", "role": "Chief Technology Officer", "source_ids": lead},
            {"name": "John Doe", "role": "CEO", "source_ids": lead},                                        # HALLUCINATION
            {"name": "Maria Placeholder", "role": "Chief Medical Officer", "source_ids": lead},             # wrong role
        ]}

    def _gtm(self, u):
        return {"verified_evidence": [
            {"statement": "Northwind serves medical billing companies and digital health companies", "source_ids": ids_for(u, "About")},
            {"statement": "Northwind raised a Series B round led by Fabrikam Ventures", "source_ids": ids_for(u, "raises")},
            {"statement": "Northwind employs five hundred engineers worldwide", "source_ids": ids_for(u, "About")},  # HALLUCINATION
        ], "icp": ["Billing companies and digital health startups"], "business_needs": ["Faster payer connectivity"],
            "pain_points": ["Batch file transfers"], "relevant_functions": ["Partnerships"],
            "opportunities": ["Channel partnerships"], "why_this_company": "Growing API vendor expanding its partnerships team.",
            "outreach_angles": ["Partnership expansion"], "confidence": "High"}


# ---- fake HTTP layer for the Ollama client ------------------------------------------
class FakeResponse:
    def __init__(self, status=200, payload=None, text="", raises_json=False):
        self.status_code, self._payload, self.text, self._raises = status, payload, text, raises_json

    def json(self):
        if self._raises:
            raise ValueError("bad json")
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"{self.status_code}")


class FakeSession:
    """Queue of responses/exceptions; records every request."""

    def __init__(self, *responses):
        self.queue, self.requests = list(responses), []

    def _next(self):
        item = self.queue.pop(0) if len(self.queue) > 1 else self.queue[0]
        if isinstance(item, Exception):
            raise item
        return item

    def get(self, url, **kw):
        self.requests.append(("GET", url, kw))
        return self._next()

    def post(self, url, **kw):
        self.requests.append(("POST", url, kw))
        return self._next()
