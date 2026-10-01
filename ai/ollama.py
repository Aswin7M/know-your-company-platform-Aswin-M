"""Minimal Ollama client over the local HTTP API (no SDK, no cloud, no API key)."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

import requests

from config import Settings, load_settings


# ---- errors: each one carries a message that is safe to show in the UI ------------------
class OllamaError(RuntimeError):
    pass


class OllamaUnavailable(OllamaError):
    pass


class OllamaModelMissing(OllamaError):
    pass


class OllamaTimeout(OllamaError):
    pass


class OllamaBadResponse(OllamaError):
    pass


@dataclass
class OllamaStatus:
    connected: bool = False
    model: str = ""
    model_installed: bool = False
    installed_models: list[str] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ready(self) -> bool:
        return self.connected and self.model_installed


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks some reasoning models emit."""
    text = _THINK_RE.sub("", text or "")
    if "</think>" in text:                       # unterminated opening tag
        text = text.split("</think>", 1)[1]
    return text.strip()


def _same_model(a: str, b: str) -> bool:
    def norm(x: str) -> str:
        x = x.strip().lower()
        return x if ":" in x else f"{x}:latest"
    return norm(a) == norm(b)


def simplify_schema(schema: dict) -> dict:
    """Inline local $refs so the schema can be used as an Ollama structured-output format."""
    defs = schema.get("$defs", {})

    def resolve(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                return resolve(defs.get(node["$ref"].split("/")[-1], {}))
            return {k: resolve(v) for k, v in node.items() if k not in ("$defs", "title")}
        if isinstance(node, list):
            return [resolve(x) for x in node]
        return node

    return resolve(schema)


class OllamaClient:
    def __init__(self, base_url: Optional[str] = None, model: Optional[str] = None,
                 timeout: Optional[int] = None, settings: Optional[Settings] = None,
                 session: Optional[Any] = None):
        s = settings or load_settings()
        self.base_url = (base_url or s.ollama_base_url).rstrip("/")
        self.model = model or s.ollama_model
        self.timeout = timeout or s.ollama_timeout
        self.num_ctx = s.ollama_num_ctx
        self.think_mode = s.ollama_think
        self.session = session or requests.Session()

    # ---- health ---------------------------------------------------------------------
    def status(self, timeout: float = 2.0) -> OllamaStatus:
        st = OllamaStatus(model=self.model)
        try:
            r = self.session.get(f"{self.base_url}/api/tags", timeout=timeout)
            r.raise_for_status()
            payload = r.json() or {}                      # parse first: junk on the port is not "connected"
            st.installed_models = [m.get("name", "") for m in payload.get("models", [])]
            st.connected = True
            st.model_installed = any(_same_model(self.model, m) for m in st.installed_models)
            if not st.model_installed:
                st.error = f"Model '{self.model}' is not installed. Run: ollama pull {self.model}"
        except requests.ConnectionError:
            st.error = f"Cannot reach Ollama at {self.base_url}. Install/start Ollama (see OLLAMA_SETUP.md)."
        except requests.Timeout:
            st.error = f"Ollama at {self.base_url} did not respond in time."
        except (requests.RequestException, ValueError) as exc:
            st.error = f"Unexpected response from Ollama: {exc}"
        return st

    def is_available(self) -> bool:
        return self.status().ready

    # ---- chat -----------------------------------------------------------------------
    def _wants_think_flag(self) -> bool:
        return self.think_mode == "auto" and "qwen3" in self.model.lower()

    def _post_chat(self, payload: dict) -> dict:
        try:
            r = self.session.post(f"{self.base_url}/api/chat", json=payload, timeout=self.timeout)
        except requests.ConnectionError as exc:
            raise OllamaUnavailable(
                f"Cannot reach Ollama at {self.base_url}. Is Ollama running? (see OLLAMA_SETUP.md)") from exc
        except requests.Timeout as exc:
            raise OllamaTimeout(
                f"Ollama took longer than {self.timeout}s. CPU models can be slow - try a smaller model "
                "or raise OLLAMA_TIMEOUT.") from exc
        except requests.RequestException as exc:
            raise OllamaError(f"Ollama request failed: {exc}") from exc

        if r.status_code == 404:
            raise OllamaModelMissing(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
        if r.status_code >= 400:
            detail = ""
            try:
                detail = (r.json() or {}).get("error", "")
            except ValueError:
                detail = (getattr(r, "text", "") or "")[:200]
            if "not found" in detail.lower():
                raise OllamaModelMissing(f"Model '{self.model}' not found. Run: ollama pull {self.model}")
            err = OllamaBadResponse(f"Ollama returned HTTP {r.status_code}: {detail or 'no details'}")
            err.status_code = r.status_code          # type: ignore[attr-defined]
            err.detail = detail                      # type: ignore[attr-defined]
            raise err
        try:
            return r.json()
        except ValueError as exc:
            raise OllamaBadResponse("Ollama returned a response that is not valid JSON.") from exc

    def chat(self, messages: list[dict], *, json_schema: Optional[dict] = None, json_mode: bool = False,
             temperature: float = 0.2, num_predict: Optional[int] = None) -> str:
        """Send a chat request and return the assistant text (thinking blocks removed)."""
        payload: dict[str, Any] = {
            "model": self.model, "messages": messages, "stream": False,
            "options": {"temperature": temperature, "num_ctx": self.num_ctx},
        }
        if num_predict:
            payload["options"]["num_predict"] = num_predict
        if self._wants_think_flag():
            payload["think"] = False
        if json_schema is not None:
            payload["format"] = json_schema
        elif json_mode:
            payload["format"] = "json"

        try:
            data = self._post_chat(payload)
        except OllamaBadResponse as err:
            # Older Ollama builds reject structured schemas / the think flag: retry more simply.
            if getattr(err, "status_code", 0) == 400 and ("format" in payload or "think" in payload):
                payload.pop("think", None)
                if isinstance(payload.get("format"), dict):
                    payload["format"] = "json"
                data = self._post_chat(payload)
            else:
                raise

        content = ((data or {}).get("message") or {}).get("content")
        if content is None:
            content = (data or {}).get("response")            # /api/generate-style payload
        if not isinstance(content, str):
            raise OllamaBadResponse("Ollama response did not contain any message text.")
        content = strip_thinking(content)
        if not content:
            raise OllamaBadResponse("Ollama returned an empty answer. Try again or use a different model.")
        return content
