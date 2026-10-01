"""Ollama client tests. All HTTP is mocked - Ollama does not need to be running."""
import pytest
import requests
from fakes import FakeResponse, FakeSession

from ai.ollama import (OllamaBadResponse, OllamaClient, OllamaError, OllamaModelMissing, OllamaTimeout,
                       OllamaUnavailable, simplify_schema, strip_thinking)
from models.product import ProductList


def client(*responses, model="qwen3:1.7b"):
    return OllamaClient(base_url="http://localhost:11434", model=model, timeout=5, session=FakeSession(*responses))


def ok(text):
    return FakeResponse(200, {"message": {"role": "assistant", "content": text}})


def test_chat_success_returns_text():
    c = client(ok("Hello there"))
    assert c.chat([{"role": "user", "content": "hi"}]) == "Hello there"
    method, url, kw = c.session.requests[0]
    assert url == "http://localhost:11434/api/chat" and kw["json"]["stream"] is False
    assert kw["json"]["model"] == "qwen3:1.7b" and kw["json"]["options"]["num_ctx"] > 0


def test_thinking_blocks_are_removed_and_qwen3_thinking_disabled():
    c = client(ok("<think>secret reasoning</think>The answer."))
    assert c.chat([{"role": "user", "content": "q"}]) == "The answer."
    assert c.session.requests[0][2]["json"]["think"] is False


def test_think_flag_not_sent_for_other_models():
    c = client(ok("fine"), model="llama3.2:3b")
    c.chat([{"role": "user", "content": "q"}])
    assert "think" not in c.session.requests[0][2]["json"]


def test_json_mode_and_schema_payloads():
    c = client(ok("{}"))
    c.chat([{"role": "user", "content": "q"}], json_mode=True)
    assert c.session.requests[0][2]["json"]["format"] == "json"
    schema = simplify_schema(ProductList.model_json_schema())
    c2 = client(ok("{}"))
    c2.chat([{"role": "user", "content": "q"}], json_schema=schema)
    assert c2.session.requests[0][2]["json"]["format"] == schema and "$ref" not in str(schema)


def test_schema_rejected_by_old_server_retries_with_plain_json():
    c = client(FakeResponse(400, {"error": "invalid format"}), ok('{"a": 1}'))
    assert c.chat([{"role": "user", "content": "q"}], json_schema={"type": "object"}) == '{"a": 1}'
    assert c.session.requests[1][2]["json"]["format"] == "json"


def test_connection_error_is_friendly():
    c = client(requests.ConnectionError("refused"))
    with pytest.raises(OllamaUnavailable, match="Cannot reach Ollama"):
        c.chat([{"role": "user", "content": "q"}])


def test_timeout_is_friendly():
    with pytest.raises(OllamaTimeout, match="longer than"):
        client(requests.Timeout("slow")).chat([{"role": "user", "content": "q"}])


@pytest.mark.parametrize("resp", [FakeResponse(404, {"error": "x"}), FakeResponse(500, {"error": "model 'qwen3:1.7b' not found"})])
def test_missing_model_is_detected(resp):
    with pytest.raises(OllamaModelMissing, match="ollama pull"):
        client(resp).chat([{"role": "user", "content": "q"}])


def test_server_error_raises_bad_response():
    with pytest.raises(OllamaBadResponse, match="HTTP 500"):
        client(FakeResponse(500, {"error": "boom"})).chat([{"role": "user", "content": "q"}])


def test_malformed_and_empty_responses():
    with pytest.raises(OllamaBadResponse, match="not valid JSON"):
        client(FakeResponse(200, raises_json=True)).chat([{"role": "user", "content": "q"}])
    with pytest.raises(OllamaBadResponse, match="message text"):
        client(FakeResponse(200, {"unexpected": True})).chat([{"role": "user", "content": "q"}])
    with pytest.raises(OllamaBadResponse, match="empty"):
        client(ok("   ")).chat([{"role": "user", "content": "q"}])
    with pytest.raises(OllamaBadResponse, match="empty"):
        client(ok("<think>only thoughts</think>")).chat([{"role": "user", "content": "q"}])


def test_all_errors_share_a_base_class():
    for exc in (OllamaUnavailable, OllamaTimeout, OllamaModelMissing, OllamaBadResponse):
        assert issubclass(exc, OllamaError)


def test_status_connected_and_model_installed():
    st = client(FakeResponse(200, {"models": [{"name": "qwen3:1.7b"}, {"name": "llama3.2:3b"}]})).status()
    assert st.connected and st.model_installed and st.ready and st.error is None
    assert "llama3.2:3b" in st.installed_models


def test_status_model_name_without_tag_matches_latest():
    st = client(FakeResponse(200, {"models": [{"name": "mymodel:latest"}]}), model="mymodel").status()
    assert st.ready


def test_status_model_missing():
    st = client(FakeResponse(200, {"models": [{"name": "other:1b"}]})).status()
    assert st.connected and not st.model_installed and not st.ready
    assert "ollama pull qwen3:1.7b" in st.error


def test_status_unreachable():
    st = client(requests.ConnectionError("x")).status()
    assert not st.connected and "OLLAMA_SETUP.md" in st.error


def test_status_bad_payload():
    st = client(FakeResponse(200, raises_json=True)).status()
    assert not st.connected and st.error


def test_strip_thinking_variants():
    assert strip_thinking("<think>a</think>b") == "b"
    assert strip_thinking("<THINK>a\nb</THINK>\n\nanswer") == "answer"
    assert strip_thinking("dangling reasoning</think>final") == "final"
    assert strip_thinking("plain") == "plain"
