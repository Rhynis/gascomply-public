"""Tests for the Ollama-backed ``LLMProvider`` (issue #90).

Every test here runs against a stub ``httpx2.MockTransport``, never a real Ollama
daemon: CI has none, and CI fails on any skipped test (the same rule
``test_embedders.py`` follows for ``BgeM3Embedder``).
"""

import asyncio
import json
from collections.abc import Awaitable

import httpx2
import pytest

from app.llm.base import ContextOverflowError, LLMProvider, LLMUnavailableError, LLMUnreachableError
from app.llm.ollama import DEFAULT_NUM_CTX, LLMError, OllamaProvider


def _generate(provider: OllamaProvider, **kwargs: object) -> str:
    """Drive the coroutine ``OllamaProvider.generate`` to completion, as one caller would."""
    coro: Awaitable[str] = provider.generate(**kwargs)  # type: ignore[arg-type]
    return asyncio.run(coro)


def _chat_response(content: str, *, prompt_eval_count: int | None = None) -> httpx2.Response:
    body: dict[str, object] = {
        "model": "qwen2.5:7b-instruct-q4_K_M",
        "message": {"role": "assistant", "content": content},
        "done": True,
    }
    if prompt_eval_count is not None:
        body["prompt_eval_count"] = prompt_eval_count
    return httpx2.Response(200, json=body)


def test_ollama_provider_implements_the_llm_provider_interface():
    assert isinstance(OllamaProvider(base_url="http://x", model="m"), LLMProvider)


def test_request_targets_the_chat_endpoint_with_system_and_user_messages():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _chat_response('{"refusal": false, "answer": "x", "citations": []}')

    provider = OllamaProvider(
        base_url="http://ollama.local",
        model="qwen2.5:7b-instruct-q4_K_M",
        transport=httpx2.MockTransport(handler),
    )
    _generate(provider, system="Bạn là trợ lý.", user="Câu hỏi?")

    assert captured["url"] == "http://ollama.local/api/chat"
    assert captured["body"]["model"] == "qwen2.5:7b-instruct-q4_K_M"
    assert captured["body"]["messages"] == [
        {"role": "system", "content": "Bạn là trợ lý."},
        {"role": "user", "content": "Câu hỏi?"},
    ]
    assert captured["body"]["stream"] is False


def test_a_json_schema_is_passed_through_as_the_format_field():
    captured = {}
    schema = {"type": "object", "properties": {"refusal": {"type": "boolean"}}}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response('{"refusal": true, "answer": null, "citations": []}')

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u", json_schema=schema)

    assert captured["body"]["format"] == schema


def test_no_json_schema_means_no_format_field_at_all():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("{}")

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u")

    assert "format" not in captured["body"]


def test_generate_returns_message_content_exactly_as_ollama_sent_it():
    reply_text = '{"refusal": false, "answer": "12 m2", "citations": [{"unit_id": "x#dieu-1"}]}'

    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_response(reply_text)

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    assert _generate(provider, system="s", user="u") == reply_text


def test_raises_llm_error_when_the_response_has_no_message_content():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(200, json={"model": "m", "done": True})  # no "message" at all

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    with pytest.raises(LLMError, match=r"no message\.content"):
        _generate(provider, system="s", user="u")


def test_raises_on_a_non_2xx_status_not_caught_here():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return httpx2.Response(404, json={"error": "model 'm' not found"})

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    with pytest.raises(httpx2.HTTPStatusError):
        _generate(provider, system="s", user="u")


def test_a_connection_failure_raises_llm_unreachable_error_not_the_raw_httpx_exception():
    # Issue #183: a connection failure is an operational fact about the deployment (Ollama
    # restarting under another project's LaunchAgent, measured in the issue), not a bug in
    # this codebase, so it is converted exactly like the timeout #129 already converts.
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ConnectError("connection refused", request=request)

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    with pytest.raises(LLMUnreachableError):
        _generate(provider, system="s", user="u")


def test_a_dropped_connection_mid_reply_also_raises_llm_unreachable_error():
    # Issue #182's own "Outage during the candidate run": the in-flight request got exactly
    # this error when another project's LaunchAgent restarted Ollama mid-generation.
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.RemoteProtocolError("Server disconnected without sending a response")

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    with pytest.raises(LLMUnreachableError):
        _generate(provider, system="s", user="u")


def test_llm_unreachable_error_is_a_subclass_of_llm_unavailable_error():
    # Issue #183's own point: every existing caller that catches LLMUnavailableError
    # (app.services.generator's except clauses) must catch this too, with no change there.
    assert issubclass(LLMUnreachableError, LLMUnavailableError)


def test_thinking_mode_is_always_disabled_regardless_of_model():
    """``think`` is always sent as ``False``, regardless of the configured model.

    Issue #112: a thinking-capable model (qwen3) must not emit a reasoning passage
    before its JSON answer -- ``generate_answer`` depends on the raw reply parsing as
    strict JSON, and a thinking model's own flag is not trusted without this.
    """
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("{}")

    provider = OllamaProvider(
        base_url="http://x", model="qwen3:8b", transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u")

    assert captured["body"]["think"] is False


def test_temperature_is_low_but_not_left_to_ollamas_own_default():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("{}")

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u")

    assert 0.0 <= captured["body"]["options"]["temperature"] <= 0.3


# --- num_ctx, sent on every call (issue #129) ----------------------------------------------


def test_num_ctx_defaults_to_16384_when_not_configured():
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response("{}")

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u")

    assert captured["body"]["options"]["num_ctx"] == DEFAULT_NUM_CTX == 16384


def test_num_ctx_is_sent_as_configured_on_every_call():
    captured_bodies: list[dict] = []

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured_bodies.append(json.loads(request.content))
        return _chat_response("{}")

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=8192, transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u")
    _generate(provider, system="s2", user="u2", json_schema={"type": "object"})

    assert [body["options"]["num_ctx"] for body in captured_bodies] == [8192, 8192]


# --- context overflow: never answer from a truncated prompt (issue #129) ------------------


def test_a_prompt_eval_count_well_below_the_reply_budget_does_not_raise():
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_response(
            '{"refusal": true, "answer": null, "citations": []}', prompt_eval_count=100
        )

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=1000, transport=httpx2.MockTransport(handler)
    )
    reply = _generate(provider, system="s", user="u")

    assert reply == '{"refusal": true, "answer": null, "citations": []}'
    assert provider.last_prompt_eval_count == 100


def test_a_prompt_eval_count_at_the_reply_budget_boundary_raises_context_overflow():
    # num_ctx=1000, reply budget 256 (the module's own _REPLY_BUDGET_TOKENS): a
    # prompt_eval_count of exactly 744 leaves exactly the budget and must still be refused --
    # the issue's own rule is "within the reply budget", inclusive of the boundary.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_response("{}", prompt_eval_count=744)

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=1000, transport=httpx2.MockTransport(handler)
    )

    with pytest.raises(ContextOverflowError, match="744"):
        _generate(provider, system="s", user="u")


def test_a_prompt_eval_count_at_num_ctx_minus_one_certainly_truncated_raises():
    # The issue's own measurement: every truncated case logged a context limit of exactly
    # num_ctx - 1 (e.g. "limit=4095" at Ollama's default num_ctx=4096).
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_response("{}", prompt_eval_count=4095)

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=4096, transport=httpx2.MockTransport(handler)
    )

    with pytest.raises(ContextOverflowError):
        _generate(provider, system="s", user="u")


def test_overflow_is_logged_at_error(caplog: pytest.LogCaptureFixture):
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_response("{}", prompt_eval_count=4095)

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=4096, transport=httpx2.MockTransport(handler)
    )

    with caplog.at_level("ERROR", logger="app.llm.ollama"), pytest.raises(ContextOverflowError):
        _generate(provider, system="s", user="u")

    assert any("4095" in record.getMessage() for record in caplog.records)


def test_no_prompt_eval_count_at_all_does_not_raise():
    # A malformed or minimal real-world reply (missing the field entirely) must not be
    # treated as an overflow -- only an explicit, too-large count is proof of one.
    def handler(request: httpx2.Request) -> httpx2.Response:
        return _chat_response("{}")  # no prompt_eval_count

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=1000, transport=httpx2.MockTransport(handler)
    )

    _generate(provider, system="s", user="u")  # must not raise

    assert provider.last_prompt_eval_count is None


def test_the_verifiers_short_call_still_gets_the_same_num_ctx_and_check():
    """Trap 3: a verifier call is just another ``generate`` call through this same method.

    So it is exercised exactly the same way as the main generation call -- no
    special-casing needed, and none exists.
    """
    captured = {}

    def handler(request: httpx2.Request) -> httpx2.Response:
        captured["body"] = json.loads(request.content)
        return _chat_response('{"supported": true}', prompt_eval_count=50)

    provider = OllamaProvider(
        base_url="http://x", model="m", num_ctx=1000, transport=httpx2.MockTransport(handler)
    )
    _generate(provider, system="s", user="u", json_schema={"properties": {"supported": {}}})

    assert captured["body"]["options"]["num_ctx"] == 1000
    assert provider.last_prompt_eval_count == 50


# --- a timeout and a connection failure are distinct exceptions (issues #129/#183) --------


def test_a_timeout_raises_llm_unavailable_error_not_the_raw_httpx_exception():
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )

    with pytest.raises(LLMUnavailableError):
        _generate(provider, system="s", user="u")


def test_a_timeout_is_not_mistaken_for_llm_unreachable_error():
    # Trap #1: TimeoutException is itself a TransportError, so the timeout branch must be
    # checked first -- a timeout raising LLMUnreachableError instead would wrongly fold a
    # genuine timeout into the connection-failure code (issue #183).
    def handler(request: httpx2.Request) -> httpx2.Response:
        raise httpx2.ReadTimeout("timed out", request=request)

    provider = OllamaProvider(
        base_url="http://x", model="m", transport=httpx2.MockTransport(handler)
    )

    with pytest.raises(LLMUnavailableError) as excinfo:
        _generate(provider, system="s", user="u")
    assert not isinstance(excinfo.value, LLMUnreachableError)


# --- create_llm_provider ----------------------------------------------------------------


def test_create_llm_provider_defaults_to_mock():
    from app.core.config import Settings
    from app.llm.factory import create_llm_provider
    from app.llm.mock import MockLLM

    provider = create_llm_provider(Settings(_env_file=None))
    assert isinstance(provider, MockLLM)


def test_create_llm_provider_builds_ollama_when_configured():
    from app.core.config import LLMProvider as LLMProviderKind
    from app.core.config import Settings
    from app.llm.factory import create_llm_provider

    provider = create_llm_provider(
        Settings(
            llm_provider=LLMProviderKind.OLLAMA,
            ollama_base_url="http://ollama.local",
            ollama_model="qwen2.5:7b-instruct-q4_K_M",
            ollama_num_ctx=8192,
            _env_file=None,
        )
    )
    assert isinstance(provider, OllamaProvider)
    assert provider.num_ctx == 8192


def test_settings_defaults_ollama_num_ctx_to_16384():
    from app.core.config import Settings

    assert Settings(_env_file=None).ollama_num_ctx == 16384


def test_settings_defaults_ollama_model_to_qwen3_8b():
    """Issue #156: #143's descent (issue #146) is only safe with this default.

    qwen2.5:7b-instruct-q4_K_M cites the wrong appendix item 6/6 on the flagship question;
    qwen3:8b cites the right one 6/6 (see #146's PR for the measurement).
    """
    from app.core.config import Settings

    assert Settings(_env_file=None).ollama_model == "qwen3:8b"
