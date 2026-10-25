"""Tool round trips and token counting use the same multimodal messages."""

import copy
import json
from typing import Any

import pytest

from capicove.model.vlm import parse_chat_request
from tests.unit.model.test_openai_vlm import client
from tests.unit.model.test_openai_vlm import upstream as upstream


def tool_request() -> dict[str, Any]:
    return {
        "messages": [{"role": "user", "content": "What is 6 times 7?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "multiply",
                    "description": "Multiply two integers.",
                    "parameters": {
                        "type": "object",
                        "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
                        "required": ["a", "b"],
                        "additionalProperties": False,
                    },
                    "strict": True,
                },
            }
        ],
        "tool_choice": "required",
        "parallel_tool_calls": False,
    }


def tool_call() -> dict[str, Any]:
    return {
        "id": "call_1",
        "type": "function",
        "function": {"name": "multiply", "arguments": '{"a":6,"b":7}'},
    }


def test_tool_definitions_and_results_roundtrip_without_mutable_aliases() -> None:
    raw = tool_request()
    parsed = parse_chat_request(raw)
    raw["tools"][0]["function"]["parameters"]["required"].clear()
    assert parsed.to_openai("model")["tools"] == tool_request()["tools"]
    history = tool_request()
    history["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
    )
    assert parse_chat_request(history).to_openai("model")["messages"] == history["messages"]


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown_result",
        "missing_result",
        "duplicate_result",
        "duplicate_tool",
        "unknown_choice",
        "invalid_arguments",
    ],
)
def test_invalid_tool_conversations_fail_before_network(mutation: str) -> None:
    raw = tool_request()
    raw["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
    )
    if mutation == "unknown_result":
        raw["messages"][-1]["tool_call_id"] = "missing"
    elif mutation == "missing_result":
        raw["messages"].pop()
    elif mutation == "duplicate_result":
        raw["messages"].append(copy.deepcopy(raw["messages"][-1]))
    elif mutation == "duplicate_tool":
        raw["tools"] *= 2
    elif mutation == "unknown_choice":
        raw["tool_choice"] = {"type": "function", "function": {"name": "unknown"}}
    else:
        raw["messages"][1]["tool_calls"][0]["function"]["arguments"] = "{"
    with pytest.raises(ValueError):
        parse_chat_request(raw)


def test_provider_preserves_tool_calls_and_accepts_result_turn(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["body"]["choices"][0].update(
        message={"role": "assistant", "content": None, "tool_calls": [tool_call()]},
        finish_reason="tool_calls",
    )
    with client(endpoint) as provider:
        result = provider.generate(parse_chat_request(tool_request()))
    assert json.loads(json.dumps(result.to_openai()))["choices"][0]["message"]["tool_calls"] == [
        tool_call()
    ]
    assert result.finish_reason == "tool_calls"


def test_vllm_count_uses_chat_template_tools_and_no_generation(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    state["body"] = {"count": 3, "tokens": [1, 2, 3], "max_model_len": 4096}
    raw = tool_request()
    raw["tool_choice"] = "auto"
    with client(endpoint, token_counting="vllm") as provider:
        count = provider.count_tokens(parse_chat_request(raw))
    assert state["path"] == "/tokenize"
    assert state["request"]["tools"] == raw["tools"]
    assert state["request"]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "max_tokens" not in state["request"]
    assert count.input_tokens == 3 and count.context_limit == 4096


@pytest.mark.parametrize(
    "count,tokens,limit",
    [(True, [1], 4096), (-1, [], 4096), (2, [1], 4096), (1, [True], 4096), (1, [1], 0)],
)
def test_invalid_token_counts_rejected(
    upstream: tuple[str, dict[str, Any]], count: Any, tokens: Any, limit: Any
) -> None:
    from capicove.model.providers.openai_vlm import VlmError

    endpoint, state = upstream
    state["body"] = {"count": count, "tokens": tokens, "max_model_len": limit}
    with client(endpoint, token_counting="vllm") as provider:
        with pytest.raises(VlmError, match="invalid_upstream_response"):
            provider.count_tokens(parse_chat_request(tool_request()))


def test_count_is_explicit_opt_in(upstream: tuple[str, dict[str, Any]]) -> None:
    from capicove.model.providers.openai_vlm import VlmError

    endpoint, state = upstream
    with client(endpoint) as provider:
        with pytest.raises(VlmError, match="not_configured") as error:
            provider.count_tokens(parse_chat_request(tool_request()))
    assert error.value.status == 501 and "request" not in state


@pytest.mark.parametrize(
    "change",
    [
        {
            "id": "call_2",
            "type": "function",
            "function": {"name": "unregistered", "arguments": "{}"},
        },
        {"id": "call_1", "type": "function", "function": {"name": "multiply", "arguments": "{"}},
    ],
)
def test_unexpected_or_invalid_tool_results_rejected(
    upstream: tuple[str, dict[str, Any]], change: dict[str, Any]
) -> None:
    from capicove.model.providers.openai_vlm import VlmError

    endpoint, state = upstream
    state["body"]["choices"][0].update(
        message={"role": "assistant", "tool_calls": [change], "content": None},
        finish_reason="tool_calls",
    )
    with client(endpoint) as provider, pytest.raises(VlmError, match="invalid_upstream_response"):
        provider.generate(parse_chat_request(tool_request()))


def test_named_tool_choice_preserves_reserved_name() -> None:
    raw = tool_request()
    raw["tools"][0]["function"]["name"] = "auto"
    raw["tool_choice"] = {"type": "function", "function": {"name": "auto"}}
    assert parse_chat_request(raw).to_openai("model")["tool_choice"] == raw["tool_choice"]


def test_assistant_reasoning_survives_tool_history() -> None:
    raw = tool_request()
    raw["messages"].extend(
        [
            {
                "role": "assistant",
                "content": None,
                "reasoning_content": "I should multiply.",
                "tool_calls": [tool_call()],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": ""},
        ]
    )
    assert parse_chat_request(raw).to_openai("model")["messages"] == raw["messages"]


def test_missing_function_name_is_a_validation_error() -> None:
    raw = tool_request()
    del raw["tools"][0]["function"]["name"]
    with pytest.raises(ValueError):
        parse_chat_request(raw)


def test_reused_historical_tool_call_id_is_rejected_before_execution(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    from capicove.model.vlm import VlmError

    endpoint, state = upstream
    raw = tool_request()
    raw["messages"].extend(
        [
            {"role": "assistant", "content": None, "tool_calls": [tool_call()]},
            {"role": "tool", "tool_call_id": "call_1", "content": "42"},
        ]
    )
    state["body"]["choices"][0].update(
        message={"role": "assistant", "content": None, "tool_calls": [tool_call()]},
        finish_reason="tool_calls",
    )
    with client(endpoint) as provider, pytest.raises(VlmError, match="invalid_upstream_response"):
        provider.generate(parse_chat_request(raw))


def test_named_vllm_call_with_stop_reason_remains_executable_tool_proposal(
    upstream: tuple[str, dict[str, Any]],
) -> None:
    endpoint, state = upstream
    raw = tool_request()
    raw["tool_choice"] = {"type": "function", "function": {"name": "multiply"}}
    state["body"]["choices"][0].update(
        message={"role": "assistant", "content": "", "tool_calls": [tool_call()]},
        finish_reason="stop",
    )
    with client(endpoint) as provider:
        result = provider.generate(parse_chat_request(raw))
    assert result.finish_reason == "tool_calls"
    assert (
        json.loads(json.dumps(result.to_openai()))["capicove"]["upstream_finish_reason"] == "stop"
    )


def test_overflowing_json_number_is_not_a_finite_tool_argument() -> None:
    from capicove.model.vlm import parse_json

    with pytest.raises(ValueError, match="nonfinite"):
        parse_json(b'{"value":1e999}')
