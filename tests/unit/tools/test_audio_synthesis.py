"""Speech Tools enforce authority, admission, and canonical failure envelopes."""

from collections.abc import Mapping
from dataclasses import replace
from datetime import UTC, datetime
from threading import Lock

import pytest

from capicove.contracts import InvocationContext, ToolErrorCode, ToolStatus
from capicove.mcp.bindings import default_context
from capicove.processing.audio.synthesis import SpeechRequest, SynthesisError
from capicove.tools.audio_synthesis import SpeechToolHandler


def test_denied_invocation_never_calls_service() -> None:
    def forbidden(request: SpeechRequest, deadline: datetime) -> Mapping[str, object]:
        pytest.fail("denied calls must not reach speech execution")

    handler = SpeechToolHandler(forbidden, clone=False)
    context = replace(default_context(handler.spec), permissions=frozenset())
    result = handler(context, {"text": "Hello"})
    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED


def test_busy_admission_does_not_queue_or_release_another_requests_lock() -> None:
    admission = Lock()
    admission.acquire()
    handler = SpeechToolHandler(lambda *_: {}, clone=False, admission=admission)
    result = handler(default_context(handler.spec), {"text": "Hello"})
    assert result.error is not None and result.error.message == "server_busy"
    assert admission.locked()
    admission.release()


@pytest.mark.parametrize(
    ("domain", "outer"),
    [
        ("timeout", ToolErrorCode.TIMEOUT),
        ("voice_unauthorized", ToolErrorCode.POLICY_REJECTED),
        ("invalid_reference", ToolErrorCode.INVALID_INPUT),
        ("completion_unverified", ToolErrorCode.UNAVAILABLE),
    ],
)
def test_domain_failures_preserve_invocation_identity_and_release_admission(
    domain: str,
    outer: ToolErrorCode,
) -> None:
    def fail(request: SpeechRequest, deadline: datetime) -> Mapping[str, object]:
        raise SynthesisError(domain)

    handler = SpeechToolHandler(fail, clone=False)
    context = default_context(handler.spec)
    result = handler(context, {"text": "Hello"})
    assert result.error is not None and result.error.code is outer
    assert result.request_id == context.request_id
    assert result.trace_id == context.trace_id
    assert not handler.admission.locked()


@pytest.mark.parametrize(
    "arguments",
    [
        {},
        {"text": 123},
        {"text": "Hello", "executable": "/foreign"},
        {"text": "Hello", "language": None},
    ],
)
def test_malformed_arguments_fail_before_execution(arguments: dict[str, object]) -> None:
    def forbidden(request: SpeechRequest, deadline: datetime) -> Mapping[str, object]:
        pytest.fail("invalid calls must not reach speech execution")

    handler = SpeechToolHandler(forbidden, clone=False)
    result = handler(default_context(handler.spec), arguments)
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT


def test_expired_tool_deadline_is_rejected_before_service_execution() -> None:
    def forbidden(request: SpeechRequest, deadline: datetime) -> Mapping[str, object]:
        pytest.fail("expired calls must not acquire a model")

    handler = SpeechToolHandler(forbidden, clone=False)
    context: InvocationContext = replace(default_context(handler.spec), deadline=datetime.now(UTC))
    result = handler(context, {"text": "Hello"})
    assert result.error is not None and result.error.code is ToolErrorCode.TIMEOUT
