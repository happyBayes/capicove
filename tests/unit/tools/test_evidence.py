"""Evidence access is actor-scoped and section-bounded."""

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts import InvocationContext, ToolErrorCode, ToolStatus
from capicove.tools.evidence import EvidenceToolHandler


class Reader:
    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]:
        if context.actor_id != "allowed":
            raise PermissionError
        return {"preview": {"resource_ref": "resource:preview:k", "media_type": "image/jpeg"}}


def test_permission_and_preview_bounds() -> None:
    handler = EvidenceToolHandler(Reader())

    def context(actor: str, permissions: frozenset[str]) -> InvocationContext:
        return InvocationContext(
            "r",
            "t",
            "w",
            actor,
            "e",
            datetime.now(UTC) + timedelta(seconds=5),
            permissions=permissions,
        )

    assert (
        handler(context("allowed", frozenset()), {"evidence_id": "e"}).status is ToolStatus.FAILED
    )
    assert (
        handler(context("denied", frozenset({"evidence:read"})), {"evidence_id": "e"}).status
        is ToolStatus.FAILED
    )
    result = handler(
        context("allowed", frozenset({"evidence:read"})),
        {"evidence_id": "e", "sections": ["preview"]},
    )
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result == {
        "evidence_id": "e",
        "preview": {"resource_ref": "resource:preview:k", "media_type": "image/jpeg"},
    }


def authorized_context() -> InvocationContext:
    return InvocationContext(
        "r",
        "t",
        "w",
        "allowed",
        "e",
        datetime.now(UTC) + timedelta(seconds=5),
        permissions=frozenset({"evidence:read"}),
    )


@pytest.mark.parametrize(
    "arguments",
    [
        {"evidence_id": "e", "extra": 1},
        {"evidence_id": ""},
        {"evidence_id": "x" * 513},
        {"evidence_id": "e", "sections": []},
        {"evidence_id": "e", "sections": ["unknown"]},
        {"evidence_id": "e", "sections": ["metadata", "metadata"]},
    ],
)
def test_invalid_evidence_arguments_are_rejected(arguments: dict[str, object]) -> None:
    result = EvidenceToolHandler(Reader())(authorized_context(), arguments)
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT


@pytest.mark.parametrize(
    "payload,section",
    [
        ({"ocr": "unrequested"}, "metadata"),
        ({"metadata": "x" * 4097}, "metadata"),
        ({"metadata": [0] * 33}, "metadata"),
        ({"preview": "not-a-preview"}, "preview"),
        (
            {
                "preview": {
                    "resource_ref": "resource:preview:/private/path",
                    "media_type": "image/png",
                }
            },
            "preview",
        ),
        ({"preview": {"resource_ref": "resource:preview:k", "media_type": "text/html"}}, "preview"),
    ],
)
def test_provider_output_cannot_widen_sections_or_expose_paths(
    payload: dict[str, object],
    section: str,
) -> None:
    class OutputReader:
        def read(self, **kwargs: object) -> Mapping[str, object]:
            return payload

    result = EvidenceToolHandler(OutputReader())(
        authorized_context(), {"evidence_id": "e", "sections": [section]}
    )
    assert result.error is not None and result.error.code is ToolErrorCode.INTERNAL
    assert result.result is None


@pytest.mark.parametrize(
    "failure,code",
    [
        (LookupError, ToolErrorCode.UNAVAILABLE),
        (TimeoutError, ToolErrorCode.TIMEOUT),
    ],
)
def test_reader_failures_are_sanitized(failure: type[Exception], code: ToolErrorCode) -> None:
    class FailedReader:
        def read(self, **kwargs: object) -> Mapping[str, object]:
            raise failure("private provider detail")

    result = EvidenceToolHandler(FailedReader())(authorized_context(), {"evidence_id": "e"})
    assert result.error is not None and result.error.code is code
    assert "private" not in result.error.message


def test_diagnostics_require_additional_permission() -> None:
    result = EvidenceToolHandler(Reader())(
        authorized_context(), {"evidence_id": "e", "sections": ["provider_diagnostics"]}
    )
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED


def test_expired_evidence_context_fails_without_read() -> None:
    from dataclasses import replace

    invocation = replace(authorized_context(), deadline=datetime.now(UTC) - timedelta(seconds=1))
    result = EvidenceToolHandler(Reader())(invocation, {"evidence_id": "e"})
    assert result.error is not None and result.error.code is ToolErrorCode.TIMEOUT
