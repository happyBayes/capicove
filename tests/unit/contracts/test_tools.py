from datetime import UTC, datetime, timedelta

import pytest

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)


def make_spec(**overrides: object) -> ToolSpec:
    values: dict[str, object] = {
        "tool_id": "capicove.search",
        "contract_version": 1,
        "description": "Search indexed evidence.",
        "input_schema": {"type": "object"},
        "output_schema": {"type": "object"},
        "modalities": ("text", "image", "audio", "video"),
        "required_permissions": frozenset({"evidence:read"}),
        "privacy_class": PrivacyClass.CONFIDENTIAL,
        "side_effect": SideEffect.READ,
        "timeout_seconds": 30.0,
    }
    values.update(overrides)
    return ToolSpec(**values)  # type: ignore[arg-type]


def test_tool_spec_exposes_a_stable_registry_key() -> None:
    assert make_spec().registry_key == ("capicove.search", 1)


def test_tool_spec_declares_immutable_execution_resources() -> None:
    resources = ToolResources(
        filesystem_read=True,
        filesystem_write=True,
        subprocesses=("ffmpeg", "ffprobe"),
    )

    spec = make_spec(resources=resources)

    assert spec.resources == resources
    assert spec.resources.subprocesses == ("ffmpeg", "ffprobe")


@pytest.mark.parametrize(
    "executable",
    ["/usr/bin/ffmpeg", "ffmpeg -y", "ffmpeg;echo", "", "../ffmpeg"],
)
def test_tool_resources_rejects_noncanonical_executable_names(executable: str) -> None:
    with pytest.raises(ValueError, match="subprocess"):
        ToolResources(subprocesses=(executable,))


def test_tool_resources_rejects_duplicate_executable_names() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        ToolResources(subprocesses=("ffmpeg", "ffmpeg"))


def test_tool_spec_copies_schema_mappings() -> None:
    input_schema: dict[str, object] = {"type": "object"}
    spec = make_spec(input_schema=input_schema)

    input_schema["type"] = "array"

    assert spec.input_schema["type"] == "object"


@pytest.mark.parametrize("tool_id", ["search", "Capicove.search", "capicove."])
def test_tool_spec_rejects_noncanonical_ids(tool_id: str) -> None:
    with pytest.raises(ValueError, match="tool_id"):
        make_spec(tool_id=tool_id)


@pytest.mark.parametrize(
    ("field", "value"),
    [("contract_version", 0), ("description", "  "), ("timeout_seconds", 0.0)],
)
def test_tool_spec_rejects_invalid_required_values(field: str, value: object) -> None:
    with pytest.raises(ValueError, match=field):
        make_spec(**{field: value})


def test_invocation_context_requires_a_timezone_aware_deadline() -> None:
    with pytest.raises(ValueError, match="deadline"):
        InvocationContext(
            request_id="request-1",
            trace_id="trace-1",
            workspace_id="workspace-1",
            actor_id="agent-1",
            environment_id="environment-1",
            deadline=datetime.now(),
        )


def test_success_result_cannot_contain_an_error() -> None:
    started = datetime.now(UTC)
    with pytest.raises(ValueError, match="successful result"):
        ToolResult(
            request_id="request-1",
            trace_id="trace-1",
            tool_id="capicove.search",
            tool_version=1,
            status=ToolStatus.SUCCEEDED,
            error=ToolError(
                code=ToolErrorCode.INTERNAL,
                message="unexpected",
                retryable=False,
            ),
            started_at=started,
            completed_at=started + timedelta(milliseconds=1),
        )


def test_failed_result_requires_an_error() -> None:
    started = datetime.now(UTC)
    with pytest.raises(ValueError, match="failed or rejected result"):
        ToolResult(
            request_id="request-1",
            trace_id="trace-1",
            tool_id="capicove.search",
            tool_version=1,
            status=ToolStatus.FAILED,
            started_at=started,
            completed_at=started + timedelta(milliseconds=1),
        )


def test_result_rejects_completion_before_start() -> None:
    started = datetime.now(UTC)
    with pytest.raises(ValueError, match="completed_at"):
        ToolResult(
            request_id="request-1",
            trace_id="trace-1",
            tool_id="capicove.search",
            tool_version=1,
            status=ToolStatus.SUCCEEDED,
            started_at=started,
            completed_at=started - timedelta(milliseconds=1),
        )
