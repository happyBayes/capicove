"""Verify canonical clipping authorization and registry contracts."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts import InvocationContext, ToolStatus


def context(*, permissions: frozenset[str] = frozenset()) -> InvocationContext:
    return InvocationContext(
        "request",
        "trace",
        "workspace",
        "actor",
        "environment",
        datetime.now(UTC) + timedelta(seconds=30),
        permissions,
    )


def test_missing_permission_never_invokes_backend(tmp_path: Path) -> None:
    from capicove.tools.audio_clip import AudioClipToolHandler

    class Forbidden:
        def clip_file(self, *args: Any, **kwargs: Any) -> dict[str, object]:
            raise AssertionError("unauthorized media execution")

    handler = AudioClipToolHandler(
        Forbidden(),  # type: ignore[arg-type]
        read_roots=(tmp_path,),
        output_root=tmp_path,
        store_root=tmp_path / "evidence",
    )
    result = handler(context(), {})
    assert result.status is ToolStatus.REJECTED
    assert result.error and result.error.code.value == "permission_denied"


@pytest.mark.parametrize(
    "field,value",
    [("extra", 1), ("channels", None), ("start_seconds", True), ("end_seconds", float("inf"))],
)
def test_invalid_tool_arguments_create_no_output(tmp_path: Path, field: str, value: object) -> None:
    from capicove.tools.audio_clip import AUDIO_CLIP_SPEC, AudioClipToolHandler

    arguments: dict[str, object] = {
        "source": {"type": "path", "path": str(tmp_path / "input.wav")},
        "destination": {"type": "path", "path": str(tmp_path / "clip.wav")},
        "start_seconds": 0,
        "end_seconds": 1,
    }
    arguments[field] = value
    handler = AudioClipToolHandler(
        None,  # type: ignore[arg-type]
        read_roots=(tmp_path,),
        output_root=tmp_path,
        store_root=tmp_path / "evidence",
    )
    result = handler(context(permissions=AUDIO_CLIP_SPEC.required_permissions), arguments)
    assert result.status is ToolStatus.FAILED
    assert (
        result.error and result.error.code.value == "invalid_input" and not result.error.retryable
    )
    assert not (tmp_path / "clip.wav").exists()


def test_clip_registry_and_transport_schema_have_one_owner() -> None:
    from capicove.tools.audio_clip import AUDIO_CLIP_SPEC, register_audio_clip_tool
    from capicove.tools.registry import ToolRegistry

    registry = ToolRegistry()
    register_audio_clip_tool(registry, lambda ctx, arguments: None)  # type: ignore[arg-type]
    assert registry.resolve("capicove.audio.clip", 1).spec is AUDIO_CLIP_SPEC
    assert AUDIO_CLIP_SPEC.input_schema["additionalProperties"] is False
    assert AUDIO_CLIP_SPEC.output_schema["additionalProperties"] is False
    assert not AUDIO_CLIP_SPEC.idempotent and not AUDIO_CLIP_SPEC.network_access
