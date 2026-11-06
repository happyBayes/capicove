"""CLI arguments and result statuses for video media quality inspection."""

import json
from collections.abc import Mapping
from typing import Any

import pytest

from capicove.cli.app import _build_parser
from capicove.cli.commands.video_quality import run
from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolResult, ToolStatus
from capicove.tools.video_quality import VIDEO_QUALITY_SPEC


@pytest.mark.parametrize(
    ("verdict", "status", "exit_code"),
    [
        ("observed", ToolStatus.SUCCEEDED, 0),
        ("review", ToolStatus.SUCCEEDED, 0),
        ("fail", ToolStatus.SUCCEEDED, 1),
        ("incomplete", ToolStatus.FAILED, 2),
    ],
)
def test_cli_exit_and_json_preserve_canonical_verdict(
    verdict: str,
    status: ToolStatus,
    exit_code: int,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        return ToolResult(
            context.request_id,
            context.trace_id,
            VIDEO_QUALITY_SPEC.tool_id,
            1,
            status,
            result={"verdict": verdict, "checks": []},
            error=None
            if status is ToolStatus.SUCCEEDED
            else ToolError(ToolErrorCode.TIMEOUT, "deadline", False),
        )

    args = _build_parser().parse_args(
        ["video", "quality", "--input", "/authorized/video.mp4", "--output", "json"]
    )
    assert run(args, handler=handler) == exit_code
    payload: dict[str, Any] = json.loads(capsys.readouterr().out)
    assert payload["result"]["verdict"] == verdict
