import io
import json
from collections.abc import Mapping
from datetime import UTC, datetime

import pytest
from pytest import CaptureFixture

from capicove.cli.app import _build_parser
from capicove.cli.commands.video_split import run
from capicove.contracts import InvocationContext, ToolResult, ToolStatus
from capicove.mcp.video_split import encode_mcp_result


pytestmark = pytest.mark.e2e


def make_result(*, degraded: bool) -> ToolResult:
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.split",
        tool_version=1,
        status=ToolStatus.SUCCEEDED,
        result={
            "status": "succeeded",
            "original": {"sha256": "a" * 64},
            "segments": [{"index": 0}],
            "warnings": ([{"code": "segment_failed"}] if degraded else []),
            "degraded": degraded,
            "lineage": {"trace_id": "trace-1"},
            "failure": None,
        },
        evidence_refs=("source-1",),
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


class SameResultHandler:
    def __init__(self, result: ToolResult) -> None:
        self.result = result

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        return self.result


def test_cli_json_and_mcp_preserve_the_same_canonical_result(
    capsys: CaptureFixture[str],
) -> None:
    result = make_result(degraded=True)
    namespace = _build_parser().parse_args(
        [
            "video",
            "split",
            "--input",
            "-",
            "--strategy",
            "keyframe",
            "--output",
            "json",
        ]
    )
    run(namespace, handler=SameResultHandler(result), stdin=io.BytesIO(b"video"))
    cli = json.loads(capsys.readouterr().out)
    mcp = encode_mcp_result(result)

    assert cli["status"] == mcp["status"]
    assert cli["result"] == mcp["result"]
    assert cli["error"] == mcp["error"]
    assert cli["evidence_refs"] == mcp["evidence_refs"]
    assert cli["trace_id"] == mcp["trace_id"]
