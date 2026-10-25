import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pytest import CaptureFixture

from capicove.cli.commands.video_filter import run
from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolResult, ToolStatus
from capicove.mcp.video_filter import encode_mcp_result


class FakeHandler:
    def __init__(self, result: ToolResult) -> None:
        self.result = result

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        del context, arguments
        return self.result


def test_cli_json_and_mcp_preserve_filter_failure_diagnostics(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    criteria = tmp_path / "criteria.json"
    criteria.write_text(
        '[{"type":"metadata","criterion_id":"metadata","min_width":1}]',
        encoding="utf-8",
    )
    error = ToolError(ToolErrorCode.UNAVAILABLE, "local model missing", False)
    result = ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.filter",
        tool_version=1,
        status=ToolStatus.FAILED,
        result={
            "status": "failed",
            "keep": None,
            "items": [{"item_id": "video-1"}],
            "criteria": [{"criterion_id": "metadata", "passed": True}],
            "warnings": [],
            "failures": [{"code": "backend_unavailable"}],
            "aggregation": "all",
            "lineage": {"trace_id": "trace-1"},
        },
        evidence_refs=("source-1",),
        error=error,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    namespace = argparse.Namespace(
        input=["/tmp/video.mp4"],
        text=[],
        criteria=str(criteria),
        aggregation="all",
        output="json",
    )

    assert run(namespace, handler=FakeHandler(result)) == 1

    cli = json.loads(capsys.readouterr().out)
    mcp = encode_mcp_result(result)
    assert cli["status"] == mcp["status"]
    assert cli["result"] == mcp["result"]
    assert cli["error"] == mcp["error"]
    assert cli["evidence_refs"] == mcp["evidence_refs"]
