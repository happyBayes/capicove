import argparse
import json
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from pytest import CaptureFixture, MonkeyPatch

from capicove.cli.app import _build_parser, main
from capicove.cli.commands.video_filter import run
from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolResult, ToolStatus


class FakeHandler:
    def __init__(self, result: ToolResult) -> None:
        self.result = result
        self.calls: list[tuple[InvocationContext, Mapping[str, object]]] = []

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        self.calls.append((context, arguments))
        return self.result


def _result(
    status: ToolStatus = ToolStatus.SUCCEEDED,
    *,
    keep: bool | None = True,
) -> ToolResult:
    error = (
        ToolError(ToolErrorCode.UNAVAILABLE, "model unavailable", False)
        if status is ToolStatus.FAILED
        else None
    )
    return ToolResult(
        request_id="request-1",
        trace_id="trace-1",
        tool_id="capicove.video.filter",
        tool_version=1,
        status=status,
        result={
            "status": "succeeded" if status is ToolStatus.SUCCEEDED else "failed",
            "keep": keep,
            "items": [{"item_id": "video-1", "evidence_ref": "source-1"}],
            "criteria": [{"criterion_id": "metadata", "passed": keep}],
            "warnings": [],
            "failures": (
                [{"code": "backend_unavailable", "message": "model unavailable"}]
                if error is not None
                else []
            ),
            "aggregation": "all",
            "lineage": {"trace_id": "trace-1"},
        },
        evidence_refs=("source-1",),
        error=error,
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )


def _parse(arguments: list[str]) -> argparse.Namespace:
    return _build_parser().parse_args(["video", "filter", *arguments])


def _criteria_file(tmp_path: Path) -> Path:
    path = tmp_path / "criteria.json"
    path.write_text(
        json.dumps(
            [
                {
                    "type": "metadata",
                    "criterion_id": "metadata",
                    "min_duration_seconds": 1.0,
                }
            ]
        ),
        encoding="utf-8",
    )
    return path


def test_cli_model_config_passes_a_scoped_factory_to_video_tool(
    tmp_path: Path, monkeypatch: MonkeyPatch, capsys: CaptureFixture[str]
) -> None:
    import capicove.cli.commands.video_filter as command
    from tests.unit.model.test_bindings import ocr_factory

    factory = ocr_factory(tmp_path)
    monkeypatch.setattr("capicove.model.load_configuration", lambda path: factory.configuration)
    monkeypatch.setattr("capicove.model.ModelFactory", lambda config: factory)
    captured: list[object] = []

    def build(*args: object, **kwargs: object) -> FakeHandler:
        captured.append(kwargs["model_factory"])
        return FakeHandler(_result())

    monkeypatch.setattr(command, "build_video_filter_tool_handler", build)
    args = _parse(
        [
            "--input",
            str(tmp_path / "video.mp4"),
            "--criteria",
            str(_criteria_file(tmp_path)),
            "--model-config",
            str(tmp_path / "models.json"),
            "--output",
            "json",
        ]
    )
    assert run(args) == 0
    assert captured == [factory]
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"
    import pytest

    with pytest.raises(RuntimeError, match="closed"):
        with factory.image_ocr("video.ocr"):
            pass


def test_cli_maps_multiple_paths_text_and_criteria_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    handler = FakeHandler(_result())
    criteria = _criteria_file(tmp_path)
    arguments = _parse(
        [
            "--input",
            "/tmp/one.mp4",
            "--input",
            "/tmp/two.mp4",
            "--text",
            "first description",
            "--text",
            "second description",
            "--criteria",
            str(criteria),
            "--output",
            "json",
        ]
    )

    assert run(arguments, handler=handler) == 0

    canonical = handler.calls[0][1]
    sources = canonical["sources"]
    assert isinstance(sources, list)
    assert sources[0] == {
        "item_id": "video-1",
        "source": {"type": "path", "path": "/tmp/one.mp4"},
        "text": "first description",
    }
    assert sources[1]["item_id"] == "video-2"
    assert canonical["aggregation"] == "all"
    assert json.loads(capsys.readouterr().out)["result"]["keep"] is True


def test_cli_preserves_keep_false_as_success(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    handler = FakeHandler(_result(keep=False))

    exit_code = run(
        _parse(
            [
                "--input",
                "/tmp/video.mp4",
                "--criteria",
                str(_criteria_file(tmp_path)),
            ]
        ),
        handler=handler,
    )

    assert exit_code == 0
    assert "keep: false" in capsys.readouterr().out


def test_cli_failed_result_retains_diagnostics(capsys: CaptureFixture[str], tmp_path: Path) -> None:
    handler = FakeHandler(_result(ToolStatus.FAILED, keep=None))

    exit_code = run(
        _parse(
            [
                "--input",
                "/tmp/video.mp4",
                "--criteria",
                str(_criteria_file(tmp_path)),
                "--output",
                "json",
            ]
        ),
        handler=handler,
    )

    payload = json.loads(capsys.readouterr().out)
    assert exit_code == 1
    assert payload["result"]["failures"][0]["code"] == "backend_unavailable"


def test_app_dispatches_nested_video_filter_command(monkeypatch: MonkeyPatch) -> None:
    observed: list[argparse.Namespace] = []

    def fake_run(namespace: argparse.Namespace) -> int:
        observed.append(namespace)
        return 0

    monkeypatch.setattr("capicove.cli.app.run_video_filter", fake_run)

    assert main(["video", "filter", "--input", "a.mp4", "--criteria", "rules.json"]) == 0
    assert observed[0].video_command == "filter"
