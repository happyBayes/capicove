"""Tests for the optional unified directory batch CLI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Callable, Mapping
from pathlib import Path

import pytest
from pytest import CaptureFixture

from capicove.batch import (
    BatchItem,
    BatchModality,
    BatchOperationError,
    ImageToolBatchOperation,
    OperationFingerprint,
    VideoFilterBatchOperation,
    VideoSplitBatchOperation,
)
from capicove.cli.app import _build_parser
from capicove.cli.commands.batch import (
    _build_default_operation,
    _float,
    _integer,
    _non_negative_float,
    _positive_float,
    _retry_count,
    _sha256,
    _worker_count,
    run,
)


_PROVIDER_SHA = "b" * 64


class _Operation:
    def __init__(
        self,
        *,
        fail: bool = False,
        modalities: frozenset[BatchModality] = frozenset({BatchModality.IMAGE}),
    ) -> None:
        self._fail = fail
        self.calls: list[BatchItem] = []
        self._fingerprint = OperationFingerprint(
            operation_id="capicove.test.image-batch",
            operation_version="1",
            modalities=modalities,
            parameters_sha256="a" * 64,
            provider_fingerprint=_PROVIDER_SHA,
        )

    @property
    def fingerprint(self) -> OperationFingerprint:
        return self._fingerprint

    def execute(self, item: BatchItem, /) -> Mapping[str, object]:
        self.calls.append(item)
        if self._fail:
            raise BatchOperationError("INVALID_IMAGE", "invalid image", retryable=False)
        return {"relative_path": item.relative_path.as_posix(), "sha256": item.sha256}


def test_importing_batch_cli_does_not_load_media_backends() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; import capicove.cli.commands.batch; "
                "assert 'cv2' not in sys.modules; assert 'av' not in sys.modules"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr


def _arguments(tmp_path: Path, *, output: str = "json") -> argparse.Namespace:
    input_root = tmp_path / "input"
    input_root.mkdir(parents=True)
    (input_root / "b.png").write_bytes(b"b")
    (input_root / "a.jpg").write_bytes(b"a")
    (input_root / "ignored.txt").write_text("ignored", encoding="utf-8")
    parameters = tmp_path / "parameters.json"
    parameters.write_text("{}", encoding="utf-8")
    return _build_parser().parse_args(
        [
            "batch",
            "run",
            "--input",
            str(input_root),
            "--operation",
            "image_size_filter",
            "--parameters",
            str(parameters),
            "--state",
            str(tmp_path / "state" / "batch.sqlite3"),
            "--provider-fingerprint",
            _PROVIDER_SHA,
            "--workers",
            "2",
            "--modality-workers",
            "1",
            "--max-retries",
            "0",
            "--output",
            output,
        ]
    )


def test_batch_parser_registers_required_explicit_execution_inputs(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)

    assert arguments.command == "batch"
    assert arguments.batch_command == "run"
    assert arguments.operation == "image_size_filter"
    assert arguments.provider_fingerprint == _PROVIDER_SHA
    assert arguments.workers == 2
    assert arguments.modality_workers == 1


def test_batch_run_discovers_supported_files_and_renders_deterministic_json(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    operation = _Operation()

    status = run(_arguments(tmp_path), operation=operation)

    payload = json.loads(capsys.readouterr().out)
    assert status == 0
    assert [item.relative_path.as_posix() for item in operation.calls] == ["a.jpg", "b.png"]
    assert payload["total"] == 2
    assert payload["cache_hits"] == 0
    assert payload["counts"] == {
        "failed": 0,
        "pending": 0,
        "running": 0,
        "skipped": 0,
        "source_changed": 0,
        "succeeded": 2,
    }
    assert [item["result"]["relative_path"] for item in payload["items"]] == [
        "a.jpg",
        "b.png",
    ]


def test_batch_run_reuses_persistent_content_cache_on_resume(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    arguments = _arguments(tmp_path)
    first = _Operation()
    assert run(arguments, operation=first) == 0
    capsys.readouterr()
    resumed = _Operation(fail=True)

    assert run(arguments, operation=resumed) == 0

    payload = json.loads(capsys.readouterr().out)
    assert resumed.calls == []
    assert payload["cache_hits"] == 2
    assert all(item["cache_hit"] for item in payload["items"])


def test_batch_run_returns_failure_when_any_item_fails(
    tmp_path: Path,
    capsys: CaptureFixture[str],
) -> None:
    status = run(_arguments(tmp_path), operation=_Operation(fail=True))

    payload = json.loads(capsys.readouterr().out)
    assert status == 1
    assert payload["counts"]["failed"] == 2
    assert payload["items"][0]["failure"] == {
        "code": "INVALID_IMAGE",
        "message": "invalid image",
        "retryable": False,
    }


def test_batch_run_rejects_non_object_parameters(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.parameters.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="parameters JSON must contain an object"):
        run(arguments)


def test_batch_run_rejects_modality_limit_above_global_limit(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments.modality_workers = 3

    with pytest.raises(ValueError, match="must not exceed --workers"):
        run(arguments, operation=_Operation())


def test_batch_run_renders_human_summary(tmp_path: Path, capsys: CaptureFixture[str]) -> None:
    assert run(_arguments(tmp_path, output="human"), operation=_Operation()) == 0

    output = capsys.readouterr().out
    assert "plan: plan:" in output
    assert "total: 2" in output
    assert "succeeded: 2" in output


def test_batch_run_maps_discovery_and_store_errors_to_cli_values(tmp_path: Path) -> None:
    missing_arguments = _arguments(tmp_path / "missing-root")
    missing_arguments.input = tmp_path / "does-not-exist"
    with pytest.raises(ValueError, match="input root"):
        run(missing_arguments, operation=_Operation())

    store_arguments = _arguments(tmp_path / "bad-store")
    store_arguments.state = store_arguments.input
    with pytest.raises(ValueError, match="database initialization failed"):
        run(store_arguments, operation=_Operation())


def test_batch_run_rejects_multi_modality_and_audio_operations(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path / "multi")
    with pytest.raises(ValueError, match="exactly one modality"):
        run(
            arguments,
            operation=_Operation(modalities=frozenset({BatchModality.IMAGE, BatchModality.VIDEO})),
        )

    audio_arguments = _arguments(tmp_path / "audio")
    with pytest.raises(ValueError, match="supports image and video"):
        run(
            audio_arguments,
            operation=_Operation(modalities=frozenset({BatchModality.AUDIO})),
        )


@pytest.mark.parametrize(
    ("name", "parameters"),
    [
        ("image_aspect_ratio_filter", {"min_ratio": 0.5, "max_ratio": 2.0}),
        ("image_shape_filter", {"min_width": 2, "min_height": 2}),
        ("image_size_filter", {"min_size": "1B", "max_size": "1MiB"}),
        ("image_subplot_filter", {}),
    ],
)
def test_default_factory_composes_each_local_image_operation(
    tmp_path: Path,
    name: str,
    parameters: Mapping[str, object],
) -> None:
    operation = _build_default_operation(
        name,
        parameters,
        input_root=tmp_path.resolve(),
        artifact_root=None,
        provider_fingerprint=_PROVIDER_SHA,
    )

    assert isinstance(operation, ImageToolBatchOperation)
    assert operation.fingerprint.modalities == frozenset({BatchModality.IMAGE})


def test_default_factory_composes_video_filter_and_split_operations(tmp_path: Path) -> None:
    video_filter = _build_default_operation(
        "video_filter",
        {"criteria": [{"type": "metadata", "criterion_id": "duration"}]},
        input_root=tmp_path.resolve(),
        artifact_root=None,
        provider_fingerprint=_PROVIDER_SHA,
    )
    video_split = _build_default_operation(
        "video_split",
        {
            "strategy": "duration",
            "config": {"split_duration": 30.0},
            "backend": "ffmpeg",
            "write_mode": "copy",
        },
        input_root=(tmp_path / "input").resolve(),
        artifact_root=tmp_path / "artifacts",
        provider_fingerprint=_PROVIDER_SHA,
    )

    assert isinstance(video_filter, VideoFilterBatchOperation)
    assert isinstance(video_split, VideoSplitBatchOperation)


@pytest.mark.parametrize(
    ("name", "parameters", "artifact_root", "message"),
    [
        ("video_filter", {"criteria": []}, None, "non-empty array"),
        ("video_filter", {"criteria": [1]}, None, "criterion must be an object"),
        (
            "video_filter",
            {"criteria": [{}], "aggregation": ""},
            None,
            "aggregation must be a nonblank string",
        ),
        (
            "video_filter",
            {"criteria": [{}], "unknown": True},
            None,
            "unknown operation parameters",
        ),
        (
            "video_split",
            {"strategy": "duration", "config": {}},
            None,
            "--artifact-root is required",
        ),
        (
            "video_split",
            {"strategy": "duration", "config": []},
            Path("artifacts"),
            "config must be an object",
        ),
        (
            "video_split",
            {"strategy": "", "config": {}},
            Path("artifacts"),
            "strategy must be a nonblank string",
        ),
    ],
)
def test_default_factory_rejects_invalid_video_parameters(
    tmp_path: Path,
    name: str,
    parameters: Mapping[str, object],
    artifact_root: Path | None,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_default_operation(
            name,
            parameters,
            input_root=(tmp_path / "input").resolve(),
            artifact_root=artifact_root,
            provider_fingerprint=_PROVIDER_SHA,
        )


def test_default_factory_rejects_split_outputs_inside_input_and_unknown_names(
    tmp_path: Path,
) -> None:
    input_root = (tmp_path / "input").resolve()
    parameters = {"strategy": "duration", "config": {}}

    with pytest.raises(ValueError, match="outside the input tree"):
        _build_default_operation(
            "video_split",
            parameters,
            input_root=input_root,
            artifact_root=input_root / "segments",
            provider_fingerprint=_PROVIDER_SHA,
        )
    with pytest.raises(ValueError, match="unsupported batch operation"):
        _build_default_operation(
            "unknown",
            {},
            input_root=input_root,
            artifact_root=None,
            provider_fingerprint=_PROVIDER_SHA,
        )


def test_default_factory_rejects_invalid_image_configuration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid parameters"):
        _build_default_operation(
            "image_shape_filter",
            {"unknown": True},
            input_root=tmp_path.resolve(),
            artifact_root=None,
            provider_fingerprint=_PROVIDER_SHA,
        )


@pytest.mark.parametrize(
    ("validator", "value"),
    [
        (_sha256, "ABC"),
        (_worker_count, "0"),
        (_retry_count, "11"),
        (_non_negative_float, "-1"),
        (_positive_float, "0"),
    ],
)
def test_batch_argument_validators_reject_invalid_values(
    validator: Callable[[str], object],
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        validator(value)


@pytest.mark.parametrize(("validator", "value"), [(_integer, "x"), (_float, "x"), (_float, "nan")])
def test_named_numeric_validators_reject_invalid_values(
    validator: Callable[[str, str], object],
    value: str,
) -> None:
    with pytest.raises(argparse.ArgumentTypeError):
        validator(value, "value")


def test_batch_run_rejects_unreadable_or_invalid_parameter_json(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path / "invalid-json")
    arguments.parameters.write_text("{", encoding="utf-8")
    with pytest.raises(ValueError, match="readable UTF-8 JSON"):
        run(arguments)

    arguments.parameters = tmp_path / "missing.json"
    with pytest.raises(ValueError, match="readable UTF-8 JSON"):
        run(arguments)


def test_batch_parser_rejects_unsafe_worker_and_retry_bounds(tmp_path: Path) -> None:
    parameters = tmp_path / "parameters.json"
    parameters.write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            [
                "batch",
                "run",
                "--input",
                str(tmp_path),
                "--operation",
                "image_size_filter",
                "--parameters",
                str(parameters),
                "--state",
                str(tmp_path / "batch.sqlite3"),
                "--provider-fingerprint",
                _PROVIDER_SHA,
                "--workers",
                "65",
            ]
        )
