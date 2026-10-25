from __future__ import annotations

import json
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from capicove.processing.audio import AudioFailureCode
from capicove.processing.audio._adapters.ffprobe import (
    FfprobeAudioStreamInspector,
    _ProbeOutputTooLarge,
    _read_bounded_output,
    _run_probe,
)
from capicove.processing.audio.ports import AudioValidationError


def _payload(*global_indexes: int) -> bytes:
    return json.dumps({"streams": [{"index": index} for index in global_indexes]}).encode()


def test_ffprobe_inspector_uses_fixed_arguments_and_audio_array_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "meeting.m4a"
    source.touch()

    with patch(
        "capicove.processing.audio._adapters.ffprobe._run_probe",
        return_value=(0, _payload(1, 3)),
    ) as run_probe:
        inventory = FfprobeAudioStreamInspector("trusted-ffprobe").inspect(source)

    assert inventory.audio_stream_count == 2
    run_probe.assert_called_once_with(
        [
            "trusted-ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            str(source),
        ],
        300.0,
    )


@pytest.mark.parametrize(
    ("returncode", "payload"),
    [
        (1, b"private decoder diagnostic"),
        (0, b"not-json"),
        (0, b"{}"),
        (0, b'{"streams": {}}'),
        (0, b'{"streams": [null]}'),
        (0, _payload(*range(65))),
        (0, b"x" * (64 * 1024 + 1)),
    ],
)
def test_ffprobe_inspector_rejects_invalid_inventory_without_exposing_output(
    tmp_path: Path,
    returncode: int,
    payload: bytes,
) -> None:
    with (
        patch(
            "capicove.processing.audio._adapters.ffprobe._run_probe",
            return_value=(returncode, payload),
        ),
        pytest.raises(
            AudioValidationError, match=r"^audio stream inventory is invalid$"
        ) as captured,
    ):
        FfprobeAudioStreamInspector().inspect(tmp_path / "private-name.m4a")

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO
    assert "private" not in str(captured.value)


@pytest.mark.parametrize(
    ("executable", "timeout_seconds", "message"),
    [
        ("", 300.0, "executable must not be blank"),
        ("ffprobe", 0.0, "timeout_seconds must be positive"),
    ],
)
def test_ffprobe_inspector_rejects_invalid_configuration(
    executable: str,
    timeout_seconds: float,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        FfprobeAudioStreamInspector(executable, timeout_seconds)


@pytest.mark.parametrize(
    ("error", "expected_code", "message"),
    [
        (
            _ProbeOutputTooLarge("private output"),
            AudioFailureCode.INVALID_AUDIO,
            "audio stream inventory is invalid",
        ),
        (
            OSError("private operating-system detail"),
            AudioFailureCode.DECODER_UNAVAILABLE,
            "audio stream decoder is unavailable",
        ),
    ],
)
def test_ffprobe_inspector_maps_bounded_runner_failures_without_details(
    tmp_path: Path,
    error: Exception,
    expected_code: AudioFailureCode,
    message: str,
) -> None:
    with (
        patch(
            "capicove.processing.audio._adapters.ffprobe._run_probe",
            side_effect=error,
        ),
        pytest.raises(AudioValidationError, match=f"^{message}$") as captured,
    ):
        FfprobeAudioStreamInspector().inspect(tmp_path / "private-name.m4a")

    assert captured.value.code is expected_code
    assert "private" not in str(captured.value)


@pytest.mark.parametrize("global_index", [True, -1, "1"])
def test_ffprobe_inspector_rejects_invalid_container_global_indexes(
    tmp_path: Path,
    global_index: object,
) -> None:
    payload = json.dumps({"streams": [{"index": global_index}]}).encode()
    with (
        patch(
            "capicove.processing.audio._adapters.ffprobe._run_probe",
            return_value=(0, payload),
        ),
        pytest.raises(AudioValidationError) as captured,
    ):
        FfprobeAudioStreamInspector().inspect(tmp_path / "meeting.m4a")

    assert captured.value.code is AudioFailureCode.INVALID_AUDIO


def test_ffprobe_inspector_maps_missing_executable_and_timeout(tmp_path: Path) -> None:
    source = tmp_path / "meeting.m4a"

    with (
        patch(
            "capicove.processing.audio._adapters.ffprobe._run_probe",
            side_effect=FileNotFoundError,
        ),
        pytest.raises(AudioValidationError, match="decoder is unavailable") as missing,
    ):
        FfprobeAudioStreamInspector().inspect(source)
    assert missing.value.code is AudioFailureCode.DECODER_UNAVAILABLE

    with (
        patch(
            "capicove.processing.audio._adapters.ffprobe._run_probe",
            side_effect=subprocess.TimeoutExpired(["ffprobe"], timeout=1),
        ),
        pytest.raises(AudioValidationError, match="inventory timed out") as timed_out,
    ):
        FfprobeAudioStreamInspector().inspect(source)
    assert timed_out.value.code is AudioFailureCode.TIMEOUT


def test_ffprobe_inspector_bounds_probe_by_remaining_deadline(tmp_path: Path) -> None:
    deadline = datetime.now(UTC) + timedelta(seconds=1)
    with patch(
        "capicove.processing.audio._adapters.ffprobe._run_probe",
        return_value=(0, _payload(0)),
    ) as run_probe:
        FfprobeAudioStreamInspector(timeout_seconds=300).inspect(
            tmp_path / "meeting.m4a",
            deadline=deadline,
        )

    timeout = run_probe.call_args.args[1]
    assert 0 < timeout <= 1


def test_ffprobe_inspector_rejects_expired_deadline_without_subprocess(
    tmp_path: Path,
) -> None:
    with (
        patch("capicove.processing.audio._adapters.ffprobe._run_probe") as run_probe,
        pytest.raises(AudioValidationError, match="inventory timed out") as captured,
    ):
        FfprobeAudioStreamInspector().inspect(
            tmp_path / "meeting.m4a",
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT
    run_probe.assert_not_called()


def test_ffprobe_inspector_rejects_naive_deadline_without_subprocess(
    tmp_path: Path,
) -> None:
    with (
        patch("capicove.processing.audio._adapters.ffprobe._run_probe") as run_probe,
        pytest.raises(ValueError, match="deadline must be timezone-aware"),
    ):
        FfprobeAudioStreamInspector().inspect(
            tmp_path / "meeting.m4a",
            deadline=datetime.now(),
        )

    run_probe.assert_not_called()


def test_ffprobe_probe_runner_is_shell_free_and_caps_stdout() -> None:
    class _Process:
        stdout = object()

    process = _Process()
    with (
        patch("subprocess.Popen", return_value=process) as popen,
        patch(
            "capicove.processing.audio._adapters.ffprobe._read_bounded_output",
            return_value=(0, b'{"streams": []}'),
        ) as read_output,
    ):
        assert _run_probe(["trusted-ffprobe", "fixed"], 2.0) == (
            0,
            b'{"streams": []}',
        )

    popen.assert_called_once_with(
        ["trusted-ffprobe", "fixed"],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        shell=False,
    )
    read_output.assert_called_once_with(process, 2.0)


def test_ffprobe_probe_runner_terminates_output_beyond_64_kib() -> None:
    with pytest.raises(_ProbeOutputTooLarge):
        _run_probe(
            [
                sys.executable,
                "-c",
                "import sys; sys.stdout.buffer.write(b'x' * 65537)",
            ],
            2.0,
        )


def test_ffprobe_bounded_reader_rejects_missing_stdout_pipe() -> None:
    process = MagicMock(spec=subprocess.Popen)
    process.stdout = None

    with (
        patch("capicove.processing.audio._adapters.ffprobe._terminate") as terminate,
        pytest.raises(OSError, match="probe stdout pipe is unavailable"),
    ):
        _read_bounded_output(process, 2.0)

    terminate.assert_called_once_with(process)
