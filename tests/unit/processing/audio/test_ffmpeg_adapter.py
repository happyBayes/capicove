from __future__ import annotations

import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import patch

import pytest

from capicove.processing.audio import AudioFailureCode
from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
from capicove.processing.audio.ports import AudioValidationError


def test_ffmpeg_validator_uses_fixed_arguments_without_a_shell(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.touch()
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")

    with patch("subprocess.run", return_value=completed) as run:
        FfmpegAudioValidator(executable="trusted-ffmpeg").validate(
            source,
            audio_stream_index=1,
        )

    run.assert_called_once_with(
        [
            "trusted-ffmpeg",
            "-v",
            "error",
            "-xerror",
            "-nostdin",
            "-i",
            str(source),
            "-map",
            "0:a:1",
            "-f",
            "null",
            "-",
        ],
        check=False,
        capture_output=True,
        timeout=300.0,
        start_new_session=True,
        shell=False,
    )


def test_ffmpeg_validator_rejects_decode_failure_without_exposing_diagnostic(
    tmp_path: Path,
) -> None:
    source = tmp_path / "private-name.wav"
    completed = subprocess.CompletedProcess(
        args=[], returncode=1, stdout=b"", stderr=b"private decoder diagnostic"
    )

    with (
        patch("subprocess.run", return_value=completed),
        pytest.raises(
            AudioValidationError,
            match=r"^audio stream is not decodable$",
        ) as captured,
    ):
        FfmpegAudioValidator().validate(source, audio_stream_index=1)

    assert captured.value.code is AudioFailureCode.SELECTED_AUDIO_STREAM_UNDECODABLE


def test_ffmpeg_validator_reports_unavailable_decoder(tmp_path: Path) -> None:
    with (
        patch("subprocess.run", side_effect=FileNotFoundError),
        pytest.raises(AudioValidationError, match="decoder is unavailable") as captured,
    ):
        FfmpegAudioValidator().validate(tmp_path / "audio.wav", audio_stream_index=0)

    assert captured.value.code is AudioFailureCode.DECODER_UNAVAILABLE


def test_ffmpeg_validator_maps_permission_error_to_unavailable_decoder(
    tmp_path: Path,
) -> None:
    with (
        patch("subprocess.run", side_effect=PermissionError("not executable")),
        pytest.raises(
            AudioValidationError,
            match=r"^audio decoder is unavailable$",
        ) as captured,
    ):
        FfmpegAudioValidator().validate(tmp_path / "audio.wav", audio_stream_index=0)

    assert captured.value.code is AudioFailureCode.DECODER_UNAVAILABLE


def test_ffmpeg_validator_reports_timeout(tmp_path: Path) -> None:
    with (
        patch(
            "subprocess.run",
            side_effect=subprocess.TimeoutExpired(cmd=["ffmpeg"], timeout=1),
        ),
        pytest.raises(AudioValidationError, match="validation timed out") as captured,
    ):
        FfmpegAudioValidator(timeout_seconds=1).validate(
            tmp_path / "audio.wav",
            audio_stream_index=0,
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT


def test_ffmpeg_validator_bounds_subprocess_by_remaining_deadline(tmp_path: Path) -> None:
    source = tmp_path / "meeting.wav"
    source.touch()
    completed = subprocess.CompletedProcess(args=[], returncode=0, stdout=b"", stderr=b"")
    deadline = datetime.now(UTC) + timedelta(seconds=1)

    with patch("subprocess.run", return_value=completed) as run:
        FfmpegAudioValidator(timeout_seconds=300).validate(
            source,
            audio_stream_index=0,
            deadline=deadline,
        )

    timeout = run.call_args.kwargs["timeout"]
    assert 0 < timeout <= 1


def test_ffmpeg_validator_rejects_expired_deadline_without_subprocess(tmp_path: Path) -> None:
    with (
        patch("subprocess.run") as run,
        pytest.raises(AudioValidationError, match="timed out") as captured,
    ):
        FfmpegAudioValidator().validate(
            tmp_path / "meeting.wav",
            audio_stream_index=0,
            deadline=datetime.now(UTC) - timedelta(seconds=1),
        )

    assert captured.value.code is AudioFailureCode.TIMEOUT
    run.assert_not_called()
