"""Bounded FFprobe adapter for audio-stream inventory."""

from __future__ import annotations

import json
import os
import selectors
import subprocess  # nosec B404
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

from capicove.processing.audio._adapters.ffmpeg import restricted_input_arguments
from capicove.processing.audio.models import AudioFailureCode, AudioStreamInventory
from capicove.processing.audio.ports import AudioValidationError


_MAX_PROBE_OUTPUT_BYTES = 64 * 1024
_MAX_AUDIO_STREAMS = 64


class _ProbeOutputTooLarge(ValueError):
    pass


class FfprobeAudioStreamInspector:
    """Count audio streams without selecting by container metadata."""

    def __init__(
        self,
        executable: str = "ffprobe",
        timeout_seconds: float = 300.0,
        *,
        restrict_inputs: bool = False,
    ) -> None:
        if not executable:
            raise ValueError("executable must not be blank")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._executable = executable
        self._timeout_seconds = timeout_seconds
        self._restrict_inputs = restrict_inputs

    def inspect(
        self,
        source: Path,
        *,
        deadline: datetime | None = None,
    ) -> AudioStreamInventory:
        """Return audio array length while discarding metadata and diagnostics."""
        timeout = _remaining_timeout(deadline, self._timeout_seconds)
        arguments = [
            self._executable,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=index",
            "-of",
            "json",
            *(restricted_input_arguments(source) if self._restrict_inputs else []),
            str(source),
        ]
        try:
            returncode, output = _run_probe(arguments, timeout)
        except FileNotFoundError as error:
            raise AudioValidationError(
                AudioFailureCode.DECODER_UNAVAILABLE,
                "audio stream decoder is unavailable",
            ) from error
        except subprocess.TimeoutExpired as error:
            raise AudioValidationError(
                AudioFailureCode.TIMEOUT,
                "audio stream inventory timed out",
            ) from error
        except _ProbeOutputTooLarge as error:
            raise AudioValidationError(
                AudioFailureCode.INVALID_AUDIO,
                "audio stream inventory is invalid",
            ) from error
        except OSError as error:
            raise AudioValidationError(
                AudioFailureCode.DECODER_UNAVAILABLE,
                "audio stream decoder is unavailable",
            ) from error
        if returncode != 0 or len(output) > _MAX_PROBE_OUTPUT_BYTES:
            raise AudioValidationError(
                AudioFailureCode.INVALID_AUDIO,
                "audio stream inventory is invalid",
            )
        try:
            payload = json.loads(output)
            root = _mapping(payload)
            streams = _sequence(root["streams"])
            if len(streams) > _MAX_AUDIO_STREAMS:
                raise ValueError("audio stream inventory exceeds the limit")
            for stream in streams:
                item = _mapping(stream)
                global_index = item["index"]
                if (
                    not isinstance(global_index, int)
                    or isinstance(global_index, bool)
                    or global_index < 0
                ):
                    raise ValueError("audio stream index is invalid")
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            raise AudioValidationError(
                AudioFailureCode.INVALID_AUDIO,
                "audio stream inventory is invalid",
            ) from error
        return AudioStreamInventory(len(streams))


def _remaining_timeout(deadline: datetime | None, configured_timeout: float) -> float:
    if deadline is None:
        return configured_timeout
    if deadline.tzinfo is None or deadline.utcoffset() is None:
        raise ValueError("deadline must be timezone-aware")
    remaining = (deadline - datetime.now(UTC)).total_seconds()
    if remaining <= 0:
        raise AudioValidationError(
            AudioFailureCode.TIMEOUT,
            "audio stream inventory timed out",
        )
    return min(configured_timeout, remaining)


def _run_probe(arguments: list[str], timeout: float) -> tuple[int, bytes]:
    process = subprocess.Popen(  # nosec B603
        arguments,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
        shell=False,
    )
    return _read_bounded_output(process, timeout)


def _read_bounded_output(
    process: subprocess.Popen[bytes],
    timeout: float,
) -> tuple[int, bytes]:
    stdout = process.stdout
    if stdout is None:
        _terminate(process)
        raise OSError("probe stdout pipe is unavailable")
    output = bytearray()
    timeout_at = time.monotonic() + timeout
    selector = selectors.DefaultSelector()
    try:
        selector.register(stdout, selectors.EVENT_READ)
        while True:
            remaining = timeout_at - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(process.args, timeout)
            if not selector.select(remaining):
                raise subprocess.TimeoutExpired(process.args, timeout)
            read_size = min(8192, _MAX_PROBE_OUTPUT_BYTES + 1 - len(output))
            chunk = os.read(stdout.fileno(), read_size)
            if not chunk:
                break
            output.extend(chunk)
            if len(output) > _MAX_PROBE_OUTPUT_BYTES:
                raise _ProbeOutputTooLarge("probe output exceeds the limit")
        remaining = timeout_at - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout)
        return process.wait(timeout=remaining), bytes(output)
    except BaseException:
        _terminate(process)
        raise
    finally:
        selector.close()
        stdout.close()


def _terminate(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.kill()
    try:
        process.wait(timeout=1)
    except subprocess.TimeoutExpired:
        pass


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError("probe object is invalid")
    return cast(Mapping[str, object], value)


def _sequence(value: object) -> Sequence[object]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("probe streams are invalid")
    return cast(Sequence[object], value)
