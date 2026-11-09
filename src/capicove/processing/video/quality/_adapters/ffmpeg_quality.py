"""Streaming FFprobe/FFmpeg adapters for full selected-stream inspection."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import selectors
import shutil
import signal
import subprocess  # nosec B404
from collections.abc import Callable, Mapping
from datetime import datetime
from functools import partial
from itertools import pairwise
from pathlib import Path
from typing import Any

from capicove.processing.video.quality._adapters.local_media import check_deadline
from capicove.processing.video.quality.video_quality_contracts import (
    QualityError,
    QualityLimits,
    QualityRequest,
)
from capicove.processing.video.quality.video_quality_ports import QualitySource
from capicove.processing.video.quality.video_quality_rules import union_duration


_NUMBER = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def finite(value: object) -> float | None:
    """Keep absent or nonfinite backend numbers unknown."""
    try:
        number = float(str(value))
        return number if math.isfinite(number) else None
    except (ValueError, TypeError):
        return None


def _executable(value: str) -> Path:
    found = shutil.which(value)
    if found is None:
        raise QualityError("backend_unavailable", "FFmpeg and FFprobe must be installed locally.")
    return Path(found).absolute()


def run_process(
    command: list[str],
    deadline: datetime,
    limits: QualityLimits,
    on_stderr: Callable[[str], None] | None = None,
    on_stdout: Callable[[str], None] | None = None,
) -> tuple[int, str, str]:
    """Drain bounded binary pipes, enforce the deadline, and reap on every exit."""
    check_deadline(deadline)
    try:
        child = subprocess.Popen(  # nosec B603
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env={**os.environ, "LC_ALL": "C", "LANG": "C", "AV_LOG_FORCE_NOCOLOR": "1"},
            start_new_session=True,
        )
    except OSError as error:
        raise QualityError(
            "backend_unavailable", "The configured media executable could not start."
        ) from error
    collected = {"stdout": bytearray(), "stderr": bytearray()}
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    records = 0
    try:
        with selectors.DefaultSelector() as selector:
            assert child.stdout is not None and child.stderr is not None
            selector.register(child.stdout, selectors.EVENT_READ, "stdout")
            selector.register(child.stderr, selectors.EVENT_READ, "stderr")
            while selector.get_map():
                check_deadline(deadline)
                for key, _ in selector.select(0.1):
                    name = key.data
                    chunk = os.read(key.fd, 65536)
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    callback = on_stdout if name == "stdout" else on_stderr
                    collected[name].extend(chunk)
                    maximum = (
                        limits.max_probe_bytes
                        if name == "stdout" and callback is None
                        else limits.max_diagnostic_bytes
                    )
                    if len(collected[name]) > maximum:
                        if name == "stdout" and callback is None:
                            raise QualityError(
                                "resource_limit", "Media provider output exceeds its byte budget."
                            )
                        del collected[name][:-maximum]
                    if callback is not None:
                        buffers[name].extend(chunk)
                        while b"\n" in buffers[name]:
                            line, _, rest = buffers[name].partition(b"\n")
                            buffers[name] = bytearray(rest)
                            records += 1
                            if records > limits.max_records or len(line) > 65536:
                                raise QualityError(
                                    "resource_limit",
                                    "Media analysis records exceed their parsing budget.",
                                )
                            callback(line.decode("utf-8", errors="replace").rstrip("\r"))
                        if len(buffers[name]) > 65536:
                            raise QualityError(
                                "resource_limit", "Media analysis record exceeds its byte budget."
                            )
            for name in buffers:
                callback = on_stdout if name == "stdout" else on_stderr
                if callback is not None and buffers[name]:
                    callback(buffers[name].decode("utf-8", errors="replace"))
        remaining = max(0.001, (deadline - datetime.now(deadline.tzinfo)).total_seconds())
        try:
            code = child.wait(timeout=remaining)
        except subprocess.TimeoutExpired as error:
            raise QualityError("timeout", "Media inspection exceeded its deadline.") from error
        return (
            code,
            collected["stdout"].decode(errors="replace"),
            collected["stderr"].decode(errors="replace"),
        )
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        if child.stdout is not None:
            child.stdout.close()
        if child.stderr is not None:
            child.stderr.close()


class ScanParser:
    """Parse per-track timestamps, filter events and a final EBU summary."""

    def __init__(
        self, limits: QualityLimits, selection: Mapping[str, Any], black_min_seconds: float = 2.0
    ) -> None:
        self.limits = limits
        self.black_min_seconds = black_min_seconds
        self.selection = selection
        self.video: dict[str, Any] = {
            "start": None,
            "end": None,
            "last": None,
            "duration": None,
            "frames": 0,
        }
        self.audio: dict[str, Any] = {"start": None, "end": None, "samples": 0}
        self.events: dict[str, list[dict[str, Any]]] = {
            k: [] for k in ("black", "freeze", "silence")
        }
        self.intervals: dict[str, dict[int | None, list[tuple[float, float]]]] = {
            k: {} for k in self.events
        }
        self.pending: dict[tuple[str, int | None], float] = {}
        self.totals: dict[str, dict[str, Any]] = {
            k: {"count": 0, "longest": 0.0} for k in self.events
        }
        self.loudness: dict[str, float | None] = {}
        self.ebu_summary = False
        self.ebu_peak = False
        self.progress_end = False
        self.decode_errors = 0
        self.failure_code: str | None = None

    def progress(self, line: str) -> None:
        if line == "progress=end":
            self.progress_end = True

    def _timestamp(self, track: dict[str, Any], pts: float, duration: float | None) -> None:
        previous = track.get("last")
        if previous is not None and pts < previous - 1e-6:
            raise QualityError(
                "decode_failed", "Decoded presentation timestamps are not monotonic."
            )
        if track["start"] is None:
            track["start"] = pts
        if pts - track["start"] > self.limits.max_duration_seconds:
            raise QualityError("resource_limit", "Decoded media exceeds the duration budget.")
        track["last"] = pts
        track["duration"] = duration
        track["end"] = pts + duration if duration is not None and duration > 0 else None

    def frame_line(self, line: str, kind: str) -> None:
        """Read structured FFprobe frames, independent of concurrent human logs."""
        if not line.strip():
            return
        if not line.startswith("frame|"):
            raise QualityError("parse_failed", "FFprobe emitted an unrecognized frame record.")
        fields = dict(part.split("=", 1) for part in line.split("|")[1:] if "=" in part)
        expected = self.selection["video_index" if kind == "video" else "audio_index"]
        if fields.get("stream_index") != str(expected):
            raise QualityError("parse_failed", "FFprobe emitted an unselected frame stream.")
        pts = finite(fields.get("pts_time"))
        if pts is None:
            pts = finite(fields.get("best_effort_timestamp_time"))
        if pts is None:
            raise QualityError("parse_failed", "Decoded frame is missing a presentation timestamp.")
        duration = finite(fields.get("duration_time"))
        if duration is None:
            duration = finite(fields.get("pkt_duration_time"))
        if kind == "video":
            self._timestamp(self.video, pts, duration)
            self.video["frames"] += 1
        else:
            try:
                samples = int(fields["nb_samples"])
            except (KeyError, ValueError) as error:
                raise QualityError(
                    "parse_failed", "Decoded audio frame has no sample count."
                ) from error
            if samples <= 0:
                raise QualityError(
                    "parse_failed", "Decoded audio frame has an invalid sample count."
                )
            rate = self.selection.get("audio_sample_rate")
            if duration is None and rate and rate > 0:
                duration = samples / rate
            self._timestamp(self.audio, pts, duration)
            self.audio["samples"] += samples

    def _event(
        self, kind: str, start: float, end: float, channel: int | None, eof: bool = False
    ) -> None:
        if end < start or not math.isfinite(end - start):
            raise QualityError("parse_failed", "A detector emitted an invalid interval.")
        if end == start:
            return
        self.totals[kind]["count"] += 1
        self.totals[kind]["longest"] = max(self.totals[kind]["longest"], end - start)
        # Merge online per channel; retained interval memory is separately bounded.
        intervals = self.intervals[kind].setdefault(channel, [])
        if intervals and start <= intervals[-1][1]:
            intervals[-1] = (intervals[-1][0], max(intervals[-1][1], end))
        else:
            intervals.append((start, end))
        if sum(len(v) for v in self.intervals[kind].values()) > self.limits.max_records:
            raise QualityError(
                "resource_limit", "Detector interval aggregation exceeds its budget."
            )
        if len(self.events[kind]) < self.limits.max_events:
            self.events[kind].append(
                {
                    "type": kind,
                    "stream_index": self.selection[
                        "audio_index" if kind == "silence" else "video_index"
                    ],
                    "channel": channel,
                    "start_seconds": start,
                    "end_seconds": end,
                    "duration_seconds": end - start,
                    "closed_by_eof": eof,
                }
            )

    def line(self, line: str) -> None:
        """Separate concurrent filter records that share a physical stderr line."""
        prefixes = list(
            re.finditer(
                r"\[(?:Parsed_)?(?:ashowinfo|showinfo|metadata|freezedetect|silencedetect|ebur128)"
                r"(?:_\d+)? @ [^]]+\]",
                line,
            )
        )
        if len(prefixes) > 1:
            positions = [0, *(match.start() for match in prefixes[1:]), len(line)]
            for start, end in pairwise(positions):
                self._record(line[start:end])
        else:
            self._record(line)

    def _record(self, line: str) -> None:
        key: tuple[str, int | None]
        lower = line.lower()
        if line.startswith("[") and any(
            s in lower
            for s in (
                "error while decoding",
                "error submitting",
                "invalid data found",
                "corrupt",
                "concealing",
                "decode_slice_header error",
                "file ended prematurely",
                "partial file",
                "invalid nal",
                "error parsing",
            )
        ):
            self.decode_errors += 1
        for event in ("start", "end"):
            match = re.fullmatch(
                r"\[(?:Parsed_)?metadata(?:_\d+)? @ [^]]+\] lavfi\.black_"
                + event
                + r"=("
                + _NUMBER
                + r")",
                line.strip(),
            )
            if match:
                key = ("black", None)
                if event == "start":
                    self.pending[key] = float(match[1])
                elif key in self.pending:
                    start, end = self.pending.pop(key), float(match[1])
                    if end - start >= self.black_min_seconds - 1e-6:
                        self._event("black", start, end, None)
                else:
                    raise QualityError("parse_failed", "Black interval ended without a start.")
        if "freezedetect" in line:
            for event in ("start", "end"):
                match = re.search(
                    r"lavfi\.freezedetect\.freeze_" + event + r":\s*(" + _NUMBER + r")", line
                )
                if match:
                    key = ("freeze", None)
                    if event == "start":
                        self.pending[key] = float(match[1])
                    elif key in self.pending:
                        self._event("freeze", self.pending.pop(key), float(match[1]), None)
                    else:
                        raise QualityError("parse_failed", "Static interval ended without a start.")
        if "silencedetect" in line:
            channel_match = re.search(r"channel:\s*(\d+)", line)
            channel = int(channel_match[1]) if channel_match else None
            for event in ("start", "end"):
                match = re.search(r"silence_" + event + r":\s*(" + _NUMBER + r")", line)
                if match:
                    key = ("silence", channel)
                    if event == "start":
                        self.pending[key] = float(match[1])
                    elif key in self.pending:
                        self._event("silence", self.pending.pop(key), float(match[1]), channel)
                    else:
                        raise QualityError("parse_failed", "Quiet interval ended without a start.")
        if "ebur128" in line and "Summary:" in line:
            self.ebu_summary, self.ebu_peak = True, False
            self.loudness = {}
        elif self.ebu_summary:
            if "True peak:" in line:
                self.ebu_peak = True
            for field, pattern in (
                ("integrated_lufs", r"^\s*I:\s*(\S+)\s+LUFS"),
                ("range_lu", r"^\s*LRA:\s*(\S+)\s+LU"),
                ("true_peak_dbtp", r"^\s*Peak:\s*(\S+)\s+dBFS"),
            ):
                match = re.search(pattern, line)
                if match and (field != "true_peak_dbtp" or self.ebu_peak):
                    self.loudness[field] = finite(match[1])

    def finish(self, complete: bool) -> dict[str, Any]:
        """Close pending events only at verified track EOF and label coverage."""
        complete = complete and self.progress_end and not self.decode_errors
        if complete:
            for (kind, channel), start in list(self.pending.items()):
                track = self.audio if kind == "silence" else self.video
                if track["end"] is not None:
                    if kind != "black" or track["end"] - start >= self.black_min_seconds - 1e-6:
                        self._event(kind, start, track["end"], channel, True)
                    del self.pending[(kind, channel)]
        origin_values = [v["start"] for v in (self.video, self.audio) if v["start"] is not None]
        origin = min(origin_values) if origin_values else None
        for kind, events in self.events.items():
            for event in events:
                if origin is not None:
                    event["start_seconds"] -= origin
                    event["end_seconds"] -= origin
                track = self.audio if kind == "silence" else self.video
                if (
                    complete
                    and track["end"] is not None
                    and abs((event["end_seconds"] + (origin or 0)) - track["end"]) < 0.001
                ):
                    event["closed_by_eof"] = True
        totals = {}
        for kind, channels in self.intervals.items():
            totals[kind] = {
                **self.totals[kind],
                "union_seconds": union_duration(
                    [i for values in channels.values() for i in values]
                ),
                "events_truncated": self.totals[kind]["count"] > len(self.events[kind]),
            }
        return {
            "complete": complete,
            "video": self.video,
            "audio": self.audio,
            "origin_seconds": origin,
            "events": self.events,
            "totals": totals,
            "pending": sorted({k[0] for k in self.pending}),
            "loudness": self.loudness,
            "decode_errors": self.decode_errors,
            "failure_code": self.failure_code,
        }


class FfmpegQualityBackend:
    """Run composition-owned providers with fixed local-only command arguments."""

    def __init__(
        self, limits: QualityLimits, *, ffmpeg: str = "ffmpeg", ffprobe: str = "ffprobe"
    ) -> None:
        self.limits = limits
        self._ffmpeg = ffmpeg
        self._ffprobe = ffprobe

    def probe(self, source: QualitySource, deadline: datetime) -> dict[str, Any]:
        executable = _executable(self._ffprobe)
        code, output, _ = run_process(
            [
                str(executable),
                "-v",
                "error",
                "-protocol_whitelist",
                "file",
                "-f",
                source.demuxer,
                "-show_program_version",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(source.staged_path),
            ],
            deadline,
            self.limits,
        )
        if code:
            raise QualityError("decode_failed", "FFprobe could not inspect the staged media.")
        try:
            value = json.loads(output)
        except ValueError as error:
            raise QualityError("parse_failed", "FFprobe returned invalid JSON.") from error
        if not isinstance(value, dict) or not isinstance(value.get("streams"), list):
            raise QualityError("parse_failed", "FFprobe returned an invalid stream inventory.")
        with executable.open("rb") as stream:
            value["ffprobe_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        return dict(value)

    def scan(
        self,
        source: QualitySource,
        selection: Mapping[str, Any],
        request: QualityRequest,
        limits: QualityLimits,
        deadline: datetime,
    ) -> dict[str, Any]:
        executable = _executable(self._ffmpeg)
        code, filters, _ = run_process(
            [str(executable), "-hide_banner", "-filters"], deadline, limits
        )
        required = {"null"}
        if selection["audio_index"] is not None:
            required.add("anull")
            if "silence" in request.checks:
                required.add("silencedetect")
            if "loudness" in request.checks:
                required.add("ebur128")
        required.update({"blackdetect", "metadata"} if "black" in request.checks else set())
        required.update({"freezedetect"} if "freeze" in request.checks else set())
        names = {parts[1] for line in filters.splitlines() if len(parts := line.split()) >= 3}
        if code or not required.issubset(names):
            raise QualityError(
                "backend_unavailable", "The local FFmpeg build lacks a requested analysis filter."
            )
        code, version, _ = run_process([str(executable), "-version"], deadline, limits)
        if code:
            raise QualityError("backend_unavailable", "FFmpeg version discovery failed.")
        opts = request.detector_options
        vf: list[str] = []
        if "black" in request.checks:
            vf.extend(
                [
                    f"blackdetect=d={opts['black_min_seconds']}:pic_th={opts['black_picture_ratio']}:pix_th={opts['black_pixel_threshold']}",
                    "metadata=mode=print:key=lavfi.black_start",
                    "metadata=mode=print:key=lavfi.black_end",
                ]
            )
        if "freeze" in request.checks:
            vf.append(f"freezedetect=n={opts['freeze_noise']}:d={opts['freeze_min_seconds']}")
        af: list[str] = []
        if "silence" in request.checks:
            af.append(
                f"silencedetect=n={opts['silence_threshold_dbfs']}dB:d={opts['silence_min_seconds']}:mono={int(opts['silence_per_channel'])}"
            )
        if "loudness" in request.checks:
            af.append("ebur128=peak=true:framelog=verbose")
        command = [
            str(executable),
            "-hide_banner",
            "-nostdin",
            "-nostats",
            "-loglevel",
            "info",
            "-xerror",
            "-err_detect",
            "explode",
            "-threads",
            str(limits.threads),
            "-filter_threads",
            str(limits.threads),
            "-protocol_whitelist",
            "file",
            "-f",
            source.demuxer,
            "-hwaccel",
            "none",
            "-copyts",
            "-i",
            str(source.staged_path),
            "-map",
            f"0:{selection['video_index']}",
            "-vf",
            ",".join(vf) or "null",
            "-fps_mode",
            "passthrough",
        ]
        if selection["audio_index"] is not None:
            command.extend(
                ["-map", f"0:{selection['audio_index']}", "-af", ",".join(af) or "anull"]
            )
        command.extend(["-threads", str(limits.threads), "-progress", "pipe:1", "-f", "null", "-"])
        parser = ScanParser(limits, selection, opts["black_min_seconds"])
        passes: list[dict[str, Any]] = []
        try:
            probe_executable = _executable(self._ffprobe)
            for kind in ("video", "audio"):
                index = selection["video_index" if kind == "video" else "audio_index"]
                if index is None:
                    continue
                timing_command = [
                    str(probe_executable),
                    "-v",
                    "warning",
                    "-threads",
                    str(limits.threads),
                    "-err_detect",
                    "explode",
                    "-protocol_whitelist",
                    "file",
                    "-f",
                    source.demuxer,
                    "-select_streams",
                    str(index),
                    "-show_frames",
                    "-show_entries",
                    "frame=stream_index,pts_time,best_effort_timestamp_time,duration_time,pkt_duration_time,nb_samples:frame_side_data=",
                    "-of",
                    "compact=p=1:nk=0",
                    str(source.staged_path),
                ]
                passes.append(
                    {
                        "backend": "ffprobe",
                        "purpose": f"{kind}_frame_timing",
                        "stream_indexes": [index],
                        "complete": False,
                    }
                )
                timing_code, _, _ = run_process(
                    timing_command,
                    deadline,
                    limits,
                    parser.line,
                    partial(parser.frame_line, kind=kind),
                )
                passes[-1]["complete"] = timing_code == 0
                if timing_code:
                    raise QualityError(
                        "decode_failed", "Structured selected-stream frame inspection failed."
                    )
            passes.append(
                {
                    "backend": "ffmpeg",
                    "purpose": "detectors_and_integrity",
                    "stream_indexes": [
                        i
                        for i in (selection["video_index"], selection["audio_index"])
                        if i is not None
                    ],
                    "complete": False,
                }
            )
            code, _, _ = run_process(command, deadline, limits, parser.line, parser.progress)
            passes[-1]["complete"] = code == 0 and parser.progress_end and not parser.decode_errors
            complete = code == 0
        except QualityError as error:
            parser.failure_code = error.code
            complete = False
        result = parser.finish(complete)
        result["passes"] = passes
        result["ffmpeg_version"] = version.splitlines()[0]
        with executable.open("rb") as stream:
            result["ffmpeg_sha256"] = hashlib.file_digest(stream, "sha256").hexdigest()
        return result
