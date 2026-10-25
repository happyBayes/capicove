"""Configure and qualify the EmbeddingGemma 2 Q8_0 GGUF + llama.cpp CPU runtime.

Two explicit operator steps; neither downloads anything or opens a network socket:

``receipt``
    Validate operator-supplied GGUF files (magic, version, architecture and Q8_0
    quantization), hash them and the ``llama-server`` binary, record the pinned
    llama.cpp revision and build options, the upstream revisions and license, and
    write a local receipt plus a model configuration usable by ``ModelFactory``.

``benchmark``
    Start the provider from that receipt on the CPU and measure model load time,
    first and warm embedding latency for one configured audio window, text query
    latency, peak resident memory of the llama.cpp child and batch indexing
    throughput. It verifies that text and audio share one 768-dimensional space
    and that the audio encoder accepts the prepared waveform directly.

Loading alone does not qualify CPU inference: the benchmark must succeed and its
results file must be retained. Vectors from this runtime are not claimed
interchangeable with a Transformers/Sentence Transformers checkpoint.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import resource
import statistics
import struct
import subprocess  # nosec B404
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, BinaryIO


RECEIPT_SCHEMA = "capicove.embeddinggemma2.cpu-receipt.v1"
RESULTS_SCHEMA = "capicove.embeddinggemma2.cpu-benchmark.v1"
TEXT_ARCHITECTURE = "gemma-embedding2"
PROJECTOR_ARCHITECTURE = "clip"
Q8_0_FILE_TYPE = 7
_MAX_METADATA_ENTRIES = 4096
_MAX_STRING_BYTES = 16 * 1024 * 1024
_SCALARS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}
_BUILD_KEYS = (
    "CMAKE_BUILD_TYPE",
    "CMAKE_C_COMPILER",
    "CMAKE_CXX_COMPILER",
    "BUILD_SHARED_LIBS",
    "GGML_NATIVE",
    "GGML_AVX",
    "GGML_AVX2",
    "GGML_AVX512",
    "GGML_FMA",
    "GGML_F16C",
    "GGML_OPENMP",
    "GGML_BLAS",
    "GGML_CUDA",
    "GGML_METAL",
    "GGML_VULKAN",
    "LLAMA_CURL",
    "LLAMA_OPENSSL",
)


def _read(stream: BinaryIO, size: int) -> bytes:
    data = stream.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF metadata")
    return data


def _string(stream: BinaryIO) -> str:
    (length,) = struct.unpack("<Q", _read(stream, 8))
    if length > _MAX_STRING_BYTES:
        raise ValueError("GGUF string exceeds the metadata limit")
    return _read(stream, length).decode("utf-8")


def _value(stream: BinaryIO, kind: int) -> object:
    if kind in _SCALARS:
        layout = _SCALARS[kind]
        return struct.unpack(layout, _read(stream, struct.calcsize(layout)))[0]
    if kind == 8:
        return _string(stream)
    if kind == 9:
        item_kind, length = struct.unpack("<IQ", _read(stream, 12))
        if item_kind == 9:
            raise ValueError("nested GGUF arrays are unsupported")
        if item_kind in _SCALARS:
            stream.seek(struct.calcsize(_SCALARS[item_kind]) * length, os.SEEK_CUR)
        elif item_kind == 8:
            for _ in range(length):
                (size,) = struct.unpack("<Q", _read(stream, 8))
                stream.seek(size, os.SEEK_CUR)
        else:
            raise ValueError("unknown GGUF array type")
        return {"array_type": item_kind, "length": length}
    raise ValueError("unknown GGUF metadata type")


def read_gguf_metadata(path: Path) -> dict[str, object]:
    """Return scalar metadata and array summaries from a GGUF v2/v3 header."""
    with path.open("rb") as stream:
        if _read(stream, 4) != b"GGUF":
            raise ValueError("not a GGUF file")
        version, _tensors, entries = struct.unpack("<IQQ", _read(stream, 20))
        if version not in (2, 3):
            raise ValueError("unsupported GGUF version")
        if entries > _MAX_METADATA_ENTRIES:
            raise ValueError("GGUF metadata entry count exceeds the limit")
        metadata: dict[str, object] = {"gguf.version": version}
        for _ in range(entries):
            key = _string(stream)
            (kind,) = struct.unpack("<I", _read(stream, 4))
            if key in metadata:
                raise ValueError("duplicate GGUF metadata key")
            metadata[key] = _value(stream, kind)
        return metadata


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def describe_asset(path: Path, role: str) -> dict[str, object]:
    """Validate one GGUF role and return its receipt entry."""
    metadata = read_gguf_metadata(path)
    expected = TEXT_ARCHITECTURE if role == "model" else PROJECTOR_ARCHITECTURE
    architecture = metadata.get("general.architecture")
    if architecture != expected:
        raise ValueError(f"{role} architecture must be {expected}")
    if metadata.get("general.file_type") != Q8_0_FILE_TYPE:
        raise ValueError(f"{role} must be quantized as Q8_0")
    if role == "model" and metadata.get("gemma-embedding2.embedding_length_out") != 768:
        raise ValueError("model must project to the 768-dimensional shared space")
    if role == "mmproj" and metadata.get("clip.has_audio_encoder") is not True:
        raise ValueError("mmproj must contain the audio encoder")
    summary = {
        key: value
        for key, value in metadata.items()
        if not isinstance(value, dict)
        and isinstance(value, (int, float, str, bool))
        and not key.startswith("tokenizer.")
        and (not isinstance(value, str) or len(value) <= 256)
    }
    return {
        "name": path.name,
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "architecture": architecture,
        "quantization": "Q8_0",
        "gguf_version": metadata["gguf.version"],
        "metadata": summary,
    }


def _git_revision(source: Path) -> str:
    """Read HEAD without executing Git, hooks or configuration."""
    head = (source / ".git/HEAD").read_text().strip()
    if head.startswith("ref: "):
        reference = head[5:]
        loose = source / ".git" / reference
        if loose.is_file():
            head = loose.read_text().strip()
        else:
            for line in (source / ".git/packed-refs").read_text().splitlines():
                if line.endswith(" " + reference):
                    head = line.split()[0]
                    break
    if len(head) != 40 or any(c not in "0123456789abcdef" for c in head):
        raise ValueError("llama.cpp source revision is not a pinned commit")
    return head


def _build_options(server: Path) -> dict[str, str]:
    cache = server.parent.parent / "CMakeCache.txt"
    options: dict[str, str] = {}
    for line in cache.read_text().splitlines():
        name, _, value = line.partition("=")
        key = name.split(":", 1)[0]
        if key in _BUILD_KEYS:
            options[key] = value
    return options


def _run_version(server: Path) -> str:
    completed = subprocess.run(  # nosec B603
        [str(server), "--version"],
        capture_output=True,
        timeout=30,
        check=False,
        env={"PATH": "/usr/bin:/bin"},
    )
    lines = (completed.stdout + completed.stderr).decode("utf-8", "replace").splitlines()
    return "\n".join(line for line in lines if line.strip())[:1024]


def build_receipt(args: argparse.Namespace) -> dict[str, Any]:
    """Validate and hash every pinned input; no file is modified."""
    server = args.llama_server.resolve()
    model, mmproj = args.model.resolve(), args.mmproj.resolve()
    if model.parent != mmproj.parent:
        raise ValueError("model and mmproj must share one model root directory")
    return {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "model_root": str(model.parent),
        "assets": {
            "model": describe_asset(model, "model"),
            "mmproj": describe_asset(mmproj, "mmproj"),
        },
        "upstream": {
            "model_repository": args.upstream_repository,
            "model_revision": args.upstream_revision,
            "gguf_repository": args.gguf_repository,
            "gguf_revision": args.gguf_revision,
            "license": args.license,
            "conversion": "upstream GGUF conversion by the GGUF repository publisher",
        },
        "llama_cpp": {
            "release": args.llama_cpp_release,
            "revision": _git_revision(args.llama_cpp_source.resolve()),
            "server": str(server),
            "server_sha256": _sha256(server),
            "server_version": _run_version(server),
            "build_options": _build_options(server),
        },
        "space": {
            "dimension": 768,
            "modalities": ["text", "image", "audio"],
            "interchangeability": "not qualified against Transformers checkpoints",
        },
    }


def profile_options(receipt: dict[str, Any], *, threads: int) -> dict[str, object]:
    """Project a receipt to provider options pinned by the same digests."""
    assets, runtime, upstream = receipt["assets"], receipt["llama_cpp"], receipt["upstream"]
    return {
        "model": assets["model"]["name"],
        "model_sha256": assets["model"]["sha256"],
        "mmproj": assets["mmproj"]["name"],
        "mmproj_sha256": assets["mmproj"]["sha256"],
        "llama_server": runtime["server"],
        "llama_server_sha256": runtime["server_sha256"],
        "backend_version": f"llama.cpp {runtime['release']} {runtime['revision']}",
        "model_revision": f"{upstream['gguf_repository']}@{upstream['gguf_revision']}",
        "threads": threads,
    }


def profile_manifest(receipt: dict[str, Any], *, threads: int) -> str:
    """Return the manifest digest the provider will verify."""
    from capicove.model.providers.embeddinggemma import manifest_digest

    return manifest_digest(profile_options(receipt, threads=threads))


def model_configuration(
    receipt: dict[str, Any], *, threads: int, code: bool = False
) -> dict[str, object]:
    """Return a version-1 model configuration with one shared-space or code binding."""
    provider, capability, binding = (
        ("embeddinggemma2-llamacpp-code", "text_embedding", "code")
        if code
        else ("embeddinggemma2-llamacpp", "multimodal_embedding", "unified")
    )
    return {
        "schema_version": 1,
        "profiles": {
            "embeddinggemma2-q8-cpu": {
                "provider": provider,
                "capability": capability,
                "root": receipt["model_root"],
                "manifest_sha256": profile_manifest(receipt, threads=threads),
                "device": "cpu",
                "options": profile_options(receipt, threads=threads),
            }
        },
        "bindings": {binding: "embeddinggemma2-q8-cpu"},
    }


def _synthetic_window(seconds: int, rate: int) -> bytes:
    """Return a deterministic tone/noise mix for latency only, never semantics."""
    frames = seconds * rate
    state = 0x12345678
    samples = []
    for index in range(frames):
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        noise = (state / 0x7FFFFFFF - 0.5) * 0.2
        tone = 0.4 * math.sin(2 * math.pi * 440 * index / rate)
        sweep = 0.2 * math.sin(2 * math.pi * (200 + 600 * index / frames) * index / rate)
        samples.append(int(max(-1.0, min(1.0, tone + sweep + noise)) * 32767))
    return struct.pack(f"<{frames}h", *samples)


def _prepared_window(path: Path, scratch: Path) -> tuple[bytes, dict[str, object]]:
    from capicove.processing.audio import AudioProcessor
    from capicove.processing.audio._adapters.audio_windows import FfmpegAudioWindowBackend
    from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
    from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
    from capicove.processing.audio.retrieval import AudioWindowPreparer

    store, staging = scratch / "store", scratch / "staging"
    for directory in (store, staging):
        directory.mkdir(mode=0o700, exist_ok=True)
    preparer = AudioWindowPreparer(
        AudioProcessor(
            FfprobeAudioStreamInspector("/usr/bin/ffprobe", restrict_inputs=True),
            FfmpegAudioValidator("/usr/bin/ffmpeg", restrict_inputs=True),
        ),
        FfmpegAudioWindowBackend("/usr/bin/ffmpeg"),
    )
    deadline = datetime.now(UTC) + timedelta(minutes=10)
    with preparer.prepare(path, store, staging, deadline=deadline) as prepared:
        receipt = prepared.receipts[0]
        return prepared.window_pcm(0), {
            "kind": "operator_audio_file",
            "source_sha256": prepared.evidence.sha256,
            "window_receipt_sha256": receipt.digest,
            "frames": receipt.end_frame - receipt.start_frame,
            "decoder": receipt.decoder,
        }


def _child_peak_rss_kib(pid: int) -> int | None:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmHWM:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _timed(call: Any) -> tuple[float, Any]:
    started = time.perf_counter()
    value = call()
    return time.perf_counter() - started, value


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "count": len(ordered),
        "min_seconds": ordered[0],
        "p50_seconds": statistics.median(ordered),
        "p95_seconds": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
        "max_seconds": ordered[-1],
    }


def _cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    return math.fsum(a * b for a, b in zip(left, right, strict=True))


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Measure real CPU inference from a receipt; every failure aborts the run."""
    from capicove.model.configuration import ModelConfiguration
    from capicove.model.factory import ModelFactory
    from capicove.model.multimodal_embedding import PcmAudio, TextDocument

    receipt = json.loads(args.receipt.read_text())
    if receipt.get("schema") != RECEIPT_SCHEMA:
        raise ValueError("unsupported receipt schema")
    configuration = ModelConfiguration.from_mapping(
        model_configuration(receipt, threads=args.threads)
    )
    rate, seconds = 16_000, args.window_seconds
    if args.audio is not None:
        window, audio_source = _prepared_window(args.audio.resolve(), args.scratch)
    else:
        window = _synthetic_window(seconds, rate)
        audio_source = {"kind": "synthetic_signal", "frames": len(window) // 2}
    clip = PcmAudio(window, rate)
    deadline = datetime.now(UTC) + timedelta(seconds=args.timeout)
    factory = ModelFactory(configuration)
    started = time.perf_counter()
    try:
        with factory.shared_space_encoder("unified") as encoder:
            load_wall = time.perf_counter() - started
            pid = encoder.pid  # type: ignore[attr-defined]
            first_audio, audio_vectors = _timed(
                lambda: encoder.encode_audio([clip], deadline=deadline)
            )
            warm_audio = []
            repeat_vectors = []
            for _ in range(args.repeats):
                elapsed, vectors = _timed(lambda: encoder.encode_audio([clip], deadline=deadline))
                warm_audio.append(elapsed)
                repeat_vectors.append(vectors[0])
            first_text, text_vectors = _timed(
                lambda: encoder.encode_text_queries(["a dog barking outside"], deadline=deadline)
            )
            warm_text = [
                _timed(
                    lambda: encoder.encode_text_queries(["an alarm ringing"], deadline=deadline)
                )[0]
                for _ in range(args.repeats)
            ]
            _document_seconds, documents = _timed(
                lambda: encoder.encode_text_documents(
                    [TextDocument("A dog barks repeatedly in a backyard.")], deadline=deadline
                )
            )
            batch_size = encoder.max_batch  # type: ignore[attr-defined]
            indexed = 0
            batch_started = time.perf_counter()
            while indexed < args.batch_windows:
                count = min(batch_size, args.batch_windows - indexed)
                encoder.encode_audio([clip] * count, deadline=deadline)
                indexed += count
            batch_elapsed = time.perf_counter() - batch_started
            peak_rss = _child_peak_rss_kib(pid)
            space = encoder.space
            load_seconds = encoder.load_seconds  # type: ignore[attr-defined]
    finally:
        factory.close()
    audio_vector, text_vector = audio_vectors[0], text_vectors[0]
    dimensions = {len(audio_vector), len(text_vector), len(documents[0])}
    if dimensions != {768} or space.dimension != 768:
        raise RuntimeError("text and audio outputs do not share the 768-dimensional space")
    repeatability = min(_cosine(audio_vector, vector) for vector in repeat_vectors)
    return {
        "schema": RESULTS_SCHEMA,
        "completed_at": datetime.now(UTC).isoformat(),
        "status": "succeeded",
        "receipt_sha256": hashlib.sha256(args.receipt.read_bytes()).hexdigest(),
        "profile_identity": configuration.profiles["embeddinggemma2-q8-cpu"].identity,
        "space": {
            "space_id": space.space_id,
            "model_revision": space.model_revision,
            "dimension": space.dimension,
            "normalization": space.normalization.value,
        },
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": _cpu_model(),
            "logical_cpus": os.cpu_count(),
            "memory_kib": _memory_kib(),
            "python": sys.version.split()[0],
            "threads": args.threads,
        },
        "audio_input": {
            **audio_source,
            "sample_rate_hz": rate,
            "window_seconds": len(window) / 2 / rate,
        },
        "measurements": {
            "model_load_seconds": load_seconds,
            "model_load_wall_seconds": load_wall,
            "first_audio_embedding_seconds": first_audio,
            "warm_audio_embedding": _summary(warm_audio),
            "first_text_query_seconds": first_text,
            "warm_text_query": _summary(warm_text),
            "llama_server_peak_rss_kib": peak_rss,
            "benchmark_process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "batch_indexing": {
                "windows": args.batch_windows,
                "batch_size": batch_size,
                "elapsed_seconds": batch_elapsed,
                "windows_per_second": args.batch_windows / batch_elapsed,
                "audio_seconds_per_wall_second": args.batch_windows
                * len(window)
                / 2
                / rate
                / batch_elapsed,
            },
        },
        "checks": {
            "shared_dimension": 768,
            "audio_waveform_accepted_directly": True,
            "audio_repeat_min_cosine": repeatability,
            "text_audio_cosine": _cosine(audio_vector, text_vector),
            "audio_document_cosine": _cosine(audio_vector, documents[0]),
        },
        "limitations": [
            "Latency/memory qualification only; no semantic retrieval quality is measured.",
            "A synthetic signal establishes timing, not recognition of real sounds."
            if audio_source["kind"] == "synthetic_signal"
            else "One operator audio window establishes timing, not retrieval quality.",
            "llama.cpp vectors are not claimed interchangeable with other runtimes.",
        ],
    }


def _cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or "unknown"


def _memory_kib() -> int | None:
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            if line.startswith("MemTotal:"):
                return int(line.split()[1])
    except OSError:
        return None
    return None


def _write_new(path: Path, payload: dict[str, Any]) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def main(argv: list[str] | None = None) -> int:
    """Run one explicit subcommand; outputs are created exclusively, never overwritten."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    receipt = commands.add_parser("receipt", help="Validate and hash pinned local assets.")
    receipt.add_argument("--model", type=Path, required=True)
    receipt.add_argument("--mmproj", type=Path, required=True)
    receipt.add_argument("--llama-server", type=Path, required=True)
    receipt.add_argument("--llama-cpp-source", type=Path, required=True)
    receipt.add_argument("--llama-cpp-release", required=True)
    receipt.add_argument("--gguf-repository", required=True)
    receipt.add_argument("--gguf-revision", required=True)
    receipt.add_argument("--upstream-repository", required=True)
    receipt.add_argument("--upstream-revision", required=True)
    receipt.add_argument("--license", required=True)
    receipt.add_argument("--threads", type=int, default=4)
    receipt.add_argument("--output", type=Path, required=True)
    receipt.add_argument("--model-config-output", type=Path, required=True)
    receipt.add_argument(
        "--code-model-config-output",
        type=Path,
        help="Also write a text_embedding configuration bound as 'code' for code search.",
    )
    benchmark = commands.add_parser("benchmark", help="Measure real CPU inference.")
    benchmark.add_argument("--receipt", type=Path, required=True)
    benchmark.add_argument("--audio", type=Path)
    benchmark.add_argument("--scratch", type=Path, required=True)
    benchmark.add_argument("--threads", type=int, default=4)
    benchmark.add_argument("--window-seconds", type=int, default=30)
    benchmark.add_argument("--repeats", type=int, default=5)
    benchmark.add_argument("--batch-windows", type=int, default=16)
    benchmark.add_argument("--timeout", type=int, default=1800)
    benchmark.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "receipt":
        built = build_receipt(args)
        _write_new(args.output, built)
        _write_new(args.model_config_output, model_configuration(built, threads=args.threads))
        if args.code_model_config_output is not None:
            _write_new(
                args.code_model_config_output,
                model_configuration(built, threads=args.threads, code=True),
            )
        print(json.dumps({"receipt": str(args.output), "assets": list(built["assets"])}))
        return 0
    if args.repeats < 1 or args.batch_windows < 1 or not 1 <= args.window_seconds <= 300:
        parser.error("repeats, batch windows and window seconds must be positive and bounded")
    args.scratch.mkdir(mode=0o700, parents=True, exist_ok=True)
    results = run_benchmark(args)
    _write_new(args.output, results)
    print(json.dumps(results["measurements"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
