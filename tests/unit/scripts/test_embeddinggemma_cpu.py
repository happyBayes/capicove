"""Offline GGUF validation and receipt rules of the EmbeddingGemma 2 CPU script."""

from __future__ import annotations

import hashlib
import importlib.util
import struct
from pathlib import Path
from types import ModuleType

import pytest


ROOT = Path(__file__).resolve().parents[3]


def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "embeddinggemma_cpu", ROOT / "scripts/embeddinggemma_cpu.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _gguf(path: Path, metadata: list[tuple[str, int, object]], *, version: int = 3) -> Path:
    body = b"GGUF" + struct.pack("<IQQ", version, 0, len(metadata))
    for key, kind, value in metadata:
        body += _string(key) + struct.pack("<I", kind)
        if kind == 8:
            body += _string(str(value))
        elif kind == 4:
            body += struct.pack("<I", value)
        elif kind == 7:
            body += struct.pack("<?", value)
        elif kind == 9:
            items = list(value)  # type: ignore[call-overload]
            body += struct.pack("<IQ", 8, len(items)) + b"".join(_string(i) for i in items)
        else:
            raise AssertionError(kind)
    path.write_bytes(body)
    return path


def test_reads_bounded_metadata_and_skips_large_arrays(tmp_path: Path) -> None:
    path = _gguf(
        tmp_path / "m.gguf",
        [
            ("general.architecture", 8, "gemma-embedding2"),
            ("general.file_type", 4, 7),
            ("tokenizer.ggml.tokens", 9, ["a", "b", "c"]),
            ("gemma-embedding2.embedding_length", 4, 512),
        ],
    )
    metadata = script().read_gguf_metadata(path)
    assert metadata["general.architecture"] == "gemma-embedding2"
    assert metadata["general.file_type"] == 7
    assert metadata["tokenizer.ggml.tokens"] == {"array_type": 8, "length": 3}


@pytest.mark.parametrize(
    "payload",
    [b"GGML" + b"\x00" * 20, b"GGUF" + struct.pack("<IQQ", 1, 0, 0), b"GGUF\x03"],
)
def test_rejects_non_gguf_old_versions_and_truncation(tmp_path: Path, payload: bytes) -> None:
    path = tmp_path / "bad.gguf"
    path.write_bytes(payload)
    with pytest.raises(ValueError):
        script().read_gguf_metadata(path)


def test_model_role_requires_q8_text_model_and_q8_projector(tmp_path: Path) -> None:
    module = script()
    text = _gguf(
        tmp_path / "t.gguf",
        [
            ("general.architecture", 8, "gemma-embedding2"),
            ("general.file_type", 4, 7),
            ("gemma-embedding2.embedding_length_out", 4, 768),
        ],
    )
    projector = _gguf(
        tmp_path / "p.gguf",
        [
            ("general.architecture", 8, "clip"),
            ("general.file_type", 4, 7),
            ("clip.has_audio_encoder", 7, True),
        ],
    )
    assert module.describe_asset(text, "model")["quantization"] == "Q8_0"
    assert module.describe_asset(projector, "mmproj")["architecture"] == "clip"
    with pytest.raises(ValueError, match="Q8_0"):
        module.describe_asset(
            _gguf(
                tmp_path / "f16.gguf",
                [("general.architecture", 8, "gemma-embedding2"), ("general.file_type", 4, 1)],
            ),
            "model",
        )
    with pytest.raises(ValueError, match="architecture"):
        module.describe_asset(projector, "model")
    with pytest.raises(ValueError, match="audio encoder"):
        module.describe_asset(
            _gguf(
                tmp_path / "vision-only.gguf",
                [("general.architecture", 8, "clip"), ("general.file_type", 4, 7)],
            ),
            "mmproj",
        )
    with pytest.raises(ValueError, match="768"):
        module.describe_asset(
            _gguf(
                tmp_path / "t512.gguf",
                [("general.architecture", 8, "gemma-embedding2"), ("general.file_type", 4, 7)],
            ),
            "model",
        )
    described = module.describe_asset(text, "model")
    assert described["sha256"] == hashlib.sha256(text.read_bytes()).hexdigest()


def test_profile_options_reproduce_the_provider_manifest(tmp_path: Path) -> None:
    from capicove.model.providers.embeddinggemma import manifest_digest

    module = script()
    receipt = {
        "assets": {
            "model": {"name": "m.gguf", "sha256": "a" * 64},
            "mmproj": {"name": "p.gguf", "sha256": "b" * 64},
        },
        "llama_cpp": {
            "server": "/opt/llama-server",
            "server_sha256": "c" * 64,
            "revision": "9" * 40,
            "release": "b11765",
        },
        "upstream": {
            "gguf_repository": "ggml-org/embeddinggemma-2-GGUF",
            "gguf_revision": "1" * 40,
        },
    }
    options = module.profile_options(receipt, threads=4)
    assert options["llama_server"] == "/opt/llama-server"
    assert options["backend_version"] == "llama.cpp b11765 " + "9" * 40
    assert module.profile_manifest(receipt, threads=4) == manifest_digest(options)
