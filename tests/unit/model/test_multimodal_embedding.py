"""EmbeddingGemma 2 llama.cpp provider: local-only load, payloads and output checks.

The recording fake server observes the process boundary; it cannot qualify real
inference, which ``tests/model/test_embeddinggemma_llamacpp.py`` covers separately.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.model.configuration import ModelCapability, ModelConfiguration, ModelProfile
from capicove.model.multimodal_embedding import (
    MultimodalEmbeddingError,
    PcmAudio,
    TextDocument,
    validate_vectors,
)
from capicove.model.providers.embeddinggemma import (
    PROVIDER_ID,
    create_embeddinggemma,
    manifest_digest,
    space_for,
    validate_embeddinggemma,
)


FAKE = Path(__file__).with_name("fake_llama_server.py")
pytestmark = pytest.mark.skipif(os.name != "posix", reason="requires Unix sockets")


def _deadline(seconds: float = 30) -> datetime:
    return datetime.now(UTC) + timedelta(seconds=seconds)


def _profile(tmp_path: Path, mode: str = "ok", **overrides: object) -> tuple[ModelProfile, Path]:
    root = tmp_path / "assets"
    root.mkdir(exist_ok=True)
    (root / "model.gguf").write_bytes(b"GGUF-model")
    (root / "mmproj.gguf").write_bytes(b"GGUF-mmproj")
    log = tmp_path / "server.log"
    server = tmp_path / "llama-server"
    server.write_text(
        f'#!/bin/sh\nexec "{sys.executable}" -I "{FAKE}" --fake-mode {mode} --fake-log "{log}" "$@"\n'
    )
    server.chmod(0o700)
    options: dict[str, object] = {
        "model": "model.gguf",
        "model_sha256": hashlib.sha256(b"GGUF-model").hexdigest(),
        "mmproj": "mmproj.gguf",
        "mmproj_sha256": hashlib.sha256(b"GGUF-mmproj").hexdigest(),
        "llama_server": str(server),
        "llama_server_sha256": hashlib.sha256(server.read_bytes()).hexdigest(),
        "backend_version": "llama.cpp b11765 9df0b73c707900b7277a46b5ae52ff7e7e567d13",
        "model_revision": "ggml-org/embeddinggemma-2-GGUF@58faf909",
        "startup_timeout_seconds": 20,
    }
    options.update(overrides)
    return (
        ModelProfile(
            "shared",
            PROVIDER_ID,
            ModelCapability.MULTIMODAL_EMBEDDING,
            root.resolve(),
            manifest_digest(options),
            "cpu",
            options,  # type: ignore[arg-type]
        ),
        log,
    )


def _requests(log: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in log.read_text().splitlines()]


def test_profile_validation_rejects_unpinned_or_implicit_settings(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path)
    validate_embeddinggemma(profile)
    base = dict(profile.options)
    for change in (
        {"llama_server": "llama-server"},
        {"model_sha256": "0" * 63},
        {"model": "../model.gguf"},
        {"mmproj": "mmproj.bin"},
        {"threads": 0},
        {"unexpected": True},
    ):
        options = {**base, **change}
        with pytest.raises(ValueError):
            validate_embeddinggemma(
                ModelProfile(
                    "shared",
                    PROVIDER_ID,
                    ModelCapability.MULTIMODAL_EMBEDDING,
                    profile.root,
                    manifest_digest(options)
                    if "model" in change or "mmproj" in change
                    else profile.manifest_sha256,
                    "cpu",
                    options,
                )
            )
    with pytest.raises(ValueError, match="manifest"):
        validate_embeddinggemma(
            ModelProfile(
                "shared",
                PROVIDER_ID,
                ModelCapability.MULTIMODAL_EMBEDDING,
                profile.root,
                "f" * 64,
                "cpu",
                base,
            )
        )
    with pytest.raises(ValueError):
        validate_embeddinggemma(
            ModelProfile(
                "shared",
                PROVIDER_ID,
                ModelCapability.MULTIMODAL_EMBEDDING,
                profile.root,
                profile.manifest_sha256,
                "cuda",
                base,
            )
        )


def test_tampered_asset_fails_before_any_process_starts(tmp_path: Path) -> None:
    profile, log = _profile(tmp_path)
    (profile.root / "mmproj.gguf").write_bytes(b"GGUF-tampered")
    with pytest.raises(ValueError, match="digest"):
        create_embeddinggemma(profile)
    assert not log.exists()


def test_server_is_offline_unix_socket_only_and_payloads_are_exact(tmp_path: Path) -> None:
    profile, log = _profile(tmp_path)
    encoder = create_embeddinggemma(profile)
    try:
        argv = _requests(log)[0]["argv"]
        host = argv[argv.index("--host") + 1]
        assert host.endswith(".sock") and "--offline" in argv and "--embedding" in argv
        assert "--port" not in argv
        assert encoder.space == space_for(profile)  # type: ignore[attr-defined]
        pcm = struct.pack("<4h", 1, -2, 3, -4)
        queries = encoder.encode_text_queries(["barking dog"], deadline=_deadline())  # type: ignore[attr-defined]
        documents = encoder.encode_text_documents(  # type: ignore[attr-defined]
            [TextDocument("notes"), TextDocument("body", title="Title")], deadline=_deadline()
        )
        audio = encoder.encode_audio([PcmAudio(pcm, 16_000)], deadline=_deadline())  # type: ignore[attr-defined]
        image = encoder.encode_images([b"\x89PNG image"], deadline=_deadline())  # type: ignore[attr-defined]
        assert len(queries) == 1 and len(documents) == 2 and len(audio) == 1 and len(image) == 1
        assert all(len(vector) == 768 for vector in queries + documents + audio + image)
        assert documents[0] != documents[1]
        sent = [entry for entry in _requests(log) if "items" in entry]
        assert sent[0]["items"] == [{"text": "task: search result | query: barking dog"}]
        assert sent[1]["items"] == [
            {"text": "title: none | text: notes"},
            {"text": "title: Title | text: body"},
        ]
        assert sent[2]["items"] == [
            {
                "prompt": "<marker>",
                "rate": 16_000,
                "channels": 1,
                "width": 2,
                "pcm_sha256": hashlib.sha256(pcm).hexdigest(),
            }
        ]
        assert sent[3]["items"][0]["prompt"] == "<marker>"
        assert {entry["normalize"] for entry in sent} == {2}
        pid = encoder.pid  # type: ignore[attr-defined]
        directory = Path(host).parent
    finally:
        encoder.close()  # type: ignore[attr-defined]
    assert not directory.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


@pytest.mark.parametrize("mode", ["nan", "dimension", "count", "zero", "unnormalized", "duplicate"])
def test_malformed_outputs_are_rejected_without_repair(tmp_path: Path, mode: str) -> None:
    profile, _ = _profile(tmp_path, mode)
    encoder = create_embeddinggemma(profile)
    try:
        with pytest.raises(MultimodalEmbeddingError) as error:
            encoder.encode_text_queries(["a", "b"], deadline=_deadline())  # type: ignore[attr-defined]
        assert error.value.code == "invalid_output"
    finally:
        encoder.close()  # type: ignore[attr-defined]


def test_deadline_and_server_failure_are_typed(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path, "slow")
    encoder = create_embeddinggemma(profile)
    try:
        with pytest.raises(MultimodalEmbeddingError) as error:
            encoder.encode_text_queries(["a"], deadline=_deadline(0.5))  # type: ignore[attr-defined]
        assert error.value.code == "deadline_exceeded"
        with pytest.raises(MultimodalEmbeddingError) as expired:
            encoder.encode_text_queries(["a"], deadline=datetime.now(UTC) - timedelta(seconds=1))  # type: ignore[attr-defined]
        assert expired.value.code == "deadline_exceeded"
    finally:
        encoder.close()  # type: ignore[attr-defined]
    failing = tmp_path / "failing"
    failing.mkdir()
    profile, _ = _profile(failing, "http500")
    encoder = create_embeddinggemma(profile)
    try:
        with pytest.raises(MultimodalEmbeddingError) as failure:
            encoder.encode_text_queries(["a"], deadline=_deadline())  # type: ignore[attr-defined]
        assert failure.value.code == "unavailable"
    finally:
        encoder.close()  # type: ignore[attr-defined]


def test_crashing_server_is_unavailable_and_leaves_no_directory(tmp_path: Path) -> None:
    profile, _ = _profile(tmp_path, "crash")
    before = set(Path(os.environ.get("TMPDIR", "/tmp")).glob("capicove-eg2-*"))
    with pytest.raises(MultimodalEmbeddingError) as error:
        create_embeddinggemma(profile)
    assert error.value.code == "unavailable"
    assert set(Path(os.environ.get("TMPDIR", "/tmp")).glob("capicove-eg2-*")) <= before


def test_batch_and_input_bounds_are_enforced_before_requests(tmp_path: Path) -> None:
    profile, log = _profile(tmp_path, max_batch=2)
    encoder = create_embeddinggemma(profile)
    try:
        calls: tuple[Callable[[], object], ...] = (
            lambda: encoder.encode_text_queries([], deadline=_deadline()),  # type: ignore[attr-defined]
            lambda: encoder.encode_text_queries(["a", "b", "c"], deadline=_deadline()),  # type: ignore[attr-defined]
            lambda: encoder.encode_text_queries([" "], deadline=_deadline()),  # type: ignore[attr-defined]
            lambda: encoder.encode_audio([b"pcm"], deadline=_deadline()),  # type: ignore[attr-defined]
            lambda: encoder.encode_text_queries(  # type: ignore[attr-defined]
                [f"x {encoder.media_marker}"],  # type: ignore[attr-defined]
                deadline=_deadline(),
            ),
            lambda: encoder.encode_text_documents(  # type: ignore[attr-defined]
                [TextDocument("t", title=encoder.media_marker)],  # type: ignore[attr-defined]
                deadline=_deadline(),
            ),
        )
        for call in calls:
            with pytest.raises(MultimodalEmbeddingError) as error:
                call()
            assert error.value.code == "invalid_input"
        assert not [entry for entry in _requests(log) if "items" in entry]
    finally:
        encoder.close()  # type: ignore[attr-defined]
    with pytest.raises(ValueError):
        PcmAudio(b"\x00", 16_000)
    with pytest.raises(ValueError):
        TextDocument("text", title=" ")


def test_factory_owns_provider_lifecycle(tmp_path: Path) -> None:
    from capicove.model.factory import ModelFactory

    profile, log = _profile(tmp_path)
    configuration = ModelConfiguration({"shared": profile}, {"unified": "shared"})
    factory = ModelFactory(configuration)
    with factory.shared_space_encoder("unified") as encoder:
        assert encoder.space.dimension == 768
        pid = encoder.pid  # type: ignore[attr-defined]
    factory.close()
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
    assert len([entry for entry in _requests(log) if "argv" in entry]) == 1


def test_validate_vectors_rejects_every_malformed_shape() -> None:
    profile = ModelProfile(
        "shared",
        PROVIDER_ID,
        ModelCapability.MULTIMODAL_EMBEDDING,
        Path("/"),
        "0" * 64,
        "cpu",
        {},
    )
    space = space_for(
        ModelProfile(
            profile.name,
            profile.provider,
            profile.capability,
            profile.root,
            profile.manifest_sha256,
            "cpu",
            {"model_revision": "r", "backend_version": "b"},
        )
    )
    unit = [1.0] + [0.0] * 767
    assert validate_vectors([unit], count=1, space=space) == (tuple(unit),)
    for raw in ([unit, unit], [[True] + [0.0] * 767], "x" * 768, [[1.0] * 768]):
        with pytest.raises(MultimodalEmbeddingError):
            validate_vectors(raw, count=1, space=space)
