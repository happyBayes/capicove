"""Speech publication, consent, and failure contracts with authored PCM audio."""

import hashlib
import json
import os
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.audio.synthesis import (
    SpeechRequest,
    SpeechService,
    SynthesisError,
    file_digest,
    verify_provenance,
)
from capicove.runtime.speech import SpeechRuntime
from tests.speech_support import AuthoredSpeechProvider, speech_fixture, write_wave


def deadline() -> datetime:
    return datetime.now(UTC) + timedelta(seconds=30)


def test_plain_speech_publishes_authenticated_artifact_without_raw_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    runtime = SpeechRuntime(fixture.config_path)
    try:
        result = runtime.generate(SpeechRequest("Private authored text"), deadline())
    finally:
        runtime.close()
    wav = Path(str(result["path"]))
    receipt = Path(str(result["provenance_path"]))
    record = json.loads(receipt.read_text())
    assert wav.parent == receipt.parent
    assert verify_provenance(record, fixture.key)
    assert record["output_sha256"] == file_digest(wav)
    assert "Private authored text" not in receipt.read_text()
    assert result["warnings"] == ["UNCONDITIONED_VOICE"]
    record["language"] = "tampered"
    assert not verify_provenance(record, fixture.key)
    assert not list(fixture.output_root.glob(".tts-*"))


@pytest.mark.parametrize("text", ["", "  ", "a" * 201, "private\0text"])
def test_invalid_text_never_invokes_provider(tmp_path: Path, text: str) -> None:
    provider = AuthoredSpeechProvider()
    service = SpeechService(provider, tmp_path / "output", (), dict, b"x" * 32)
    with pytest.raises(SynthesisError, match="invalid_input"):
        service.generate(SpeechRequest(text), deadline=deadline())
    assert provider.requests == []
    assert not service.output_root.exists()


@pytest.mark.parametrize(
    "changes",
    [
        {"language": "unsupported"},
        {"reference_text": "unsupported"},
        {"voice": "unknown"},
    ],
)
def test_unsupported_features_are_not_silently_ignored(
    tmp_path: Path,
    changes: dict[str, str],
) -> None:
    provider = AuthoredSpeechProvider()
    service = SpeechService(provider, tmp_path / "output", (), dict, b"x" * 32)
    request = replace(SpeechRequest("Hello"), **changes)  # type: ignore[arg-type]
    with pytest.raises(SynthesisError, match="unsupported_feature"):
        service.generate(request, deadline=deadline())
    assert provider.requests == []


def test_expired_deadline_never_creates_output(tmp_path: Path) -> None:
    provider = AuthoredSpeechProvider()
    service = SpeechService(provider, tmp_path / "output", (), dict, b"x" * 32)
    with pytest.raises(SynthesisError, match="timeout"):
        service.generate(SpeechRequest("Hello"), deadline=datetime.now(UTC) - timedelta(seconds=1))
    assert provider.requests == []
    assert not service.output_root.exists()


@pytest.mark.parametrize("mode", ["empty", "silent", "malformed"])
def test_invalid_generated_audio_leaves_no_published_or_partial_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    fixture.provider.output_mode = mode
    runtime = SpeechRuntime(fixture.config_path)
    try:
        with pytest.raises(SynthesisError, match="output_invalid"):
            runtime.generate(SpeechRequest("Hello"), deadline())
    finally:
        runtime.close()
    assert list(fixture.output_root.iterdir()) == []


def test_clone_stages_authorized_reference_and_preserves_original(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    before = fixture.reference.read_bytes()
    runtime = SpeechRuntime(fixture.config_path)
    try:
        result = runtime.generate(
            SpeechRequest("Hello", reference=fixture.reference, authorization="authored-grant"),
            deadline(),
        )
    finally:
        runtime.close()
    staged = fixture.provider.requests[0].reference
    assert staged != fixture.reference
    assert staged is not None and not staged.exists()
    assert fixture.reference.read_bytes() == before
    record = json.loads(Path(str(result["provenance_path"])).read_text())
    assert record["reference_sha256"] == hashlib.sha256(before).hexdigest()
    assert result["warnings"] == []


@pytest.mark.parametrize("grant", [None, "caller-self-asserted-consent"])
def test_missing_or_forged_authorization_is_rejected_before_generation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    grant: str | None,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)
    runtime = SpeechRuntime(fixture.config_path)
    try:
        with pytest.raises(SynthesisError, match="voice_unauthorized"):
            runtime.generate(
                SpeechRequest("Hello", reference=fixture.reference, authorization=grant), deadline()
            )
    finally:
        runtime.close()
    assert fixture.provider.requests == []
    assert list(fixture.output_root.iterdir()) == []


def test_revocation_during_generation_prevents_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = speech_fixture(tmp_path, monkeypatch)

    def revoke() -> None:
        config = json.loads(fixture.config_path.read_text())
        config["authorizations"] = {}
        fixture.config_path.write_text(json.dumps(config))

    fixture.provider.after_generation = revoke
    runtime = SpeechRuntime(fixture.config_path)
    try:
        with pytest.raises(SynthesisError, match="voice_unauthorized"):
            runtime.generate(
                SpeechRequest("Hello", reference=fixture.reference, authorization="authored-grant"),
                deadline(),
            )
    finally:
        runtime.close()
    assert list(fixture.output_root.iterdir()) == []
    assert fixture.reference.exists()


@pytest.mark.parametrize("seconds", [1, 31])
def test_reference_duration_is_bounded_before_provider_call(tmp_path: Path, seconds: int) -> None:
    reference = tmp_path.resolve() / "reference.wav"
    write_wave(reference, seconds=seconds)
    provider = AuthoredSpeechProvider()
    grants = {file_digest(reference): "authored-grant"}
    service = SpeechService(provider, tmp_path / "output", (tmp_path,), lambda: grants, b"x" * 32)
    with pytest.raises(SynthesisError, match="invalid_reference"):
        service.generate(
            SpeechRequest("Hello", reference=reference, authorization="authored-grant"),
            deadline=deadline(),
        )
    assert provider.requests == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX link policy; native Windows has separate gates")
@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_linked_reference_is_rejected(tmp_path: Path, kind: str) -> None:
    original = tmp_path.resolve() / "original.wav"
    write_wave(original, seconds=3)
    linked = tmp_path / "linked.wav"
    if kind == "symlink":
        linked.symlink_to(original)
    else:
        os.link(original, linked)
    provider = AuthoredSpeechProvider()
    service = SpeechService(provider, tmp_path / "output", (tmp_path,), dict, b"x" * 32)
    with pytest.raises(SynthesisError, match="unsafe_path"):
        service.generate(
            SpeechRequest("Hello", reference=linked, authorization="authored-grant"),
            deadline=deadline(),
        )
    assert provider.requests == []
