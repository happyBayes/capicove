"""Dependency-free speech contracts and bounded local artifact publication."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import stat
import tempfile
import uuid
import wave
from collections.abc import Callable, Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import BinaryIO, Protocol


class SynthesisError(ValueError):
    """Stable speech-domain failure without private input details."""


@dataclass(frozen=True)
class SpeechFeatures:
    """Declare only features implemented by a selected provider."""

    languages: tuple[str, ...]
    reference_audio: bool = False
    reference_text: bool = False
    preset_voice: bool = False


@dataclass(frozen=True)
class SpeechRequest:
    """Short speech request independent of model SDKs and transports."""

    text: str
    language: str = "zh"
    reference: Path | None = None
    authorization: str | None = None
    reference_text: str | None = None
    voice: str | None = None


class SpeechProvider(Protocol):
    """Write a completed PCM WAV into a service-owned temporary location."""

    model_id: str
    features: SpeechFeatures
    lineage: Mapping[str, object]

    def synthesize(self, request: SpeechRequest, output: Path, *, deadline: datetime) -> None: ...

    def close(self) -> None: ...


def file_digest(path: Path) -> str:
    """Read one regular unlinked file through a pinned descriptor."""
    with open_regular(path) as source:
        digest = hashlib.sha256()
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
        return digest.hexdigest()


def open_regular(path: Path) -> AbstractContextManager[BinaryIO]:
    """Reject linked ancestors and Windows reparse points before opening."""
    if os.name == "nt":
        from capicove._windows_files import open_windows_secure_stream

        return open_windows_secure_stream(path)
    if not path.is_absolute() or ".." in path.parts:
        raise SynthesisError("unsafe_path")
    for part in (path, *path.parents):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise SynthesisError("unsafe_path")
    directory = os.open(path.anchor, os.O_RDONLY | os.O_DIRECTORY)
    try:
        for component in path.parts[1:-1]:
            child = os.open(
                component, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory
            )
            os.close(directory)
            directory = child
        descriptor = os.open(path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=directory)
    finally:
        os.close(directory)
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        os.close(descriptor)
        raise SynthesisError("unsafe_path")
    return os.fdopen(descriptor, "rb")


def canonical(value: Mapping[str, object]) -> bytes:
    """Canonical bytes used for local HMAC provenance verification."""
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def verify_provenance(record: Mapping[str, object], key: bytes) -> bool:
    """Authenticate a local provenance sidecar; this is not a public-key signature."""
    body = dict(record)
    signature = body.pop("signature", None)
    return isinstance(signature, str) and hmac.compare_digest(
        signature, hmac.digest(key, canonical(body), "sha256").hex()
    )


class SpeechService:
    """Validate one request and publish a generated artifact with provenance.

    Roots and authorization records come from trusted startup configuration.
    A new private directory contains WAV and receipt, published by directory rename.
    """

    def __init__(
        self,
        provider: SpeechProvider,
        output_root: Path,
        read_roots: tuple[Path, ...],
        authorizations: Callable[[], Mapping[str, str]],
        signing_key: bytes,
    ) -> None:
        if len(signing_key) < 32:
            raise ValueError("provenance key must have at least 32 bytes")
        self.provider = provider
        self.output_root = output_root
        self.read_roots = read_roots
        self.authorizations = authorizations
        self.signing_key = signing_key

    def generate(self, request: SpeechRequest, *, deadline: datetime) -> dict[str, object]:
        """Generate at most 200 characters; reference input is staged privately."""
        if deadline.tzinfo is None or deadline <= datetime.now(UTC):
            raise SynthesisError("timeout")
        if (
            not isinstance(request.text, str)
            or not request.text.strip()
            or len(request.text) > 200
            or "\0" in request.text
        ):
            raise SynthesisError("invalid_input")
        features = self.provider.features
        if (
            request.language not in features.languages
            or (request.reference and not features.reference_audio)
            or (request.reference_text is not None and not features.reference_text)
            or (request.voice is not None and not features.preset_voice)
        ):
            raise SynthesisError("unsupported_feature")
        if (
            (request.reference and request.voice)
            or (request.authorization and not request.reference)
            or (request.reference_text and not request.reference)
        ):
            raise SynthesisError("invalid_input")
        self.output_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if os.name == "nt":
            from capicove._windows_files import validate_windows_path

            validate_windows_path(self.output_root, directory=True, private_acl=True)
        elif self.output_root.stat().st_mode & 0o077:
            raise SynthesisError("unsafe_path")
        if self.output_root.resolve() != self.output_root:
            raise SynthesisError("unsafe_path")
        reference_hash = None
        with tempfile.TemporaryDirectory(dir=self.output_root, prefix=".tts-") as temp:
            directory = Path(temp)
            staged = request
            if request.reference is not None:
                source = request.reference
                if not any(source.is_relative_to(root) for root in self.read_roots):
                    raise SynthesisError("voice_unauthorized")
                with open_regular(source) as stream:
                    if os.fstat(stream.fileno()).st_size > 16 * 1024 * 1024:
                        raise SynthesisError("invalid_reference")
                    data = stream.read(16 * 1024 * 1024 + 1)
                reference_hash = hashlib.sha256(data).hexdigest()
                if (
                    not request.authorization
                    or self.authorizations().get(reference_hash) != request.authorization
                ):
                    raise SynthesisError("voice_unauthorized")
                local = directory / "reference.wav"
                local.write_bytes(data)
                try:
                    with wave.open(str(local), "rb") as audio:
                        duration = audio.getnframes() / audio.getframerate()
                        if (
                            audio.getnchannels() != 1
                            or audio.getsampwidth() != 2
                            or not 3 <= duration <= 30
                        ):
                            raise SynthesisError("invalid_reference")
                except (wave.Error, EOFError) as error:
                    raise SynthesisError("invalid_reference") from error
                staged = SpeechRequest(
                    request.text,
                    request.language,
                    local,
                    request.authorization,
                    request.reference_text,
                    request.voice,
                )
            output = directory / "speech.wav"
            self.provider.synthesize(staged, output, deadline=deadline)
            if deadline <= datetime.now(UTC):
                raise SynthesisError("timeout")
            if (
                reference_hash
                and self.authorizations().get(reference_hash) != request.authorization
            ):
                raise SynthesisError("voice_unauthorized")
            try:
                with open_regular(output) as stream:
                    if os.fstat(stream.fileno()).st_size > 16 * 1024 * 1024:
                        raise SynthesisError("output_invalid")
                    with wave.open(stream, "rb") as audio:
                        duration = audio.getnframes() / audio.getframerate()
                        sample_rate = audio.getframerate()
                        if (
                            audio.getnchannels() != 1
                            or audio.getsampwidth() != 2
                            or not 0 < duration <= 60
                            or audio.getframerate() > 96000
                            or not (frames := audio.readframes(audio.getnframes()))
                            or len(frames) != audio.getnframes() * 2
                            or not any(frames)
                        ):
                            raise SynthesisError("output_invalid")
            except (OSError, wave.Error, EOFError) as error:
                raise SynthesisError("output_invalid") from error
            identifier = uuid.uuid4().hex
            record: dict[str, object] = {
                "schema_version": 1,
                "request_id": identifier,
                "generated_at": datetime.now(UTC).isoformat(),
                "model": dict(self.provider.lineage),
                "text_sha256": hashlib.sha256(request.text.encode()).hexdigest(),
                "reference_text_sha256": hashlib.sha256(request.reference_text.encode()).hexdigest()
                if request.reference_text is not None
                else None,
                "reference_sha256": reference_hash,
                "voice_id": reference_hash if reference_hash else request.voice,
                "authorization_reference": request.authorization,
                "output_sha256": file_digest(output),
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "language": request.language,
                "signature_algorithm": "hmac-sha256",
            }
            record["signature"] = hmac.digest(self.signing_key, canonical(record), "sha256").hex()
            (directory / "provenance.json").write_bytes(canonical(record))
            if staged.reference:
                staged.reference.unlink()
            destination = self.output_root / identifier
            directory.rename(destination)
            return {
                "path": str(destination / "speech.wav"),
                "provenance_path": str(destination / "provenance.json"),
                "duration_seconds": duration,
                "sample_rate": sample_rate,
                "channels": 1,
                "warnings": ["UNCONDITIONED_VOICE"]
                if not request.reference and not request.voice
                else [],
                "lineage": record,
            }
