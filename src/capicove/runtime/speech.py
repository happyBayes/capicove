"""Explicit local speech startup composition, shared by CLI and MCP."""

from __future__ import annotations

import json
import os
import stat
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Any

from capicove.model.configuration import ModelCapability, ModelProfile, load_configuration
from capicove.model.factory import ModelFactory
from capicove.processing.audio.synthesis import SpeechRequest, SpeechService, open_regular
from capicove.tools.audio_synthesis import SpeechToolHandler


MAX_SPEECH_CONFIG_BYTES = 65_536


def read_speech_configuration(path: Path) -> dict[str, Any]:
    """Read bounded operator-owned configuration, including refreshed grants."""
    with open_regular(path) as source:
        info = os.fstat(source.fileno())
        if info.st_size > MAX_SPEECH_CONFIG_BYTES or (
            os.name != "nt" and (info.st_uid != os.getuid() or stat.S_IMODE(info.st_mode) & 0o022)
        ):
            raise ValueError(
                "speech configuration must be operator-owned and not writable by others"
            )
        value = json.load(source)
    if not isinstance(value, dict):
        raise ValueError("speech configuration must be an object")
    return value


class SpeechRuntime:
    """Own one configured factory and immutable startup execution authority."""

    def __init__(self, path: Path, *, signing_key_file: Path | None = None) -> None:
        self.path = path
        config = read_speech_configuration(path)
        required = {
            "schema_version",
            "model_config",
            "binding",
            "output_root",
            "read_roots",
            "authorizations",
            "signing_key_env",
        }
        if (
            set(config) != required
            or type(config["schema_version"]) is not int
            or config["schema_version"] != 1
        ):
            raise ValueError("invalid speech runtime configuration")
        self.output_root = Path(config["output_root"])
        self.read_roots = tuple(Path(root) for root in config["read_roots"])
        for root in (self.output_root, *self.read_roots):
            if not root.is_absolute() or root.resolve() != root:
                raise ValueError("explicit unlinked speech roots required")
        if path.is_relative_to(self.output_root) or any(
            path.is_relative_to(root) for root in self.read_roots
        ):
            raise ValueError("speech configuration must be outside input/output roots")
        key_name = config["signing_key_env"]
        if not isinstance(key_name, str) or not key_name.startswith("CAPICOVE_"):
            raise ValueError("speech signing key must use CAPICOVE_ environment reference")
        self.key = bytes.fromhex(
            os.environ.get(key_name, "")
            if signing_key_file is None
            else _read_signing_key(signing_key_file)
        )
        if len(self.key) < 32:
            raise ValueError("speech provenance key must contain at least 32 bytes")
        self.binding = config["binding"]
        self.factory = ModelFactory(load_configuration(Path(config["model_config"])))
        self.factory.profile(self.binding, ModelCapability.AUDIO_SYNTHESIS)
        admission = Lock()
        self.synthesize = SpeechToolHandler(self.generate, clone=False, admission=admission)
        self.clone = SpeechToolHandler(self.generate, clone=True, admission=admission)

    def authorizations(self) -> dict[str, str]:
        """Reread only trusted authorization grants to observe revocations."""
        data = read_speech_configuration(self.path)
        grants = data["authorizations"]
        if not isinstance(grants, dict) or any(
            not isinstance(k, str) or not isinstance(v, str) for k, v in grants.items()
        ):
            raise ValueError("invalid voice authorizations")
        return grants

    def generate(self, request: SpeechRequest, deadline: datetime) -> dict[str, object]:
        """Lease one model and delegate policy to the audio application."""
        profile = self.factory.profile(self.binding, ModelCapability.AUDIO_SYNTHESIS)
        assert isinstance(profile, ModelProfile)
        deadline = min(
            deadline,
            datetime.now(UTC) + timedelta(seconds=int(profile.options.get("timeout_seconds", 240))),
        )
        with self.factory.speech(self.binding) as provider:
            return SpeechService(
                provider, self.output_root, self.read_roots, self.authorizations, self.key
            ).generate(request, deadline=deadline)

    def capabilities(self) -> dict[str, object]:
        """Verify provisioned assets and describe features without native inference."""
        with self.factory.speech(self.binding) as provider:
            return {
                "available": True,
                "model_id": provider.model_id,
                "languages": list(provider.features.languages),
                "reference_audio": provider.features.reference_audio,
                "reference_text": provider.features.reference_text,
                "preset_voice": provider.features.preset_voice,
                "streaming": False,
            }

    def close(self) -> None:
        """Release only idle model instances."""
        self.factory.close()


def _read_signing_key(path: Path) -> str:
    """Read a bounded, owner-private key selected explicitly by a local caller."""
    if not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("speech signing key must be a regular file")
    if os.name == "nt":
        from capicove._windows_files import read_windows_secure_file

        return read_windows_secure_file(path, max_bytes=256, private_acl=True).decode("ascii")
    with open_regular(path) as source:
        info = os.fstat(source.fileno())
        if info.st_uid != os.getuid() or info.st_mode & 0o077 or info.st_size > 256:
            raise ValueError("speech signing key file must be owner-private")
        return source.read(256).decode("ascii")
