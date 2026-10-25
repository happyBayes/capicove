"""Canonical local evidence for downloaded image models."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from capicove.__about__ import __version__
from capicove.model.manifest import ImageModelSpec


RECEIPT_NAME = "capicove-model-receipt.json"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class VerifiedModelInstallation:
    """Bind runtime model lineage to a trusted lock entry and file manifest."""

    path: Path
    model: ImageModelSpec
    source_manifest_sha256: str
    model_manifest_sha256: str
    receipt_sha256: str | None = None
    directory_identity: tuple[int, int] | None = None

    @property
    def repository(self) -> str:
        """Return the repository identity from the trusted lock entry."""
        return self.model.repository

    @property
    def revision(self) -> str:
        """Return the immutable revision from the trusted lock entry."""
        return self.model.revision

    @property
    def preprocess_revision(self) -> str:
        """Return the preprocessing contract from the trusted lock entry."""
        return self.model.preprocess_revision

    def verify(self) -> None:
        """Revalidate identity and exact file bytes immediately before model loading."""
        _require_directory_identity(self.path, self.directory_identity)
        receipt = read_receipt(self.path, expected_sha256=self.receipt_sha256)
        if not receipt_matches(
            receipt,
            self.model,
            manifest_sha256=self.source_manifest_sha256,
        ):
            raise ValueError("installed model receipt does not match its trusted lock entry")
        if receipt.get("model_manifest_sha256") != self.model_manifest_sha256:
            raise ValueError(
                "installed model receipt does not match the trusted model manifest digest"
            )
        verify_receipt_files(self.path, receipt)
        if self.receipt_sha256 is not None:
            final_receipt = read_receipt(self.path, expected_sha256=self.receipt_sha256)
            if final_receipt != receipt:
                raise ValueError("installed model receipt changed while being verified")
        _require_directory_identity(self.path, self.directory_identity)


def verify_model_installation(
    directory: Path,
    model: ImageModelSpec,
    *,
    manifest_sha256: str,
    expected_model_manifest_sha256: str,
    expected_receipt_sha256: str | None = None,
) -> VerifiedModelInstallation:
    """Create runtime evidence anchored by trusted lock and file-manifest digests."""
    if _SHA256_PATTERN.fullmatch(manifest_sha256) is None:
        raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")
    if _SHA256_PATTERN.fullmatch(expected_model_manifest_sha256) is None:
        raise ValueError("expected model manifest must be a lowercase SHA-256 digest")
    if (
        expected_receipt_sha256 is not None
        and _SHA256_PATTERN.fullmatch(expected_receipt_sha256) is None
    ):
        raise ValueError("expected receipt must be a lowercase SHA-256 digest")
    path = Path(directory).expanduser().resolve()
    installation = VerifiedModelInstallation(
        path=path,
        model=model,
        source_manifest_sha256=manifest_sha256,
        model_manifest_sha256=expected_model_manifest_sha256,
        receipt_sha256=expected_receipt_sha256,
        directory_identity=_verified_directory_identity(path),
    )
    installation.verify()
    return installation


def build_receipt(
    model: ImageModelSpec,
    directory: Path,
    *,
    manifest_sha256: str,
    hf_version: str,
) -> dict[str, Any]:
    """Build canonical evidence for regular files below a model directory."""
    files = _file_records(directory)
    if not files:
        raise ValueError("downloaded model contains no regular files")
    encoded_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "receipt_version": 1,
        "model_key": model.key,
        "repository": model.repository,
        "revision": model.revision,
        "license": model.license,
        "license_url": model.license_url,
        "capability": model.capability,
        "adapter": model.adapter,
        "architecture": model.architecture,
        "preprocess_revision": model.preprocess_revision,
        "settings": dict(model.settings),
        "completed_at": datetime.now(UTC).isoformat(),
        "files": files,
        "model_manifest_sha256": hashlib.sha256(encoded_files).hexdigest(),
        "capicove_version": __version__,
        "hf_version": hf_version,
        "source_manifest_sha256": manifest_sha256,
    }


def verify_receipt_files(directory: Path, receipt: dict[str, Any]) -> None:
    """Reject modified, missing, extra, or symlinked installed model content."""
    files = _file_records(directory)
    encoded_files = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    digest = hashlib.sha256(encoded_files).hexdigest()
    if receipt.get("files") != files or receipt.get("model_manifest_sha256") != digest:
        raise ValueError("existing model contents do not match its receipt")


def _file_records(directory: Path) -> list[dict[str, str | int]]:
    files: list[dict[str, str | int]] = []
    for path in sorted(directory.rglob("*"), key=lambda item: item.as_posix()):
        if path.is_symlink():
            raise ValueError(f"downloaded model contains a symlink: {path.name}")
        if not path.is_file() or path.name == RECEIPT_NAME:
            continue
        relative = path.relative_to(directory).as_posix()
        files.append(
            {
                "path": relative,
                "sha256": _file_sha256(path),
                "size": path.stat().st_size,
            }
        )
    return files


def write_receipt(directory: Path, receipt: dict[str, Any]) -> Path:
    """Write canonical JSON only after model verification succeeds."""
    target = directory / RECEIPT_NAME
    payload = json.dumps(receipt, sort_keys=True, separators=(",", ":")) + "\n"
    target.write_text(payload, encoding="utf-8")
    return target


def read_receipt(
    directory: Path,
    *,
    expected_sha256: str | None = None,
) -> dict[str, Any]:
    """Read a completed receipt or reject an untrusted directory."""
    if expected_sha256 is not None and _SHA256_PATTERN.fullmatch(expected_sha256) is None:
        raise ValueError("expected receipt must be a lowercase SHA-256 digest")
    target = directory / RECEIPT_NAME
    try:
        payload = target.read_bytes()
        if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise ValueError("installed model receipt digest does not match trusted provisioning")
    except FileNotFoundError as error:
        raise ValueError("existing model directory lacks a valid Capicove model receipt") from error
    return parse_receipt(payload)


def parse_receipt(payload: bytes) -> dict[str, Any]:
    """Parse exact receipt bytes already read by a trusted caller."""
    if not isinstance(payload, bytes):
        raise TypeError("payload must be bytes")
    try:
        raw = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("existing model directory lacks a valid Capicove model receipt") from error
    if not isinstance(raw, dict) or raw.get("receipt_version") != 1:
        raise ValueError("existing model directory lacks a valid Capicove model receipt")
    return cast(dict[str, Any], raw)


def receipt_matches(
    receipt: dict[str, Any],
    model: ImageModelSpec,
    *,
    manifest_sha256: str,
) -> bool:
    """Return whether a receipt identifies the exact normalized lock entry."""
    expected = {
        "model_key": model.key,
        "repository": model.repository,
        "revision": model.revision,
        "license": model.license,
        "license_url": model.license_url,
        "capability": model.capability,
        "adapter": model.adapter,
        "architecture": model.architecture,
        "preprocess_revision": model.preprocess_revision,
        "settings": dict(model.settings),
        "source_manifest_sha256": manifest_sha256,
    }
    return all(receipt.get(name) == value for name, value in expected.items())


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verified_directory_identity(path: Path) -> tuple[int, int]:
    try:
        metadata = path.stat(follow_symlinks=False)
    except OSError as error:
        raise ValueError("installed model directory is unavailable") from error
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError("installed model path must be a non-symlink directory")
    return metadata.st_dev, metadata.st_ino


def _require_directory_identity(
    path: Path,
    expected: tuple[int, int] | None,
) -> None:
    observed = _verified_directory_identity(path)
    if expected is not None and observed != expected:
        raise ValueError("installed model directory identity changed after verification")
