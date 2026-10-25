"""Authored verified model installations for adapter unit tests."""

from __future__ import annotations

from pathlib import Path

from capicove.model.manifest import ImageModelSpec
from capicove.model.receipt import (
    VerifiedModelInstallation,
    build_receipt,
    verify_model_installation,
    write_receipt,
)


def verified_model_installation(
    directory: Path,
    *,
    adapter: str,
    preprocess_revision: str,
    capability: str = "test_image_capability",
) -> VerifiedModelInstallation:
    """Create deterministic trusted model evidence without downloading weights."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text("{}", encoding="utf-8")
    model = ImageModelSpec(
        key=f"test-{adapter.replace('_', '-').replace('huggingface-', '')}",
        repository=f"test-owner/{adapter}",
        revision="a" * 40,
        license="apache-2.0",
        license_url="https://example.test/license",
        capability=capability,
        adapter=adapter,
        architecture="TestArchitecture",
        preprocess_revision=preprocess_revision,
        default=False,
        restricted=False,
    )
    manifest_sha256 = "b" * 64
    receipt = build_receipt(
        model,
        directory,
        manifest_sha256=manifest_sha256,
        hf_version="hf-test",
    )
    write_receipt(directory, receipt)
    return verify_model_installation(
        directory,
        model,
        manifest_sha256=manifest_sha256,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )
