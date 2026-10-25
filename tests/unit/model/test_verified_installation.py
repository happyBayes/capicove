"""Tests for trusted runtime model-installation evidence."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.model.manifest import ImageModelSpec
from capicove.model.receipt import (
    build_receipt,
    verify_model_installation,
    write_receipt,
)


def _model() -> ImageModelSpec:
    return ImageModelSpec(
        key="test-model",
        repository="test-owner/test-model",
        revision="a" * 40,
        license="apache-2.0",
        license_url="https://example.test/license",
        capability="image_aesthetics_filter",
        adapter="huggingface-aesthetics",
        architecture="TestArchitecture",
        preprocess_revision="test-preprocess-v1",
        default=False,
        restricted=False,
    )


def test_verified_installation_requires_trusted_file_manifest_digest(tmp_path: Path) -> None:
    model = _model()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    receipt = build_receipt(model, tmp_path, manifest_sha256="b" * 64, hf_version="hf-test")
    write_receipt(tmp_path, receipt)

    with pytest.raises(ValueError, match="trusted model manifest digest"):
        verify_model_installation(
            tmp_path,
            model,
            manifest_sha256="b" * 64,
            expected_model_manifest_sha256="c" * 64,
        )


def test_verified_installation_derives_lineage_from_trusted_model_spec(tmp_path: Path) -> None:
    model = _model()
    (tmp_path / "config.json").write_text("{}", encoding="utf-8")
    receipt = build_receipt(model, tmp_path, manifest_sha256="b" * 64, hf_version="hf-test")
    write_receipt(tmp_path, receipt)

    installation = verify_model_installation(
        tmp_path,
        model,
        manifest_sha256="b" * 64,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )

    assert installation.repository == model.repository
    assert installation.revision == model.revision
    assert installation.preprocess_revision == model.preprocess_revision
    assert installation.path == tmp_path.resolve()
    assert installation.receipt_sha256 is None
    installation.verify()
