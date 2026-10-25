"""Tests for immutable Hugging Face image-model manifests."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.model.manifest import load_manifest, load_packaged_manifest


VALID_MANIFEST = """
manifest_version = 1

[[models]]
key = "custom-aesthetics"
repository = "example/aesthetics"
revision = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
license = "apache-2.0"
license_url = "https://example.invalid/license"
capability = "image_aesthetics_filter"
adapter = "huggingface-aesthetics"
architecture = "ExampleForImageClassification"
preprocess_revision = "example-v1"
default = false
restricted = false

[models.settings]
target_label = "aesthetic"
"""

VALID_ONNX_MANIFEST = VALID_MANIFEST.replace(
    'adapter = "huggingface-aesthetics"',
    'adapter = "onnx-image-embedding"',
)


def test_packaged_manifest_pins_expected_models_and_restricts_agpl() -> None:
    manifest = load_packaged_manifest()
    models = {model.key: model for model in manifest.models}

    assert manifest.manifest_version == 1
    assert len(manifest.source_sha256) == 64
    assert set(models) == {
        "aesthetics-cafe",
        "blip-caption-base",
        "blip-itm-coco",
        "clip-vit-b32",
        "nsfw-falconsai",
        "siglip2-base-patch16-224",
        "smolvlm-256m-instruct",
        "watermark-siglip2",
    }
    assert models["aesthetics-cafe"].revision == (
        "48a343764f786abc1ca8aaddfcd60a688a70da9b"  # pragma: allowlist secret
    )
    assert models["aesthetics-cafe"].restricted is True
    assert models["aesthetics-cafe"].default is False
    assert dict(models["aesthetics-cafe"].settings) == {"target_label": "aesthetic"}
    assert models["clip-vit-b32"].revision == (
        "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"  # pragma: allowlist secret
    )
    assert models["siglip2-base-patch16-224"].revision == (
        "75de2d55ec2d0b4efc50b3e9ad70dba96a7b2fa2"  # pragma: allowlist secret
    )
    assert models["siglip2-base-patch16-224"].adapter == ("huggingface-siglip2-retrieval")


def test_custom_manifest_allows_alternative_aesthetics_model(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(VALID_MANIFEST, encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.models[0].key == "custom-aesthetics"
    assert manifest.models[0].adapter == "huggingface-aesthetics"
    assert dict(manifest.models[0].settings) == {"target_label": "aesthetic"}


def test_custom_manifest_allows_offline_onnx_image_embedding_adapter(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(VALID_ONNX_MANIFEST, encoding="utf-8")

    manifest = load_manifest(path)

    assert manifest.models[0].adapter == "onnx-image-embedding"


def test_custom_manifest_rejects_moving_revision(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(VALID_MANIFEST.replace("a" * 40, "main"), encoding="utf-8")

    with pytest.raises(ValueError, match="40-character"):
        load_manifest(path)


def test_custom_manifest_rejects_unknown_fields(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    path.write_text(
        VALID_MANIFEST.replace("restricted = false", "restricted = false\ntypo = true"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unknown model fields: typo"):
        load_manifest(path)


def test_custom_manifest_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "models.toml"
    duplicate = VALID_MANIFEST + VALID_MANIFEST.split("[[models]]", maxsplit=1)[1].join(
        ("\n[[models]]", "")
    )
    path.write_text(duplicate, encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate model key"):
        load_manifest(path)
