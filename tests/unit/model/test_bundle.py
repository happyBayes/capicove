"""Tests for verified offline ONNX image-encoder bundles."""

from __future__ import annotations

import builtins
import hashlib
import json
import socket
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import ModuleType

import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.bundle import load_verified_image_encoder_bundle
from capicove.model.manifest import ImageModelSpec
from capicove.model.receipt import (
    VerifiedModelInstallation,
    build_receipt,
    verify_model_installation,
    write_receipt,
)


_SOURCE_MANIFEST_SHA256 = "a" * 64
_REVISION = "1" * 40


def _embedding_space(*, dimension: int = 3) -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="test-image-space-v1",
        provider_id="test-onnx-image-encoder",
        model_revision=_REVISION,
        preprocessing_version="test-preprocess-v1",
        dimension=dimension,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _space_json(*, dimension: int = 3) -> dict[str, object]:
    space = _embedding_space(dimension=dimension)
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": [QueryModality.IMAGE.value],
        "target_modalities": [TargetType.IMAGE.value],
        "schema_version": space.schema_version,
    }


def _model() -> ImageModelSpec:
    return ImageModelSpec(
        key="test-onnx-image-encoder",
        repository="owner/model",
        revision=_REVISION,
        license="apache-2.0",
        license_url="https://example.invalid/license",
        capability="image_embedding",
        adapter="onnx-image-embedding",
        architecture="SyntheticImageEncoder",
        preprocess_revision="test-preprocess-v1",
        default=False,
        restricted=False,
    )


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def _file_entry(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _valid_manifest(
    checksums_payload: bytes,
    model_payload: bytes,
    license_payload: bytes,
) -> dict[str, object]:
    return {
        "bundle_id": "test-image-encoder-v1",
        "repository": "owner/model",
        "revision": _REVISION,
        "license": "apache-2.0",
        "preprocessing": {
            "color_mode": "RGB",
            "apply_exif_orientation": True,
            "resize_strategy": "exact",
            "resize_size": [8, 8],
            "crop_strategy": "none",
            "crop_size": None,
            "interpolation": "bilinear",
            "value_range": [0.0, 1.0],
            "channel_mean": [0.5, 0.5, 0.5],
            "channel_std": [0.5, 0.5, 0.5],
            "tensor_layout": "NCHW",
            "tensor_dtype": "float32",
            "schema_version": 1,
        },
        "input": {
            "name": "pixel_values",
            "rank": 4,
            "shape": ["batch", 3, 8, 8],
            "dtype": "float32",
        },
        "output": {
            "name": "image_embedding",
            "rank": 2,
            "selection": "only",
            "pooling": "none",
            "dimension": 3,
            "dtype": "float32",
            "l2_normalized": True,
            "metric": "cosine",
            "vector_schema": "image-vector-v1",
        },
        "embedding_space": _space_json(),
        "artifacts": [
            {"role": "onnx-model", **_file_entry("vision_encoder.onnx", model_payload)},
            {"role": "checksums", **_file_entry("checksums.json", checksums_payload)},
            {"role": "license", **_file_entry("LICENSE", license_payload)},
        ],
        "runtime": {
            "version_specifier": ">=1.20,<2",
            "execution_provider": "CPUExecutionProvider",
            "graph_optimization": "all",
            "custom_operators_allowed": False,
            "provider_fallback_allowed": False,
        },
        "network_behavior": "offline",
        "schema_version": 1,
    }


def _write_verified_bundle_fixture(
    tmp_path: Path,
    *,
    mutate_manifest: Callable[[dict[str, object]], None] | None = None,
    mutate_checksums: Callable[[dict[str, object]], None] | None = None,
) -> VerifiedModelInstallation:
    root = tmp_path / "bundle"
    root.mkdir()
    model_payload = b"synthetic-onnx-model"
    license_payload = b"Apache License 2.0\n"
    (root / "vision_encoder.onnx").write_bytes(model_payload)
    (root / "LICENSE").write_bytes(license_payload)
    checksums: dict[str, object] = {
        "schema_version": 1,
        "files": [
            _file_entry("vision_encoder.onnx", model_payload),
            _file_entry("LICENSE", license_payload),
        ],
    }
    if mutate_checksums is not None:
        mutate_checksums(checksums)
    checksums_payload = _canonical_json(checksums)
    (root / "checksums.json").write_bytes(checksums_payload)
    manifest = _valid_manifest(checksums_payload, model_payload, license_payload)
    if mutate_manifest is not None:
        mutate_manifest(manifest)
    (root / "manifest.json").write_bytes(_canonical_json(manifest))
    receipt = build_receipt(
        _model(),
        root,
        manifest_sha256=_SOURCE_MANIFEST_SHA256,
        hf_version="not-applicable",
    )
    write_receipt(root, receipt)
    return verify_model_installation(
        root,
        _model(),
        manifest_sha256=_SOURCE_MANIFEST_SHA256,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )


def test_verified_bundle_binds_lock_receipt_assets_and_embedding_space(tmp_path: Path) -> None:
    installation = _write_verified_bundle_fixture(tmp_path)

    bundle = load_verified_image_encoder_bundle(installation)

    assert bundle.manifest.embedding_space == _embedding_space()
    assert bundle.manifest.network_behavior == "offline"
    assert bundle.manifest.input.name == "pixel_values"
    assert bundle.manifest.output.name == "image_embedding"
    assert bundle.artifact_path("onnx-model") == installation.path / "vision_encoder.onnx"
    bundle.verify()


@pytest.mark.parametrize(
    ("path", "role"),
    [
        ("/tmp/model.onnx", "onnx-model"),
        ("../model.onnx", "onnx-model"),
        ("file:model.onnx", "onnx-model"),
        ("sub\\model.onnx", "onnx-model"),
        ("vision_encoder.onnx", "license"),
    ],
)
def test_bundle_rejects_unsafe_paths_and_duplicate_roles(
    tmp_path: Path,
    path: str,
    role: str,
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        artifacts = manifest["artifacts"]
        assert isinstance(artifacts, list)
        artifact = artifacts[0]
        assert isinstance(artifact, dict)
        artifact["path"] = path
        artifact["role"] = role

    installation = _write_verified_bundle_fixture(tmp_path, mutate_manifest=mutate)

    with pytest.raises(ValueError):
        load_verified_image_encoder_bundle(installation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("network_behavior", "allow-network"),
        ("schema_version", 2),
        ("revision", "main"),
        ("repository", "other/model"),
        ("license", "other"),
    ],
)
def test_bundle_rejects_invalid_or_mismatched_top_level_identity(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    installation = _write_verified_bundle_fixture(
        tmp_path,
        mutate_manifest=lambda manifest: manifest.__setitem__(field, value),
    )

    with pytest.raises(ValueError):
        load_verified_image_encoder_bundle(installation)


def test_bundle_rejects_unknown_nested_fields_and_unsafe_runtime(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        runtime = manifest["runtime"]
        assert isinstance(runtime, dict)
        runtime["remote_code_allowed"] = True

    installation = _write_verified_bundle_fixture(tmp_path, mutate_manifest=mutate)

    with pytest.raises(ValueError, match="unknown"):
        load_verified_image_encoder_bundle(installation)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("execution_provider", "CUDAExecutionProvider"),
        ("version_specifier", ">=1"),
        ("graph_optimization", "unbounded"),
        ("custom_operators_allowed", True),
        ("provider_fallback_allowed", True),
    ],
)
def test_bundle_rejects_unqualified_runtime_policy(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        runtime = manifest["runtime"]
        assert isinstance(runtime, dict)
        runtime[field] = value

    installation = _write_verified_bundle_fixture(tmp_path, mutate_manifest=mutate)

    with pytest.raises(ValueError):
        load_verified_image_encoder_bundle(installation)


def test_bundle_rejects_incomplete_or_inconsistent_embedding_space(tmp_path: Path) -> None:
    def mutate(manifest: dict[str, object]) -> None:
        space = manifest["embedding_space"]
        assert isinstance(space, dict)
        space["dimension"] = 4

    installation = _write_verified_bundle_fixture(tmp_path, mutate_manifest=mutate)

    with pytest.raises(ValueError, match="dimension"):
        load_verified_image_encoder_bundle(installation)


def test_bundle_rejects_checksum_set_mismatch(tmp_path: Path) -> None:
    def mutate(checksums: dict[str, object]) -> None:
        files = checksums["files"]
        assert isinstance(files, list)
        files.pop()

    installation = _write_verified_bundle_fixture(tmp_path, mutate_checksums=mutate)

    with pytest.raises(ValueError, match="checksums"):
        load_verified_image_encoder_bundle(installation)


def test_bundle_rejects_asset_tamper_after_load(tmp_path: Path) -> None:
    installation = _write_verified_bundle_fixture(tmp_path)
    bundle = load_verified_image_encoder_bundle(installation)
    (installation.path / "vision_encoder.onnx").write_bytes(b"tampered")

    with pytest.raises(ValueError):
        bundle.verify()


def test_bundle_rejects_symlinked_assets(tmp_path: Path) -> None:
    installation = _write_verified_bundle_fixture(tmp_path)
    model = installation.path / "vision_encoder.onnx"
    replacement = tmp_path / "replacement.onnx"
    replacement.write_bytes(model.read_bytes())
    model.unlink()
    try:
        model.symlink_to(replacement)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="symlink"):
        load_verified_image_encoder_bundle(installation)


def test_bundle_verification_imports_no_model_runtime_and_opens_no_network(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installation = _write_verified_bundle_fixture(tmp_path)
    blocked = {"onnxruntime", "torch", "transformers", "faiss"}
    real_import = builtins.__import__

    def guarded_import(
        name: str,
        globals: Mapping[str, object] | None = None,
        locals: Mapping[str, object] | None = None,
        fromlist: Sequence[str] | None = (),
        level: int = 0,
    ) -> ModuleType:
        if name.split(".", maxsplit=1)[0] in blocked:
            raise AssertionError(f"unexpected optional import: {name}")
        return real_import(name, globals, locals, fromlist, level)

    def fail_socket(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("bundle verification must not open a network socket")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "socket", fail_socket)

    bundle = load_verified_image_encoder_bundle(installation)
    bundle.verify()
