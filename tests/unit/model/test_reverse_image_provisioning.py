"""Tests for content-addressed personal reverse-image model provisioning."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import shutil
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import ModuleType

import pytest

from capicove.model.manifest import ImageModelSpec, load_manifest
from capicove.model.provisioning import load_reverse_image_model_provisioning
from capicove.model.receipt import (
    VerifiedModelInstallation,
    build_receipt,
    verify_model_installation,
    write_receipt,
)


@dataclass(frozen=True, slots=True)
class _ProvisioningFixture:
    root: Path
    relative_path: PurePosixPath
    sha256: str


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _file_entry(path: str, payload: bytes) -> dict[str, object]:
    return {
        "path": path,
        "size_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _lock_payload() -> bytes:
    entries = []
    for lane, revision in (("instance", "1" * 40), ("concept", "2" * 40)):
        entries.append(
            f"""
[[models]]
key = "reverse-image-{lane}"
repository = "owner/reverse-image-{lane}"
revision = "{revision}"
license = "apache-2.0"
license_url = "https://example.invalid/{lane}/license"
capability = "image_embedding"
adapter = "onnx-image-embedding"
architecture = "Synthetic{lane.title()}Encoder"
preprocess_revision = "rgb-exact-2-v1"
default = false
restricted = false
"""
        )
    return ("manifest_version = 1\n" + "".join(entries)).encode()


def _bundle_manifest(
    lane: str,
    model: ImageModelSpec,
    *,
    model_payload: bytes,
    checksums_payload: bytes,
    license_payload: bytes,
) -> dict[str, object]:
    return {
        "artifacts": [
            {"role": "onnx-model", **_file_entry("vision.onnx", model_payload)},
            {"role": "checksums", **_file_entry("checksums.json", checksums_payload)},
            {"role": "license", **_file_entry("LICENSE", license_payload)},
        ],
        "bundle_id": f"reverse-image-{lane}-v1",
        "embedding_space": {
            "dimension": 3,
            "distance_metric": "cosine",
            "model_revision": model.revision,
            "normalization": "l2",
            "preprocessing_version": model.preprocess_revision,
            "provider_id": f"reverse-image-{lane}-onnx",
            "query_modalities": ["image"],
            "schema_version": 1,
            "space_id": f"reverse-image-{lane}-v1",
            "target_modalities": ["image"],
        },
        "input": {
            "dtype": "float32",
            "name": "pixel_values",
            "rank": 4,
            "shape": ["batch", 3, 2, 2],
        },
        "license": model.license,
        "network_behavior": "offline",
        "output": {
            "dimension": 3,
            "dtype": "float32",
            "l2_normalized": True,
            "metric": "cosine",
            "name": "image_embedding",
            "pooling": "none",
            "rank": 2,
            "selection": "only",
            "vector_schema": "image-vector-v1",
        },
        "preprocessing": {
            "apply_exif_orientation": True,
            "channel_mean": [0.0, 0.0, 0.0],
            "channel_std": [1.0, 1.0, 1.0],
            "color_mode": "RGB",
            "crop_size": None,
            "crop_strategy": "none",
            "interpolation": "nearest",
            "resize_size": [2, 2],
            "resize_strategy": "exact",
            "schema_version": 1,
            "tensor_dtype": "float32",
            "tensor_layout": "NCHW",
            "value_range": [0.0, 1.0],
        },
        "repository": model.repository,
        "revision": model.revision,
        "runtime": {
            "custom_operators_allowed": False,
            "execution_provider": "CPUExecutionProvider",
            "graph_optimization": "all",
            "provider_fallback_allowed": False,
            "version_specifier": ">=1.20,<2",
        },
        "schema_version": 1,
    }


def _write_fixture(tmp_path: Path) -> _ProvisioningFixture:
    root = (tmp_path / "models").resolve()
    lock_path = root / "locks" / "image-models.toml"
    lock_path.parent.mkdir(parents=True)
    lock_path.write_bytes(_lock_payload())
    lock = load_manifest(lock_path)
    entries: list[dict[str, object]] = []
    for lane, model in zip(("instance", "concept"), lock.models, strict=True):
        installation = root / "bundles" / lane
        installation.mkdir(parents=True)
        model_payload = f"synthetic ONNX graph for {lane}\n".encode()
        license_payload = b"Apache License 2.0\n"
        (installation / "vision.onnx").write_bytes(model_payload)
        (installation / "LICENSE").write_bytes(license_payload)
        checksums_payload = _canonical_json(
            {
                "files": [
                    _file_entry("vision.onnx", model_payload),
                    _file_entry("LICENSE", license_payload),
                ],
                "schema_version": 1,
            }
        )
        (installation / "checksums.json").write_bytes(checksums_payload)
        manifest_payload = _canonical_json(
            _bundle_manifest(
                lane,
                model,
                model_payload=model_payload,
                checksums_payload=checksums_payload,
                license_payload=license_payload,
            )
        )
        (installation / "manifest.json").write_bytes(manifest_payload)
        receipt = build_receipt(
            model,
            installation,
            manifest_sha256=lock.source_sha256,
            hf_version="not-applicable",
        )
        receipt_path = write_receipt(installation, receipt)
        receipt_payload = receipt_path.read_bytes()
        entries.append(
            {
                "bundle_manifest_sha256": hashlib.sha256(manifest_payload).hexdigest(),
                "installation_relative_path": f"bundles/{lane}",
                "lane": lane,
                "model_key": model.key,
                "receipt_relative_path": (f"bundles/{lane}/capicove-model-receipt.json"),
                "receipt_sha256": hashlib.sha256(receipt_payload).hexdigest(),
            }
        )
    document = {
        "entries": entries,
        "model_lock_relative_path": "locks/image-models.toml",
        "model_lock_sha256": lock.source_sha256,
        "provisioning_id": "personal-reverse-image-v1",
        "schema_version": 1,
    }
    relative_path = PurePosixPath("reverse-image-provisioning.json")
    payload = _canonical_json(document)
    (root / relative_path).write_bytes(payload)
    return _ProvisioningFixture(root, relative_path, hashlib.sha256(payload).hexdigest())


def _rewrite_provisioning(
    fixture: _ProvisioningFixture,
    mutate: Callable[[dict[str, object]], None],
) -> str:
    path = fixture.root.joinpath(*fixture.relative_path.parts)
    document = json.loads(path.read_bytes())
    assert isinstance(document, dict)
    mutate(document)
    payload = _canonical_json(document)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def _entries(document: dict[str, object]) -> list[dict[str, object]]:
    raw = document["entries"]
    assert isinstance(raw, list)
    return raw


def test_loads_exact_instance_and_concept_bundles_offline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
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
        raise AssertionError("model provisioning must remain offline")

    monkeypatch.setattr(builtins, "__import__", guarded_import)
    monkeypatch.setattr(socket, "socket", fail_socket)

    provisioned = load_reverse_image_model_provisioning(
        fixture.root,
        fixture.relative_path,
        expected_manifest_sha256=fixture.sha256,
    )

    assert provisioned.manifest.source_sha256 == fixture.sha256
    assert provisioned.instance_bundle.manifest.bundle_id == "reverse-image-instance-v1"
    assert provisioned.concept_bundle.manifest.bundle_id == "reverse-image-concept-v1"
    assert (
        provisioned.instance_bundle.installation.receipt_sha256
        == provisioned.manifest.entries[0].receipt_sha256
    )
    assert (
        provisioned.instance_bundle.manifest.embedding_space
        != provisioned.concept_bundle.manifest.embedding_space
    )


def test_rejects_unpinned_or_noncanonical_provisioning_manifest(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    with pytest.raises(ValueError, match="startup policy"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256="f" * 64,
        )

    path = fixture.root.joinpath(*fixture.relative_path.parts)
    document = json.loads(path.read_bytes())
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    changed_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="canonical JSON"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=changed_sha256,
        )


def test_rejects_model_lock_digest_mismatch(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    lock_path = fixture.root / "locks" / "image-models.toml"
    lock_path.write_bytes(lock_path.read_bytes() + b"\n")

    with pytest.raises(ValueError, match="model lock digest"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=fixture.sha256,
        )


@pytest.mark.parametrize(
    "mutation",
    [
        lambda document: document.__setitem__("model_lock_relative_path", "../lock.toml"),
        lambda document: _entries(document)[0].__setitem__(
            "installation_relative_path", "/tmp/instance"
        ),
        lambda document: _entries(document)[0].__setitem__(
            "receipt_relative_path", "file:receipt.json"
        ),
    ],
)
def test_rejects_unsafe_relative_paths(
    tmp_path: Path,
    mutation: Callable[[dict[str, object]], None],
) -> None:
    fixture = _write_fixture(tmp_path)
    expected = _rewrite_provisioning(fixture, mutation)

    with pytest.raises(ValueError, match="safe relative POSIX path"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=expected,
        )


@pytest.mark.parametrize("lanes", [("instance",), ("instance", "instance")])
def test_requires_exactly_one_instance_and_one_concept_entry(
    tmp_path: Path,
    lanes: tuple[str, ...],
) -> None:
    fixture = _write_fixture(tmp_path)

    def mutate(document: dict[str, object]) -> None:
        entries = _entries(document)
        document["entries"] = [{**entries[index], "lane": lane} for index, lane in enumerate(lanes)]

    expected = _rewrite_provisioning(fixture, mutate)

    with pytest.raises(ValueError, match="instance and concept"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=expected,
        )


def test_rejects_changed_model_lock_even_when_provisioning_digest_is_updated(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    lock_path = fixture.root / "locks" / "image-models.toml"
    lock_path.write_bytes(lock_path.read_bytes().replace(b"SyntheticInstanceEncoder", b"Other"))
    changed_lock_sha256 = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    expected = _rewrite_provisioning(
        fixture,
        lambda document: document.__setitem__("model_lock_sha256", changed_lock_sha256),
    )

    with pytest.raises(ValueError, match="trusted lock entry"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=expected,
        )


def test_rejects_receipt_that_is_readdressed_but_no_longer_matches_the_lock(
    tmp_path: Path,
) -> None:
    fixture = _write_fixture(tmp_path)
    receipt_path = fixture.root / "bundles" / "instance" / "capicove-model-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert isinstance(receipt, dict)
    receipt["architecture"] = "Other"
    receipt_payload = _canonical_json(receipt)
    receipt_path.write_bytes(receipt_payload)

    def mutate(document: dict[str, object]) -> None:
        _entries(document)[0]["receipt_sha256"] = hashlib.sha256(receipt_payload).hexdigest()

    expected = _rewrite_provisioning(fixture, mutate)

    with pytest.raises(ValueError, match="trusted lock entry"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=expected,
        )


def test_rejects_receipt_outside_its_declared_installation(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)

    def mutate(document: dict[str, object]) -> None:
        entries = _entries(document)
        entries[0]["receipt_relative_path"] = "bundles/concept/alternate-receipt.json"

    expected = _rewrite_provisioning(fixture, mutate)

    with pytest.raises(ValueError, match="installation-owned"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=expected,
        )


def test_rejects_wrong_bundle_manifest_identity(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    expected = _rewrite_provisioning(
        fixture,
        lambda document: _entries(document)[0].__setitem__("bundle_manifest_sha256", "f" * 64),
    )

    with pytest.raises(ValueError, match="bundle manifest digest"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=expected,
        )


def test_rejects_symlinked_model_lock(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    lock_path = fixture.root / "locks" / "image-models.toml"
    replacement = tmp_path / "replacement.toml"
    replacement.write_bytes(lock_path.read_bytes())
    lock_path.unlink()
    try:
        lock_path.symlink_to(replacement)
    except OSError:
        pytest.skip("symlinks are unavailable")

    with pytest.raises(ValueError, match="symlink"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=fixture.sha256,
        )


def test_provisioned_bundle_rejects_exact_receipt_drift_after_startup(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    provisioned = load_reverse_image_model_provisioning(
        fixture.root,
        fixture.relative_path,
        expected_manifest_sha256=fixture.sha256,
    )
    receipt_path = fixture.root / "bundles" / "instance" / "capicove-model-receipt.json"
    receipt = json.loads(receipt_path.read_bytes())
    assert isinstance(receipt, dict)
    receipt["hf_version"] = "tampered-after-startup"
    receipt_path.write_bytes(_canonical_json(receipt))

    with pytest.raises(ValueError, match="receipt digest"):
        provisioned.instance_bundle.verify()


def test_rejects_installation_swap_to_external_symlink_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    installation = fixture.root / "bundles" / "instance"
    parked = fixture.root / "bundles" / "instance.parked"
    outside = tmp_path / "outside-instance"
    shutil.copytree(installation, outside)
    probe = tmp_path / "symlink-probe"
    try:
        probe.symlink_to(outside, target_is_directory=True)
    except OSError:
        pytest.skip("symlinks are unavailable")
    else:
        probe.unlink()

    def swap_during_verify(
        directory: Path,
        model: ImageModelSpec,
        *,
        manifest_sha256: str,
        expected_model_manifest_sha256: str,
        expected_receipt_sha256: str | None = None,
    ) -> VerifiedModelInstallation:
        if directory != installation:
            return verify_model_installation(
                directory,
                model,
                manifest_sha256=manifest_sha256,
                expected_model_manifest_sha256=expected_model_manifest_sha256,
                expected_receipt_sha256=expected_receipt_sha256,
            )
        installation.rename(parked)
        installation.symlink_to(outside, target_is_directory=True)
        try:
            verified = verify_model_installation(
                directory,
                model,
                manifest_sha256=manifest_sha256,
                expected_model_manifest_sha256=expected_model_manifest_sha256,
                expected_receipt_sha256=expected_receipt_sha256,
            )
        finally:
            installation.unlink()
            parked.rename(installation)
        return verified

    monkeypatch.setattr(
        "capicove.model.provisioning.verify_model_installation",
        swap_during_verify,
    )

    with pytest.raises(ValueError, match="escaped or changed"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=fixture.sha256,
        )


def test_unsupported_secure_directory_api_fails_with_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    monkeypatch.setattr(os, "supports_dir_fd", set())

    with pytest.raises(RuntimeError, match="dir_fd/no-follow support"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=fixture.sha256,
        )


def test_missing_secure_open_flag_fails_with_actionable_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_fixture(tmp_path)
    monkeypatch.delattr(os, "O_DIRECTORY")

    with pytest.raises(RuntimeError, match="O_DIRECTORY"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=fixture.sha256,
        )


def test_canonical_manifest_uses_ascii_escaped_unicode_values(tmp_path: Path) -> None:
    fixture = _write_fixture(tmp_path)
    escaped_sha256 = _rewrite_provisioning(
        fixture,
        lambda document: document.__setitem__("provisioning_id", "个人图像检索"),
    )

    provisioned = load_reverse_image_model_provisioning(
        fixture.root,
        fixture.relative_path,
        expected_manifest_sha256=escaped_sha256,
    )

    assert provisioned.manifest.provisioning_id == "个人图像检索"
    path = fixture.root.joinpath(*fixture.relative_path.parts)
    document = json.loads(path.read_bytes())
    raw_unicode_payload = (
        json.dumps(
            document,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    path.write_bytes(raw_unicode_payload)

    with pytest.raises(ValueError, match="canonical JSON"):
        load_reverse_image_model_provisioning(
            fixture.root,
            fixture.relative_path,
            expected_manifest_sha256=hashlib.sha256(raw_unicode_payload).hexdigest(),
        )
