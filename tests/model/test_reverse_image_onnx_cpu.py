"""Real synthetic ONNX parity for independent query and target adapters."""

from __future__ import annotations

import hashlib
import json
import socket
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image


onnx = pytest.importorskip("onnx", reason="install the model-test dependency group")
pytest.importorskip("onnxruntime", reason="install capicove[retrieval-cpu]")

from capicove.contracts.retrieval import (  # noqa: E402
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.bundle import (  # noqa: E402
    VerifiedImageEncoderBundle,
    load_verified_image_encoder_bundle,
)
from capicove.model.manifest import ImageModelSpec  # noqa: E402
from capicove.model.receipt import (  # noqa: E402
    build_receipt,
    verify_model_installation,
    write_receipt,
)
from capicove.processing.image._adapters.onnx_target import (  # noqa: E402
    OnnxImageTargetEncoder,
)
from capicove.processing.image.ports import (  # noqa: E402
    ImageSemanticLane,
    SemanticTargetIdentity,
)
from capicove.retrieval._adapters.onnx_image import OnnxImageQueryEncoder  # noqa: E402


pytestmark = [pytest.mark.model_onnx, pytest.mark.cpu]

FUTURE = datetime.now(UTC) + timedelta(hours=1)
REVISION = "1" * 40
SOURCE_MANIFEST_SHA256 = "a" * 64


def _space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="synthetic-rgb-mean-v1",
        provider_id="synthetic-onnx",
        model_revision=REVISION,
        preprocessing_version="rgb-exact-2-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _model_spec() -> ImageModelSpec:
    return ImageModelSpec(
        key="synthetic-onnx-image-encoder",
        repository="owner/synthetic-model",
        revision=REVISION,
        license="apache-2.0",
        license_url="https://example.invalid/license",
        capability="image_embedding",
        adapter="onnx-image-embedding",
        architecture="SyntheticRgbMean",
        preprocess_revision="rgb-exact-2-v1",
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
        "sha256": hashlib.sha256(payload).hexdigest(),
        "size_bytes": len(payload),
    }


def _write_synthetic_bundle(tmp_path: Path) -> VerifiedImageEncoderBundle:
    root = tmp_path / "bundle"
    root.mkdir()
    model_path = root / "vision.onnx"
    model_input = onnx.helper.make_tensor_value_info(
        "pixel_values",
        onnx.TensorProto.FLOAT,
        ["batch", 3, 2, 2],
    )
    model_output = onnx.helper.make_tensor_value_info(
        "image_embedding",
        onnx.TensorProto.FLOAT,
        ["batch", 3],
    )
    graph = onnx.helper.make_graph(
        [
            onnx.helper.make_node(
                "ReduceMean",
                ["pixel_values"],
                ["image_embedding"],
                axes=[2, 3],
                keepdims=0,
            )
        ],
        "synthetic_rgb_spatial_mean",
        [model_input],
        [model_output],
    )
    model = onnx.helper.make_model(
        graph,
        ir_version=9,
        opset_imports=[onnx.helper.make_opsetid("", 17)],
    )
    onnx.save(model, model_path)
    model_payload = model_path.read_bytes()
    license_payload = b"Apache License 2.0\n"
    (root / "LICENSE").write_bytes(license_payload)
    checksums_payload = _canonical_json(
        {
            "files": [
                _file_entry("vision.onnx", model_payload),
                _file_entry("LICENSE", license_payload),
            ],
            "schema_version": 1,
        }
    )
    (root / "checksums.json").write_bytes(checksums_payload)
    space = _space()
    manifest = {
        "artifacts": [
            {"role": "onnx-model", **_file_entry("vision.onnx", model_payload)},
            {"role": "checksums", **_file_entry("checksums.json", checksums_payload)},
            {"role": "license", **_file_entry("LICENSE", license_payload)},
        ],
        "bundle_id": "synthetic-rgb-mean-v1",
        "embedding_space": {
            "dimension": space.dimension,
            "distance_metric": space.distance_metric.value,
            "model_revision": space.model_revision,
            "normalization": space.normalization.value,
            "preprocessing_version": space.preprocessing_version,
            "provider_id": space.provider_id,
            "query_modalities": [QueryModality.IMAGE.value],
            "schema_version": space.schema_version,
            "space_id": space.space_id,
            "target_modalities": [TargetType.IMAGE.value],
        },
        "input": {
            "dtype": "float32",
            "name": "pixel_values",
            "rank": 4,
            "shape": ["batch", 3, 2, 2],
        },
        "license": "apache-2.0",
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
        "repository": "owner/synthetic-model",
        "revision": REVISION,
        "runtime": {
            "custom_operators_allowed": False,
            "execution_provider": "CPUExecutionProvider",
            "graph_optimization": "all",
            "provider_fallback_allowed": False,
            "version_specifier": ">=1.20,<2",
        },
        "schema_version": 1,
    }
    (root / "manifest.json").write_bytes(_canonical_json(manifest))
    receipt = build_receipt(
        _model_spec(),
        root,
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        hf_version="not-applicable",
    )
    write_receipt(root, receipt)
    installation = verify_model_installation(
        root,
        _model_spec(),
        manifest_sha256=SOURCE_MANIFEST_SHA256,
        expected_model_manifest_sha256=str(receipt["model_manifest_sha256"]),
    )
    return load_verified_image_encoder_bundle(installation)


def test_real_cpu_query_and_target_adapters_have_vector_parity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny_socket(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("ONNX image encoding must remain offline")

    bundle = _write_synthetic_bundle(tmp_path)
    query = OnnxImageQueryEncoder(bundle, intra_op_threads=2, inter_op_threads=1)
    target = OnnxImageTargetEncoder(
        bundle,
        lane=ImageSemanticLane.INSTANCE,
        intra_op_threads=2,
        inter_op_threads=1,
    )
    image = Image.new("RGB", (2, 2), (255, 128, 64))
    monkeypatch.setattr(socket, "socket", deny_socket)

    query_vector = query.encode(image, bundle.manifest.embedding_space, FUTURE)
    target_vectors = target.encode_images((image,), bundle.manifest.embedding_space, FUTURE)

    assert query.execution_provider == "CPUExecutionProvider"
    assert target.execution_provider == "CPUExecutionProvider"
    assert target.identity == SemanticTargetIdentity(
        ImageSemanticLane.INSTANCE,
        bundle.manifest.embedding_space,
        bundle.manifest_sha256,
    )
    assert target_vectors[0] == pytest.approx(query_vector.values, abs=1e-6)
    assert "torch" not in sys.modules
    assert "transformers" not in sys.modules
