"""Opt-in production CLIP/SigLIP2 ONNX parity and complete retrieval acceptance."""

from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from capicove.model.bundle import VerifiedImageEncoderBundle, load_verified_image_encoder_bundle
from capicove.model.manifest import load_packaged_manifest
from capicove.model.receipt import verify_model_installation
from capicove.processing.image._adapters.onnx_target import OnnxImageTargetEncoder
from capicove.processing.image.ports import ImageSemanticLane
from capicove.retrieval._adapters.onnx_image import OnnxImageQueryEncoder


pytestmark = [pytest.mark.model_small, pytest.mark.cpu]


@pytest.fixture(scope="module")
def bundles() -> dict[str, VerifiedImageEncoderBundle]:
    selected = os.environ.get("CAPICOVE_PRODUCTION_ONNX_QUALIFICATION")
    if not selected:
        pytest.skip("Select hash-pinned, locally exported production ONNX bundles")
    configuration = json.loads(Path(selected).read_text())
    manifest = load_packaged_manifest()
    result = {}
    for lane in ("instance", "concept"):
        entry = configuration[lane]
        source = next(model for model in manifest.models if model.key == entry["source_model"])
        spec = replace(
            source,
            key=source.key + "-onnx",
            adapter="onnx-image-embedding",
            capability="image_embedding",
            preprocess_revision=entry["preprocess_revision"],
            default=False,
            settings=(),
        )
        installation = verify_model_installation(
            Path(entry["root"]),
            spec,
            manifest_sha256=entry["source_manifest_sha256"],
            expected_model_manifest_sha256=entry["model_manifest_sha256"],
        )
        result[lane] = load_verified_image_encoder_bundle(installation)
        assert result[lane].manifest_sha256 == entry["bundle_manifest_sha256"]
    return result


@pytest.mark.parametrize("lane", ["instance", "concept"])
def test_production_query_and_target_vectors_match(
    bundles: dict[str, VerifiedImageEncoderBundle], lane: str
) -> None:
    bundle = bundles[lane]
    query = OnnxImageQueryEncoder(bundle, intra_op_threads=2, inter_op_threads=1)
    target = OnnxImageTargetEncoder(
        bundle, lane=ImageSemanticLane(lane), intra_op_threads=2, inter_op_threads=1
    )
    deadline = datetime.now(UTC) + timedelta(seconds=60)
    image = Image.new("RGB", (320, 240), (90, 120, 200))
    vector = query.encode(image, bundle.manifest.embedding_space, deadline)
    targets = target.encode_images((image, image), bundle.manifest.embedding_space, deadline)
    assert len(vector.values) in (512, 768)
    assert targets[0] == pytest.approx(vector.values, abs=1e-5)
    assert targets[1] == pytest.approx(vector.values, abs=1e-5)


def test_production_models_complete_build_activation_search_and_mcp(
    bundles: dict[str, VerifiedImageEncoderBundle], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.model import test_reverse_image_onnx_runtime_vertical as vertical

    monkeypatch.setattr(vertical, "_write_bundle", lambda root, lane: bundles[lane])
    vertical.test_real_onnx_runtime_completes_reverse_image_vertical(tmp_path, monkeypatch)


def test_production_models_reopen_through_cli_and_stdio_mcp(
    bundles: dict[str, VerifiedImageEncoderBundle], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.model import test_reverse_image_onnx_runtime_vertical as vertical

    def copied_bundle(root: Path, lane: str) -> VerifiedImageEncoderBundle:
        original = bundles[lane].installation
        target = root / "bundles" / lane
        shutil.copytree(original.path, target)
        installation = verify_model_installation(
            target,
            original.model,
            manifest_sha256=original.source_manifest_sha256,
            expected_model_manifest_sha256=original.model_manifest_sha256,
        )
        return load_verified_image_encoder_bundle(installation)

    monkeypatch.setattr(vertical, "_write_bundle", copied_bundle)
    vertical.test_image_cli_and_real_stdio_reopen_activated_index(tmp_path)
