"""Unit tests for the processing-owned lazy ONNX target adapter."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Protocol, cast

import numpy as np
import pytest
from PIL import Image

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model.bundle import (
    BundleArtifact,
    ImageEncoderBundleManifest,
    ImagePreprocessingContract,
    OnnxOutputContract,
    OnnxRuntimeContract,
    OnnxTensorContract,
    VerifiedImageEncoderBundle,
)
from capicove.processing.image._adapters.onnx_target import (
    OnnxImageTargetEncoder,
    _TargetOnnxSession,
    _TargetOnnxValueInfo,
)
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity


FUTURE = datetime.now(UTC) + timedelta(hours=1)
SHA_A = "a" * 64


def _space(space_id: str = "instance-space") -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id=space_id,
        provider_id="synthetic-onnx",
        model_revision="1" * 40,
        preprocessing_version="rgb-exact-2-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )


def _manifest() -> ImageEncoderBundleManifest:
    return ImageEncoderBundleManifest(
        bundle_id="synthetic-target-v1",
        repository="owner/model",
        revision="1" * 40,
        license="apache-2.0",
        preprocessing=ImagePreprocessingContract(
            color_mode="RGB",
            apply_exif_orientation=True,
            resize_strategy="exact",
            resize_size=(2, 2),
            crop_strategy="none",
            crop_size=None,
            interpolation="nearest",
            value_range=(0.0, 1.0),
            channel_mean=(0.0, 0.0, 0.0),
            channel_std=(1.0, 1.0, 1.0),
            tensor_layout="NCHW",
            tensor_dtype="float32",
        ),
        input=OnnxTensorContract("pixel_values", 4, ("batch", 3, 2, 2), "float32"),
        output=OnnxOutputContract(
            "image_embedding",
            2,
            "only",
            "none",
            2,
            "float32",
            True,
            "cosine",
            "image-vector-v1",
        ),
        embedding_space=_space(),
        artifacts=(
            BundleArtifact("onnx-model", PurePosixPath("vision.onnx"), 1, SHA_A),
            BundleArtifact("checksums", PurePosixPath("checksums.json"), 1, SHA_A),
            BundleArtifact("license", PurePosixPath("LICENSE"), 1, SHA_A),
        ),
        runtime=OnnxRuntimeContract(
            ">=1.20,<2",
            "CPUExecutionProvider",
            "all",
            False,
            False,
        ),
        network_behavior="offline",
    )


class FakeBundle:
    def __init__(self, root: Path, events: list[str]) -> None:
        self.manifest = _manifest()
        self.manifest_sha256 = SHA_A
        self.installation = SimpleNamespace(path=root)
        self._events = events

    def verify(self) -> None:
        self._events.append("verify")


@dataclass
class FakeValueInfo(_TargetOnnxValueInfo):
    name: str
    shape: tuple[int | str | None, ...]
    type: str = "tensor(float)"


class FakeSession(_TargetOnnxSession):
    def __init__(
        self,
        output: object,
        *,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        self._output = output
        self._providers = providers
        self.input_feed: dict[str, object] | None = None

    def get_inputs(self) -> tuple[FakeValueInfo, ...]:
        return (FakeValueInfo("pixel_values", ("batch", 3, 2, 2)),)

    def get_outputs(self) -> tuple[FakeValueInfo, ...]:
        return (FakeValueInfo("image_embedding", ("batch", 2)),)

    def get_providers(self) -> tuple[str, ...]:
        return self._providers

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> tuple[object, ...]:
        assert tuple(output_names) == ("image_embedding",)
        self.input_feed = dict(input_feed)
        return (self._output,)


class SessionOptionsView(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int


def _bundle(root: Path, events: list[str]) -> VerifiedImageEncoderBundle:
    (root / "vision.onnx").write_bytes(b"x")
    return cast(VerifiedImageEncoderBundle, FakeBundle(root, events))


def test_target_adapter_preserves_batch_order_identity_and_preprocessing(tmp_path: Path) -> None:
    events: list[str] = []
    session = FakeSession(
        np.asarray(
            [
                [3.0, 4.0],
                [5.0, 12.0],
            ],
            dtype=np.float32,
        )
    )

    def session_factory(
        model_path: Path,
        *,
        session_options: object,
        providers: tuple[str, ...],
    ) -> FakeSession:
        assert model_path == tmp_path / "vision.onnx"
        events.append("session")
        options = cast(SessionOptionsView, session_options)
        assert options.intra_op_num_threads == 2
        assert options.inter_op_num_threads == 1
        assert providers == ("CPUExecutionProvider",)
        return session

    bundle = _bundle(tmp_path, events)
    adapter = OnnxImageTargetEncoder(
        bundle,
        lane=ImageSemanticLane.INSTANCE,
        intra_op_threads=2,
        session_factory=session_factory,
    )

    vectors = adapter.encode_images(
        (
            Image.new("RGB", (1, 1), (255, 0, 0)),
            Image.new("RGB", (1, 1), (0, 255, 0)),
        ),
        bundle.manifest.embedding_space,
        FUTURE,
    )

    assert events == ["verify", "session"]
    assert vectors[0] == pytest.approx((0.6, 0.8))
    assert vectors[1] == pytest.approx((5.0 / 13.0, 12.0 / 13.0))
    assert adapter.identity == SemanticTargetIdentity(
        ImageSemanticLane.INSTANCE,
        bundle.manifest.embedding_space,
        SHA_A,
    )
    assert session.input_feed is not None
    tensor = session.input_feed["pixel_values"]
    assert isinstance(tensor, np.ndarray)
    assert tensor.shape == (2, 3, 2, 2)
    np.testing.assert_array_equal(tensor[0, :, 0, 0], (1.0, 0.0, 0.0))
    np.testing.assert_array_equal(tensor[1, :, 0, 0], (0.0, 1.0, 0.0))


def test_target_verify_ready_does_not_create_a_session(tmp_path: Path) -> None:
    events: list[str] = []
    bundle = _bundle(tmp_path, events)
    adapter = OnnxImageTargetEncoder(
        bundle,
        lane=ImageSemanticLane.CONCEPT,
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: events.append("session"),  # type: ignore[arg-type]
    )

    adapter.verify_ready(deadline=FUTURE)

    assert events == ["verify"]


def test_target_rejects_empty_batch_space_and_deadline_before_session(tmp_path: Path) -> None:
    events: list[str] = []
    bundle = _bundle(tmp_path, events)
    adapter = OnnxImageTargetEncoder(
        bundle,
        lane=ImageSemanticLane.INSTANCE,
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32)
        ),
    )

    with pytest.raises(ValueError, match="batch"):
        adapter.encode_images((), adapter.identity.space, FUTURE)
    with pytest.raises(ValueError, match="embedding space"):
        adapter.encode_images((Image.new("RGB", (2, 2)),), _space("other"), FUTURE)
    with pytest.raises(TimeoutError, match="deadline"):
        adapter.encode_images(
            (Image.new("RGB", (2, 2)),),
            adapter.identity.space,
            datetime.now(UTC) - timedelta(seconds=1),
        )

    assert events == []


@pytest.mark.parametrize(
    "output",
    (
        np.asarray([[3.0, 4.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0], [5.0, 12.0], [8.0, 15.0]], dtype=np.float32),
        np.asarray([[3.0], [4.0]], dtype=np.float32),
        np.asarray([[float("inf"), 1.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=np.float64),
    ),
)
def test_target_rejects_invalid_batch_output(tmp_path: Path, output: np.ndarray) -> None:
    bundle = _bundle(tmp_path, [])
    adapter = OnnxImageTargetEncoder(
        bundle,
        lane=ImageSemanticLane.INSTANCE,
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: FakeSession(output),
    )

    with pytest.raises(ValueError, match=r"output|row|dimension|finite|norm|float32"):
        adapter.encode_images(
            (Image.new("RGB", (2, 2)), Image.new("RGB", (2, 2))),
            adapter.identity.space,
            FUTURE,
        )


def test_target_rejects_unavailable_cpu_provider(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path, [])
    adapter = OnnxImageTargetEncoder(
        bundle,
        lane=ImageSemanticLane.INSTANCE,
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            providers=("AzureExecutionProvider",),
        ),
    )

    with pytest.raises(ValueError, match="provider"):
        adapter.encode_images(
            (Image.new("RGB", (2, 2)),),
            adapter.identity.space,
            FUTURE,
        )


@pytest.mark.parametrize("lane", ("instance", None, True))
def test_target_requires_closed_semantic_lane(tmp_path: Path, lane: object) -> None:
    with pytest.raises(TypeError, match="lane"):
        OnnxImageTargetEncoder(
            _bundle(tmp_path, []),
            lane=lane,  # type: ignore[arg-type]
            intra_op_threads=1,
        )
