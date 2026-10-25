"""Unit tests for the lazy ONNX image-query adapter."""

from __future__ import annotations

import sys
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
from capicove.retrieval._adapters.onnx_image import (
    OnnxImageQueryEncoder,
    _QueryOnnxSession,
    _QueryOnnxValueInfo,
)


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
        bundle_id="synthetic-query-v1",
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
        input=OnnxTensorContract(
            name="pixel_values",
            rank=4,
            shape=("batch", 3, 2, 2),
            dtype="float32",
        ),
        output=OnnxOutputContract(
            name="image_embedding",
            rank=2,
            selection="only",
            pooling="none",
            dimension=2,
            dtype="float32",
            l2_normalized=True,
            metric="cosine",
            vector_schema="image-vector-v1",
        ),
        embedding_space=_space(),
        artifacts=(
            BundleArtifact("onnx-model", PurePosixPath("vision.onnx"), 1, SHA_A),
            BundleArtifact("checksums", PurePosixPath("checksums.json"), 1, SHA_A),
            BundleArtifact("license", PurePosixPath("LICENSE"), 1, SHA_A),
        ),
        runtime=OnnxRuntimeContract(
            version_specifier=">=1.20,<2",
            execution_provider="CPUExecutionProvider",
            graph_optimization="all",
            custom_operators_allowed=False,
            provider_fallback_allowed=False,
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
class FakeValueInfo(_QueryOnnxValueInfo):
    name: str
    shape: tuple[int | str | None, ...]
    type: str = "tensor(float)"


class FakeSession(_QueryOnnxSession):
    def __init__(
        self,
        output: object,
        *,
        input_info: FakeValueInfo | None = None,
        output_info: FakeValueInfo | None = None,
        providers: tuple[str, ...] = ("CPUExecutionProvider",),
    ) -> None:
        self._output = output
        self._input = input_info or FakeValueInfo("pixel_values", ("batch", 3, 2, 2))
        self._output_info = output_info or FakeValueInfo("image_embedding", ("batch", 2))
        self._providers = providers
        self.input_feed: dict[str, object] | None = None
        self.output_names: tuple[str, ...] | None = None

    def get_inputs(self) -> tuple[FakeValueInfo, ...]:
        return (self._input,)

    def get_outputs(self) -> tuple[FakeValueInfo, ...]:
        return (self._output_info,)

    def get_providers(self) -> tuple[str, ...]:
        return self._providers

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> tuple[object, ...]:
        self.output_names = tuple(output_names)
        self.input_feed = dict(input_feed)
        return (self._output,)


class SessionOptionsView(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int


def _bundle(root: Path, events: list[str]) -> VerifiedImageEncoderBundle:
    (root / "vision.onnx").write_bytes(b"x")
    return cast(VerifiedImageEncoderBundle, FakeBundle(root, events))


def test_query_adapter_is_lazy_reverifies_and_preprocesses_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delitem(sys.modules, "onnxruntime", raising=False)
    events: list[str] = []
    session = FakeSession(np.asarray([[3.0, 4.0]], dtype=np.float32))
    captured: dict[str, object] = {}

    def session_factory(
        model_path: Path,
        *,
        session_options: object,
        providers: tuple[str, ...],
    ) -> FakeSession:
        events.append("session")
        captured.update(
            model_path=model_path,
            session_options=session_options,
            providers=providers,
        )
        return session

    adapter = OnnxImageQueryEncoder(
        _bundle(tmp_path, events),
        intra_op_threads=2,
        inter_op_threads=1,
        session_factory=session_factory,
    )
    assert events == []
    assert "onnxruntime" not in sys.modules

    vector = adapter.encode(
        Image.new("RGB", (1, 1), (255, 128, 0)),
        adapter.embedding_space,
        FUTURE,
    )

    assert events == ["verify", "session"]
    assert vector.values == pytest.approx((0.6, 0.8))
    assert captured["model_path"] == tmp_path / "vision.onnx"
    assert captured["providers"] == ("CPUExecutionProvider",)
    options = cast(SessionOptionsView, captured["session_options"])
    assert options.intra_op_num_threads == 2
    assert options.inter_op_num_threads == 1
    assert session.output_names == ("image_embedding",)
    assert session.input_feed is not None
    tensor = session.input_feed["pixel_values"]
    assert isinstance(tensor, np.ndarray)
    assert tensor.dtype == np.float32
    assert tensor.shape == (1, 3, 2, 2)
    np.testing.assert_allclose(tensor[0, :, 0, 0], (1.0, 128.0 / 255.0, 0.0))


def test_query_rejects_space_and_deadline_before_bundle_or_session(tmp_path: Path) -> None:
    events: list[str] = []
    adapter = OnnxImageQueryEncoder(
        _bundle(tmp_path, events),
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32)
        ),
    )

    with pytest.raises(ValueError, match="embedding space"):
        adapter.encode(Image.new("RGB", (2, 2)), _space("different"), FUTURE)
    with pytest.raises(TimeoutError, match="deadline"):
        adapter.encode(
            Image.new("RGB", (2, 2)),
            adapter.embedding_space,
            datetime.now(UTC) - timedelta(seconds=1),
        )

    assert events == []


@pytest.mark.parametrize(
    "session",
    (
        FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            input_info=FakeValueInfo("wrong", ("batch", 3, 2, 2)),
        ),
        FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            output_info=FakeValueInfo("wrong", ("batch", 2)),
        ),
        FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            input_info=FakeValueInfo("pixel_values", ("batch", 3, 3, 2)),
        ),
        FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            output_info=FakeValueInfo("image_embedding", ("batch", 3)),
        ),
        FakeSession(
            np.asarray([[1.0, 0.0]], dtype=np.float32),
            providers=("AzureExecutionProvider",),
        ),
    ),
)
def test_query_rejects_session_metadata_mismatch(tmp_path: Path, session: FakeSession) -> None:
    adapter = OnnxImageQueryEncoder(
        _bundle(tmp_path, []),
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: session,
    )

    with pytest.raises(ValueError, match=r"input|output|provider"):
        adapter.encode(Image.new("RGB", (2, 2)), adapter.embedding_space, FUTURE)

    assert session.input_feed is None


@pytest.mark.parametrize(
    "output",
    (
        np.asarray([3.0, 4.0], dtype=np.float32),
        np.asarray([[3.0, 4.0], [5.0, 12.0]], dtype=np.float32),
        np.asarray([[3.0]], dtype=np.float32),
        np.asarray([[float("nan"), 1.0]], dtype=np.float32),
        np.asarray([[0.0, 0.0]], dtype=np.float32),
        np.asarray([[3.0, 4.0]], dtype=np.float64),
    ),
)
def test_query_rejects_invalid_output(tmp_path: Path, output: np.ndarray) -> None:
    session = FakeSession(output)
    adapter = OnnxImageQueryEncoder(
        _bundle(tmp_path, []),
        intra_op_threads=1,
        session_factory=lambda *_args, **_kwargs: session,
    )

    with pytest.raises(ValueError, match=r"output|finite|norm|float32"):
        adapter.encode(Image.new("RGB", (2, 2)), adapter.embedding_space, FUTURE)


@pytest.mark.parametrize(("intra", "inter"), ((0, 1), (1, 0), (-1, 1), (True, 1)))
def test_query_rejects_invalid_thread_budgets(tmp_path: Path, intra: object, inter: object) -> None:
    with pytest.raises(ValueError, match="thread"):
        OnnxImageQueryEncoder(
            _bundle(tmp_path, []),
            intra_op_threads=intra,  # type: ignore[arg-type]
            inter_op_threads=inter,  # type: ignore[arg-type]
        )
