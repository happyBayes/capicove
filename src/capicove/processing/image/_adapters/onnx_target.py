"""Lazy offline ONNX Runtime adapter for build-side image embeddings."""

from __future__ import annotations

import importlib
import math
import re
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol, cast

import numpy as np
from PIL import Image, ImageOps

from capicove.contracts.retrieval import EmbeddingSpaceRef
from capicove.model.bundle import (
    ImagePreprocessingContract,
    VerifiedImageEncoderBundle,
)
from capicove.processing.image.ports import ImageSemanticLane, SemanticTargetIdentity


_CPU_PROVIDER = "CPUExecutionProvider"
_ORT_VERSION_PATTERN = re.compile(r"^(\d+)\.(\d+)(?:\.(\d+))?")
_SPECIFIER_PATTERN = re.compile(r"^>=(\d+)\.(\d+)(?:\.(\d+))?,<(\d+)(?:\.(\d+))?(?:\.(\d+))?$")


class _TargetOnnxValueInfo(Protocol):
    name: str
    shape: Sequence[int | str | None]
    type: str


class _TargetOnnxSession(Protocol):
    def get_inputs(self) -> Sequence[_TargetOnnxValueInfo]: ...

    def get_outputs(self) -> Sequence[_TargetOnnxValueInfo]: ...

    def get_providers(self) -> Sequence[str]: ...

    def run(
        self,
        output_names: Sequence[str],
        input_feed: Mapping[str, object],
    ) -> Sequence[object]: ...


class _TargetOnnxSessionFactory(Protocol):
    def __call__(
        self,
        model_path: Path,
        *,
        session_options: object,
        providers: tuple[str, ...],
    ) -> _TargetOnnxSession: ...


class _OrtSessionOptions(Protocol):
    intra_op_num_threads: int
    inter_op_num_threads: int
    execution_mode: object
    graph_optimization_level: object
    use_per_session_threads: bool
    use_deterministic_compute: bool

    def add_session_config_entry(self, key: str, value: str) -> None: ...


class _OrtExecutionMode(Protocol):
    ORT_SEQUENTIAL: object


class _OrtGraphOptimizationLevel(Protocol):
    ORT_DISABLE_ALL: object
    ORT_ENABLE_BASIC: object
    ORT_ENABLE_EXTENDED: object
    ORT_ENABLE_ALL: object


class _OrtModule(Protocol):
    __version__: str
    ExecutionMode: _OrtExecutionMode
    GraphOptimizationLevel: _OrtGraphOptimizationLevel

    def SessionOptions(self) -> _OrtSessionOptions: ...

    def get_available_providers(self) -> Sequence[str]: ...

    def InferenceSession(
        self,
        model_path: str,
        *,
        sess_options: object,
        providers: Sequence[str],
    ) -> _TargetOnnxSession: ...


@dataclass(frozen=True, slots=True)
class _TargetSessionOptionsSpec:
    intra_op_num_threads: int
    inter_op_num_threads: int
    graph_optimization: str
    allow_spinning: bool = False
    use_per_session_threads: bool = True
    use_deterministic_compute: bool = True


class OnnxImageTargetEncoder:
    """Encode ordered target batches with a separate verified CPU-only session."""

    execution_provider = _CPU_PROVIDER

    def __init__(
        self,
        bundle: VerifiedImageEncoderBundle,
        lane: ImageSemanticLane,
        *,
        intra_op_threads: int,
        inter_op_threads: int = 1,
        session_factory: _TargetOnnxSessionFactory | None = None,
    ) -> None:
        """Configure a lazy target encoder under an explicit positive CPU budget."""
        if not isinstance(lane, ImageSemanticLane):
            raise TypeError("lane must be an ImageSemanticLane")
        _require_positive_threads("intra_op_threads", intra_op_threads)
        _require_positive_threads("inter_op_threads", inter_op_threads)
        self._bundle = bundle
        self.identity = SemanticTargetIdentity(
            lane=lane,
            space=bundle.manifest.embedding_space,
            bundle_manifest_sha256=bundle.manifest_sha256,
        )
        self._intra_op_threads = intra_op_threads
        self._inter_op_threads = inter_op_threads
        self._session_factory = session_factory
        self._session: _TargetOnnxSession | None = None
        self._session_lock = threading.Lock()

    def verify_ready(self, *, deadline: datetime) -> None:
        """Reverify the local bundle and CPU runtime without creating a session."""
        _check_deadline(deadline)
        _verified_model_path(self._bundle)
        if self._session_factory is None:
            _verify_runtime(_load_onnxruntime(), self._bundle)
        _check_deadline(deadline)

    def encode_images(
        self,
        images: tuple[Image.Image, ...],
        space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[tuple[float, ...], ...]:
        """Encode one non-empty ordered target batch in the exact verified space."""
        if not isinstance(images, tuple):
            raise TypeError("images must be a tuple")
        if not images:
            raise ValueError("images batch must not be empty")
        if any(not isinstance(image, Image.Image) for image in images):
            raise TypeError("images batch must contain only PIL Images")
        if space != self.identity.space:
            raise ValueError("requested embedding space does not match the verified bundle")
        _check_deadline(deadline)
        session = self._get_session(deadline)
        _check_deadline(deadline)
        tensor = _preprocess_images(images, self._bundle.manifest.preprocessing)
        _check_deadline(deadline)
        output = session.run(
            (self._bundle.manifest.output.name,),
            {self._bundle.manifest.input.name: tensor},
        )
        _check_deadline(deadline)
        return _normalized_output_rows(
            output,
            expected_rows=len(images),
            expected_dimension=self.identity.space.dimension,
        )

    def _get_session(self, deadline: datetime) -> _TargetOnnxSession:
        session = self._session
        if session is not None:
            return session
        with self._session_lock:
            if self._session is None:
                _check_deadline(deadline)
                self._session = self._create_session()
            return self._session

    def _create_session(self) -> _TargetOnnxSession:
        model_path = _verified_model_path(self._bundle)
        providers = (_CPU_PROVIDER,)
        if self._session_factory is None:
            runtime = _load_onnxruntime()
            _verify_runtime(runtime, self._bundle)
            options = _runtime_session_options(
                runtime,
                intra_op_threads=self._intra_op_threads,
                inter_op_threads=self._inter_op_threads,
                graph_optimization=self._bundle.manifest.runtime.graph_optimization,
            )
            session = runtime.InferenceSession(
                str(model_path),
                sess_options=options,
                providers=providers,
            )
        else:
            options_spec = _TargetSessionOptionsSpec(
                intra_op_num_threads=self._intra_op_threads,
                inter_op_num_threads=self._inter_op_threads,
                graph_optimization=self._bundle.manifest.runtime.graph_optimization,
            )
            session = self._session_factory(
                model_path,
                session_options=options_spec,
                providers=providers,
            )
        _verify_session_contract(session, self._bundle)
        return session


def _verified_model_path(bundle: VerifiedImageEncoderBundle) -> Path:
    bundle.verify()
    artifacts = tuple(
        artifact for artifact in bundle.manifest.artifacts if artifact.role == "onnx-model"
    )
    if len(artifacts) != 1:
        raise ValueError("verified bundle must contain exactly one ONNX model")
    return bundle.installation.path.joinpath(*artifacts[0].relative_path.parts)


def _load_onnxruntime() -> _OrtModule:
    try:
        module = importlib.import_module("onnxruntime")
    except ImportError as error:
        raise RuntimeError("install capicove[retrieval-cpu] for ONNX image retrieval") from error
    return cast(_OrtModule, module)


def _verify_runtime(runtime: _OrtModule, bundle: VerifiedImageEncoderBundle) -> None:
    if not _version_satisfies(
        runtime.__version__,
        bundle.manifest.runtime.version_specifier,
    ):
        raise RuntimeError("installed ONNX Runtime version is outside the verified bundle range")
    if _CPU_PROVIDER not in runtime.get_available_providers():
        raise RuntimeError("ONNX Runtime CPUExecutionProvider is unavailable")


def _runtime_session_options(
    runtime: _OrtModule,
    *,
    intra_op_threads: int,
    inter_op_threads: int,
    graph_optimization: str,
) -> _OrtSessionOptions:
    options = runtime.SessionOptions()
    options.intra_op_num_threads = intra_op_threads
    options.inter_op_num_threads = inter_op_threads
    options.use_per_session_threads = True
    options.use_deterministic_compute = True
    options.execution_mode = runtime.ExecutionMode.ORT_SEQUENTIAL
    optimization_levels = {
        "disabled": runtime.GraphOptimizationLevel.ORT_DISABLE_ALL,
        "basic": runtime.GraphOptimizationLevel.ORT_ENABLE_BASIC,
        "extended": runtime.GraphOptimizationLevel.ORT_ENABLE_EXTENDED,
        "all": runtime.GraphOptimizationLevel.ORT_ENABLE_ALL,
    }
    options.graph_optimization_level = optimization_levels[graph_optimization]
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    options.add_session_config_entry("session.inter_op.allow_spinning", "0")
    options.add_session_config_entry("session.force_spinning_stop", "1")
    return options


def _verify_session_contract(
    session: _TargetOnnxSession,
    bundle: VerifiedImageEncoderBundle,
) -> None:
    inputs = tuple(session.get_inputs())
    outputs = tuple(session.get_outputs())
    input_contract = bundle.manifest.input
    output_contract = bundle.manifest.output
    if len(inputs) != 1:
        raise ValueError("ONNX session must expose exactly one input")
    if len(outputs) != 1:
        raise ValueError("ONNX session must expose exactly one output")
    observed_input = inputs[0]
    observed_output = outputs[0]
    if observed_input.name != input_contract.name:
        raise ValueError("ONNX input name does not match the bundle manifest")
    if observed_output.name != output_contract.name:
        raise ValueError("ONNX output name does not match the bundle manifest")
    if observed_input.type != "tensor(float)" or observed_output.type != "tensor(float)":
        raise ValueError("ONNX input and output must be float32 tensors")
    input_shape = tuple(observed_input.shape)
    output_shape = tuple(observed_output.shape)
    if len(input_shape) != input_contract.rank or tuple(input_shape[1:]) != tuple(
        input_contract.shape[1:]
    ):
        raise ValueError("ONNX input shape does not match the bundle manifest")
    if len(output_shape) != output_contract.rank or output_shape[1] != output_contract.dimension:
        raise ValueError("ONNX output shape does not match the bundle manifest")
    if not _is_dynamic_batch_dimension(input_shape[0]) or not _is_dynamic_batch_dimension(
        output_shape[0]
    ):
        raise ValueError("ONNX input and output must retain a dynamic batch dimension")
    if tuple(session.get_providers()) != (_CPU_PROVIDER,):
        raise ValueError("ONNX session provider must be exactly CPUExecutionProvider")


def _preprocess_images(
    images: tuple[Image.Image, ...],
    contract: ImagePreprocessingContract,
) -> np.ndarray:
    rows: list[np.ndarray] = []
    for image in images:
        prepared = (
            ImageOps.exif_transpose(image) if contract.apply_exif_orientation else image.copy()
        )
        prepared = prepared.convert("RGB")
        prepared = _resize_and_crop(prepared, contract)
        pixels = np.asarray(prepared, dtype=np.float32)
        if pixels.ndim != 3 or pixels.shape[2] != 3:
            raise ValueError("preprocessed image must contain exactly three RGB channels")
        lower, upper = contract.value_range
        pixels = lower + (pixels / np.float32(255.0)) * (upper - lower)
        mean = np.asarray(contract.channel_mean, dtype=np.float32)
        std = np.asarray(contract.channel_std, dtype=np.float32)
        pixels = (pixels - mean) / std
        rows.append(np.transpose(pixels, (2, 0, 1)))
    return np.ascontiguousarray(np.stack(rows), dtype=np.float32)


def _resize_and_crop(
    image: Image.Image,
    contract: ImagePreprocessingContract,
) -> Image.Image:
    target_height, target_width = contract.resize_size
    resampling = {
        "nearest": Image.Resampling.NEAREST,
        "bilinear": Image.Resampling.BILINEAR,
        "bicubic": Image.Resampling.BICUBIC,
    }[contract.interpolation]
    if contract.resize_strategy == "exact":
        resized = image.resize((target_width, target_height), resampling)
    else:
        scale = max(target_width / image.width, target_height / image.height)
        resized = image.resize(
            (
                max(target_width, round(image.width * scale)),
                max(target_height, round(image.height * scale)),
            ),
            resampling,
        )
    if contract.crop_strategy == "center":
        if contract.crop_size is None:
            raise ValueError("center crop requires a crop_size")
        crop_height, crop_width = contract.crop_size
        left = (resized.width - crop_width) // 2
        top = (resized.height - crop_height) // 2
        resized = resized.crop((left, top, left + crop_width, top + crop_height))
    expected_height, expected_width = contract.crop_size or contract.resize_size
    if resized.size != (expected_width, expected_height):
        raise ValueError("preprocessed image shape does not match the ONNX input contract")
    return resized


def _normalized_output_rows(
    output: Sequence[object],
    *,
    expected_rows: int,
    expected_dimension: int,
) -> tuple[tuple[float, ...], ...]:
    if len(output) != 1:
        raise ValueError("ONNX inference must return exactly one output")
    array = np.asarray(output[0])
    if array.dtype != np.dtype(np.float32):
        raise ValueError("ONNX output must have float32 dtype")
    if array.ndim != 2:
        raise ValueError("ONNX output must have rank 2")
    if array.shape != (expected_rows, expected_dimension):
        raise ValueError("ONNX output row count or dimension does not match the request")
    if not np.isfinite(array).all():
        raise ValueError("ONNX output must contain only finite values")
    normalized: list[tuple[float, ...]] = []
    for row in array:
        values = tuple(float(value) for value in row)
        norm = math.sqrt(math.fsum(value * value for value in values))
        if not math.isfinite(norm) or norm <= 0.0:
            raise ValueError("ONNX output row must have a positive finite norm")
        normalized.append(tuple(value / norm for value in values))
    return tuple(normalized)


def _version_satisfies(version: str, specifier: str) -> bool:
    version_match = _ORT_VERSION_PATTERN.match(version)
    specifier_match = _SPECIFIER_PATTERN.fullmatch(specifier)
    if version_match is None or specifier_match is None:
        return False
    current = tuple(int(value or 0) for value in version_match.groups())
    values = tuple(int(value or 0) for value in specifier_match.groups())
    lower = values[:3]
    upper = values[3:]
    return lower <= current < upper


def _is_dynamic_batch_dimension(value: int | str | None) -> bool:
    return value is None or (isinstance(value, str) and bool(value.strip()))


def _require_positive_threads(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive thread count")


def _check_deadline(deadline: datetime) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise ValueError("deadline must be a timezone-aware datetime")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise TimeoutError("ONNX image encoding deadline exceeded")
