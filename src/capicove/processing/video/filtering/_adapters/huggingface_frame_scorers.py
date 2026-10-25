"""Local-only Hugging Face adapters for sampled video frame criteria."""

from __future__ import annotations

import hashlib
import importlib
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

from capicove.processing.video.filtering.video_filter_contracts import (
    AestheticsCriterion,
    FrameMetric,
    FrameTextSimilarityCriterion,
    LocalModelReference,
    NsfwCriterion,
    VideoCriterionResult,
    VideoFilterCriterion,
    VideoFilterItem,
    VideoFilterMetadata,
    WatermarkCriterion,
)
from capicove.processing.video.filtering.video_filter_ports import (
    BackendUnavailableFilterError,
    DecodeFailedFilterError,
    InvalidVideoFilterInputError,
    ModelIncompatibleFilterError,
    SampledFrame,
    VideoFilterTimeoutError,
)
from capicove.processing.video.filtering.video_filter_rules import in_range, reduce_values
from capicove.processing.video.models import AcceptedVideoSource


@dataclass(frozen=True, slots=True)
class ResolvedHuggingFaceModel:
    """Bind private provider handles to deterministic local model lineage."""

    reference: str
    revision: str | None
    manifest_hash: str
    model: object
    processor: object
    id2label: Mapping[int, str]


class LocalModelResolverProtocol(Protocol):
    """Resolve one local model without provider-specific public values."""

    def resolve(
        self,
        reference: LocalModelReference,
        *,
        model_kind: str,
    ) -> ResolvedHuggingFaceModel: ...


ClassificationInference = Callable[
    [ResolvedHuggingFaceModel, tuple[SampledFrame, ...]],
    list[list[float]],
]
ClipInference = Callable[
    [ResolvedHuggingFaceModel, tuple[SampledFrame, ...], str],
    tuple[list[float], list[float]],
]


class HuggingFaceLocalModelResolver:
    """Resolve only local or already-cached Hugging Face model assets."""

    def __init__(self, *, transformers_module: object | None = None) -> None:
        self._transformers_module = transformers_module

    def resolve(
        self,
        reference: LocalModelReference,
        *,
        model_kind: str,
    ) -> ResolvedHuggingFaceModel:
        """Load a classification or CLIP model without network or remote code."""
        reference_path = Path(reference.reference)
        if not reference_path.exists() and reference.cache_directory is None:
            raise BackendUnavailableFilterError(
                "A non-path model ID requires an explicit authorized cache_directory."
            )
        load_reference = (
            str(reference_path.resolve()) if reference_path.exists() else reference.reference
        )
        try:
            transformers = self._transformers_module or importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise BackendUnavailableFilterError(
                "Transformers is unavailable; install capicove[video-filter-models]."
            ) from error

        if model_kind == "classification":
            processor_factory = getattr(transformers, "AutoImageProcessor", None)
            model_factory = getattr(transformers, "AutoModelForImageClassification", None)
        elif model_kind == "clip":
            processor_factory = getattr(transformers, "AutoProcessor", None)
            model_factory = getattr(transformers, "CLIPModel", None)
        else:
            raise ValueError(f"unsupported Hugging Face model kind: {model_kind}")
        if processor_factory is None or model_factory is None:
            raise ModelIncompatibleFilterError(
                f"Transformers does not expose the required {model_kind} factories."
            )

        kwargs: dict[str, object] = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        if reference.revision is not None:
            kwargs["revision"] = reference.revision
        if reference.cache_directory is not None:
            kwargs["cache_dir"] = reference.cache_directory
        try:
            processor = processor_factory.from_pretrained(load_reference, **kwargs)
            model = model_factory.from_pretrained(load_reference, **kwargs)
        except (OSError, ValueError) as error:
            raise BackendUnavailableFilterError(
                f"The model {reference.reference!r} is not available locally."
            ) from error
        config = getattr(model, "config", None)
        id2label = _normalize_id2label(getattr(config, "id2label", {}))
        return ResolvedHuggingFaceModel(
            reference=reference.reference,
            revision=reference.revision,
            manifest_hash=_manifest_hash(reference),
            model=model,
            processor=processor,
            id2label=id2label,
        )


class HuggingFaceClassificationEvaluator:
    """Evaluate aesthetics and target-class probabilities from local models."""

    def __init__(
        self,
        resolver: LocalModelResolverProtocol | None = None,
        *,
        inference: ClassificationInference | None = None,
    ) -> None:
        self._resolver = resolver or HuggingFaceLocalModelResolver()
        self._inference = inference or _classification_inference
        self._cache: dict[LocalModelReference, ResolvedHuggingFaceModel] = {}

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        """Evaluate one supported local classification criterion."""
        del source, metadata
        if not isinstance(criterion, (AestheticsCriterion, NsfwCriterion, WatermarkCriterion)):
            raise InvalidVideoFilterInputError(
                "The Hugging Face classification evaluator received an unsupported criterion."
            )
        if not frames:
            raise DecodeFailedFilterError("Model scoring requires at least one sampled frame.")
        if datetime.now(UTC) >= deadline:
            raise VideoFilterTimeoutError("The model scoring deadline expired.")
        resolved = self._cache.get(criterion.model)
        if resolved is None:
            resolved = self._resolver.resolve(criterion.model, model_kind="classification")
            self._cache[criterion.model] = resolved
        logits = self._inference(resolved, frames)
        if (
            len(logits) != len(frames)
            or any(not row for row in logits)
            or len({len(row) for row in logits}) != 1
            or any(not math.isfinite(value) for row in logits for value in row)
        ):
            raise ModelIncompatibleFilterError(
                "The model output must contain one equally sized, non-empty, finite row per frame."
            )

        if isinstance(criterion, AestheticsCriterion):
            target_index: int | None = None
            if len(logits[0]) == 1:
                raw_scores = tuple(float(row[0]) for row in logits)
                scores = raw_scores
            else:
                matches = [
                    index
                    for index, label in resolved.id2label.items()
                    if label.casefold() == "aesthetic"
                ]
                if len(matches) != 1 or not 0 <= matches[0] < len(logits[0]):
                    raise ModelIncompatibleFilterError(
                        "Aesthetics requires scalar regression or a unique aesthetic class label."
                    )
                target_index = matches[0]
                raw_scores = tuple(float(row[target_index]) for row in logits)
                scores = tuple(_softmax(row)[target_index] for row in logits)
            minimum, maximum = criterion.min_score, criterion.max_score
        else:
            target_index = _target_index(resolved.id2label, criterion)
            if any(not 0 <= target_index < len(row) for row in logits):
                raise ModelIncompatibleFilterError(
                    "The target label index is outside the model output shape."
                )
            raw_scores = tuple(float(row[target_index]) for row in logits)
            scores = tuple(_softmax(row)[target_index] for row in logits)
            if isinstance(criterion, NsfwCriterion):
                minimum, maximum = criterion.min_score, criterion.max_score
            else:
                minimum, maximum = None, criterion.max_probability

        score = reduce_values(scores, criterion.reduction)
        metrics = tuple(
            FrameMetric(
                frame_index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                value=value,
                raw_value=raw,
            )
            for frame, value, raw in zip(frames, scores, raw_scores, strict=True)
        )
        return VideoCriterionResult(
            item_id=item.item_id,
            criterion_id=criterion.criterion_id,
            passed=in_range(score, minimum, maximum),
            score=score,
            frame_metrics=metrics,
            details={
                "model_reference": resolved.reference,
                "model_revision": resolved.revision,
                "model_manifest_hash": resolved.manifest_hash,
                "target_index": target_index,
                "reduction": criterion.reduction.value,
                "backend": "huggingface-local",
            },
        )


class HuggingFaceClipSimilarityEvaluator:
    """Evaluate normalized local CLIP frame-text similarity."""

    def __init__(
        self,
        resolver: LocalModelResolverProtocol | None = None,
        *,
        inference: ClipInference | None = None,
    ) -> None:
        self._resolver = resolver or HuggingFaceLocalModelResolver()
        self._inference = inference or _clip_inference
        self._cache: dict[LocalModelReference, ResolvedHuggingFaceModel] = {}

    def evaluate(
        self,
        item: VideoFilterItem,
        source: AcceptedVideoSource,
        metadata: VideoFilterMetadata,
        frames: tuple[SampledFrame, ...],
        criterion: VideoFilterCriterion,
        deadline: datetime,
    ) -> VideoCriterionResult:
        """Evaluate one video's frames against its explicitly associated text."""
        del source, metadata
        if not isinstance(criterion, FrameTextSimilarityCriterion):
            raise InvalidVideoFilterInputError("CLIP requires a frame-text criterion.")
        if item.text is None or not item.text.strip():
            raise InvalidVideoFilterInputError(
                "Associated text is required for frame-text similarity."
            )
        if not frames:
            raise DecodeFailedFilterError("CLIP similarity requires at least one frame.")
        if datetime.now(UTC) >= deadline:
            raise VideoFilterTimeoutError("The CLIP similarity deadline expired.")
        resolved = self._cache.get(criterion.model)
        if resolved is None:
            resolved = self._resolver.resolve(criterion.model, model_kind="clip")
            self._cache[criterion.model] = resolved
        similarities, raw_logits = self._inference(resolved, frames, item.text)
        if len(similarities) != len(frames) or len(raw_logits) != len(frames):
            raise ModelIncompatibleFilterError(
                "CLIP must return one similarity and raw logit per frame."
            )
        if any(not math.isfinite(value) for value in similarities + raw_logits):
            raise ModelIncompatibleFilterError("CLIP returned a non-finite score.")
        score = reduce_values(tuple(similarities), criterion.reduction)
        metrics = tuple(
            FrameMetric(
                frame_index=frame.index,
                timestamp_seconds=frame.timestamp_seconds,
                value=similarity,
                raw_value=raw_logit,
            )
            for frame, similarity, raw_logit in zip(
                frames,
                similarities,
                raw_logits,
                strict=True,
            )
        )
        return VideoCriterionResult(
            item_id=item.item_id,
            criterion_id=criterion.criterion_id,
            passed=in_range(score, criterion.min_score, criterion.max_score),
            score=score,
            frame_metrics=metrics,
            details={
                "model_reference": resolved.reference,
                "model_revision": resolved.revision,
                "model_manifest_hash": resolved.manifest_hash,
                "reduction": criterion.reduction.value,
                "backend": "huggingface-clip-local",
            },
        )


def _rgb_images(frames: tuple[SampledFrame, ...]) -> list[object]:
    import numpy as np

    images: list[object] = []
    for frame in frames:
        payload = frame.payload
        if not isinstance(payload, np.ndarray) or payload.ndim != 3 or payload.shape[2] != 3:
            raise DecodeFailedFilterError("Model frames must contain BGR image arrays.")
        images.append(np.ascontiguousarray(payload[..., ::-1]))
    return images


def _classification_inference(
    resolved: ResolvedHuggingFaceModel,
    frames: tuple[SampledFrame, ...],
) -> list[list[float]]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise BackendUnavailableFilterError(
            "PyTorch is unavailable; install capicove[video-filter-models]."
        ) from error
    images = _rgb_images(frames)
    inputs = resolved.processor(images=images, return_tensors="pt")  # type: ignore[operator]
    device = getattr(resolved.model, "device", None)
    if device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(device)
    with torch.inference_mode():
        outputs = resolved.model(**inputs)  # type: ignore[operator]
    logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ModelIncompatibleFilterError("The classification model returned no logits.")
    if hasattr(logits, "detach"):
        logits = logits.detach().cpu().tolist()
    return [[float(value) for value in row] for row in logits]


def _clip_inference(
    resolved: ResolvedHuggingFaceModel,
    frames: tuple[SampledFrame, ...],
    text: str,
) -> tuple[list[float], list[float]]:
    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise BackendUnavailableFilterError(
            "PyTorch is unavailable; install capicove[video-filter-models]."
        ) from error
    images = _rgb_images(frames)
    inputs = resolved.processor(  # type: ignore[operator]
        text=[text],
        images=images,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    device = getattr(resolved.model, "device", None)
    if device is not None and hasattr(inputs, "to"):
        inputs = inputs.to(device)
    try:
        with torch.inference_mode():
            image_features = resolved.model.get_image_features(  # type: ignore[attr-defined]
                pixel_values=inputs["pixel_values"]
            )
            text_arguments = {"input_ids": inputs["input_ids"]}
            if "attention_mask" in inputs:
                text_arguments["attention_mask"] = inputs["attention_mask"]
            text_features = resolved.model.get_text_features(**text_arguments)  # type: ignore[attr-defined]
            image_features = getattr(image_features, "pooler_output", image_features)
            text_features = getattr(text_features, "pooler_output", text_features)
            for features, count in ((image_features, len(frames)), (text_features, 1)):
                if (
                    features.ndim != 2
                    or features.shape[0] != count
                    or features.shape[1] == 0
                    or not torch.isfinite(features).all()
                    or not (features.norm(p=2, dim=-1) > 0).all()
                ):
                    raise ModelIncompatibleFilterError("CLIP returned invalid embeddings.")
            image_features = image_features / image_features.norm(
                p=2,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            text_features = text_features / text_features.norm(
                p=2,
                dim=-1,
                keepdim=True,
            ).clamp_min(1e-12)
            similarities = (image_features @ text_features.transpose(0, 1)).squeeze(-1)
            logit_scale = getattr(resolved.model, "logit_scale", None)
            raw_logits = similarities if logit_scale is None else similarities * logit_scale.exp()
    except (AttributeError, KeyError, RuntimeError, TypeError) as error:
        raise ModelIncompatibleFilterError(
            "The local CLIP model does not expose compatible image and text features."
        ) from error
    normalized_values = similarities.detach().cpu().tolist()
    raw_values = raw_logits.detach().cpu().tolist()
    return (
        [float(value) for value in normalized_values],
        [float(value) for value in raw_values],
    )


def _target_index(
    labels: Mapping[int, str],
    criterion: NsfwCriterion | WatermarkCriterion,
) -> int:
    explicit = criterion.target_label
    if explicit is not None:
        matches = [
            index for index, label in labels.items() if label.casefold() == explicit.casefold()
        ]
    else:
        token = "nsfw" if isinstance(criterion, NsfwCriterion) else "watermark"
        matches = [index for index, label in labels.items() if token in label.casefold()]
    if len(matches) != 1:
        raise ModelIncompatibleFilterError(
            "The model target label could not be resolved uniquely; configure target_label."
        )
    return matches[0]


def _softmax(values: list[float]) -> tuple[float, ...]:
    maximum = max(values)
    exponentials = tuple(math.exp(float(value) - maximum) for value in values)
    total = sum(exponentials)
    return tuple(value / total for value in exponentials)


def _normalize_id2label(value: object) -> Mapping[int, str]:
    if not isinstance(value, Mapping):
        return {}
    normalized: dict[int, str] = {}
    for key, label in value.items():
        try:
            index = int(key)
        except (TypeError, ValueError):
            continue
        normalized[index] = str(label)
    return normalized


def _manifest_hash(reference: LocalModelReference) -> str:
    digest = hashlib.sha256()
    path = Path(reference.reference)
    files: tuple[Path, ...]
    if path.is_file():
        files = (path,)
        root = path.parent
    elif path.is_dir():
        files = tuple(sorted(candidate for candidate in path.rglob("*") if candidate.is_file()))
        root = path
    else:
        digest.update(reference.reference.encode())
        digest.update((reference.revision or "").encode())
        return digest.hexdigest()
    for file_path in files:
        relative = file_path.relative_to(root).as_posix()
        digest.update(relative.encode())
        digest.update(str(file_path.stat().st_size).encode())
        with file_path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()
