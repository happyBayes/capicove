"""Lazy, local-only Hugging Face adapters for image scoring."""

from __future__ import annotations

import importlib
import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import numpy as np
from PIL import Image

from capicove.model.receipt import VerifiedModelInstallation


class HuggingFaceUnavailableError(RuntimeError):
    """Raised when the optional local model backend is unavailable."""


class HuggingFaceAestheticsScorer:
    """Return normalized regression or classification aesthetics predictions."""

    provider_id = "huggingface-transformers"
    provider_version: str | None = None
    model_revision: str | None = None

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        score_scale: float = 10.0,
        *,
        target_label: str | None = None,
        device: str | None = None,
    ) -> None:
        if not math.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("score_scale must be positive and finite")
        self._installation = _require_installation(installation, "huggingface-aesthetics")
        self._model_path = self._installation.path
        self._score_scale = score_scale
        self._target_label = (
            _required_nonblank("target_label", target_label).casefold()
            if target_label is not None
            else None
        )
        self.model_id = self._installation.repository
        self.model_revision = self._installation.revision
        self.model_manifest_sha256 = self._installation.model_manifest_sha256
        self.preprocess_revision = self._installation.preprocess_revision
        self.device = device or "cpu"
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def score(self, images: tuple[Image.Image, ...]) -> tuple[float, ...]:
        """Return one scaled regression score or named-class probability per image."""
        if not images:
            return ()
        self._load()
        processor, model, torch = _require_loaded(
            self._processor,
            self._model,
            self._torch,
        )
        inputs = processor(images=list(images), return_tensors="pt")
        inputs = _move_inputs(inputs, self._device)
        with torch.no_grad():
            outputs = model(**inputs)
        logits = np.asarray(outputs.logits.detach().cpu().numpy(), dtype=np.float64)
        logits = logits.reshape(len(images), -1)
        if self._target_label is None:
            if logits.shape[1] != 1:
                raise ValueError("aesthetics regression model must return one logit per image")
            return tuple(float(value / self._score_scale) for value in logits[:, 0])
        label_index = _model_label_index(model, self._target_label, logits.shape[1])
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        return tuple(float(value) for value in probabilities[:, label_index])

    def _load(self) -> None:
        if self._model is not None:
            return
        self._installation.verify()
        transformers, torch = _model_modules()
        self._processor = transformers.AutoImageProcessor.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        )
        self._model = transformers.AutoModelForImageClassification.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        ).eval()
        if self._device is not None:
            self._model = self._model.to(self._device)
        self._installation.verify()
        self._torch = torch
        self.provider_version = _module_version(transformers)


class _LocalClipScorer:
    """Return CLIP image-text cosine similarity from an existing local model."""

    provider_id = "huggingface-transformers-clip"
    provider_version: str | None = None
    model_revision: str | None = None

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        *,
        device: str | None = None,
    ) -> None:
        self._installation = _require_installation(
            installation,
            "huggingface-clip-similarity",
        )
        self._model_path = self._installation.path
        self.model_id = self._installation.repository
        self.model_revision = self._installation.revision
        self.model_manifest_sha256 = self._installation.model_manifest_sha256
        self.preprocess_revision = self._installation.preprocess_revision
        self.device = device or "cpu"
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        self._installation.verify()
        transformers, torch = _model_modules()
        processor = transformers.CLIPProcessor.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        )
        model = transformers.CLIPModel.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        ).eval()
        if self._device is not None:
            model = model.to(self._device)
        self._installation.verify()
        self._processor, self._model, self._torch = processor, model, torch
        self.provider_version = _module_version(transformers)


class HuggingFaceClipImageTextSimilarityScorer(_LocalClipScorer):
    """Return cosine similarity from verified local CLIP embeddings."""

    def score(self, images: tuple[Image.Image, ...], text: str) -> tuple[float, ...]:
        """Return one cosine similarity per image against the same text."""
        if not text.strip():
            raise ValueError("text must not be blank")
        if not images:
            return ()
        self._load()
        processor, model, torch = _require_loaded(
            self._processor,
            self._model,
            self._torch,
        )
        inputs = processor(
            text=[text] * len(images),
            images=list(images),
            return_tensors="pt",
            padding=True,
        )
        inputs = _move_inputs(inputs, self._device)
        with torch.no_grad():
            outputs = model(**inputs)
        image_vectors = _normalized_rows(outputs.image_embeds)
        text_vectors = _normalized_rows(outputs.text_embeds)
        if image_vectors.shape != text_vectors.shape or image_vectors.shape[0] != len(images):
            raise ValueError("CLIP image and text embeddings must have matching batch shapes")
        return tuple(float(value) for value in np.sum(image_vectors * text_vectors, axis=1))


class _LocalPipelineScorer:
    """Share local-only pipeline construction without eager imports."""

    provider_id = "huggingface-transformers"
    provider_version: str | None = None

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        task: str,
        *,
        expected_adapter: str,
        device: int | str | None = None,
    ) -> None:
        self._installation = _require_installation(installation, expected_adapter)
        self._model_path = self._installation.path
        self._task = task
        self._pipeline: Callable[..., Any] | None = None
        self.model_id = self._installation.repository
        self.model_revision = self._installation.revision
        self.model_manifest_sha256 = self._installation.model_manifest_sha256
        self.preprocess_revision = self._installation.preprocess_revision
        self.device = str(device) if device is not None else "cpu"
        self._device = device

    def _get_pipeline(self) -> Callable[..., Any]:
        if self._pipeline is None:
            self._installation.verify()
            try:
                transformers = importlib.import_module("transformers")
            except ModuleNotFoundError as error:
                raise HuggingFaceUnavailableError(
                    "Transformers is unavailable; install capicove[models-dev]"
                ) from error
            factory = cast(Callable[..., Callable[..., Any]], transformers.pipeline)
            pipeline_kwargs: dict[str, object] = {
                "model": transformers.AutoModelForImageClassification.from_pretrained(
                    str(self._model_path), local_files_only=True, trust_remote_code=False
                ),
                "image_processor": transformers.AutoImageProcessor.from_pretrained(
                    str(self._model_path), local_files_only=True, trust_remote_code=False
                ),
                "trust_remote_code": False,
            }
            if self._device is not None:
                pipeline_kwargs["device"] = self._device
            pipeline = factory(
                self._task,
                **pipeline_kwargs,
            )
            self._installation.verify()
            self._pipeline = pipeline
            version = getattr(transformers, "__version__", None)
            if isinstance(version, str) and version.strip():
                self.provider_version = version
        return self._pipeline


class HuggingFaceImageClassificationScorer(_LocalPipelineScorer):
    """Return a named class probability from a local image classifier."""

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        target_label: str,
        *,
        device: int | str | None = None,
    ) -> None:
        super().__init__(
            installation,
            "image-classification",
            expected_adapter="huggingface-image-classification",
            device=device,
        )
        if not target_label.strip():
            raise ValueError("target_label must not be blank")
        self._target_label = target_label.strip().casefold()

    def score(self, images: tuple[Image.Image, ...]) -> tuple[float, ...]:
        """Score each image by semantic label instead of class index."""
        raw = self._get_pipeline()(list(images))
        batches = _classification_batches(raw, len(images))
        values: list[float] = []
        for batch in batches:
            matches = [
                _score_value(item.get("score"))
                for item in batch
                if str(item.get("label", "")).strip().casefold() == self._target_label
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"model output does not contain target label: {self._target_label}"
                )
            values.append(matches[0])
        return tuple(values)


class HuggingFaceBlipImageTextMatchingScorer:
    """Return BLIP image-text matching probabilities from a local model."""

    provider_id = "huggingface-transformers-blip-itm"
    provider_version: str | None = None
    model_revision: str | None = None

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        *,
        device: str | None = None,
    ) -> None:
        self._installation = _require_installation(installation, "huggingface-blip-itm")
        self._model_path = self._installation.path
        self.model_id = self._installation.repository
        self.model_revision = self._installation.revision
        self.model_manifest_sha256 = self._installation.model_manifest_sha256
        self.preprocess_revision = self._installation.preprocess_revision
        self.device = device or "cpu"
        self._device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def score(self, images: tuple[Image.Image, ...], text: str) -> tuple[float, ...]:
        """Return the probability of BLIP's matched class for each image."""
        if not text.strip():
            raise ValueError("text must not be blank")
        if not images:
            return ()
        self._load()
        processor, model, torch = _require_loaded(
            self._processor,
            self._model,
            self._torch,
        )
        inputs = processor(
            images=list(images),
            text=[text] * len(images),
            padding=True,
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, self._device)
        with torch.no_grad():
            outputs = model(**inputs, use_itm_head=True)
        logits = np.asarray(outputs.itm_score.detach().cpu().numpy(), dtype=np.float64)
        if logits.shape != (len(images), 2):
            raise ValueError("BLIP ITM model must return two class logits per image-text pair")
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        return tuple(float(value) for value in probabilities[:, 1])

    def _load(self) -> None:
        if self._model is not None:
            return
        self._installation.verify()
        transformers, torch = _model_modules()
        self._processor = transformers.BlipProcessor.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        )
        self._model = transformers.BlipForImageTextRetrieval.from_pretrained(
            str(self._model_path), local_files_only=True, trust_remote_code=False
        ).eval()
        if self._device is not None:
            self._model = self._model.to(self._device)
        self._installation.verify()
        self._torch = torch
        self.provider_version = _module_version(transformers)


class HuggingFaceImagePairScorer(_LocalClipScorer):
    """Compute cosine similarity using the CLIP image tower only."""

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        *,
        device: int | str | None = None,
    ) -> None:
        normalized_device = (
            ("cpu" if device < 0 else f"cuda:{device}") if isinstance(device, int) else device
        )
        super().__init__(installation, device=normalized_device)
        self.preprocess_revision += "|clip-image-tower-cosine-v1"

    def score(self, left: Image.Image, right: Image.Image) -> float:
        """Return cosine similarity for two decoded images."""
        self._load()
        processor, model, torch = _require_loaded(self._processor, self._model, self._torch)
        inputs = processor(images=[left.convert("RGB"), right.convert("RGB")], return_tensors="pt")
        inputs = _move_inputs(inputs, self._device)
        with torch.no_grad():
            features = model.get_image_features(**inputs)
        vectors = _normalized_rows(getattr(features, "pooler_output", features))
        if vectors.shape[0] != 2:
            raise ValueError("CLIP must return exactly two image embeddings")
        return float(np.clip(np.dot(vectors[0], vectors[1]), -1.0, 1.0))


def _classification_batches(
    raw: object, expected: int
) -> tuple[tuple[Mapping[str, object], ...], ...]:
    sequence = cast(Sequence[object], raw)
    if expected == 1 and sequence and isinstance(sequence[0], Mapping):
        return (_label_scores(sequence),)
    batches = tuple(_label_scores(item) for item in sequence)
    if len(batches) != expected:
        raise ValueError("classifier must return one result batch per image")
    return batches


def _label_scores(raw: object) -> tuple[Mapping[str, object], ...]:
    sequence = cast(Sequence[object], raw)
    if not all(isinstance(item, Mapping) for item in sequence):
        raise ValueError("classifier returned an invalid label-score payload")
    return tuple(cast(Mapping[str, object], item) for item in sequence)


def _score_value(value: object) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("classifier score must be numeric")
    return float(value)


def _model_label_index(model: object, target_label: str, class_count: int) -> int:
    config = getattr(model, "config", None)
    labels = getattr(config, "id2label", None)
    if not isinstance(labels, Mapping):
        raise ValueError("aesthetics classification model must expose config.id2label")
    matches = [
        int(index)
        for index, label in labels.items()
        if str(label).strip().casefold() == target_label
    ]
    if len(matches) != 1 or not 0 <= matches[0] < class_count:
        raise ValueError(f"aesthetics model does not contain target label: {target_label}")
    return matches[0]


def _require_installation(
    installation: VerifiedModelInstallation,
    expected_adapter: str,
) -> VerifiedModelInstallation:
    if not isinstance(installation, VerifiedModelInstallation):
        raise TypeError("image model adapter requires a VerifiedModelInstallation")
    if installation.model.adapter != expected_adapter:
        raise ValueError(
            f"model adapter {installation.model.adapter} is incompatible with {expected_adapter}"
        )
    installation.verify()
    return installation


def _model_modules() -> tuple[Any, Any]:
    try:
        return importlib.import_module("transformers"), importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise HuggingFaceUnavailableError(
            "Transformers and PyTorch are unavailable; install capicove[models-dev]"
        ) from error


def _module_version(module: object) -> str | None:
    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) and version.strip() else None


def _require_loaded(
    processor: Any | None,
    model: Any | None,
    torch: Any | None,
) -> tuple[Any, Any, Any]:
    """Return initialized lazy components or fail safely in optimized Python."""
    if processor is None or model is None or torch is None:
        raise RuntimeError("Hugging Face adapter did not initialize all model components")
    return processor, model, torch


def _normalized_rows(tensor: Any) -> np.ndarray:
    values = np.asarray(tensor.detach().cpu().numpy(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("model embeddings must be a non-empty two-dimensional array")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.all(np.isfinite(values)) or not np.all(np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("model embeddings must be finite and have non-zero finite norm")
    return cast(np.ndarray, values / norms)


def _required_nonblank(name: str, value: str) -> str:
    if not value.strip():
        raise ValueError(f"{name} must not be blank")
    return value


def _move_inputs(inputs: Any, device: str | None) -> Any:
    if device is None:
        return inputs
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in inputs.items()
        }
    raise TypeError("model inputs do not support device placement")
