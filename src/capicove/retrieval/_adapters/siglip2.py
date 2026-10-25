"""Lazy, local-only Hugging Face SigLIP2 retrieval provider."""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping
from typing import Any, cast

import numpy as np
from PIL import Image

from capicove.model.receipt import VerifiedModelInstallation
from capicove.retrieval.models import ProviderMatch


class Siglip2UnavailableError(RuntimeError):
    """Raised when optional SigLIP2 runtime dependencies are unavailable."""


class HuggingFaceSiglip2Provider:
    """Provide aligned vectors and ranking with a verified local SigLIP2 model."""

    provider_id = "huggingface-transformers-siglip2"
    provider_version: str | None = None

    def __init__(
        self,
        installation: VerifiedModelInstallation,
        *,
        device: str = "auto",
    ) -> None:
        if not isinstance(installation, VerifiedModelInstallation):
            raise TypeError("SigLIP2 provider requires a VerifiedModelInstallation")
        if installation.model.adapter != "huggingface-siglip2-retrieval":
            raise ValueError(
                f"model adapter {installation.model.adapter} is incompatible with "
                "huggingface-siglip2-retrieval"
            )
        if device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError("device must be one of: auto, cpu, cuda, mps")
        installation.verify()
        self._installation = installation
        self._requested_device = device
        self.model_id = installation.repository
        self.model_revision = installation.revision
        self.model_manifest_sha256 = installation.model_manifest_sha256
        self.preprocess_revision = installation.preprocess_revision
        self.device = device
        self._processor: Any | None = None
        self._model: Any | None = None
        self._torch: Any | None = None

    def close(self) -> None:
        """Drop owned runtime references after the caller finishes its lease."""
        self._processor = None
        self._model = None
        self._torch = None

    def load(self) -> None:
        """Load optional dependencies and verified local model files once."""
        if self._model is not None:
            return
        self._installation.verify()
        try:
            transformers = importlib.import_module("transformers")
            torch = importlib.import_module("torch")
        except ModuleNotFoundError as error:
            raise Siglip2UnavailableError(
                "SigLIP2 requires PyTorch and Transformers; install capicove[model-serving]"
            ) from error
        device = _select_device(torch, self._requested_device)
        load_options = {
            "local_files_only": True,
            "trust_remote_code": False,
        }
        processor = transformers.AutoProcessor.from_pretrained(
            str(self._installation.path),
            **load_options,
        )
        model = transformers.AutoModel.from_pretrained(
            str(self._installation.path),
            **load_options,
        ).eval()
        model = model.to(device)
        self._installation.verify()
        self._processor = processor
        self._model = model
        self._torch = torch
        self.device = device
        version = getattr(transformers, "__version__", None)
        if isinstance(version, str) and version.strip():
            self.provider_version = version

    def rank(
        self,
        image: Image.Image,
        texts: tuple[str, ...],
    ) -> tuple[ProviderMatch, ...]:
        """Return model-native logits and sigmoid match probabilities."""
        processor, model, torch = self._loaded()
        inputs = processor(
            images=image,
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, self.device)
        with torch.inference_mode():
            outputs = model(**inputs)
        logits = _array(outputs.logits_per_image)
        if logits.shape != (1, len(texts)):
            raise ValueError("SigLIP2 must return one logit per candidate text")
        return tuple(
            ProviderMatch(
                logit=float(value),
                match_probability=_sigmoid(float(value)),
            )
            for value in logits[0]
        )

    def embed_images(
        self,
        images: tuple[Image.Image, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Return L2-normalized image embeddings."""
        if not images:
            return ()
        processor, model, torch = self._loaded()
        inputs = processor(images=list(images), return_tensors="pt")
        inputs = _move_inputs(inputs, self.device)
        with torch.inference_mode():
            vectors = model.get_image_features(**inputs)
        return _normalized_vectors(_feature_tensor(vectors), len(images))

    def embed_texts(
        self,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        """Return L2-normalized text embeddings."""
        if not texts:
            return ()
        processor, model, torch = self._loaded()
        inputs = processor(
            text=list(texts),
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        inputs = _move_inputs(inputs, self.device)
        with torch.inference_mode():
            vectors = model.get_text_features(**inputs)
        return _normalized_vectors(_feature_tensor(vectors), len(texts))

    def _loaded(self) -> tuple[Any, Any, Any]:
        self.load()
        if self._processor is None or self._model is None or self._torch is None:
            raise RuntimeError("SigLIP2 provider did not initialize all components")
        return self._processor, self._model, self._torch


def _select_device(torch: Any, requested: str) -> str:
    if requested == "auto":
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise Siglip2UnavailableError("CUDA was requested but is unavailable")
    if requested == "mps":
        mps = getattr(getattr(torch, "backends", None), "mps", None)
        if mps is None or not mps.is_available():
            raise Siglip2UnavailableError("MPS was requested but is unavailable")
    return requested


def _move_inputs(inputs: Any, device: str) -> Any:
    if hasattr(inputs, "to"):
        return inputs.to(device)
    if isinstance(inputs, Mapping):
        return {
            name: value.to(device) if hasattr(value, "to") else value
            for name, value in inputs.items()
        }
    raise TypeError("processor inputs do not support device placement")


def _array(tensor: Any) -> np.ndarray:
    values = np.asarray(tensor.detach().float().cpu().numpy(), dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("SigLIP2 output must contain finite values")
    return cast(np.ndarray, values)


def _feature_tensor(output: Any) -> Any:
    """Accept feature tensors from Transformers 4 or pooled outputs from 5."""
    pooled = getattr(output, "pooler_output", None)
    return pooled if pooled is not None else output


def _normalized_vectors(
    tensor: Any,
    expected_rows: int,
) -> tuple[tuple[float, ...], ...]:
    values = _array(tensor)
    if values.ndim != 2 or values.shape[0] != expected_rows or values.shape[1] == 0:
        raise ValueError("SigLIP2 embeddings have an unexpected shape")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("SigLIP2 embeddings must have non-zero norm")
    normalized = values / norms
    return tuple(tuple(float(value) for value in row) for row in normalized)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)
