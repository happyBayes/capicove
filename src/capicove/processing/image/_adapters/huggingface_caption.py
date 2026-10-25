"""Lazy, local-only Hugging Face difference-caption adapter."""

from __future__ import annotations

import hashlib
import importlib
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

from PIL import Image

from capicove.model.receipt import VerifiedModelInstallation


class HuggingFaceDifferenceCaptionProvider:
    """Caption paired crops with local structured Transformers pipelines."""

    provider_id = "huggingface-transformers"
    provider_version: str | None = None
    model_revision: str | None = None

    def __init__(
        self,
        caption_installation: VerifiedModelInstallation,
        difference_installation: VerifiedModelInstallation,
        *,
        device: int | str | None = None,
    ) -> None:
        self._caption_installation = _require_installation(
            caption_installation,
            "huggingface-region-caption",
        )
        self._difference_installation = _require_installation(
            difference_installation,
            "huggingface-difference-caption",
        )
        self._caption_model_path = self._caption_installation.path
        self._difference_model_path = self._difference_installation.path
        self.model_id = (
            f"{self._caption_installation.repository}|{self._difference_installation.repository}"
        )
        self.model_revision = (
            f"{self._caption_installation.revision}|{self._difference_installation.revision}"
        )
        model_manifests = (
            f"{self._caption_installation.model_manifest_sha256}:"
            f"{self._difference_installation.model_manifest_sha256}"
        )
        self.model_manifest_sha256 = hashlib.sha256(model_manifests.encode("ascii")).hexdigest()
        self.preprocess_revision = (
            f"{self._caption_installation.preprocess_revision}|"
            f"{self._difference_installation.preprocess_revision}|unconditional-embedded-chat-v1"
        )
        self.device = str(device) if device is not None else "cpu"
        self._device = device
        self._caption_pipeline: Callable[..., Any] | None = None
        self._difference_pipeline: Callable[..., Any] | None = None

    def caption(self, image: Image.Image) -> str:
        """Return the structured generated text for one crop."""
        output = self._captioner()(
            images=image.convert("RGB"),
            text="",
            max_new_tokens=256,
            return_full_text=False,
        )
        return _extract_text(output, "generated_text", maximum_length=2048)

    def describe_difference(
        self,
        left: Image.Image,
        right: Image.Image,
        left_caption: str,
        right_caption: str,
    ) -> str:
        """Describe a change with a model that natively accepts two images."""
        instruction = (
            "The first image is the before region and the second image is the after region. "
            "Describe only the visible change from the first image to the second image. "
            f"Left caption: {left_caption[:512]}. Right caption: {right_caption[:512]}."
        )
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": left.convert("RGB")},
                    {"type": "image", "image": right.convert("RGB")},
                    {"type": "text", "text": instruction},
                ],
            }
        ]
        output = self._difference_captioner()(
            text=messages,
            max_new_tokens=256,
            return_full_text=False,
        )
        return _extract_text(output, "generated_text", maximum_length=4096)

    def _captioner(self) -> Callable[..., Any]:
        if self._caption_pipeline is None:
            self._caption_pipeline = self._build_pipeline(
                "image-text-to-text",
                self._caption_installation,
            )
        return self._caption_pipeline

    def _difference_captioner(self) -> Callable[..., Any]:
        if self._difference_pipeline is None:
            self._difference_pipeline = self._build_pipeline(
                "image-text-to-text",
                self._difference_installation,
            )
        return self._difference_pipeline

    def _build_pipeline(
        self,
        task: str,
        installation: VerifiedModelInstallation,
    ) -> Callable[..., Any]:
        installation.verify()
        try:
            transformers = importlib.import_module("transformers")
        except ModuleNotFoundError as error:
            raise RuntimeError(
                "Transformers is unavailable; install capicove[models-dev]"
            ) from error
        factory = cast(Callable[..., Callable[..., Any]], transformers.pipeline)
        pipeline_kwargs: dict[str, object] = {
            "model": transformers.AutoModelForImageTextToText.from_pretrained(
                str(installation.path), local_files_only=True, trust_remote_code=False
            ),
            "processor": transformers.AutoProcessor.from_pretrained(
                str(installation.path), local_files_only=True, trust_remote_code=False
            ),
            "trust_remote_code": False,
        }
        if self._device is not None:
            pipeline_kwargs["device"] = self._device
        pipeline = factory(
            task,
            **pipeline_kwargs,
        )
        installation.verify()
        version = getattr(transformers, "__version__", None)
        if isinstance(version, str) and version.strip():
            self.provider_version = version
        return pipeline


def _require_installation(
    installation: VerifiedModelInstallation,
    expected_adapter: str,
) -> VerifiedModelInstallation:
    if not isinstance(installation, VerifiedModelInstallation):
        raise TypeError("caption adapter requires a VerifiedModelInstallation")
    if installation.model.adapter != expected_adapter:
        raise ValueError(
            f"model adapter {installation.model.adapter} is incompatible with {expected_adapter}"
        )
    installation.verify()
    return installation


def _extract_text(output: object, key: str, *, maximum_length: int) -> str:
    item: Mapping[str, object]
    if isinstance(output, Mapping):
        item = output
    else:
        if not isinstance(output, Sequence) or isinstance(output, (str, bytes, bytearray)):
            raise ValueError("caption model returned an invalid structured payload")
        sequence = cast(Sequence[object], output)
        if len(sequence) != 1 or not isinstance(sequence[0], Mapping):
            raise ValueError("caption model returned an invalid structured payload")
        item = cast(Mapping[str, object], sequence[0])
    value = _generated_text(item.get(key))
    if value is None or not value.strip():
        raise ValueError(f"caption model output is missing nonblank {key}")
    normalized = value.strip()
    if len(normalized) > maximum_length:
        raise ValueError(f"caption model output exceeds the {key} length limit")
    return normalized


def _generated_text(value: object) -> str | None:
    if isinstance(value, str):
        return value
    if not isinstance(value, Sequence):
        return None
    for message in reversed(value):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, Sequence):
            continue
        fragments = [
            str(block["text"])
            for block in content
            if isinstance(block, Mapping)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
        ]
        if fragments:
            return " ".join(fragments)
    return None
