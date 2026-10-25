"""Tests for lazy local Hugging Face difference captioning."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from capicove.processing.image._adapters.huggingface_caption import (
    HuggingFaceDifferenceCaptionProvider,
)
from tests.unit.processing.image.model_installations import verified_model_installation


def test_difference_caption_provider_uses_structured_pipeline_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    caption_model = tmp_path / "caption"
    difference_model = tmp_path / "difference"
    caption_model.mkdir()
    difference_model.mkdir()
    created_tasks: list[str] = []
    created_kwargs: list[dict[str, object]] = []
    caption_inputs: list[dict[str, object]] = []
    difference_inputs: list[dict[str, object]] = []

    def pipeline(task: str, **kwargs: object) -> Callable[..., object]:
        created_kwargs.append(dict(kwargs))
        created_tasks.append(task)
        if kwargs["model"] == "caption-model":

            def caption(**inputs: object) -> object:
                caption_inputs.append(dict(inputs))
                return [{"generated_text": "a bright object"}]

            return caption

        def describe(**inputs: object) -> object:
            difference_inputs.append(dict(inputs))
            return {
                "generated_text": [
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "the object became dark"}],
                    }
                ]
            }

        return describe

    def load_local(path: str, **kwargs: object) -> object:
        assert kwargs == {"local_files_only": True, "trust_remote_code": False}
        return "caption-model" if path == str(caption_model.resolve()) else "difference-model"

    monkeypatch.setattr(
        "capicove.processing.image._adapters.huggingface_caption.importlib.import_module",
        lambda name: SimpleNamespace(
            pipeline=pipeline,
            __version__="5.0",
            AutoModelForImageTextToText=SimpleNamespace(from_pretrained=load_local),
            AutoProcessor=SimpleNamespace(from_pretrained=load_local),
        ),
    )
    caption_installation = verified_model_installation(
        caption_model,
        adapter="huggingface-region-caption",
        preprocess_revision="transformers-caption-pipelines-v2",
    )
    difference_installation = verified_model_installation(
        difference_model,
        adapter="huggingface-difference-caption",
        preprocess_revision="transformers-caption-pipelines-v2",
    )
    provider = HuggingFaceDifferenceCaptionProvider(
        caption_installation,
        difference_installation,
        device=0,
    )
    image = Image.new("RGB", (4, 4), "white")

    assert created_tasks == []
    assert provider.caption(image) == "a bright object"
    assert provider.describe_difference(image, image, "bright", "dark") == (
        "the object became dark"
    )
    assert created_tasks == ["image-text-to-text", "image-text-to-text"]
    assert caption_inputs[0]["images"] == image
    assert caption_inputs[0]["return_full_text"] is False
    assert "images" not in difference_inputs[0]
    assert caption_inputs[0]["text"] == ""
    assert "image" not in difference_inputs[0]
    assert difference_inputs[0]["return_full_text"] is False
    messages = difference_inputs[0]["text"]
    assert isinstance(messages, list)
    assert [item["type"] for item in messages[0]["content"]] == ["image", "image", "text"]
    assert [block["image"] for block in messages[0]["content"][:2]] == [image, image]
    assert all("model_kwargs" not in kwargs and "processor" in kwargs for kwargs in created_kwargs)
    assert provider.model_revision == f"{'a' * 40}|{'a' * 40}"
    expected_manifest = hashlib.sha256(
        (
            f"{caption_installation.model_manifest_sha256}:"
            f"{difference_installation.model_manifest_sha256}"
        ).encode("ascii")
    ).hexdigest()
    assert provider.model_manifest_sha256 == expected_manifest


@pytest.mark.parametrize(
    "payload",
    [
        None,
        3,
        {},
        [],
        [{"generated_text": " "}],
        [{"generated_text": [{"role": "user", "content": "Echoed prompt"}]}],
    ],
)
def test_caption_rejects_invalid_or_prompt_only_output(payload: object) -> None:
    from capicove.processing.image._adapters.huggingface_caption import _extract_text

    with pytest.raises(ValueError):
        _extract_text(payload, "generated_text", maximum_length=100)
