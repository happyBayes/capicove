"""Opt-in real local models; no inference substitutions or automatic downloads.

Set CAPICOVE_MODEL_QUALIFICATION to an operator-owned JSON file containing
model_root, digests (model key to trusted SHA-256), and data_root (the authored
evaluation fixture directory). Missing configuration is an explicit skip.
"""

from __future__ import annotations

import json
import math
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from capicove.model.manifest import load_packaged_manifest
from capicove.model.receipt import VerifiedModelInstallation, verify_model_installation


pytestmark = pytest.mark.model_full


@pytest.fixture(scope="module")
def configuration() -> dict[str, Any]:
    path = os.environ.get("CAPICOVE_MODEL_QUALIFICATION")
    if not path:
        pytest.skip("Set CAPICOVE_MODEL_QUALIFICATION to qualify verified local HF models")
    return dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _installation(configuration: dict[str, Any], key: str) -> VerifiedModelInstallation:
    manifest = load_packaged_manifest()
    model = next(model for model in manifest.models if model.key == key)
    return verify_model_installation(
        Path(configuration["model_root"]) / key,
        model,
        manifest_sha256=manifest.source_sha256,
        expected_model_manifest_sha256=configuration["digests"][key],
    )


def _photo(configuration: dict[str, Any], key: str = "cat") -> Image.Image:
    manifest = json.loads((Path(configuration["data_root"]) / "manifest.json").read_text())
    record = next(item for item in manifest["photos"] if item["key"] == key)
    with Image.open(record["path"]) as image:
        return image.convert("RGB")


@pytest.mark.parametrize(
    "key, label", [("nsfw-falconsai", "nsfw"), ("watermark-siglip2", "Watermark")]
)
def test_real_classification(configuration: dict[str, Any], key: str, label: str) -> None:
    from capicove.processing.image._adapters.huggingface import HuggingFaceImageClassificationScorer

    scorer = HuggingFaceImageClassificationScorer(
        _installation(configuration, key),
        target_label=label,
        device=configuration.get("device", "cpu"),
    )
    scores = scorer.score((_photo(configuration),))
    assert len(scores) == 1 and math.isfinite(scores[0]) and 0 <= scores[0] <= 1


def test_real_clip_pair_and_text(configuration: dict[str, Any]) -> None:
    from capicove.processing.image._adapters.huggingface import (
        HuggingFaceClipImageTextSimilarityScorer,
        HuggingFaceImagePairScorer,
    )

    installation = _installation(configuration, "clip-vit-b32")
    image = _photo(configuration)
    pair = HuggingFaceImagePairScorer(installation, device=configuration.get("device", "cpu"))
    assert pair.score(image, image) == pytest.approx(1.0, abs=1e-5)
    assert pair.score(image, _photo(configuration, "coffee")) < 0.95
    text = HuggingFaceClipImageTextSimilarityScorer(
        installation, device=configuration.get("device", "cpu")
    )
    assert text.score((image,), "a photo of a cat")[0] > text.score((image,), "a cup of coffee")[0]


def test_real_captions(configuration: dict[str, Any]) -> None:
    from capicove.processing.image._adapters.huggingface_caption import (
        HuggingFaceDifferenceCaptionProvider,
    )

    provider = HuggingFaceDifferenceCaptionProvider(
        _installation(configuration, "blip-caption-base"),
        _installation(configuration, "smolvlm-256m-instruct"),
        device=configuration.get("device", "cpu"),
    )
    image = _photo(configuration)
    caption = provider.caption(image)
    assert "cat" in caption.lower()
    # This checks inference completion, not semantic difference-description quality.
    assert provider.describe_difference(
        image, _photo(configuration, "coffee"), caption, "a cup of coffee"
    ).strip()


def test_real_aesthetics_and_itm(configuration: dict[str, Any]) -> None:
    from capicove.processing.image._adapters.huggingface import (
        HuggingFaceAestheticsScorer,
        HuggingFaceBlipImageTextMatchingScorer,
    )

    image = _photo(configuration)
    aesthetics = HuggingFaceAestheticsScorer(
        _installation(configuration, "aesthetics-cafe"), target_label="aesthetic"
    )
    assert 0 <= aesthetics.score((image,))[0] <= 1
    itm = HuggingFaceBlipImageTextMatchingScorer(_installation(configuration, "blip-itm-coco"))
    assert itm.score((image,), "a cat")[0] > itm.score((image,), "a cup of coffee")[0]


@pytest.mark.parametrize(
    "kind, key, options, filename, text, keep",
    [
        (
            "nsfw",
            "nsfw-falconsai",
            {"target_label": "nsfw", "max_score": 0.5},
            "testsrc_10s.mp4",
            None,
            True,
        ),
        (
            "watermark",
            "watermark-siglip2",
            {"target_label": "Watermark", "max_probability": 0.8},
            "testsrc_10s.mp4",
            None,
            True,
        ),
        ("aesthetics", "aesthetics-cafe", {"min_score": 0.0}, "testsrc_10s.mp4", None, True),
        (
            "frame_text_similarity",
            "clip-vit-b32",
            {"min_score": 0.245},
            "text_overlay_4s.mp4",
            "black text on a white background",
            True,
        ),
        (
            "frame_text_similarity",
            "clip-vit-b32",
            {"min_score": 0.245},
            "static_6s.mp4",
            "a photo of a cat",
            False,
        ),
    ],
)
def test_real_video_cli(
    configuration: dict[str, Any],
    kind: str,
    key: str,
    options: dict[str, object],
    filename: str,
    text: str | None,
    keep: bool,
) -> None:
    installation = _installation(configuration, key)
    # The standalone CLI authorizes the criteria directory as a model read root.
    criteria_file = tempfile.NamedTemporaryFile(
        prefix="capicove-qualification-",
        suffix=".json",
        dir=configuration["criteria_root"],
        delete=False,
    )
    criteria = Path(criteria_file.name)
    criteria_file.close()
    try:
        criteria.write_text(
            json.dumps(
                [
                    {
                        "type": kind,
                        "criterion_id": kind,
                        "model": {"reference": str(installation.path)},
                        "sampling": {"method": "uniform", "frame_count": 3},
                        **options,
                    }
                ]
            )
        )
        command = [
            sys.executable,
            "-m",
            "capicove",
            "video",
            "filter",
            "--input",
            str(Path(configuration["data_root"]) / "video" / filename),
            "--criteria",
            str(criteria),
            "--output",
            "json",
        ]
        if text:
            command.extend(["--text", text])
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=180, check=False
        )
        assert completed.returncode == 0, completed.stderr + completed.stdout
        payload = json.loads(completed.stdout)
        assert payload["status"] == "succeeded", payload
        assert payload["result"]["keep"] is keep, payload
    finally:
        criteria.unlink(missing_ok=True)
