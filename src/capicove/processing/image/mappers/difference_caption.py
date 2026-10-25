"""Generate structured captions for image-pair difference regions."""

from __future__ import annotations

import math
from difflib import SequenceMatcher

from capicove.processing.image._validation import require_non_blank
from capicove.processing.image.invocation import (
    check_image_deadline,
    consume_image_model_calls,
)
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    BoundingBox,
    DifferenceAreaResult,
    DifferenceCaptionRegion,
    DifferenceCaptionResult,
    ImageSource,
)
from capicove.processing.image.ports import DifferenceCaptionProvider, ImageTextScorer


_MAX_REGION_CAPTION_LENGTH = 2048
_MAX_DIFFERENCE_CAPTION_LENGTH = 4096


class ImgDiffDifferenceCaptionGeneratorMapper:
    """Caption paired in-memory crops without random files or string parsing."""

    def __init__(
        self,
        provider: DifferenceCaptionProvider,
        *,
        validator: ImageTextScorer,
        min_caption_match: float = 0.1,
        min_caption_difference: float = 0.05,
    ) -> None:
        if not math.isfinite(min_caption_match) or not -1 <= min_caption_match <= 1:
            raise ValueError("min_caption_match must be within [-1, 1]")
        if not math.isfinite(min_caption_difference) or not 0 <= min_caption_difference <= 1:
            raise ValueError("min_caption_difference must be within [0, 1]")
        self._provider = provider
        self._validator = validator
        self._min_caption_match = min_caption_match
        self._min_caption_difference = min_caption_difference

    def map(
        self,
        sources: tuple[ImageSource, ...],
        areas: DifferenceAreaResult,
    ) -> DifferenceCaptionResult:
        """Generate one structured caption record per validated difference box."""
        if len(sources) != 2:
            raise ValueError("difference-caption generation requires exactly two images")
        left, right = (load_image(source) for source in sources)
        source_hashes = (left.sha256, right.sha256)
        if source_hashes != areas.source_hashes:
            raise ValueError("difference areas do not belong to the supplied image pair")
        right_image = right.image
        if right_image.size != left.image.size:
            raise ValueError("difference-caption images must have matching dimensions")

        regions: list[DifferenceCaptionRegion] = []
        for box in areas.boxes:
            check_image_deadline()
            crop_box = _crop_coordinates(box, left.width, left.height)
            left_crop = left.image.crop(crop_box)
            right_crop = right_image.crop(crop_box)
            consume_image_model_calls()
            left_caption = self._provider.caption(left_crop).strip()
            check_image_deadline()
            consume_image_model_calls()
            right_caption = self._provider.caption(right_crop).strip()
            check_image_deadline()
            require_non_blank("left_caption", left_caption)
            require_non_blank("right_caption", right_caption)
            if len(left_caption) > _MAX_REGION_CAPTION_LENGTH:
                raise ValueError("left caption exceeds the output-length limit")
            if len(right_caption) > _MAX_REGION_CAPTION_LENGTH:
                raise ValueError("right caption exceeds the output-length limit")
            left_match_score = self._caption_match_score(left_crop, left_caption)
            if left_match_score < self._min_caption_match:
                continue
            right_match_score = self._caption_match_score(right_crop, right_caption)
            if right_match_score < self._min_caption_match:
                continue
            difference_score = (
                1.0
                - SequenceMatcher(
                    None,
                    left_caption.casefold(),
                    right_caption.casefold(),
                ).ratio()
            )
            if difference_score < self._min_caption_difference:
                continue
            consume_image_model_calls()
            difference_caption = self._provider.describe_difference(
                left_crop,
                right_crop,
                left_caption,
                right_caption,
            ).strip()
            check_image_deadline()
            require_non_blank("difference_caption", difference_caption)
            if len(difference_caption) > _MAX_DIFFERENCE_CAPTION_LENGTH:
                raise ValueError("difference caption exceeds the output-length limit")
            regions.append(
                DifferenceCaptionRegion(
                    box=box,
                    left_caption=left_caption,
                    right_caption=right_caption,
                    difference_caption=difference_caption,
                    left_match_score=left_match_score,
                    right_match_score=right_match_score,
                    caption_difference_score=difference_score,
                )
            )
        return DifferenceCaptionResult(
            source_hashes=source_hashes,
            regions=tuple(regions),
            provider_id=self._provider.provider_id,
            provider_version=self._provider.provider_version,
            model_id=self._provider.model_id,
            model_revision=self._provider.model_revision,
            model_manifest_sha256=self._provider.model_manifest_sha256,
            preprocess_revision=self._provider.preprocess_revision,
            device=self._provider.device,
            operation_version=1,
            parameters=(
                ("caption_gate", "sequence-matcher-v1"),
                ("min_caption_difference", self._min_caption_difference),
                ("min_caption_match", self._min_caption_match),
                ("prompt_revision", "difference-caption-v2-multi-image"),
                ("validator_model_id", self._validator.model_id),
                ("validator_model_revision", self._validator.model_revision),
                ("validator_model_manifest_sha256", self._validator.model_manifest_sha256),
                ("validator_preprocess_revision", self._validator.preprocess_revision),
                ("validator_device", self._validator.device),
                ("validator_provider_id", self._validator.provider_id),
                ("validator_provider_version", self._validator.provider_version),
            ),
        )

    def _caption_match_score(self, crop: object, caption: str) -> float:
        consume_image_model_calls()
        scores = self._validator.score((crop,), caption)  # type: ignore[arg-type]
        if len(scores) != 1:
            raise ValueError("caption validator must return exactly one score")
        score = float(scores[0])
        if not math.isfinite(score) or not -1 <= score <= 1:
            raise ValueError("caption validation score must be within [-1, 1]")
        return score


def _crop_coordinates(box: BoundingBox, width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, math.floor(box.x))
    top = max(0, math.floor(box.y))
    right = min(width, math.ceil(box.x + box.width))
    bottom = min(height, math.ceil(box.y + box.height))
    if right <= left or bottom <= top:
        raise ValueError("difference box is outside the aligned image bounds")
    return left, top, right, bottom
