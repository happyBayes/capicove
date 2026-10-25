"""Generate deterministic pixel-difference regions for an image pair."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

from capicove.processing.image.invocation import check_image_deadline
from capicove.processing.image.io import load_image
from capicove.processing.image.models import (
    BoundingBox,
    DifferenceAreaResult,
    DifferenceStageEvidence,
    ImageSource,
)


@dataclass(frozen=True, slots=True)
class ImgDiffDifferenceAreaGeneratorMapper:
    """Find changed regions using OpenCV pixel difference and deterministic NMS."""

    pixel_threshold: int = 25
    min_area_ratio: float = 0.001
    iou_threshold: float = 0.5
    morphology_kernel_size: int = 5

    def __post_init__(self) -> None:
        if not 0 <= self.pixel_threshold <= 255:
            raise ValueError("pixel_threshold must be within [0, 255]")
        if not math.isfinite(self.min_area_ratio) or not 0 <= self.min_area_ratio <= 1:
            raise ValueError("min_area_ratio must be within [0, 1]")
        if not math.isfinite(self.iou_threshold) or not 0 <= self.iou_threshold <= 1:
            raise ValueError("iou_threshold must be within [0, 1]")
        if self.morphology_kernel_size <= 0 or self.morphology_kernel_size % 2 == 0:
            raise ValueError("morphology_kernel_size must be a positive odd integer")

    def map(self, sources: tuple[ImageSource, ...]) -> DifferenceAreaResult:
        """Generate ordered bounding boxes for exactly two image sources."""
        if len(sources) != 2:
            raise ValueError("difference-area generation requires exactly two images")
        left, right = (load_image(source) for source in sources)
        left_rgb = np.asarray(left.image)
        right_rgb = np.asarray(right.image)
        if right_rgb.shape[:2] != left_rgb.shape[:2]:
            raise ValueError("difference-area images must have matching dimensions")
        check_image_deadline()
        left_lab = cv2.cvtColor(left_rgb, cv2.COLOR_RGB2LAB)
        right_lab = cv2.cvtColor(right_rgb, cv2.COLOR_RGB2LAB)
        channel_difference = cv2.absdiff(left_lab, right_lab)
        difference = np.max(channel_difference, axis=2).astype(np.uint8)
        _, mask = cv2.threshold(
            difference,
            self.pixel_threshold,
            255,
            cv2.THRESH_BINARY,
        )
        changed_fraction = float(np.count_nonzero(mask)) / float(mask.size)
        kernel = np.ones(
            (self.morphology_kernel_size, self.morphology_kernel_size),
            dtype=np.uint8,
        )
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        check_image_deadline()
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        minimum_area = left.width * left.height * self.min_area_ratio
        candidates: list[BoundingBox] = []
        for contour in contours:
            check_image_deadline()
            if cv2.contourArea(contour) < minimum_area:
                continue
            x, y, width, height = cv2.boundingRect(contour)
            confidence = min(float(np.mean(difference[y : y + height, x : x + width])) / 255, 1.0)
            candidates.append(
                BoundingBox(
                    x=float(x),
                    y=float(y),
                    width=float(width),
                    height=float(height),
                    confidence=confidence,
                )
            )
        boxes = _non_maximum_suppression(tuple(candidates), self.iou_threshold)
        check_image_deadline()
        stage_evidence = (
            DifferenceStageEvidence(
                "pair_pre_gate",
                changed_fraction,
                changed_fraction > 0,
                "changed pixels detected" if changed_fraction > 0 else "no changed pixels",
            ),
            DifferenceStageEvidence(
                "candidate_proposal",
                float(len(contours)),
                bool(contours),
                "external contours proposed",
            ),
            DifferenceStageEvidence(
                "crop_dissimilarity",
                float(len(candidates)),
                bool(candidates),
                "candidates retained by area and mean color difference",
            ),
            DifferenceStageEvidence(
                "non_maximum_suppression",
                float(len(boxes)),
                bool(boxes),
                "overlapping candidates reduced deterministically",
            ),
        )
        return DifferenceAreaResult(
            source_hashes=(left.sha256, right.sha256),
            boxes=boxes,
            provider_id="opencv-pixel-difference",
            provider_version=cv2.__version__,
            operation_version=1,
            parameters=(
                ("color_space", "CIELAB"),
                ("iou_threshold", self.iou_threshold),
                ("min_area_ratio", self.min_area_ratio),
                ("morphology_kernel_size", self.morphology_kernel_size),
                ("pixel_threshold", self.pixel_threshold),
            ),
            stage_evidence=stage_evidence,
        )


def _non_maximum_suppression(
    boxes: tuple[BoundingBox, ...],
    iou_threshold: float,
) -> tuple[BoundingBox, ...]:
    ordered = sorted(
        boxes,
        key=lambda box: (-box.confidence, -(box.width * box.height), box.y, box.x),
    )
    kept: list[BoundingBox] = []
    for box in ordered:
        if all(_intersection_over_union(box, existing) <= iou_threshold for existing in kept):
            kept.append(box)
    return tuple(sorted(kept, key=lambda box: (box.y, box.x, box.width, box.height)))


def _intersection_over_union(left: BoundingBox, right: BoundingBox) -> float:
    x1 = max(left.x, right.x)
    y1 = max(left.y, right.y)
    x2 = min(left.x + left.width, right.x + right.width)
    y2 = min(left.y + left.height, right.y + right.height)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = left.width * left.height + right.width * right.height - intersection
    return intersection / union if union else 0.0
