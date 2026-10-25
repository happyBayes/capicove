"""Direct OpenCV analyzers for deterministic image structure."""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image


@dataclass(frozen=True, slots=True)
class SubplotMetric:
    """Record grid-line evidence for one decoded image."""

    confidence: float
    horizontal_lines: int
    vertical_lines: int


@dataclass(frozen=True, slots=True)
class OpenCvSubplotAnalyzer:
    """Detect grid-like subplot layouts using edges and line segments."""

    min_horizontal_lines: int = 3
    min_vertical_lines: int = 3
    canny_threshold1: int = 70
    canny_threshold2: int = 190
    hough_threshold: int = 110
    min_line_length: int = 110
    max_line_gap: int = 18
    angle_tolerance: float = 4.0

    def analyze(self, image: Image.Image) -> SubplotMetric:
        """Return deterministic line counts and normalized grid confidence."""
        rgb = np.asarray(image.convert("RGB"))
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, self.canny_threshold1, self.canny_threshold2)
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180,
            self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap,
        )
        if lines is None:
            return SubplotMetric(0.0, 0, 0)

        horizontal_positions: list[float] = []
        vertical_positions: list[float] = []
        for raw in lines:
            x1, y1, x2, y2 = (int(value) for value in raw[0])
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            if min(abs(angle), abs(abs(angle) - 180.0)) <= self.angle_tolerance:
                horizontal_positions.append((y1 + y2) / 2)
            elif abs(abs(angle) - 90.0) <= self.angle_tolerance:
                vertical_positions.append((x1 + x2) / 2)

        horizontal_count = len(_cluster_positions(horizontal_positions, 8.0))
        vertical_count = len(_cluster_positions(vertical_positions, 8.0))
        if horizontal_count == 0 or vertical_count == 0:
            confidence = 0.0
        else:
            horizontal_score = min(horizontal_count / self.min_horizontal_lines, 1.0)
            vertical_score = min(vertical_count / self.min_vertical_lines, 1.0)
            confidence = (horizontal_score + vertical_score) / 2
        return SubplotMetric(confidence, horizontal_count, vertical_count)


def _cluster_positions(positions: list[float], tolerance: float) -> tuple[float, ...]:
    if not positions:
        return ()
    clusters: list[list[float]] = []
    for position in sorted(positions):
        if not clusters or position - clusters[-1][-1] > tolerance:
            clusters.append([position])
        else:
            clusters[-1].append(position)
    return tuple(sum(cluster) / len(cluster) for cluster in clusters)
