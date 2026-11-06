"""Capability-owned ports for local video quality execution."""

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol

from capicove.processing.video.quality.video_quality_contracts import QualityLimits, QualityRequest


@dataclass(frozen=True, slots=True)
class QualitySource:
    """A verified private input plus its original immutable identity."""

    staged_path: Path
    original_path: Path
    sha256: str
    size_bytes: int
    demuxer: str


class QualityIntake(Protocol):
    """Authorize and stage a bounded local source without modifying it."""

    def stage(self, path: Path, deadline: datetime) -> AbstractContextManager[QualitySource]: ...


class QualityBackend(Protocol):
    """Probe and inspect the same staged source with composition-owned providers."""

    def probe(self, source: QualitySource, deadline: datetime) -> dict[str, Any]: ...

    def scan(
        self,
        source: QualitySource,
        selection: Mapping[str, Any],
        request: QualityRequest,
        limits: QualityLimits,
        deadline: datetime,
    ) -> dict[str, Any]: ...
