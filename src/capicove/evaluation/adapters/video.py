"""Video adapter: replay frozen authored lane candidates through production fusion.

The frozen fixture carries authored candidates for every lane, so no encoder,
text backend or media is required. Fusion uses ``weighted_video_rrf`` with the
``video-local-ann`` lane weights, and segment queries use the production
``TemporalCandidateAssembler``. Results measure fusion and temporal mapping only.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from capicove.contracts import (
    CharacterRange,
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    VectorNormalization,
)
from capicove.evaluation.adapters._common import profile_identity
from capicove.evaluation.harness import (
    EvalDataset,
    EvalQuery,
    EvaluationError,
    QueryOutcome,
    RankedItem,
    RunOptions,
)
from capicove.retrieval.temporal import TemporalCandidateAssembler
from capicove.retrieval.video_fusion import CandidateLane, VideoCandidate, weighted_video_rrf
from capicove.retrieval.video_profiles import VideoRetrievalProfile


_FIXTURE_SPACE = EmbeddingSpaceRef(
    space_id="fixture-visual-v1",
    provider_id="capicove.evaluation.fixture",
    model_revision="authored",
    preprocessing_version="authored-no-model-v1",
    dimension=2,
    normalization=VectorNormalization.L2,
    distance_metric=DistanceMetric.INNER_PRODUCT,
    query_modalities=frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
    target_modalities=frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
)
_LANE_TARGETS = {
    "keyframe_visual": TargetType.KEYFRAME,
    "segment_visual": TargetType.VIDEO_SEGMENT,
}


class _Catalog:
    """Immutable units and relations built solely from frozen corpus rows."""

    def __init__(self, rows: list[dict[str, Any]], offset: int, manifest_sha256: str) -> None:
        self.unit_manifest_sha256 = manifest_sha256
        self.units: dict[str, SearchableUnit] = {}
        self.relations: dict[str, tuple[str, ...]] = {}
        for row in rows:
            target = TargetType(row["type"])
            interval = None
            if target is not TargetType.KEYFRAME:
                interval = TemporalRange(row["start_ms"] + offset, row["end_ms"] + offset)
            locator = None
            if target is TargetType.TEXT_CHUNK:
                locator = TextChunkLocator(
                    TextOrigin(row["origin"]), CharacterRange(0, 10), time_range=interval
                )
            self.units[row["id"]] = SearchableUnit(
                unit_id=row["id"],
                unit_type=target,
                source_hash="a" * 64,
                artifact_ref=row["artifact"],
                evidence_ref=f"evidence:{row['id']}",
                representation_refs=(f"representation:{row['id']}",),
                processing_run_ref="fixture",
                processing_version="video-retrieval-v1",
                processing_fingerprint=ProcessingFingerprint("b" * 64),
                embedding_spaces=(_FIXTURE_SPACE,),
                timestamp_ms=row["timestamp_ms"] + offset
                if target is TargetType.KEYFRAME
                else None,
                time_range=interval if target is TargetType.VIDEO_SEGMENT else None,
                text_locator=locator,
            )
            self.relations[row["id"]] = tuple(row["segments"])
        for key, targets in self.relations.items():
            for target_id in targets:
                segment = self.units.get(target_id)
                if (
                    segment is None
                    or segment.unit_type is not TargetType.VIDEO_SEGMENT
                    or segment.artifact_ref != self.units[key].artifact_ref
                ):
                    raise EvaluationError("unknown or cross-source segment relation")

    def get(self, key: str) -> SearchableUnit | None:
        return self.units.get(key)

    def get_many(self, keys: tuple[str, ...]) -> tuple[SearchableUnit, ...]:
        return tuple(self.units[key] for key in keys)

    def segments_for_keyframe(self, key: str) -> tuple[str, ...]:
        return self.relations[key]

    def segments_for_keyframes(self, keys: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        return {key: self.relations[key] for key in keys}

    def segment_range(self, key: str) -> TemporalRange:
        interval = self.units[key].time_range
        if interval is None:
            raise EvaluationError("segment range required")
        return interval

    def segments_for_text(self, key: str) -> tuple[tuple[str, int], ...]:
        locator = self.units[key].text_locator
        if locator is None or locator.time_range is None:
            raise EvaluationError("temporal text locator required")
        interval = locator.time_range
        return tuple(
            (
                target,
                min(interval.end_ms, self.segment_range(target).end_ms)
                - max(interval.start_ms, self.segment_range(target).start_ms),
            )
            for target in self.relations[key]
        )

    def segments_for_texts(self, keys: tuple[str, ...]) -> dict[str, tuple[tuple[str, int], ...]]:
        return {key: self.segments_for_text(key) for key in keys}


@dataclass(slots=True)
class _PreparedVideo:
    corpus: list[dict[str, Any]]
    profile: VideoRetrievalProfile
    identity: Mapping[str, Any]

    def search(self, query: EvalQuery, *, k: int) -> QueryOutcome:
        row = query.payload["query"]
        offset = row["offset_ms"]
        if type(offset) is not int or offset < 0:
            raise EvaluationError("invalid timeline offset")
        catalog = _Catalog(self.corpus, offset, self.identity["index"]["unit_manifest_sha256"])
        fused = weighted_video_rrf(self._lanes(row, catalog), rrf_k=self.profile.rrf_k)
        target = TargetType(row["target"])
        if target is TargetType.KEYFRAME:
            ordered = [item.unit_id for item in fused]
        else:
            assembled = TemporalCandidateAssembler(catalog).assemble(
                fused=fused, maximum_candidates=self.profile.maximum_temporal_candidates
            )
            ordered = [item.unit.unit_id for item in assembled]
        judged = set(query.judgments)
        ranked = tuple(RankedItem(key, (key,) if key in judged else ()) for key in ordered[:k])
        return QueryOutcome(ranked, len(ordered), tuple(sorted(judged.intersection(ordered))))

    def _lanes(self, row: Mapping[str, Any], catalog: _Catalog) -> tuple[CandidateLane, ...]:
        weights = {lane.lane_id: lane.weight for lane in self.profile.lanes}
        lanes = []
        for lane in row["lanes"]:
            if set(lane) != {"id", "candidates"} or lane["id"] not in weights:
                raise EvaluationError("unknown or malformed fixture lane")
            expected = _LANE_TARGETS.get(lane["id"], TargetType.TEXT_CHUNK)
            candidates = []
            for candidate in lane["candidates"]:
                if (
                    set(candidate) != {"id", "rank", "score"}
                    or candidate["id"] not in catalog.units
                ):
                    raise EvaluationError("unknown or malformed fixture candidate")
                if catalog.units[candidate["id"]].unit_type is not expected:
                    raise EvaluationError("candidate lane target mismatch")
                candidates.append(
                    VideoCandidate(
                        candidate["id"],
                        expected,
                        lane["id"],
                        candidate["score"],
                        candidate["rank"],
                        "authored",
                        "frozen",
                        "fixture-only",
                    )
                )
            lanes.append(CandidateLane(lane["id"], weights[lane["id"]], tuple(candidates)))
        return tuple(lanes)

    def close(self) -> None:
        return None


class VideoAdapter:
    """Evaluate authored video lanes with the production fusion policy."""

    stack = "video"
    adapter_version = "1"

    def prepare(self, dataset: EvalDataset, workdir: Path, options: RunOptions) -> _PreparedVideo:
        if options.reranker != "none" or options.profile_path is not None:
            raise EvaluationError("the video adapter supports only its default profile")
        profile = VideoRetrievalProfile.video_local_ann()
        corpus = list(dataset.corpus["units"])
        return _PreparedVideo(
            corpus,
            profile,
            {
                "profile": profile_identity(profile.profile_id, 1, profile),
                "index": {
                    "kind": "authored-video-lane-fixture",
                    "unit_manifest_sha256": dataset.corpus["file_sha256"]["corpus"],
                },
                "model": {"kind": "none", "components": ["authored-lane-candidates"]},
            },
        )


def adapter(stack: str) -> VideoAdapter:
    """Return the video stack adapter."""
    return VideoAdapter()
