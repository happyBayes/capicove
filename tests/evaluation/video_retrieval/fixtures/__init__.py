"""Strict loader for project-authored, media-free evaluation metadata."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

from capicove.contracts import (
    CharacterRange,
    QueryModality,
    SearchableUnit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
)
from capicove.retrieval.temporal import TemporalCandidateAssembler
from capicove.retrieval.video_evaluation import (
    VideoEvaluationEvidence,
    VideoEvaluationEvidenceKind,
    VideoEvaluationProfile,
    VideoRankedResult,
    VideoRetrievalJudgment,
)
from capicove.retrieval.video_fusion import CandidateLane, VideoCandidate, weighted_video_rrf
from capicove.retrieval.video_profiles import VideoRetrievalProfile
from tests.unit.retrieval.test_video_models import unit


FIXTURES = Path(__file__).parent
DIGESTS = {
    # Frozen authored-fixture SHA-256 values, not credentials.
    "corpus": "333ddbc56135a31def7b9bcbbd2ecf29d4daf082b0b61ea12c76fd848ff038fe",  # pragma: allowlist secret
    "queries": "5a3ec9eac558a10af59bb687fe82206f495caa68e89d8c4f0cf4ae8b671de697",  # pragma: allowlist secret
    "judgments": "bc16f26620273bb97a42d9cfc8820fbce6913dd04a7c4740e4ddadc977e7d381",  # pragma: allowlist secret
}
FIELDS = {
    "corpus": {
        "id",
        "type",
        "artifact",
        "start_ms",
        "end_ms",
        "timestamp_ms",
        "origin",
        "segments",
    },
    "queries": {"id", "scenario", "offset_ms", "modality", "target", "lanes"},
    "judgments": {"id", "expected"},
}


def closed(row: object, fields: set[str]) -> None:
    if not isinstance(row, dict) or set(row) != fields:
        raise ValueError("fixture fields must match the closed schema")


def read_rows(root: Path, name: str) -> list[dict[str, Any]]:
    raw = (root / f"{name}.jsonl").read_bytes()
    if hashlib.sha256(raw).hexdigest() != DIGESTS[name]:
        raise ValueError("frozen fixture digest mismatch")
    rows: list[dict[str, Any]] = [json.loads(line) for line in raw.splitlines()]
    for row in rows:
        closed(row, FIELDS[name])
        if not isinstance(row["id"], str) or not row["id"]:
            raise ValueError("invalid fixture ID")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("duplicate fixture ID")
    return rows


class Catalog:
    """Build canonical units and immutable relations solely from frozen rows."""

    unit_manifest_sha256 = DIGESTS["corpus"]

    def __init__(self, rows: list[dict[str, Any]], offset: int) -> None:
        self.units: dict[str, SearchableUnit] = {}
        self.relations: dict[str, tuple[str, ...]] = {}
        for row in rows:
            target = TargetType(row["type"])
            base = replace(unit(row["id"]), artifact_ref=row["artifact"], scene_range=None)
            interval = None
            if target is not TargetType.KEYFRAME:
                interval = TemporalRange(row["start_ms"] + offset, row["end_ms"] + offset)
            locator = None
            if target is TargetType.TEXT_CHUNK:
                locator = TextChunkLocator(
                    TextOrigin(row["origin"]),
                    CharacterRange(0, 10),
                    time_range=interval,
                )
            self.units[row["id"]] = replace(
                base,
                unit_type=target,
                timestamp_ms=row["timestamp_ms"] + offset
                if target is TargetType.KEYFRAME
                else None,
                time_range=interval if target is TargetType.VIDEO_SEGMENT else None,
                text_locator=locator,
            )
            self.relations[row["id"]] = tuple(row["segments"])
        for key, targets in self.relations.items():
            source = self.units[key]
            if len(set(targets)) != len(targets):
                raise ValueError("duplicate relation")
            for target_id in targets:
                segment = self.units.get(target_id)
                if (
                    segment is None
                    or segment.unit_type is not TargetType.VIDEO_SEGMENT
                    or segment.artifact_ref != source.artifact_ref
                ):
                    raise ValueError("unknown or cross-source segment relation")

    def get(self, key: str) -> SearchableUnit | None:
        return self.units.get(key)

    def get_many(self, keys: tuple[str, ...]) -> tuple[SearchableUnit, ...]:
        return tuple(self.units[key] for key in keys)

    def segments_for_keyframe(self, key: str) -> tuple[str, ...]:
        return self.relations[key]

    def segments_for_keyframes(self, keys: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        return {key: self.segments_for_keyframe(key) for key in keys}

    def segment_range(self, key: str) -> TemporalRange:
        interval = self.units[key].time_range
        if interval is None:
            raise ValueError("segment range required")
        return interval

    def segments_for_text(self, key: str) -> tuple[tuple[str, int], ...]:
        locator = self.units[key].text_locator
        if locator is None or locator.time_range is None:
            raise ValueError("temporal text locator required")
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


def make_lanes(query: dict[str, Any], catalog: Catalog) -> tuple[CandidateLane, ...]:
    weights = {lane.lane_id: lane.weight for lane in VideoRetrievalProfile.video_local_ann().lanes}
    lanes = []
    for lane in query["lanes"]:
        closed(lane, {"id", "candidates"})
        if lane["id"] not in weights:
            raise ValueError("unknown lane")
        candidates = []
        for row in lane["candidates"]:
            closed(row, {"id", "rank", "score"})
            if row["id"] not in catalog.units:
                raise ValueError("unknown candidate")
            target = catalog.units[row["id"]].unit_type
            expected = (
                TargetType.KEYFRAME
                if lane["id"] == "keyframe_visual"
                else TargetType.VIDEO_SEGMENT
                if lane["id"] == "segment_visual"
                else TargetType.TEXT_CHUNK
            )
            if target is not expected:
                raise ValueError("candidate lane target mismatch")
            candidates.append(
                VideoCandidate(
                    row["id"],
                    target,
                    lane["id"],
                    row["score"],
                    row["rank"],
                    "authored",
                    "frozen",
                    "fixture-only",
                )
            )
        lanes.append(CandidateLane(lane["id"], weights[lane["id"]], tuple(candidates)))
    result = tuple(lanes)
    weighted_video_rrf(result)  # Validates contiguous ranks and unique lane/candidate IDs.
    return result


def load_fixture(
    root: Path = FIXTURES,
) -> tuple[VideoEvaluationProfile, list[dict[str, Any]], tuple[VideoRetrievalJudgment, ...]]:
    corpus = read_rows(root, "corpus")
    queries = read_rows(root, "queries")
    judgments = {row["id"]: row for row in read_rows(root, "judgments")}
    if set(judgments) != {query["id"] for query in queries}:
        raise ValueError("query and judgment IDs differ")
    result = []
    for query in queries:
        if type(query["offset_ms"]) is not int or query["offset_ms"] < 0:
            raise ValueError("invalid timeline offset")
        catalog = Catalog(corpus, query["offset_ms"])
        make_lanes(query, catalog)
        target = TargetType(query["target"])
        relevant = tuple(judgments[query["id"]]["expected"])
        if any(
            key not in catalog.units or catalog.units[key].unit_type is not target
            for key in relevant
        ):
            raise ValueError("unknown or mismatched judgment target")
        result.append(
            VideoRetrievalJudgment(
                query["id"],
                target,
                relevant,
                tuple(catalog.segment_range(key) for key in relevant)
                if target is TargetType.VIDEO_SEGMENT
                else (),
                QueryModality(query["modality"]),
            )
        )
    profile = VideoEvaluationProfile(
        corpus_sha256=DIGESTS["corpus"],
        query_set_sha256=hashlib.sha256(
            (DIGESTS["queries"] + DIGESTS["judgments"]).encode()
        ).hexdigest(),
        model_bundle_sha256="0" * 64,
        preprocessing_version="authored-no-model-v1",
        index_manifest_sha256="0" * 64,
        generation_set_manifest_sha256=DIGESTS["corpus"],
        fusion_profile_ref="video-weighted-rrf-v1",
        temporal_policy_ref="scene-transcript-segments-v1",
        host_fingerprint_sha256="0" * 64,
        host_class="deterministic-test",
        operating_system="independent",
        storage_class="in-memory",
        keyframe_count=2,
        embedding_dimension=1,
        network_rtt_ms=0,
        concurrency=4,
        warmup_requests=100,
        sample_requests=1000,
        evidence=VideoEvaluationEvidence(
            semantic_quality=VideoEvaluationEvidenceKind.AUTHORED_METADATA
        ),
    )
    return profile, queries, tuple(result)


def run_queries(queries: list[dict[str, Any]]) -> dict[str, tuple[VideoRankedResult, ...]]:
    corpus = read_rows(FIXTURES, "corpus")
    rankings: dict[str, tuple[VideoRankedResult, ...]] = {}
    for query in queries:
        catalog = Catalog(corpus, query["offset_ms"])
        fused = weighted_video_rrf(make_lanes(query, catalog))
        target = TargetType(query["target"])
        if target is TargetType.KEYFRAME:
            rankings[query["id"]] = tuple(VideoRankedResult(item.unit_id, target) for item in fused)
        else:
            rankings[query["id"]] = tuple(
                VideoRankedResult(item.unit.unit_id, target, item.unit.time_range)
                for item in TemporalCandidateAssembler(catalog).assemble(fused=fused)
            )
    return rankings
