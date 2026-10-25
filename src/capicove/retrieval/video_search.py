"""Bounded text/image video retrieval over one leased immutable generation set."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import secrets
from collections.abc import Callable, Mapping
from concurrent.futures import Executor, Future
from contextlib import AbstractContextManager, ExitStack
from dataclasses import asdict, replace
from datetime import UTC, datetime, timedelta
from threading import BoundedSemaphore, Lock
from typing import Protocol
from uuid import uuid4

from capicove.contracts import (
    CompactHit,
    DetailLevel,
    EmbeddingSpaceRef,
    FullHit,
    ImageQuery,
    LaneScore,
    MatchKind,
    SearchableUnit,
    SearchFilters,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TextOrigin,
    TextQuery,
)
from capicove.retrieval.ports import TextRetrieverBackend
from capicove.retrieval.temporal import TemporalCandidateAssembler
from capicove.retrieval.text_search import TextSearchError
from capicove.retrieval.video_fusion import CandidateLane, VideoCandidate, weighted_video_rrf
from capicove.retrieval.video_models import PinnedVideoGenerationSet
from capicove.retrieval.video_planning import PlannedVideoLane, VideoQueryPlanner
from capicove.retrieval.video_ports import VideoUnitCatalogReader, VisualCandidateRetriever
from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry


class VideoSearchError(TextSearchError):
    """A stable video error using the existing search Tool error boundary."""


class VideoQueryEncoder(Protocol):
    """Accept and encode a query only in the explicitly pinned space."""

    def encode(
        self,
        query: TextQuery | ImageQuery,
        *,
        embedding_space: EmbeddingSpaceRef,
        deadline: datetime,
    ) -> tuple[float, ...]: ...


class GenerationSets(Protocol):
    """Acquire a complete generation snapshot with a releaseable lease."""

    def lease_active(
        self, *, workspace_ref: str, profile_ref: str
    ) -> AbstractContextManager[PinnedVideoGenerationSet]: ...


class VideoSearchService:
    """Execute bounded lanes using a runtime-owned executor and semaphore."""

    def __init__(
        self,
        *,
        workspace_ref: str,
        generation_sets: GenerationSets,
        profiles: StaticVideoRetrievalProfileRegistry,
        catalog_resolver: Callable[[PinnedVideoGenerationSet], VideoUnitCatalogReader],
        visual_retriever: VisualCandidateRetriever,
        query_encoder: VideoQueryEncoder,
        executor: Executor,
        concurrency: BoundedSemaphore,
        text_backends: Mapping[str, TextRetrieverBackend] | None = None,
        text_spaces: Mapping[str, EmbeddingSpaceRef] | None = None,
        cursor_secret: bytes | None = None,
    ) -> None:
        self._workspace_ref = workspace_ref
        self._generation_sets = generation_sets
        self._profiles = profiles
        self._catalog_resolver = catalog_resolver
        self._visual = visual_retriever
        self._encoder = query_encoder
        self._executor = executor
        self._concurrency = concurrency
        self._text = dict(text_backends or {})
        self._text_spaces = dict(text_spaces or {})
        self._planner = VideoQueryPlanner(profiles)
        self._cursor_secret = secrets.token_bytes(32) if cursor_secret is None else cursor_secret
        if len(self._cursor_secret) < 32:
            raise ValueError("cursor secret must contain at least 32 bytes")

    def search(self, request: SearchRequest) -> SearchResult:
        """Execute using the configured deadline when no Tool context is supplied."""
        try:
            seconds = self._profiles.resolve(request.retrieval_profile_ref).deadline_seconds
        except LookupError:
            raise VideoSearchError("invalid_query") from None
        return self.search_with_deadline(
            request, deadline=datetime.now(UTC) + timedelta(seconds=seconds)
        )

    def search_with_deadline(self, request: SearchRequest, *, deadline: datetime) -> SearchResult:
        """Keep leases and concurrency permits until timed-out workers really stop."""
        if deadline.tzinfo is None:
            raise VideoSearchError("invalid_query")
        if not isinstance(request.query, (TextQuery, ImageQuery)) or not request.targets <= {
            TargetType.KEYFRAME,
            TargetType.VIDEO_SEGMENT,
        }:
            raise VideoSearchError("unsupported_query_target")
        try:
            remaining = _remaining(deadline)
        except TimeoutError:
            raise VideoSearchError("deadline_exceeded") from None
        if not self._concurrency.acquire(timeout=remaining):
            raise VideoSearchError("deadline_exceeded")
        stack = ExitStack()
        futures: list[Future[object]] = []
        try:
            pin = stack.enter_context(
                self._generation_sets.lease_active(
                    workspace_ref=self._workspace_ref, profile_ref=request.retrieval_profile_ref
                )
            )
            if pin.workspace_ref != self._workspace_ref:
                raise VideoSearchError("index_corrupt")
            return self._execute(request, pin, deadline, futures)
        except VideoSearchError:
            raise
        except PermissionError:
            raise VideoSearchError("permission_denied") from None
        except TimeoutError:
            raise VideoSearchError("deadline_exceeded") from None
        except LookupError:
            raise VideoSearchError("index_generation_unavailable") from None
        except Exception:
            raise VideoSearchError("index_corrupt") from None
        finally:
            _release_after_workers(futures, stack, self._concurrency)

    def _execute(
        self,
        request: SearchRequest,
        pin: PinnedVideoGenerationSet,
        deadline: datetime,
        futures: list[Future[object]],
    ) -> SearchResult:
        try:
            plan = self._planner.plan(request, pin)
        except ValueError as error:
            raise VideoSearchError(str(error)) from None
        profile = self._profiles.resolve(plan.profile_id)
        query_hash = hashlib.sha256(
            json.dumps(
                {"request": asdict(replace(request, cursor=None)), "profile": asdict(profile)},
                sort_keys=True,
                separators=(",", ":"),
                default=_cursor_json_value,
            ).encode()
        ).hexdigest()
        offset = self._decode_cursor(request.cursor, query_hash, pin.set_id)
        catalog = self._catalog_resolver(pin)
        if catalog.unit_manifest_sha256 != pin.unit_manifest_sha256:
            raise VideoSearchError("index_corrupt")
        query = request.query
        if not isinstance(query, (TextQuery, ImageQuery)):
            raise VideoSearchError("unsupported_query_target")
        space = pin.keyframe_visual.embedding_space
        future = self._executor.submit(
            self._encoder.encode, query, embedding_space=space, deadline=deadline
        )
        futures.append(future)  # type: ignore[arg-type]
        vector = future.result(timeout=_remaining(deadline))
        _validate_vector(vector, space)
        dense_vectors: dict[EmbeddingSpaceRef, tuple[float, ...]] = {}
        warnings: set[SearchWarningCode] = set()
        for lane in plan.lanes:
            if not lane.lane_id.endswith("dense"):
                continue
            dense_space = self._text_spaces.get(lane.lane_id)
            if dense_space is None:
                continue
            if dense_space not in dense_vectors:
                encoded = self._executor.submit(
                    self._encoder.encode, query, embedding_space=dense_space, deadline=deadline
                )
                futures.append(encoded)  # type: ignore[arg-type]
                try:
                    dense_vector = encoded.result(timeout=_remaining(deadline))
                    _validate_vector(dense_vector, dense_space)
                    dense_vectors[dense_space] = dense_vector
                except (PermissionError, TimeoutError):
                    raise
                except Exception:
                    warnings.add(SearchWarningCode.RETRIEVAL_LANE_UNAVAILABLE)
        pending: list[tuple[PlannedVideoLane, Future[CandidateLane]]] = []
        for lane in plan.lanes:
            task = self._executor.submit(
                self._retrieve, request, pin, lane, vector, dense_vectors, deadline
            )
            futures.append(task)  # type: ignore[arg-type]
            pending.append((lane, task))
        lanes: list[CandidateLane] = []
        for lane, task in pending:
            try:
                batch = task.result(timeout=_remaining(deadline))
                weighted_video_rrf((batch,), rrf_k=profile.rrf_k)
                lanes.append(batch)
            except Exception as error:
                if isinstance(error, PermissionError):
                    raise VideoSearchError("permission_denied") from None
                if isinstance(error, TimeoutError):
                    raise VideoSearchError("deadline_exceeded") from None
                if not lane.degradable and isinstance(error, (TypeError, ValueError)):
                    raise VideoSearchError("index_corrupt") from None
                if isinstance(error, VideoSearchError) and error.code in {
                    "permission_denied",
                    "deadline_exceeded",
                }:
                    raise
                if not lane.degradable:
                    if isinstance(error, VideoSearchError):
                        raise
                    raise VideoSearchError("provider_unavailable") from None
                warnings.add(SearchWarningCode.RETRIEVAL_LANE_UNAVAILABLE)
        fused = weighted_video_rrf(tuple(lanes), rrf_k=profile.rrf_k)
        projected: list[CompactHit] = []
        rank_scores: dict[str, float] = {}
        if TargetType.KEYFRAME in request.targets:
            keys = tuple(
                item for item in fused if item.contributions[0].target_type is TargetType.KEYFRAME
            )
            units = {
                unit.unit_id: unit
                for unit in catalog.get_many(tuple(item.unit_id for item in keys))
            }
            for item in keys:
                if item.unit_id not in units:
                    raise VideoSearchError("index_corrupt")
                if not _matches_projected_unit(units[item.unit_id], request.filters):
                    continue
                rank_scores[item.unit_id] = item.fused_score
                projected.append(
                    _project(
                        units[item.unit_id], item.fused_score, item.contributions, request, pin
                    )
                )
        if TargetType.VIDEO_SEGMENT in request.targets:
            refined = TemporalCandidateAssembler(catalog).assemble(
                fused=fused, maximum_candidates=profile.maximum_temporal_candidates
            )
            for refined_item in refined:
                if not _matches_projected_unit(refined_item.unit, request.filters):
                    continue
                rank_scores[refined_item.unit.unit_id] = refined_item.score
                projected.append(
                    _project(
                        refined_item.unit,
                        refined_item.score,
                        refined_item.contributions,
                        request,
                        pin,
                    )
                )
        projected.sort(key=lambda hit: (-rank_scores[hit.id], hit.id))
        _remaining(deadline)
        page = tuple(projected[offset : offset + request.top_k])
        next_offset = offset + request.top_k
        cursor = (
            self._encode_cursor(query_hash, pin.set_id, next_offset)
            if next_offset < len(projected)
            else None
        )
        return SearchResult(
            SearchStatus.SUCCEEDED,
            uuid4().hex,
            bool(warnings),
            tuple(sorted(warnings)),
            page,
            cursor,
        )

    def _retrieve(
        self,
        request: SearchRequest,
        pin: PinnedVideoGenerationSet,
        lane: PlannedVideoLane,
        vector: tuple[float, ...],
        dense_vectors: Mapping[EmbeddingSpaceRef, tuple[float, ...]],
        deadline: datetime,
    ) -> CandidateLane:
        _remaining(deadline)
        if lane.target_type is not TargetType.TEXT_CHUNK:
            generation = (
                pin.keyframe_visual
                if lane.target_type is TargetType.KEYFRAME
                else pin.segment_visual
            )
            batch = self._visual.retrieve(
                query_vector=vector,
                embedding_space=generation.embedding_space,
                filters=request.filters,
                generation=generation,
                top_k=lane.top_k,
            )
            if len(batch) > lane.top_k or any(
                c.generation_id != lane.generation_id
                or c.target_type is not lane.target_type
                or c.lane_id != lane.lane_id
                or c.provider_id != self._visual.provider_id
                for c in batch
            ):
                raise VideoSearchError("index_corrupt")
            candidates = tuple(
                VideoCandidate(
                    c.unit_id,
                    c.target_type,
                    lane.lane_id,
                    c.raw_score,
                    c.rank,
                    c.provider_id,
                    c.generation_id,
                    c.score_semantics,
                )
                for c in batch
            )
        else:
            backend = self._text.get(lane.lane_id)
            if backend is None or not isinstance(request.query, TextQuery):
                raise VideoSearchError("provider_unavailable")
            text_pin = pin.asr_text if lane.lane_id.startswith("asr") else pin.ocr_text
            if text_pin is None:
                raise VideoSearchError("index_generation_unavailable")
            origin = TextOrigin.ASR if lane.lane_id.startswith("asr") else TextOrigin.OCR
            if request.filters.text_origins and origin not in request.filters.text_origins:
                return CandidateLane(lane.lane_id, lane.weight, ())
            filters = replace(request.filters, text_origins=frozenset({origin}))
            if lane.lane_id.endswith("dense"):
                dense_space = self._text_spaces.get(lane.lane_id)
                if dense_space is None or dense_space not in dense_vectors:
                    raise VideoSearchError("embedding_space_incompatible")
                texts = backend.dense_retriever.retrieve(
                    query_vector=dense_vectors[dense_space],
                    embedding_space=dense_space,
                    filters=filters,
                    generation=text_pin,
                    top_k=lane.top_k,
                )
            else:
                texts = backend.lexical_retriever.retrieve(
                    query=request.query, filters=filters, generation=text_pin, top_k=lane.top_k
                )
            if len(texts) > lane.top_k or any(
                c.index_generation_id != lane.generation_id or c.text_locator.origin is not origin
                for c in texts
            ):
                raise VideoSearchError("index_corrupt")
            candidates = tuple(
                VideoCandidate(
                    c.unit_id,
                    TargetType.TEXT_CHUNK,
                    lane.lane_id,
                    c.raw_score,
                    c.rank,
                    c.provider_id,
                    c.index_generation_id,
                    c.score_semantics,
                )
                for c in texts
            )
        _remaining(deadline)
        return CandidateLane(lane.lane_id, lane.weight, candidates)

    def _encode_cursor(self, query_hash: str, set_id: str, offset: int) -> str:
        body = json.dumps([query_hash, set_id, offset], separators=(",", ":")).encode()
        return base64.urlsafe_b64encode(
            body + hmac.digest(self._cursor_secret, body, "sha256")
        ).decode()

    def _decode_cursor(self, cursor: str | None, query_hash: str, set_id: str) -> int:
        if cursor is None:
            return 0
        try:
            if len(cursor) > 2048:
                raise ValueError
            raw = base64.b64decode(cursor, altchars=b"-_", validate=True)
            body, signature = raw[:-32], raw[-32:]
            if not hmac.compare_digest(signature, hmac.digest(self._cursor_secret, body, "sha256")):
                raise ValueError
            values = json.loads(body)
            if (
                not isinstance(values, list)
                or len(values) != 3
                or values[:2] != [query_hash, set_id]
                or isinstance(values[2], bool)
                or not isinstance(values[2], int)
                or not 0 < values[2] <= 1000
            ):
                raise ValueError
            return int(values[2])
        except (ValueError, TypeError):
            raise VideoSearchError("stale_cursor") from None


def _cursor_json_value(value: object) -> object:
    """Normalize unordered request fields and bind inline media by content hash."""
    if isinstance(value, frozenset):
        return sorted(value)
    if isinstance(value, bytes):
        return {"sha256": hashlib.sha256(value).hexdigest()}
    raise TypeError(f"unsupported cursor identity value: {type(value).__name__}")


def _remaining(deadline: datetime) -> float:
    value = (deadline - datetime.now(UTC)).total_seconds()
    if value <= 0:
        raise TimeoutError("deadline_exceeded")
    return value


def _matches_projected_unit(unit: SearchableUnit, filters: SearchFilters) -> bool:
    """Recheck predicates available on final units after temporal mapping.

    Corpus and media-type predicates remain generation/provider-owned; providers
    must reject predicates that cannot be proved from their published metadata.
    """
    if filters.artifact_refs and unit.artifact_ref not in filters.artifact_refs:
        return False
    span = filters.time_range
    if span is None:
        return True
    if unit.timestamp_ms is not None:
        return span.start_ms <= unit.timestamp_ms < span.end_ms
    target = unit.time_range
    return target is not None and max(span.start_ms, target.start_ms) < min(
        span.end_ms, target.end_ms
    )


def _validate_vector(vector: tuple[float, ...], space: EmbeddingSpaceRef) -> None:
    if len(vector) != space.dimension or any(
        isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
        for v in vector
    ):
        raise VideoSearchError("embedding_space_incompatible")
    norm = math.sqrt(sum(v * v for v in vector))
    if not math.isclose(norm, 1.0, abs_tol=1e-4):
        raise VideoSearchError("embedding_space_incompatible")


def _project(
    unit: SearchableUnit,
    score: float,
    contributions: tuple[VideoCandidate, ...],
    request: SearchRequest,
    pin: PinnedVideoGenerationSet,
) -> CompactHit:
    previews = tuple(ref for ref in unit.representation_refs if ref.startswith("resource:preview:"))
    if len(previews) != 1 or unit.unit_type not in request.targets:
        raise VideoSearchError("index_corrupt")
    brief = min(1.0, max(0.0, score * 10))
    hit = CompactHit(
        unit.unit_id,
        unit.unit_type,
        brief,
        unit.evidence_ref,
        unit.timestamp_ms,
        unit.time_range,
        preview_resource_ref=previews[0],
    )
    if request.detail_level is DetailLevel.COMPACT:
        return hit
    by_lane = {c.lane_id: c for c in sorted(contributions, key=lambda c: -c.rank)}
    return FullHit(
        hit.id,
        hit.target_type,
        hit.brief_score,
        hit.evidence_id,
        hit.timestamp_ms,
        hit.time_range,
        artifact_ref=unit.artifact_ref,
        searchable_unit_ref=unit.unit_id,
        representation_refs=unit.representation_refs,
        scene_range=unit.scene_range,
        match_kinds=tuple(
            sorted(
                {
                    MatchKind.ASR
                    if c.lane_id.startswith("asr_")
                    else MatchKind.OCR
                    if c.lane_id.startswith("ocr_")
                    else MatchKind.VISUAL
                    for c in contributions
                },
                key=lambda kind: kind.value,
            )
        ),
        lane_scores=tuple(
            LaneScore(
                c.lane_id,
                c.raw_score,
                1.0 / c.rank,
                c.score_semantics,
                "reciprocal-rank-v1",
                c.rank,
            )
            for c in by_lane.values()
        ),
        embedding_spaces=unit.embedding_spaces,
        processing_fingerprint=unit.processing_fingerprint,
        index_generation_ids=tuple(sorted({c.generation_id for c in contributions})),
        provider_ids=tuple(sorted({c.provider_id for c in contributions})),
        lineage_refs=(unit.processing_run_ref,),
        preview_resource_ref=previews[0],
    )


def _release_after_workers(
    futures: list[Future[object]], stack: ExitStack, semaphore: BoundedSemaphore
) -> None:
    pending = [future for future in futures if not future.done()]
    if not pending:
        try:
            stack.close()
        finally:
            semaphore.release()
        return
    lock = Lock()
    remaining = len(pending)

    def finished(future: Future[object]) -> None:
        nonlocal remaining
        with lock:
            remaining -= 1
            if remaining == 0:
                try:
                    stack.close()
                finally:
                    semaphore.release()

    for future in pending:
        future.add_done_callback(finished)
        future.cancel()
