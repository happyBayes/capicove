"""Online search uses pinned generations, a shared encoder and bounded lanes."""

from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from threading import BoundedSemaphore, Event
from types import SimpleNamespace

import pytest

from capicove.contracts import (
    EmbeddingSpaceRef,
    ImageQuery,
    InlineBytesSource,
    LocalPathSource,
    SearchRequest,
    TargetType,
    TextQuery,
)
from capicove.retrieval.generations import PinnedTextIndexGeneration
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    PinnedVisualIndexGeneration,
    VisualCandidate,
)
from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry
from capicove.retrieval.video_search import (
    VideoSearchError,
    VideoSearchService,
    _release_after_workers,
)
from tests.unit.retrieval.test_video_models import unit, visual_space


def test_cleanup_failure_does_not_leak_query_capacity() -> None:
    stack = ExitStack()
    capacity = BoundedSemaphore(1)
    capacity.acquire()

    def fail() -> None:
        raise OSError("lease catalog unavailable")

    stack.callback(fail)
    with pytest.raises(OSError):
        _release_after_workers([], stack, capacity)
    assert capacity.acquire(blocking=False)
    capacity.release()


class Generations:
    leases = 0

    @contextmanager
    def lease_active(
        self, *, workspace_ref: str, profile_ref: str
    ) -> Iterator[PinnedVideoGenerationSet]:
        self.leases += 1
        try:
            yield PinnedVideoGenerationSet(
                "set",
                "a" * 64,
                workspace_ref,
                profile_ref,
                "catalog",
                "b" * 64,
                PinnedVisualIndexGeneration(
                    "kg", "c" * 64, "key-index", TargetType.KEYFRAME, visual_space()
                ),
                PinnedVisualIndexGeneration(
                    "sg", "d" * 64, "seg-index", TargetType.VIDEO_SEGMENT, visual_space()
                ),
                PinnedTextIndexGeneration("ag", "e" * 64, "asr-index"),
                None,
                "video-weighted-rrf-v1",
            )
        finally:
            self.leases -= 1


class Encoder:
    calls = 0

    def encode(
        self, query: object, *, embedding_space: EmbeddingSpaceRef, deadline: datetime
    ) -> tuple[float, ...]:
        self.calls += 1
        return (1.0, 0.0)


class Visual:
    provider_id = "p"

    def retrieve(self, **kwargs: object) -> tuple[VisualCandidate, ...]:
        pin = kwargs["generation"]
        assert isinstance(pin, PinnedVisualIndexGeneration)
        return (
            VisualCandidate(
                "k" if pin.target_type is TargetType.KEYFRAME else "s",
                pin.target_type,
                "keyframe_visual" if pin.target_type is TargetType.KEYFRAME else "segment_visual",
                0.8,
                "inner_product_similarity",
                1,
                "p",
                pin.generation_id,
            ),
        )


class Catalog:
    unit_manifest_sha256 = "b" * 64

    def get(self, unit_id: str):  # type: ignore[no-untyped-def]
        return replace(
            unit(unit_id, TargetType.KEYFRAME if unit_id == "k" else TargetType.VIDEO_SEGMENT),
            representation_refs=("resource:preview:k",),
        )

    def get_many(self, unit_ids: tuple[str, ...]):  # type: ignore[no-untyped-def]
        return tuple(self.get(key) for key in unit_ids)

    def segments_for_keyframe(self, unit_id: str) -> tuple[str, ...]:
        return ("s",)

    def segments_for_keyframes(self, unit_ids: tuple[str, ...]) -> dict[str, tuple[str, ...]]:
        return {key: ("s",) for key in unit_ids}

    def segments_for_text(self, unit_id: str) -> tuple[tuple[str, int], ...]:
        return ()

    def segments_for_texts(
        self, unit_ids: tuple[str, ...]
    ) -> dict[str, tuple[tuple[str, int], ...]]:
        return {}


def test_text_search_pins_once_and_degrades_optional_lane() -> None:
    generations, encoder = Generations(), Encoder()
    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=generations,
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=encoder,
            executor=executor,
            concurrency=BoundedSemaphore(4),
        )
        result = service.search(
            SearchRequest(
                TextQuery("slide"),
                frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
                retrieval_profile_ref="video-local-exact",
            )
        )
        assert {hit.id for hit in result.hits} == {"k", "s"}
        assert result.degraded
        assert result.warning_codes == ("retrieval_lane_unavailable",)
        assert encoder.calls == 1
        assert generations.leases == 0
        with pytest.raises(VideoSearchError):
            service.search(SearchRequest(TextQuery("slide"), frozenset({TargetType.TEXT_CHUNK})))


@pytest.mark.parametrize(
    "failure,code",
    [
        (PermissionError("denied"), "permission_denied"),
        (TimeoutError("expired"), "deadline_exceeded"),
    ],
)
def test_optional_lane_permission_and_deadline_errors_are_not_degradation(
    failure: Exception,
    code: str,
) -> None:
    def fail(**kwargs: object) -> tuple[()]:
        raise failure

    backend = SimpleNamespace(lexical_retriever=SimpleNamespace(retrieve=fail))
    with ThreadPoolExecutor(max_workers=4) as executor:
        generations = Generations()
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=generations,
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
            text_backends={"asr_lexical": backend},
        )
        with pytest.raises(VideoSearchError) as caught:
            service.search(
                SearchRequest(
                    TextQuery("slide"),
                    frozenset({TargetType.VIDEO_SEGMENT}),
                    retrieval_profile_ref="video-local-exact",
                )
            )
        assert caught.value.code == code


@pytest.mark.parametrize(
    "failure", [VideoSearchError("index_corrupt"), ValueError("malformed candidate batch")]
)
def test_malformed_optional_lane_is_discarded_and_explicitly_degraded(failure: Exception) -> None:
    def fail(**kwargs: object) -> tuple[()]:
        raise failure

    backend = SimpleNamespace(lexical_retriever=SimpleNamespace(retrieve=fail))
    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
            text_backends={"asr_lexical": backend},
        )
        result = service.search(
            SearchRequest(
                TextQuery("slide"),
                frozenset({TargetType.VIDEO_SEGMENT}),
                retrieval_profile_ref="video-local-exact",
            )
        )
        assert {hit.id for hit in result.hits} == {"s"}
        assert result.degraded
        assert result.warning_codes == ("retrieval_lane_unavailable",)


@pytest.mark.parametrize(
    "failure,code",
    [
        (PermissionError("denied"), "permission_denied"),
        (TimeoutError("expired"), "deadline_exceeded"),
    ],
)
def test_dense_query_encoding_does_not_swallow_permission_or_timeout(
    failure: Exception, code: str
) -> None:
    from capicove.retrieval.video_profiles import VideoLanePolicy, VideoRetrievalProfile

    class DenseEncoder(Encoder):
        def encode(
            self, query: object, *, embedding_space: EmbeddingSpaceRef, deadline: datetime
        ) -> tuple[float, ...]:
            if self.calls:
                raise failure
            return super().encode(query, embedding_space=embedding_space, deadline=deadline)

    baseline = VideoRetrievalProfile.video_local_exact()
    profile = replace(
        baseline, lanes=(*baseline.lanes, VideoLanePolicy("asr_dense", 1.0, True, True))
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry((profile,)),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=DenseEncoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
            text_spaces={"asr_dense": visual_space()},
        )
        with pytest.raises(VideoSearchError) as caught:
            service.search(
                SearchRequest(
                    TextQuery("slide"),
                    frozenset({TargetType.VIDEO_SEGMENT}),
                    retrieval_profile_ref=profile.profile_id,
                )
            )
        assert caught.value.code == code


def test_expired_deadline_uses_stable_search_error() -> None:
    with ThreadPoolExecutor(max_workers=1) as executor:
        permit = BoundedSemaphore(1)
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=permit,
        )
        with pytest.raises(VideoSearchError) as caught:
            service.search_with_deadline(
                SearchRequest(TextQuery("x"), frozenset({TargetType.KEYFRAME})),
                deadline=datetime.now(UTC) - timedelta(seconds=1),
            )
        assert caught.value.code == "deadline_exceeded"
        assert permit.acquire(blocking=False)
        permit.release()


def test_final_projection_rechecks_artifact_filters() -> None:
    from capicove.contracts import SearchFilters

    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
        )
        request = SearchRequest(
            TextQuery("x"),
            frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
            retrieval_profile_ref="video-local-exact",
            filters=SearchFilters(artifact_refs=frozenset({"not-in-catalog"})),
        )
        assert service.search(request).hits == ()


def test_compact_score_clipping_does_not_change_rank_order() -> None:
    from capicove.retrieval.video_profiles import VideoLanePolicy, VideoRetrievalProfile

    class KeysCatalog(Catalog):
        def get(self, unit_id: str):  # type: ignore[no-untyped-def]
            return replace(
                unit(unit_id, TargetType.KEYFRAME), representation_refs=("resource:preview:k",)
            )

    class RankedVisual(Visual):
        def retrieve(self, **kwargs: object) -> tuple[VisualCandidate, ...]:
            pin = kwargs["generation"]
            assert isinstance(pin, PinnedVisualIndexGeneration)
            return tuple(
                VisualCandidate(
                    key,
                    TargetType.KEYFRAME,
                    "keyframe_visual",
                    0.8,
                    "inner_product_similarity",
                    rank,
                    "p",
                    pin.generation_id,
                )
                for rank, key in enumerate(("z", "a"), 1)
            )

    profile = VideoRetrievalProfile(
        "video-local-exact",
        (
            VideoLanePolicy("keyframe_visual", 100.0, False),
            VideoLanePolicy("segment_visual", 1.0, False),
        ),
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry((profile,)),
            catalog_resolver=lambda pin: KeysCatalog(),
            visual_retriever=RankedVisual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
        )
        result = service.search(
            SearchRequest(
                TextQuery("x"),
                frozenset({TargetType.KEYFRAME}),
                top_k=1,
                retrieval_profile_ref="video-local-exact",
            )
        )
        assert result.hits[0].id == "z"


def test_timeout_retains_lease_and_capacity_until_worker_stops() -> None:
    released = Event()

    class SlowEncoder(Encoder):
        def encode(
            self, query: object, *, embedding_space: EmbeddingSpaceRef, deadline: datetime
        ) -> tuple[float, ...]:
            released.wait(timeout=2)
            return (1.0, 0.0)

    generations = Generations()
    capacity = BoundedSemaphore(1)
    with ThreadPoolExecutor(max_workers=1) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=generations,
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=SlowEncoder(),
            executor=executor,
            concurrency=capacity,
        )
        try:
            with pytest.raises(VideoSearchError) as caught:
                service.search_with_deadline(
                    SearchRequest(
                        TextQuery("x"),
                        frozenset({TargetType.KEYFRAME}),
                        retrieval_profile_ref="video-local-exact",
                    ),
                    deadline=datetime.now(UTC) + timedelta(milliseconds=100),
                )
            assert caught.value.code == "deadline_exceeded"
            assert generations.leases == 1
            assert not capacity.acquire(blocking=False)
        finally:
            released.set()
    assert generations.leases == 0
    assert capacity.acquire(blocking=False)
    capacity.release()


@pytest.mark.parametrize("filter_count", [0, 8, 32])
@pytest.mark.parametrize(
    "query",
    [
        TextQuery("slide"),
        ImageQuery(InlineBytesSource(b"authored-image-query", "image/png")),
        ImageQuery(LocalPathSource("/authorized/query.png")),
    ],
)
def test_full_results_and_cursor_binding(filter_count: int, query: TextQuery | ImageQuery) -> None:
    from capicove.contracts import DetailLevel, FullHit, SearchFilters
    from capicove.tools.search import encode_search_result

    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
        )
        request = SearchRequest(
            query,
            frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
            top_k=1,
            detail_level=DetailLevel.FULL,
            retrieval_profile_ref="video-local-exact",
            filters=SearchFilters(
                artifact_refs=frozenset(
                    ["artifact-1", *(f"artifact-{index}" for index in range(filter_count))]
                )
            ),
        )
        first = service.search(request)
        assert isinstance(first.hits[0], FullHit)
        assert first.hits[0].lane_scores
        assert first.hits[0].lane_scores[0].calibration_version == "reciprocal-rank-v1"
        assert (
            first.hits[0].lane_scores[0].normalized_score == 1.0 / first.hits[0].lane_scores[0].rank
        )
        assert encode_search_result(first)["hits"]
        assert first.next_cursor is not None
        # A transport can rebuild an equal set in a different insertion order.
        equivalent_filters = replace(
            request.filters,
            artifact_refs=frozenset(reversed(sorted(request.filters.artifact_refs))),
        )
        assert equivalent_filters == request.filters
        second = service.search(
            replace(request, filters=equivalent_filters, cursor=first.next_cursor)
        )
        assert first.hits[0].id != second.hits[0].id
        assert second.next_cursor is None
        for invalid in (
            replace(request, cursor="!"),
            replace(request, cursor="x" * 2049),
            replace(request, query=TextQuery("different"), cursor=first.next_cursor),
            replace(request, filters=SearchFilters(), cursor=first.next_cursor),
            replace(
                request,
                query=ImageQuery(InlineBytesSource(b"changed-image-query", "image/png")),
                cursor=first.next_cursor,
            ),
        ):
            with pytest.raises(VideoSearchError) as caught:
                service.search(invalid)
            assert caught.value.code == "stale_cursor"
