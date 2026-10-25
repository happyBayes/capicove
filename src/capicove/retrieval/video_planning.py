"""Closed query-target planning over one pinned generation set."""

from __future__ import annotations

from dataclasses import dataclass

from capicove.contracts import ImageQuery, QueryModality, SearchRequest, TargetType, TextQuery
from capicove.retrieval.video_models import PinnedVideoGenerationSet
from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry


@dataclass(frozen=True, slots=True)
class PlannedVideoLane:
    """A bounded lane bound to one physical generation."""

    lane_id: str
    target_type: TargetType
    top_k: int
    degradable: bool
    generation_id: str
    weight: float


@dataclass(frozen=True, slots=True)
class VideoQueryPlan:
    """One immutable query execution plan."""

    query_modality: QueryModality
    targets: frozenset[TargetType]
    lanes: tuple[PlannedVideoLane, ...]
    generation_set_id: str
    profile_id: str


class VideoQueryPlanner:
    """Select lanes using startup policy, never provider query knobs."""

    def __init__(self, profiles: StaticVideoRetrievalProfileRegistry) -> None:
        self._profiles = profiles

    def plan(
        self, request: SearchRequest, generation_set: PinnedVideoGenerationSet
    ) -> VideoQueryPlan:
        """Require supported modalities and an exactly compatible visual space."""
        if not isinstance(request.query, (TextQuery, ImageQuery)) or not request.targets <= {
            TargetType.KEYFRAME,
            TargetType.VIDEO_SEGMENT,
        }:
            raise ValueError("unsupported_query_target")
        profile = self._profiles.resolve(request.retrieval_profile_ref)
        if generation_set.profile_ref != profile.profile_id:
            raise ValueError("index_generation_unavailable")
        space = generation_set.keyframe_visual.embedding_space
        if generation_set.segment_visual.embedding_space != space or any(
            not space.accepts(request.query.type, target) for target in request.targets
        ):
            raise ValueError("embedding_space_incompatible")
        lanes: list[PlannedVideoLane] = []
        for lane in profile.lanes:
            name = lane.lane_id
            if name == "keyframe_visual":
                target, generation = TargetType.KEYFRAME, generation_set.keyframe_visual
            elif name == "segment_visual":
                if TargetType.VIDEO_SEGMENT not in request.targets:
                    continue
                target, generation = TargetType.VIDEO_SEGMENT, generation_set.segment_visual
            else:
                if (
                    isinstance(request.query, ImageQuery)
                    or TargetType.VIDEO_SEGMENT not in request.targets
                ):
                    continue
                text_generation = (
                    generation_set.asr_text if name.startswith("asr") else generation_set.ocr_text
                )
                if text_generation is None:
                    continue
                lanes.append(
                    PlannedVideoLane(
                        name,
                        TargetType.TEXT_CHUNK,
                        profile.text_depth(request.top_k),
                        lane.degradable,
                        text_generation.generation_id,
                        lane.weight,
                    )
                )
                continue
            lanes.append(
                PlannedVideoLane(
                    name,
                    target,
                    profile.visual_depth(request.top_k),
                    False,
                    generation.generation_id,
                    lane.weight,
                )
            )
        return VideoQueryPlan(
            request.query.type,
            request.targets,
            tuple(lanes),
            generation_set.set_id,
            profile.profile_id,
        )
