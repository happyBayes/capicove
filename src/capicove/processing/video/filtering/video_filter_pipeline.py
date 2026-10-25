"""Unified orchestration for evidence-aware local video filtering."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from capicove.processing.video.filtering.video_filter_contracts import (
    CriterionResult,
    CriterionType,
    FrameSamplingConfig,
    LocalModelReference,
    MetadataCriterion,
    TagCriterion,
    VideoCriterionResult,
    VideoFilterFailure,
    VideoFilterFailureCode,
    VideoFilterItem,
    VideoFilterItemResult,
    VideoFilterMetadata,
    VideoFilterRequest,
    VideoFilterResult,
    VideoFilterStatus,
    VideoFilterWarning,
    VideoFilterWarningCode,
)
from capicove.processing.video.filtering.video_filter_ports import (
    FrameSampleBatch,
    VideoCriterionEvaluator,
    VideoFilterCapabilityError,
    VideoFrameSampler,
    VideoMetadataProbe,
)
from capicove.processing.video.filtering.video_filter_rules import aggregate, in_range
from capicove.processing.video.models import AcceptedVideoSource
from capicove.processing.video.ports import ArtifactResolver, PathAuthorizer, PathPolicyError
from capicove.processing.video.source import VideoSourceError, accept_source


class VideoFilterPipeline:
    """Evaluate configured criteria while sharing verified source work."""

    def __init__(
        self,
        metadata_probe: VideoMetadataProbe,
        frame_sampler: VideoFrameSampler,
        evaluators: Mapping[str | CriterionType, VideoCriterionEvaluator],
        path_authorizer: PathAuthorizer,
        artifact_resolver: ArtifactResolver | None = None,
    ) -> None:
        self._metadata_probe = metadata_probe
        self._frame_sampler = frame_sampler
        self._evaluators = {
            key.value if isinstance(key, CriterionType) else key: evaluator
            for key, evaluator in evaluators.items()
        }
        self._path_authorizer = path_authorizer
        self._artifact_resolver = artifact_resolver

    def evaluate(self, request: VideoFilterRequest) -> VideoFilterResult:
        """Evaluate every required input and omit decisions after any failure."""
        failures: list[VideoFilterFailure] = []
        warnings: list[VideoFilterWarning] = []
        accepted_items: list[tuple[VideoFilterItem, AcceptedVideoSource]] = []
        item_results: list[VideoFilterItemResult] = []

        if _expired(request.deadline):
            return _failed(request, (), (), (), (_timeout_failure(),))

        for criterion in request.criteria:
            try:
                for path in _criterion_resource_paths(criterion):
                    self._path_authorizer.authorize_read(
                        request.execution_policy_ref,
                        path,
                    )
            except PathPolicyError as error:
                return _failed(
                    request,
                    (),
                    (),
                    (),
                    (
                        VideoFilterFailure(
                            VideoFilterFailureCode.PERMISSION_DENIED,
                            str(error),
                            criterion_id=criterion.criterion_id,
                        ),
                    ),
                )

        seen_hashes: set[str] = set()
        for item in request.items:
            try:
                accepted = accept_source(
                    item.source,
                    request.deadline,
                    artifact_resolver=self._artifact_resolver,
                    path_authorizer=self._path_authorizer,
                    policy_ref=request.execution_policy_ref,
                )
                metadata = self._metadata_probe.probe(accepted, request.deadline)
            except PathPolicyError as error:
                failures.append(
                    VideoFilterFailure(
                        VideoFilterFailureCode.PERMISSION_DENIED,
                        str(error),
                        item_id=item.item_id,
                    )
                )
                break
            except VideoSourceError as error:
                failures.append(
                    VideoFilterFailure(
                        _source_failure_code(error),
                        str(error),
                        item_id=item.item_id,
                    )
                )
                break
            except VideoFilterCapabilityError as error:
                failures.append(
                    VideoFilterFailure(error.failure_code, str(error), item_id=item.item_id)
                )
                if error.failure_code is VideoFilterFailureCode.PERMISSION_DENIED:
                    break
                continue
            except (OSError, ValueError) as error:
                failures.append(
                    VideoFilterFailure(
                        VideoFilterFailureCode.UNSUPPORTED_MEDIA,
                        str(error),
                        item_id=item.item_id,
                    )
                )
                continue

            if accepted.sha256 in seen_hashes:
                warnings.append(
                    VideoFilterWarning(
                        VideoFilterWarningCode.DUPLICATE_SOURCE_REUSED,
                        "Verified source content was reused within this request.",
                        item_id=item.item_id,
                    )
                )
            seen_hashes.add(accepted.sha256)
            accepted_items.append((item, accepted))
            item_results.append(
                VideoFilterItemResult(
                    item_id=item.item_id,
                    source_id=accepted.source_id,
                    source_sha256=accepted.sha256,
                    evidence_ref=accepted.artifact_ref or accepted.source_id,
                    metadata=metadata,
                )
            )

        if failures or len(accepted_items) != len(request.items):
            return _failed(
                request,
                tuple(item_results),
                (),
                tuple(warnings),
                tuple(failures),
            )

        metadata_by_item = {
            result.item_id: result.metadata
            for result in item_results
            if result.metadata is not None
        }
        frame_cache: dict[
            tuple[str, FrameSamplingConfig, tuple[str, str | None]], FrameSampleBatch
        ] = {}
        criterion_results: list[CriterionResult] = []

        for criterion in request.criteria:
            video_results: list[VideoCriterionResult] = []
            criterion_failed = False
            for item, accepted in accepted_items:
                if _expired(request.deadline):
                    failures.append(
                        VideoFilterFailure(
                            VideoFilterFailureCode.TIMEOUT,
                            "The video filter deadline expired.",
                            item_id=item.item_id,
                            criterion_id=criterion.criterion_id,
                        )
                    )
                    criterion_failed = True
                    break
                metadata = metadata_by_item[item.item_id]
                try:
                    if isinstance(criterion, MetadataCriterion):
                        observation = _evaluate_metadata(item, metadata, criterion)
                    else:
                        evaluator = self._evaluators.get(criterion.criterion_type.value)
                        if evaluator is None:
                            raise VideoFilterCapabilityError(
                                f"No evaluator is configured for {criterion.criterion_type.value}."
                            )
                        sampling = criterion.sampling
                        cache_key = (accepted.sha256, sampling, self._frame_sampler.identity)
                        batch = frame_cache.get(cache_key)
                        if batch is None:
                            batch = self._frame_sampler.sample(
                                accepted,
                                sampling,
                                request.deadline,
                            )
                            frame_cache[cache_key] = batch
                            warnings.extend(batch.warnings)
                        observation = evaluator.evaluate(
                            item,
                            accepted,
                            metadata,
                            batch.frames,
                            criterion,
                            request.deadline,
                        )
                except VideoFilterCapabilityError as error:
                    failures.append(
                        VideoFilterFailure(
                            error.failure_code,
                            str(error),
                            item_id=item.item_id,
                            criterion_id=criterion.criterion_id,
                        )
                    )
                    criterion_failed = True
                    continue
                except (OSError, ValueError) as error:
                    failures.append(
                        VideoFilterFailure(
                            VideoFilterFailureCode.INTERNAL,
                            str(error),
                            item_id=item.item_id,
                            criterion_id=criterion.criterion_id,
                        )
                    )
                    criterion_failed = True
                    continue
                video_results.append(observation)

            if not criterion_failed and len(video_results) == len(accepted_items):
                criterion_results.append(
                    CriterionResult(
                        criterion_id=criterion.criterion_id,
                        criterion_type=criterion.criterion_type,
                        video_results=tuple(video_results),
                        aggregation=criterion.videos_aggregation,
                        passed=aggregate(
                            tuple(result.passed for result in video_results),
                            criterion.videos_aggregation,
                        ),
                    )
                )

        if failures:
            return _failed(
                request,
                tuple(item_results),
                tuple(criterion_results),
                tuple(warnings),
                tuple(failures),
            )

        keep = aggregate(
            tuple(result.passed for result in criterion_results),
            request.aggregation,
        )
        return VideoFilterResult(
            status=VideoFilterStatus.SUCCEEDED,
            keep=keep,
            item_results=tuple(item_results),
            criterion_results=tuple(criterion_results),
            warnings=tuple(warnings),
            failures=(),
            aggregation=request.aggregation,
            request_id=request.request_id,
            trace_id=request.trace_id,
            attempt_id=request.attempt_id,
            operation_id=request.operation_id,
        )


def _evaluate_metadata(
    item: VideoFilterItem,
    metadata: VideoFilterMetadata,
    criterion: MetadataCriterion,
) -> VideoCriterionResult:
    checks = (
        in_range(
            metadata.duration_seconds,
            criterion.min_duration_seconds,
            criterion.max_duration_seconds,
        ),
        in_range(float(metadata.encoded_width), criterion.min_width, criterion.max_width),
        in_range(float(metadata.encoded_height), criterion.min_height, criterion.max_height),
        in_range(
            float(metadata.display_width),
            criterion.min_display_width,
            criterion.max_display_width,
        ),
        in_range(
            float(metadata.display_height),
            criterion.min_display_height,
            criterion.max_display_height,
        ),
        in_range(
            metadata.aspect_ratio,
            criterion.min_aspect_ratio,
            criterion.max_aspect_ratio,
        ),
    )
    return VideoCriterionResult(
        item_id=item.item_id,
        criterion_id=criterion.criterion_id,
        passed=all(checks),
        score=metadata.duration_seconds,
        details={
            "duration_seconds": metadata.duration_seconds,
            "encoded_width": metadata.encoded_width,
            "encoded_height": metadata.encoded_height,
            "display_width": metadata.display_width,
            "display_height": metadata.display_height,
            "aspect_ratio": metadata.aspect_ratio,
        },
    )


def _failed(
    request: VideoFilterRequest,
    item_results: tuple[VideoFilterItemResult, ...],
    criterion_results: tuple[CriterionResult, ...],
    warnings: tuple[VideoFilterWarning, ...],
    failures: tuple[VideoFilterFailure, ...],
) -> VideoFilterResult:
    return VideoFilterResult(
        status=VideoFilterStatus.FAILED,
        keep=None,
        item_results=item_results,
        criterion_results=criterion_results,
        warnings=warnings,
        failures=failures,
        aggregation=request.aggregation,
        request_id=request.request_id,
        trace_id=request.trace_id,
        attempt_id=request.attempt_id,
        operation_id=request.operation_id,
    )


def _expired(deadline: datetime) -> bool:
    return datetime.now(UTC) >= deadline


def _timeout_failure() -> VideoFilterFailure:
    return VideoFilterFailure(
        VideoFilterFailureCode.TIMEOUT,
        "The video filter deadline expired before execution.",
    )


def _source_failure_code(error: VideoSourceError) -> VideoFilterFailureCode:
    if "hash" in str(error).casefold():
        return VideoFilterFailureCode.SOURCE_CHANGED
    return VideoFilterFailureCode.UNSUPPORTED_MEDIA


def _criterion_resource_paths(criterion: object) -> tuple[Path, ...]:
    paths: list[Path] = []
    model = getattr(criterion, "model", None)
    if isinstance(model, LocalModelReference):
        reference_path = Path(model.reference)
        if model.cache_directory is None or reference_path.is_absolute():
            paths.append(reference_path.resolve())
        if model.cache_directory is not None:
            paths.append(Path(model.cache_directory).resolve())
    if isinstance(criterion, TagCriterion) and criterion.vocabulary_path is not None:
        paths.append(Path(criterion.vocabulary_path).resolve())
    return tuple(dict.fromkeys(paths))
