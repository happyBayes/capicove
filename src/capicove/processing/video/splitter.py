"""Best-effort orchestration for transport-neutral video splitting."""

from __future__ import annotations

from datetime import UTC, datetime

from capicove.processing.video.identity import configuration_hash, output_identity
from capicove.processing.video.models import (
    AcceptedVideoSource,
    DurationSplitConfig,
    KeyframeSplitConfig,
    SceneSplitConfig,
    TimeRange,
    VideoAsset,
    VideoOutputFormat,
    VideoSegment,
    VideoSplitFailureCode,
    VideoSplitRequest,
    VideoSplitResult,
    VideoSplitStatus,
    VideoSplitStrategy,
    VideoSplitWarning,
    VideoSplitWarningCode,
)
from capicove.processing.video.ports import (
    ArtifactResolver,
    BackendUnavailableError,
    OutputStore,
    PathAuthorizer,
    PathPolicyError,
    SceneDetectionError,
    SceneDetector,
    VideoBackendBundle,
    VideoBackendResolver,
    VideoProcessingError,
)
from capicove.processing.video.source import accept_source
from capicove.processing.video.strategies import (
    PlanningFailure,
    normalize_scene_ranges,
    plan_duration_ranges,
    plan_keyframe_ranges,
    validate_scene_config,
)


class VideoSplitter:
    """Plan and write video ranges while preserving successful evidence."""

    def __init__(
        self,
        backend_resolver: VideoBackendResolver,
        output_store: OutputStore,
        path_authorizer: PathAuthorizer,
        scene_detector: SceneDetector | None = None,
        artifact_resolver: ArtifactResolver | None = None,
    ) -> None:
        self._backend_resolver = backend_resolver
        self._output_store = output_store
        self._path_authorizer = path_authorizer
        self._scene_detector = scene_detector
        self._artifact_resolver = artifact_resolver

    def split(self, request: VideoSplitRequest) -> VideoSplitResult:
        """Execute one split request with visible best-effort semantics."""
        accepted = accept_source(
            request.source,
            request.deadline,
            artifact_resolver=self._artifact_resolver,
            path_authorizer=self._path_authorizer,
            policy_ref=request.execution_policy_ref,
        )

        original = _original_asset(accepted)
        normalized = _normalized_config(request)
        config_hash = configuration_hash(normalized)
        if datetime.now(UTC) >= request.deadline:
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend="unresolved",
                backend_version=None,
                failure_code=VideoSplitFailureCode.TIMEOUT,
                message="The video split deadline expired before execution.",
            )
        if request.output_directory is not None:
            try:
                self._path_authorizer.authorize_write(
                    request.execution_policy_ref,
                    request.output_directory.resolve(),
                )
            except PathPolicyError:
                return _failed_result(
                    request,
                    original,
                    normalized,
                    config_hash,
                    backend="unresolved",
                    backend_version=None,
                    failure_code=VideoSplitFailureCode.PERMISSION_DENIED,
                    message="The output path is not permitted.",
                )

        try:
            backend = self._backend_resolver.resolve(request.backend, request.write_mode)
        except BackendUnavailableError:
            warning = VideoSplitWarning(
                VideoSplitWarningCode.BACKEND_UNAVAILABLE,
                "The requested video backend is unavailable.",
            )
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend=request.backend.value,
                backend_version=None,
                failure_code=VideoSplitFailureCode.BACKEND_UNAVAILABLE,
                message="The requested video backend is unavailable.",
                warnings=(warning, _fallback_warning()),
            )

        try:
            ranges, planning_warnings, degraded, backend_identity = self._plan(
                request,
                accepted,
                backend,
            )
        except PlanningFailure as error:
            warning = VideoSplitWarning(error.warning_code, str(error))
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend=backend.name,
                backend_version=backend.version,
                failure_code=VideoSplitFailureCode.INVALID_INPUT,
                message=str(error),
                warnings=(warning, _fallback_warning()),
            )
        except BackendUnavailableError:
            warning = VideoSplitWarning(
                VideoSplitWarningCode.BACKEND_UNAVAILABLE,
                "A required video backend is unavailable.",
            )
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend=backend.name,
                backend_version=backend.version,
                failure_code=VideoSplitFailureCode.BACKEND_UNAVAILABLE,
                message="A required video backend is unavailable.",
                warnings=(warning, _fallback_warning()),
            )
        except SceneDetectionError:
            warning = VideoSplitWarning(
                VideoSplitWarningCode.SCENE_DETECTION_FAILED,
                "Scene detection failed; the original video was retained.",
            )
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend=backend.name,
                backend_version=backend.version,
                failure_code=VideoSplitFailureCode.INTERNAL,
                message="Scene detection failed.",
                warnings=(warning, _fallback_warning()),
            )
        except VideoProcessingError:
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend=backend.name,
                backend_version=backend.version,
                failure_code=VideoSplitFailureCode.INTERNAL,
                message="Video inspection failed.",
                warnings=(_fallback_warning(),),
            )

        if not ranges:
            warnings = planning_warnings
            if degraded:
                warnings += (_fallback_warning(),)
            return VideoSplitResult(
                status=VideoSplitStatus.SUCCEEDED,
                original=original,
                segments=(),
                warnings=warnings,
                degraded=degraded,
                strategy=request.strategy,
                normalized_config=normalized,
                configuration_hash=config_hash,
                backend=backend_identity,
                backend_version=backend.version,
                request_id=request.request_id,
                trace_id=request.trace_id,
                attempt_id=request.attempt_id,
                operation_id=request.operation_id,
            )

        segments: list[VideoSegment] = []
        write_warnings = list(planning_warnings)
        for index, time_range in enumerate(ranges):
            identity = output_identity(
                accepted.sha256,
                request.strategy.value,
                config_hash,
                backend_identity,
                index,
            )
            try:
                pending = self._output_store.allocate(identity, _output_suffix(accepted))
            except (VideoProcessingError, OSError) as error:
                write_warnings.append(
                    VideoSplitWarning(
                        VideoSplitWarningCode.SEGMENT_FAILED,
                        f"Segment {index} allocation failed: {type(error).__name__}.",
                        segment_index=index,
                        time_range=time_range,
                    )
                )
                continue
            finalized = False
            try:
                written = backend.writer.write(
                    accepted,
                    time_range,
                    pending,
                    request.deadline,
                )
                stored = self._output_store.finalize(pending, written)
                finalized = True
                data = (
                    self._output_store.load_bytes(stored)
                    if request.output_format is VideoOutputFormat.BYTES
                    else None
                )
                segments.append(
                    VideoSegment(
                        index=index,
                        time_range=time_range,
                        output_id=identity,
                        sha256=stored.sha256,
                        mime_type=stored.mime_type,
                        source_id=accepted.source_id,
                        strategy=request.strategy,
                        operation_id=request.operation_id,
                        configuration_hash=config_hash,
                        backend=backend_identity,
                        backend_version=backend.version,
                        path=(
                            stored.path if request.output_format is VideoOutputFormat.PATH else None
                        ),
                        data=data,
                    )
                )
            except (VideoProcessingError, OSError) as error:
                write_warnings.append(
                    VideoSplitWarning(
                        VideoSplitWarningCode.SEGMENT_FAILED,
                        f"Segment {index} failed: {type(error).__name__}.",
                        segment_index=index,
                        time_range=time_range,
                    )
                )
            finally:
                if not finalized:
                    try:
                        self._output_store.discard(pending)
                    except (VideoProcessingError, OSError):
                        write_warnings.append(
                            VideoSplitWarning(
                                VideoSplitWarningCode.OUTPUT_CLEANUP_FAILED,
                                f"Cleanup failed for segment {index}.",
                                segment_index=index,
                                time_range=time_range,
                            )
                        )

        if not segments:
            write_warnings.append(_fallback_warning())
            return _failed_result(
                request,
                original,
                normalized,
                config_hash,
                backend=backend_identity,
                backend_version=backend.version,
                failure_code=VideoSplitFailureCode.INTERNAL,
                message="All planned video segments failed.",
                warnings=tuple(write_warnings),
            )
        return VideoSplitResult(
            status=VideoSplitStatus.SUCCEEDED,
            original=original,
            segments=tuple(segments),
            warnings=tuple(write_warnings),
            degraded=degraded or len(segments) != len(ranges),
            strategy=request.strategy,
            normalized_config=normalized,
            configuration_hash=config_hash,
            backend=backend_identity,
            backend_version=backend.version,
            request_id=request.request_id,
            trace_id=request.trace_id,
            attempt_id=request.attempt_id,
            operation_id=request.operation_id,
        )

    def _plan(
        self,
        request: VideoSplitRequest,
        source: AcceptedVideoSource,
        backend: VideoBackendBundle,
    ) -> tuple[tuple[TimeRange, ...], tuple[VideoSplitWarning, ...], bool, str]:
        metadata = backend.inspector.inspect(source, request.deadline)
        backend_identity = _backend_identity(backend.name, backend.version)
        if request.strategy is VideoSplitStrategy.DURATION:
            if not isinstance(request.config, DurationSplitConfig):
                raise ValueError("strategy must match its configuration type")
            return (
                plan_duration_ranges(metadata.duration_seconds, request.config),
                (),
                False,
                backend_identity,
            )
        if request.strategy is VideoSplitStrategy.KEYFRAME:
            if not isinstance(request.config, KeyframeSplitConfig):
                raise ValueError("strategy must match its configuration type")
            plan = plan_keyframe_ranges(
                metadata.duration_seconds,
                metadata.keyframe_timestamps,
                request.config,
            )
            warnings = (plan.warning,) if plan.warning is not None else ()
            return plan.ranges, warnings, plan.degraded, backend_identity
        if not isinstance(request.config, SceneSplitConfig):
            raise ValueError("strategy must match its configuration type")
        validate_scene_config(request.config)
        if self._scene_detector is None:
            raise SceneDetectionError("scene detection backend is unavailable")
        detected = self._scene_detector.detect(source, request.config, request.deadline)
        detector_name, detector_version = self._scene_detector.identity
        composed = f"{_backend_identity(detector_name, detector_version)}+{backend_identity}"
        return normalize_scene_ranges(metadata.duration_seconds, detected), (), False, composed


def _normalized_config(request: VideoSplitRequest) -> dict[str, object]:
    if isinstance(request.config, DurationSplitConfig):
        normalized: dict[str, object] = {
            "split_duration": request.config.split_duration,
            "overlap_duration": request.config.overlap_duration,
            "min_last_split_duration": request.config.min_last_split_duration,
        }
    elif isinstance(request.config, KeyframeSplitConfig):
        normalized = {"min_segment_duration": request.config.min_segment_duration}
    else:
        normalized = {
            "detector": request.config.detector.value,
            "options": dict(request.config.options),
        }
    normalized["write_mode"] = request.write_mode.value
    return normalized


def _original_asset(source: AcceptedVideoSource) -> VideoAsset:
    if source.artifact_ref is not None:
        return VideoAsset(
            source_id=source.source_id,
            sha256=source.sha256,
            mime_type=source.mime_type,
            artifact_ref=source.artifact_ref,
        )
    return VideoAsset(
        source_id=source.source_id,
        sha256=source.sha256,
        mime_type=source.mime_type,
        path=source.path,
        data=source.data,
    )


def _backend_identity(name: str, version: str | None) -> str:
    return name if version is None else f"{name}:{version}"


def _output_suffix(source: AcceptedVideoSource) -> str:
    if source.path is not None and source.path.suffix:
        return source.path.suffix
    return ".mp4"


def _fallback_warning() -> VideoSplitWarning:
    return VideoSplitWarning(
        VideoSplitWarningCode.FALLBACK_TO_ORIGINAL,
        "The original video was retained.",
    )


def _failed_result(
    request: VideoSplitRequest,
    original: VideoAsset,
    normalized_config: dict[str, object],
    config_hash: str,
    *,
    backend: str,
    backend_version: str | None,
    failure_code: VideoSplitFailureCode,
    message: str,
    warnings: tuple[VideoSplitWarning, ...] = (),
) -> VideoSplitResult:
    return VideoSplitResult(
        status=VideoSplitStatus.FAILED,
        original=original,
        segments=(),
        warnings=warnings,
        degraded=True,
        strategy=request.strategy,
        normalized_config=normalized_config,
        configuration_hash=config_hash,
        backend=backend,
        backend_version=backend_version,
        request_id=request.request_id,
        trace_id=request.trace_id,
        attempt_id=request.attempt_id,
        operation_id=request.operation_id,
        failure_code=failure_code,
        failure_message=message,
        retryable=failure_code is VideoSplitFailureCode.BACKEND_UNAVAILABLE,
    )
