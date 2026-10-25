"""Compose selected local video Tools for the version-2 workspace host (RFC-0021).

Startup validates the closed configuration and checks only the files and Python
modules the selected Tools need. Media is staged through the shared handoff
(ADR-0026) during explicit calls; backends never read a caller-controlled path.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import stat
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import NoReturn

from capicove.contracts import (
    InvocationContext,
    ToolError,
    ToolErrorCode,
    ToolHandler,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.mcp.bindings import McpToolBinding
from capicove.processing.video.ports import (
    OutputStoreError,
    PendingOutput,
    StoredOutput,
    WrittenClip,
)
from capicove.runtime.executables import ExecutableChanged, PinnedExecutables
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.runtime.media_handoff import (
    MediaChanged,
    MediaDenied,
    MediaIntake,
    MediaInvalid,
    StagedFile,
    is_video_container,
    open_private_directory,
    restore_originals,
)


SPLIT = "capicove.video.split"
FILTER = "capicove.video.filter"
DEDUPLICATE = "capicove.video.deduplicate"
MAX_INPUT_BYTES = 64 * 1024**3
HOST_FILTER_CRITERIA = frozenset({"metadata", "motion"})
_FIELDS = frozenset(
    {
        "schema_version",
        "read_roots",
        "staging_root",
        "output_root",
        "backend",
        "ffmpeg",
        "ffprobe",
        "max_input_bytes",
    }
)
_REQUIRED = _FIELDS - {"output_root"}
_POLICY = "workspace-video"


class VideoDependencyError(ImportError):
    """A selected Tool needs an executable or module that is not available."""


@dataclass(frozen=True, slots=True)
class VideoSettings:
    """Validated startup authority for the video capability."""

    read_roots: tuple[Path, ...]
    staging_root: Path
    output_root: Path | None
    backend: str
    ffmpeg: Path | None
    ffprobe: Path | None
    max_input_bytes: int


def video_settings(config: WorkspaceHostConfiguration) -> VideoSettings:
    """Validate `capabilities.video` and the dependencies of the selected Tools."""
    if os.name != "posix":
        raise ValueError("video host Tools require POSIX no-follow file APIs in this release")
    raw = config.capabilities.get("video")
    if raw is None:
        raise ValueError("capabilities.video is not configured")
    if not _REQUIRED <= set(raw) <= _FIELDS:
        raise ValueError("capabilities.video requires exactly the documented fields")
    if raw["schema_version"] != 1:
        raise ValueError("capabilities.video schema_version must be 1")
    selected = set(config.enabled_tools)
    workspace = config.workspace_root
    roots = raw["read_roots"]
    if not isinstance(roots, tuple) or not 1 <= len(roots) <= 16 or len(set(roots)) != len(roots):
        raise ValueError("video read_roots must contain 1 to 16 unique paths")
    read_roots = tuple(_existing_directory(value, "video read_roots") for value in roots)
    staging = _private_child(raw["staging_root"], workspace, "video staging_root")
    output = (
        _private_child(raw["output_root"], workspace, "video output_root")
        if raw.get("output_root") is not None
        else None
    )
    owned = [staging, *([output] if output is not None else [])]
    for root in read_roots:
        for path in owned:
            if path.is_relative_to(root) or root.is_relative_to(path):
                raise ValueError("video staging and output roots must be disjoint from read_roots")
    if output is not None and (output.is_relative_to(staging) or staging.is_relative_to(output)):
        raise ValueError("video staging_root and output_root must be disjoint")
    backend = raw["backend"]
    if backend not in {"auto", "ffmpeg", "av"}:
        raise ValueError("video backend must be auto, ffmpeg, or av")
    ffmpeg = _executable(raw["ffmpeg"], "video ffmpeg")
    ffprobe = _executable(raw["ffprobe"], "video ffprobe")
    limit = raw["max_input_bytes"]
    if type(limit) is not int or not 1 <= limit <= MAX_INPUT_BYTES:
        raise ValueError("video max_input_bytes must be an integer from 1 through 64 GiB")
    settings = VideoSettings(read_roots, staging, output, backend, ffmpeg, ffprobe, limit)
    _check_dependencies(settings, selected)
    return settings


def _check_dependencies(settings: VideoSettings, selected: set[str]) -> None:
    if SPLIT in selected:
        if settings.output_root is None:
            raise ValueError("video output_root is required when split is selected")
        # Every split backend writes clips with FFmpeg, including the PyAV inspector.
        if settings.ffmpeg is None:
            raise VideoDependencyError("video split needs a configured ffmpeg executable")
        if settings.backend == "ffmpeg" and settings.ffprobe is None:
            raise VideoDependencyError("the ffmpeg backend needs a configured ffprobe")
        if settings.backend == "av" or (settings.backend == "auto" and settings.ffprobe is None):
            _require_modules("av")
        for executable in _split_executables(settings).values():
            _require_executable(executable)
    if FILTER in selected:
        if settings.ffprobe is None:
            raise VideoDependencyError("video filter needs a configured ffprobe")
        _require_executable(settings.ffprobe)
        _require_modules("av", "cv2", "numpy")
    if DEDUPLICATE in selected:
        _require_modules("xxhash")


def _split_executables(settings: VideoSettings) -> dict[str, Path]:
    executables = {"ffmpeg": settings.ffmpeg} if settings.ffmpeg is not None else {}
    if settings.backend != "av" and settings.ffprobe is not None:
        executables["ffprobe"] = settings.ffprobe
    return executables


@contextmanager
def compose_video_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Validate video authority and bind the selected Tools."""
    from capicove.tools.video_deduplicate import VIDEO_DEDUPLICATE_SPEC
    from capicove.tools.video_filter import VIDEO_FILTER_SPEC
    from capicove.tools.video_split import VIDEO_SPLIT_SPEC

    specs = {
        SPLIT: VIDEO_SPLIT_SPEC,
        FILTER: VIDEO_FILTER_SPEC,
        DEDUPLICATE: VIDEO_DEDUPLICATE_SPEC,
    }
    if any(tool_id not in specs for tool_id in config.enabled_tools):
        raise ValueError("video profile includes an unavailable Tool")
    for tool_id in config.enabled_tools:
        if not specs[tool_id].required_permissions.issubset(config.permissions):
            raise ValueError("configured permissions do not authorize every enabled Tool")
    settings = video_settings(config)
    # Snapshot executable identities at startup; calls run only these exact files.
    pins = {
        tool_id: PinnedExecutables(executables)
        for tool_id, executables in (
            (SPLIT, _split_executables(settings)),
            (FILTER, {"ffprobe": settings.ffprobe} if settings.ffprobe else {}),
        )
        if tool_id in config.enabled_tools
    }
    builders: dict[str, Callable[[], ToolHandler]] = {
        SPLIT: lambda: _split_handler(settings, pins[SPLIT]),
        FILTER: lambda: _filter_handler(settings, pins[FILTER]),
        DEDUPLICATE: lambda: _deduplicate_handler(settings),
    }
    yield tuple(
        McpToolBinding(
            "capicove_video_" + tool_id.rsplit(".", 1)[1],
            specs[tool_id],
            _catalog_only_handler if catalog_only else builders[tool_id](),
            context_factory,
        )
        for tool_id in sorted(config.enabled_tools)
    )


def _intake(settings: VideoSettings) -> MediaIntake:
    return MediaIntake(
        read_roots=settings.read_roots,
        staging_root=settings.staging_root,
        max_input_bytes=settings.max_input_bytes,
        accepts=lambda path, head: is_video_container(head),
    )


def _split_handler(settings: VideoSettings, pins: PinnedExecutables) -> ToolHandler:
    from capicove.processing.video import VideoSplitter
    from capicove.processing.video._adapters.ffmpeg import DefaultVideoBackendResolver
    from capicove.processing.video._adapters.filesystem import ExplicitPathAuthorizer
    from capicove.processing.video._adapters.scenedetect import PySceneDetectSceneDetector
    from capicove.tools.video_split import VIDEO_SPLIT_SPEC, VideoSplitToolHandler

    assert settings.output_root is not None
    output_root = settings.output_root
    intake = _intake(settings)
    av_available = settings.backend != "ffmpeg" and importlib.util.find_spec("av") is not None

    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        source = arguments.get("source")
        if (
            not isinstance(source, Mapping)
            or source.get("type") != "path"
            or set(source) != {"type", "path"}
            or not isinstance(source.get("path"), str)
            or "output_directory" in arguments
        ):
            return _failure(VIDEO_SPLIT_SPEC, context, ToolErrorCode.INVALID_INPUT)

        def run(staged: tuple[StagedFile, ...], executables: Mapping[str, str]) -> ToolResult:
            av_factory = None
            if av_available:
                from capicove.processing.video._adapters.pyav import create_pyav_backend

                def av_factory(mode: object) -> object:
                    return create_pyav_backend(mode, ffmpeg_executable=executables["ffmpeg"])  # type: ignore[arg-type]

            ffmpeg_pair = (
                (executables["ffmpeg"], executables["ffprobe"])
                if "ffprobe" in executables
                else None
            )
            with _PinnedOutputStore.open(output_root) as store:
                splitter = VideoSplitter(
                    backend_resolver=DefaultVideoBackendResolver(
                        av_factory,  # type: ignore[arg-type]
                        executables=ffmpeg_pair,
                        path_lookup=False,
                    ),
                    output_store=store,
                    path_authorizer=ExplicitPathAuthorizer(
                        {_POLICY: ((settings.staging_root,), (output_root,))}
                    ),
                    scene_detector=PySceneDetectSceneDetector(),
                )
                rewritten = {
                    **arguments,
                    "source": {"type": "path", "path": str(staged[0].staged)},
                }
                if rewritten.get("output_format") == "path":
                    rewritten["output_directory"] = str(output_root)
                result = VideoSplitToolHandler(splitter, _POLICY)(context, rewritten)
            original = (result.result or {}).get("original")
            if isinstance(original, Mapping) and original.get("sha256") not in (
                None,
                staged[0].sha256,
            ):
                return _failure(VIDEO_SPLIT_SPEC, context, ToolErrorCode.INVALID_INPUT)
            return result

        return _staged(VIDEO_SPLIT_SPEC, intake, pins, context, [Path(str(source["path"]))], run)

    return handler


def _filter_handler(settings: VideoSettings, pins: PinnedExecutables) -> ToolHandler:
    from capicove.processing.video._adapters.filesystem import ExplicitPathAuthorizer
    from capicove.processing.video.filtering._adapters.ffprobe_video_metadata import (
        FfprobeVideoMetadataProbe,
    )
    from capicove.processing.video.filtering._adapters.opencv_farneback_motion import (
        OpenCvFarnebackMotionEvaluator,
    )
    from capicove.processing.video.filtering._adapters.pyav_frame_sampler import (
        PyavVideoFrameSampler,
    )
    from capicove.processing.video.filtering.video_filter_pipeline import VideoFilterPipeline
    from capicove.tools.video_filter import VIDEO_FILTER_SPEC, VideoFilterToolHandler

    intake = _intake(settings)

    def handler(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
        sources = arguments.get("sources")
        criteria = arguments.get("criteria")
        if not isinstance(sources, (list, tuple)) or not isinstance(criteria, (list, tuple)):
            return _failure(VIDEO_FILTER_SPEC, context, ToolErrorCode.INVALID_INPUT)
        if any(
            not isinstance(item, Mapping) or item.get("type") not in HOST_FILTER_CRITERIA
            for item in criteria
        ):
            return _failure(VIDEO_FILTER_SPEC, context, ToolErrorCode.INVALID_INPUT)
        paths: list[Path] = []
        for item in sources:
            source = item.get("source") if isinstance(item, Mapping) else None
            if (
                not isinstance(source, Mapping)
                or source.get("type") != "path"
                or set(source) != {"type", "path"}
                or not isinstance(source.get("path"), str)
            ):
                return _failure(VIDEO_FILTER_SPEC, context, ToolErrorCode.INVALID_INPUT)
            paths.append(Path(str(source["path"])))

        def run(staged: tuple[StagedFile, ...], executables: Mapping[str, str]) -> ToolResult:
            pipeline = VideoFilterPipeline(
                metadata_probe=FfprobeVideoMetadataProbe(executables["ffprobe"]),
                frame_sampler=PyavVideoFrameSampler(),
                evaluators={"motion": OpenCvFarnebackMotionEvaluator()},
                path_authorizer=ExplicitPathAuthorizer({_POLICY: ((settings.staging_root,), ())}),
            )
            rewritten_sources = [
                {**dict(item), "source": {"type": "path", "path": str(copy.staged)}}
                for item, copy in zip(sources, staged, strict=True)
            ]
            result = VideoFilterToolHandler(pipeline, _POLICY)(
                context, {**arguments, "sources": rewritten_sources}
            )
            items = (result.result or {}).get("items", [])
            expected = {copy.sha256 for copy in staged}
            if isinstance(items, (list, tuple)) and any(
                isinstance(item, Mapping) and item.get("source_sha256") not in (None, *expected)
                for item in items
            ):
                return _failure(VIDEO_FILTER_SPEC, context, ToolErrorCode.INVALID_INPUT)
            return result

        return _staged(VIDEO_FILTER_SPEC, intake, pins, context, paths, run)

    return handler


class _PinnedOutputStore:
    """Allocate, finalize, and discard outputs only through one pinned directory.

    Backends receive ``/proc/<pid>/fd/<n>/name`` paths, so replacing ``output_root``
    with a link after startup cannot redirect where FFmpeg writes.
    """

    def __init__(self, logical_root: Path, descriptor: int) -> None:
        self._root = logical_root
        self._descriptor = descriptor
        self._bound = Path(f"/proc/{os.getpid()}/fd/{descriptor}")

    @classmethod
    @contextmanager
    def open(cls, root: Path) -> Iterator[_PinnedOutputStore]:
        descriptor = open_private_directory(root)
        try:
            yield cls(root, descriptor)
        finally:
            os.close(descriptor)

    def allocate(self, identity: str, suffix: str) -> PendingOutput:
        from capicove.processing.video._adapters.filesystem import FilesystemOutputStore

        FilesystemOutputStore._validate_identity(identity, suffix)
        name = f"{identity}{suffix}.partial"
        try:
            os.close(
                os.open(
                    name,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
                    0o600,
                    dir_fd=self._descriptor,
                )
            )
        except FileExistsError as error:
            raise OutputStoreError("a partial output already exists") from error
        return PendingOutput(identity, self._bound / name)

    def finalize(self, pending: PendingOutput, written: WrittenClip) -> StoredOutput:
        name = self._managed(pending)
        if written.path != pending.path:
            raise OutputStoreError("writer returned an unmanaged output path")
        actual = self._hash(name)
        if actual != written.sha256:
            raise OutputStoreError("written output hash verification failed")
        final = name.removesuffix(".partial")
        try:
            existing = self._hash(final)
        except FileNotFoundError:
            os.rename(name, final, src_dir_fd=self._descriptor, dst_dir_fd=self._descriptor)
        else:
            if existing != actual:
                raise OutputStoreError("output identity collision with mismatched content")
            os.unlink(name, dir_fd=self._descriptor)
        return StoredOutput(pending.identity, self._root / final, actual, written.mime_type)

    def load_bytes(self, stored: StoredOutput) -> bytes:
        payload = self._read(stored.path.name)
        if hashlib.sha256(payload).hexdigest() != stored.sha256:
            raise OutputStoreError("stored output hash verification failed")
        return payload

    def discard(self, pending: PendingOutput) -> None:
        try:
            os.unlink(self._managed(pending), dir_fd=self._descriptor)
        except FileNotFoundError:
            pass

    def _managed(self, pending: PendingOutput) -> str:
        name = pending.path.name
        if (
            pending.path.parent != self._bound
            or not name.endswith(".partial")
            or not name.startswith(pending.identity)
        ):
            raise OutputStoreError("partial output path is unmanaged")
        return name

    def _hash(self, name: str) -> str:
        return hashlib.sha256(self._read(name)).hexdigest()

    def _read(self, name: str) -> bytes:
        descriptor = os.open(
            name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=self._descriptor
        )
        with os.fdopen(descriptor, "rb") as stream:
            if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                raise OutputStoreError("output is not a regular file")
            return stream.read()


def _deduplicate_handler(settings: VideoSettings) -> ToolHandler:
    import xxhash

    from capicove.tools.video_deduplicate import DigestUnavailable, VideoDeduplicateToolHandler

    intake = _intake(settings)

    class Port:
        hash_version = str(xxhash.VERSION)

        def authorize(self, path: Path) -> None:
            try:
                intake.authorize(path)
            except MediaDenied:
                raise PermissionError("path outside read roots") from None

        def digest(self, path: Path) -> tuple[int, str, str]:
            try:
                result = intake.digest(path, fast_hash=xxhash.xxh3_128)
            except MediaDenied:
                raise DigestUnavailable("unreadable") from None
            except MediaInvalid:
                raise DigestUnavailable("too_large") from None
            except MediaChanged:
                raise DigestUnavailable("changed") from None
            except OSError:
                raise DigestUnavailable("unreadable") from None
            assert result.xxh3_128 is not None
            return result.size_bytes, result.xxh3_128, result.sha256

    return VideoDeduplicateToolHandler(Port())


def _staged(
    spec: ToolSpec,
    intake: MediaIntake,
    pins: PinnedExecutables,
    context: InvocationContext,
    paths: list[Path],
    run: Callable[[tuple[StagedFile, ...], Mapping[str, str]], ToolResult],
) -> ToolResult:
    if spec.required_permissions.difference(context.permissions):
        return _failure(spec, context, ToolErrorCode.PERMISSION_DENIED)
    try:
        for path in paths:
            intake.authorize(path)
        with pins.open() as executables, intake.stage(paths) as staged:
            result = run(staged, executables)
            restored = restore_originals(
                dict(result.result) if result.result is not None else None, staged
            )
            error = result.error
            if error is not None:
                message = restore_originals(error.message, staged)
                error = replace(error, message=str(message))
            return replace(
                result,
                result=restored if isinstance(restored, dict) else None,
                error=error,
            )
    except MediaDenied:
        return _failure(spec, context, ToolErrorCode.PERMISSION_DENIED)
    except ExecutableChanged:
        return _failure(spec, context, ToolErrorCode.UNAVAILABLE)
    except (MediaInvalid, MediaChanged):
        return _failure(spec, context, ToolErrorCode.INVALID_INPUT)


def _failure(spec: ToolSpec, context: InvocationContext, code: ToolErrorCode) -> ToolResult:
    messages = {
        ToolErrorCode.UNAVAILABLE: "A configured executable changed since startup.",
        ToolErrorCode.PERMISSION_DENIED: "The video is outside the configured authority.",
        ToolErrorCode.INVALID_INPUT: "The video request or media is not accepted by this host.",
    }
    now = datetime.now(UTC)
    return ToolResult(
        request_id=context.request_id,
        trace_id=context.trace_id,
        tool_id=spec.tool_id,
        tool_version=spec.contract_version,
        status=ToolStatus.FAILED,
        error=ToolError(code, messages[code], retryable=False),
        started_at=now,
        completed_at=now,
    )


def _catalog_only_handler(
    context: InvocationContext, arguments: Mapping[str, object], /
) -> NoReturn:
    raise RuntimeError("catalog-only video bindings cannot be invoked")


def _existing_directory(value: object, field: str) -> Path:
    path = _absolute(value, field)
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError(f"{field} must not contain symlinks")
    if not path.is_dir():
        raise ValueError(f"{field} must name existing directories")
    return path


def _private_child(value: object, workspace: Path, field: str) -> Path:
    path = _absolute(value, field)
    if path == workspace or not path.is_relative_to(workspace):
        raise ValueError(f"{field} must be strictly inside workspace_root")
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError(f"{field} must not contain symlinks")
    if path.exists():
        metadata = path.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or metadata.st_mode & 0o077:
            raise ValueError(f"{field} must be an owner-private directory")
    return path


def _absolute(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value.startswith("/") or "\0" in value:
        raise ValueError(f"{field} must be absolute paths")
    path = Path(value)
    if ".." in path.parts:
        raise ValueError(f"{field} must be normalized absolute paths")
    return path


def _executable(value: object, field: str) -> Path | None:
    if value is None:
        return None
    return _absolute(value, field)


def _require_executable(path: Path) -> None:
    if not path.is_file() or not os.access(path, os.X_OK):
        raise VideoDependencyError(f"{path.name} is not an executable file")


def _require_modules(*names: str) -> None:
    missing = [name for name in names if importlib.util.find_spec(name) is None]
    if missing:
        raise VideoDependencyError("missing Python modules: " + ", ".join(missing))


__all__ = ["VideoDependencyError", "VideoSettings", "compose_video_mcp", "video_settings"]
