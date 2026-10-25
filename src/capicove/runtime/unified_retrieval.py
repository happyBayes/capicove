"""Offline shared-space retrieval: import, build, activate, Search and Evidence (RFC-0032).

Configuration comes only from the version-2 host's ``capabilities.multimodal``
object. The ``audio-local-v1`` profile indexes native audio windows; every
supported query is encoded by one explicitly configured shared-space model.
Import, forget, build and activation are explicit operator writes. Search and
Evidence lease one complete, digest-verified generation and never create storage
or decode indexed targets.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import secrets
import shutil
import sqlite3
import stat
import uuid
from collections.abc import Iterator, Mapping, Sequence
from contextlib import closing, contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import (
    AudioClipQuery,
    CharacterRange,
    CompactHit,
    DetailLevel,
    EmbeddingSpaceRef,
    FullHit,
    ImageQuery,
    InlineBytesSource,
    LaneScore,
    LocalPathSource,
    MatchKind,
    QueryModality,
    SearchFailureCode,
    SearchFilters,
    SearchRequest,
    SearchResult,
    SearchStatus,
    SearchWarningCode,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
    TextQuery,
)
from capicove.retrieval.snippets import preview_snippet
from capicove.runtime.mcp_config import WorkspaceHostConfiguration


if TYPE_CHECKING:
    from capicove.model.factory import ModelFactory
    from capicove.processing.audio.retrieval import AudioWindowPreparer


FORMAT_VERSION = "unified-generation-v1"
INVENTORY_VERSION = 1
MAX_SOURCE_ROOTS = 16
MAX_UNITS = 10_000
MAX_SOURCE_BYTES = 4 * 1024**3
MAX_INLINE_QUERY_BYTES = 4 * 1024 * 1024
CANDIDATE_DEPTH = 100
_CONFIG_FIELDS = frozenset(
    {
        "schema_version",
        "profile",
        "source_roots",
        "index_root",
        "staging_root",
        "corpus_ref",
        "model_config",
        "model_binding",
        "ffmpeg",
        "ffprobe",
        "max_source_bytes",
    }
)
_ALIAS = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_BINDING = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_GENERATION = re.compile(r"^ug-[0-9a-f]{64}$")
_EVIDENCE = re.compile(r"^mev-[0-9a-f]{64}$")
_AUDIO_SUFFIXES = {
    ".aac": "audio/aac",
    ".flac": "audio/flac",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".ogg": "audio/ogg",
    ".opus": "audio/ogg",
    ".wav": "audio/wav",
    ".webm": "audio/webm",
}
_TEXT_SUFFIXES = {".md": "text/markdown", ".markdown": "text/markdown"}
_IMAGE_SUFFIXES = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}
_IMAGE_SIGNATURES = ((b"\x89PNG\r\n\x1a\n", "image/png"), (b"\xff\xd8\xff", "image/jpeg"))
_VIDEO_SUFFIXES = {".mp4": "video/mp4", ".mov": "video/quicktime", ".mkv": "video/x-matroska"}
_MEDIA_SUFFIXES = {
    value: key
    for key, value in reversed(
        list({**_AUDIO_SUFFIXES, **_IMAGE_SUFFIXES, **_VIDEO_SUFFIXES}.items())
    )
}
# Keyframes are sampled at a fixed interval from the first video stream; a video
# needing more frames than the budget is rejected rather than silently truncated.
KEYFRAME_POLICY = "interval-10s-max-side-896-v1"
KEYFRAME_INTERVAL_MS = 10_000
KEYFRAME_MAX_SIDE = 896
MAX_KEYFRAMES_PER_VIDEO = 360
CHUNKING_VERSION = "capicove-markdown-paragraphs-v1"
MAX_SNIPPET_CHARACTERS = 2_048
RRF_K = 60


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    """Declare one profile's allowed query modalities and targets."""

    profile_id: str
    targets: frozenset[TargetType]
    queries: frozenset[QueryModality]
    modalities: frozenset[str]


PROFILES = {
    "audio-local-v1": ProfileSpec(
        "audio-local-v1",
        frozenset({TargetType.AUDIO_SEGMENT}),
        frozenset({QueryModality.TEXT, QueryModality.AUDIO_CLIP}),
        frozenset({"audio"}),
    ),
    "unified-local-v1": ProfileSpec(
        "unified-local-v1",
        frozenset(
            {TargetType.TEXT_CHUNK, TargetType.IMAGE, TargetType.AUDIO_SEGMENT, TargetType.KEYFRAME}
        ),
        frozenset({QueryModality.TEXT, QueryModality.IMAGE, QueryModality.AUDIO_CLIP}),
        frozenset({"audio", "text", "image", "video"}),
    ),
}


class UnifiedRetrievalError(ValueError):
    """Report a stable, path-free operator failure."""


@dataclass(frozen=True, slots=True)
class UnifiedSettings:
    """Validated startup-owned shared-space retrieval authority."""

    workspace_id: str
    workspace_root: Path
    host_profile: str
    profile: ProfileSpec
    source_roots: Mapping[str, Path]
    index_root: Path
    staging_root: Path
    corpus_ref: str
    model_config: Path | None
    model_binding: str
    ffmpeg: str
    ffprobe: str
    max_source_bytes: int


def unified_settings(config: WorkspaceHostConfiguration) -> UnifiedSettings:
    """Validate the closed version-1 ``capabilities.multimodal`` object."""
    if os.name != "posix":
        raise UnifiedRetrievalError("shared-space retrieval requires POSIX no-follow file APIs")
    raw = config.capabilities.get("multimodal")
    if raw is None:
        raise UnifiedRetrievalError("capabilities.multimodal is not configured")
    if set(raw) != _CONFIG_FIELDS:
        raise UnifiedRetrievalError(
            "capabilities.multimodal requires exactly the documented fields"
        )
    if raw["schema_version"] != 1 or type(raw["schema_version"]) is not int:
        raise UnifiedRetrievalError("capabilities.multimodal schema_version must be 1")
    selected = raw["profile"]
    profile = PROFILES.get(selected) if isinstance(selected, str) else None
    if profile is None:
        raise UnifiedRetrievalError("unsupported multimodal retrieval profile")
    root = config.workspace_root
    index_root = _inside(raw["index_root"], root, "index_root")
    staging_root = _inside(raw["staging_root"], root, "staging_root")
    if (
        index_root == staging_root
        or index_root.is_relative_to(staging_root)
        or (staging_root.is_relative_to(index_root))
    ):
        raise UnifiedRetrievalError("index_root and staging_root must be disjoint")
    sources = raw["source_roots"]
    if not isinstance(sources, Mapping) or not 1 <= len(sources) <= MAX_SOURCE_ROOTS:
        raise UnifiedRetrievalError("source_roots must contain 1 to 16 aliases")
    roots: dict[str, Path] = {}
    for alias, value in sources.items():
        if not isinstance(alias, str) or _ALIAS.fullmatch(alias) is None:
            raise UnifiedRetrievalError("source root aliases must be lowercase identifiers")
        path = _absolute(value, "source_roots")
        if path.resolve() != path or not path.is_dir():
            raise UnifiedRetrievalError("source roots must be existing unlinked directories")
        for managed in (index_root, staging_root):
            if path == managed or path.is_relative_to(managed) or managed.is_relative_to(path):
                raise UnifiedRetrievalError("source roots must be disjoint from managed roots")
        roots[alias] = path
    corpus = raw["corpus_ref"]
    if not isinstance(corpus, str) or not corpus.strip() or len(corpus) > 256:
        raise UnifiedRetrievalError("corpus_ref must be a bounded nonblank string")
    model_config = raw["model_config"]
    if model_config is not None:
        model_config = _absolute(model_config, "model_config")
    binding = raw["model_binding"]
    if not isinstance(binding, str) or _BINDING.fullmatch(binding) is None:
        raise UnifiedRetrievalError("model_binding must name a configured binding")
    for name in ("ffmpeg", "ffprobe"):
        _absolute(raw[name], name)
    limit = raw["max_source_bytes"]
    if type(limit) is not int or not 1 <= limit <= MAX_SOURCE_BYTES:
        raise UnifiedRetrievalError("max_source_bytes must be an integer from 1 to 4 GiB")
    return UnifiedSettings(
        workspace_id=config.workspace_id,
        workspace_root=root,
        host_profile=config.profile,
        profile=profile,
        source_roots=roots,
        index_root=index_root,
        staging_root=staging_root,
        corpus_ref=corpus,
        model_config=model_config,
        model_binding=binding,
        ffmpeg=str(raw["ffmpeg"]),
        ffprobe=str(raw["ffprobe"]),
        max_source_bytes=limit,
    )


def _absolute(value: object, field: str) -> Path:
    if not isinstance(value, str) or not value or "\0" in value:
        raise UnifiedRetrievalError(f"{field} must be an absolute path")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        raise UnifiedRetrievalError(f"{field} must be an absolute path")
    return path


def _inside(value: object, root: Path, field: str) -> Path:
    path = _absolute(value, field)
    if path == root or not path.is_relative_to(root):
        raise UnifiedRetrievalError(f"{field} must be inside workspace_root")
    return path


def _writable(config: WorkspaceHostConfiguration) -> UnifiedSettings:
    settings = unified_settings(config)
    if settings.host_profile not in {"write", "admin"}:
        raise PermissionError("import, build and activation require a write or admin profile")
    return settings


def _private_directory(path: Path) -> None:
    path.mkdir(mode=0o700, parents=True, exist_ok=True)
    identity = path.lstat()
    if not stat.S_ISDIR(identity.st_mode) or identity.st_uid != os.geteuid():
        raise UnifiedRetrievalError("managed directories must be owned unlinked directories")
    if identity.st_mode & 0o077:
        os.chmod(path, 0o700)


@contextmanager
def _locked(settings: UnifiedSettings) -> Iterator[None]:
    """Serialize import, forget, build and activation across processes."""
    _private_directory(settings.index_root)
    descriptor = os.open(
        settings.index_root / ".lock", os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


@contextmanager
def _inventory(settings: UnifiedSettings) -> Iterator[sqlite3.Connection]:
    connection = sqlite3.connect(settings.index_root / "inventory.sqlite3", isolation_level=None)
    connection.row_factory = sqlite3.Row
    try:
        connection.execute(
            "CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT OR IGNORE INTO meta (key, value) VALUES ('version', ?)",
            (str(INVENTORY_VERSION),),
        )
        version = connection.execute("SELECT value FROM meta WHERE key = 'version'").fetchone()
        if version[0] != str(INVENTORY_VERSION):
            raise UnifiedRetrievalError("unsupported multimodal inventory version")
        connection.execute(
            "CREATE TABLE IF NOT EXISTS events ("
            " seq INTEGER PRIMARY KEY AUTOINCREMENT, source_ref TEXT NOT NULL,"
            " event TEXT NOT NULL CHECK (event IN ('import', 'forget')),"
            " sha256 TEXT, size_bytes INTEGER, media_type TEXT, modality TEXT,"
            " stream_index INTEGER, selection_method TEXT, stream_count INTEGER,"
            " recorded_at TEXT NOT NULL)"
        )
        yield connection
    finally:
        connection.close()


def _latest(connection: sqlite3.Connection, source_ref: str) -> sqlite3.Row | None:
    row: sqlite3.Row | None = connection.execute(
        "SELECT * FROM events WHERE source_ref = ? ORDER BY seq DESC LIMIT 1", (source_ref,)
    ).fetchone()
    return row


def _live(connection: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        connection.execute(
            "SELECT e.* FROM events e JOIN (SELECT source_ref, MAX(seq) AS seq FROM events"
            " GROUP BY source_ref) latest ON e.seq = latest.seq"
            " WHERE e.event = 'import' ORDER BY e.source_ref"
        )
    )


def _relative(relative_path: str) -> tuple[str, ...]:
    if not isinstance(relative_path, str) or not relative_path or "\0" in relative_path:
        raise UnifiedRetrievalError("relative_path must be a nonblank relative path")
    parts = tuple(relative_path.split("/"))
    if relative_path.startswith("/") or any(part in {"", ".", ".."} for part in parts):
        raise UnifiedRetrievalError("relative_path must not escape its source root")
    return parts


def _authorized_file(settings: UnifiedSettings, alias: str, relative_path: str) -> Path:
    root = settings.source_roots.get(alias)
    if root is None:
        raise UnifiedRetrievalError("unknown source root alias")
    path = root.joinpath(*_relative(relative_path))
    try:
        identity = path.lstat()
    except OSError:
        raise UnifiedRetrievalError("source is not a readable regular file") from None
    if path.resolve() != path or not stat.S_ISREG(identity.st_mode):
        raise UnifiedRetrievalError("source must be an unlinked regular file")
    if identity.st_size > settings.max_source_bytes:
        raise UnifiedRetrievalError("source exceeds max_source_bytes")
    return path


def _audio_preparer(settings: UnifiedSettings) -> AudioWindowPreparer:
    from capicove.processing.audio import AudioProcessor
    from capicove.processing.audio._adapters.audio_windows import FfmpegAudioWindowBackend
    from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
    from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector
    from capicove.processing.audio.retrieval import AudioWindowPreparer

    return AudioWindowPreparer(
        AudioProcessor(
            FfprobeAudioStreamInspector(settings.ffprobe, restrict_inputs=True),
            FfmpegAudioValidator(settings.ffmpeg, restrict_inputs=True),
        ),
        FfmpegAudioWindowBackend(settings.ffmpeg),
    )


def _deadline(minutes: int = 60) -> datetime:
    return datetime.now(UTC) + timedelta(minutes=minutes)


def import_media(
    config: WorkspaceHostConfiguration,
    alias: str,
    relative_path: str,
    *,
    audio_stream_index: int | None = None,
) -> dict[str, Any]:
    """Preserve one authorized source and append an import event when it changed."""
    settings = _writable(config)
    path = _authorized_file(settings, alias, relative_path)
    suffix = path.suffix.lower()
    modality = (
        "audio"
        if suffix in _AUDIO_SUFFIXES
        else "text"
        if suffix in _TEXT_SUFFIXES
        else "image"
        if suffix in _IMAGE_SUFFIXES
        else "video"
        if suffix in _VIDEO_SUFFIXES
        else None
    )
    if modality is None or modality not in settings.profile.modalities:
        allowed = ", ".join(sorted(settings.profile.modalities))
        raise UnifiedRetrievalError(f"the selected profile accepts only {allowed} sources")
    source_ref = f"{alias}:{relative_path}"
    if modality == "video":
        return _import_video(settings, path, source_ref, suffix, audio_stream_index)
    if modality != "audio":
        return _import_blob(settings, path, source_ref, modality, suffix)
    with _locked(settings):
        from capicove.processing.audio import AudioImportError, AudioProcessor
        from capicove.processing.audio._adapters.ffmpeg import FfmpegAudioValidator
        from capicove.processing.audio._adapters.ffprobe import FfprobeAudioStreamInspector

        store = settings.index_root / "evidence"
        _private_directory(store)
        processor = AudioProcessor(
            FfprobeAudioStreamInspector(settings.ffprobe, restrict_inputs=True),
            FfmpegAudioValidator(settings.ffmpeg, restrict_inputs=True),
        )
        try:
            imported = processor.import_file_with_selection(
                path,
                store,
                **(
                    {} if audio_stream_index is None else {"audio_stream_index": audio_stream_index}
                ),
                deadline=_deadline(10),
            )
        except AudioImportError as error:
            raise UnifiedRetrievalError(f"audio source rejected: {error.code.value}") from None
        evidence, selection = imported.evidence, imported.stream_selection
        with _inventory(settings) as connection:
            latest = _latest(connection, source_ref)
            unchanged = (
                latest is not None
                and latest["event"] == "import"
                and latest["sha256"] == evidence.sha256
                and latest["stream_index"] == selection.selected_audio_stream_index
            )
            if not unchanged:
                connection.execute(
                    "INSERT INTO events (source_ref, event, sha256, size_bytes, media_type,"
                    " modality, stream_index, selection_method, stream_count, recorded_at)"
                    " VALUES (?, 'import', ?, ?, ?, 'audio', ?, ?, ?, ?)",
                    (
                        source_ref,
                        evidence.sha256,
                        evidence.size_bytes,
                        _AUDIO_SUFFIXES[suffix],
                        selection.selected_audio_stream_index,
                        selection.selection_method,
                        selection.preflight_audio_stream_count,
                        datetime.now(UTC).isoformat(),
                    ),
                )
    return {
        "source_ref": source_ref,
        "imported": not unchanged,
        "sha256": evidence.sha256,
        "modality": "audio",
        "audio_stream_index": selection.selected_audio_stream_index,
    }


def _import_blob(
    settings: UnifiedSettings, path: Path, source_ref: str, modality: str, suffix: str
) -> dict[str, Any]:
    """Validate and preserve a text or image source in the content-addressed store."""
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        identity = os.fstat(stream.fileno())
        if not stat.S_ISREG(identity.st_mode) or identity.st_size > settings.max_source_bytes:
            raise UnifiedRetrievalError("source must be a bounded regular file")
        payload = stream.read(settings.max_source_bytes + 1)
    if not payload or len(payload) > settings.max_source_bytes:
        raise UnifiedRetrievalError("source must be non-empty and within max_source_bytes")
    if modality == "text":
        media_type = _TEXT_SUFFIXES[suffix]
        try:
            text = payload.decode("utf-8")
        except UnicodeDecodeError:
            raise UnifiedRetrievalError("text sources must be valid UTF-8") from None
        if not text.strip():
            raise UnifiedRetrievalError("text source must contain non-blank text")
    else:
        media_type = _IMAGE_SUFFIXES[suffix]
        if not any(
            payload.startswith(sig) and kind == media_type for sig, kind in _IMAGE_SIGNATURES
        ):
            raise UnifiedRetrievalError("image content does not match its extension")
        _verify_image(payload)
    sha256 = hashlib.sha256(payload).hexdigest()
    with _locked(settings):
        _store_blob(settings.index_root / "evidence", sha256, payload)
        with _inventory(settings) as connection:
            latest = _latest(connection, source_ref)
            unchanged = (
                latest is not None and latest["event"] == "import" and latest["sha256"] == sha256
            )
            if not unchanged:
                connection.execute(
                    "INSERT INTO events (source_ref, event, sha256, size_bytes, media_type,"
                    " modality, recorded_at) VALUES (?, 'import', ?, ?, ?, ?, ?)",
                    (
                        source_ref,
                        sha256,
                        len(payload),
                        media_type,
                        modality,
                        datetime.now(UTC).isoformat(),
                    ),
                )
    return {
        "source_ref": source_ref,
        "imported": not unchanged,
        "sha256": sha256,
        "modality": modality,
    }


def _frame_sampler(settings: UnifiedSettings) -> Any:
    from capicove.processing.video.retrieval._adapters.sampled_frames import FfmpegFrameSampler

    return FfmpegFrameSampler(settings.ffmpeg, settings.ffprobe)


def _import_video(
    settings: UnifiedSettings,
    path: Path,
    source_ref: str,
    suffix: str,
    audio_stream_index: int | None,
) -> dict[str, Any]:
    """Preserve a video container, then select its soundtrack from the preserved bytes."""
    from capicove.processing.video.retrieval._adapters.sampled_frames import (
        VIDEO_DEMUXERS,
        FrameSamplingError,
    )

    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        identity = os.fstat(stream.fileno())
        if not stat.S_ISREG(identity.st_mode) or identity.st_size > settings.max_source_bytes:
            raise UnifiedRetrievalError("source must be a bounded regular file")
        payload = stream.read(settings.max_source_bytes + 1)
    if not payload or len(payload) > settings.max_source_bytes:
        raise UnifiedRetrievalError("source must be non-empty and within max_source_bytes")
    signature = (
        payload[4:8] == b"ftyp"
        if suffix in {".mp4", ".mov"}
        else payload[:4] == b"\x1a\x45\xdf\xa3"
    )
    if not signature:
        raise UnifiedRetrievalError("video content does not match its extension")
    if audio_stream_index is not None and (
        type(audio_stream_index) is not int or audio_stream_index < 0
    ):
        raise UnifiedRetrievalError("audio_stream_index must be a nonnegative integer")
    sha256 = hashlib.sha256(payload).hexdigest()
    with _locked(settings):
        store = settings.index_root / "evidence"
        _store_blob(store, sha256, payload)
        try:
            probe = _frame_sampler(settings).probe(
                _blob_path(store, sha256), VIDEO_DEMUXERS[suffix]
            )
        except FrameSamplingError as error:
            raise UnifiedRetrievalError(f"video source rejected: {error}") from None
        if probe.video_streams < 1:
            raise UnifiedRetrievalError("video source has no video stream")
        if -(-probe.duration_ms // KEYFRAME_INTERVAL_MS) > MAX_KEYFRAMES_PER_VIDEO:
            raise UnifiedRetrievalError("video exceeds the keyframe budget")
        if audio_stream_index is not None:
            if audio_stream_index >= probe.audio_streams:
                raise UnifiedRetrievalError("audio_stream_index does not exist")
            selected, method = audio_stream_index, "explicit_index"
        elif probe.audio_streams == 1:
            selected, method = 0, "sole_stream_implicit"
        elif probe.audio_streams == 0:
            selected, method = None, "no_audio"
        else:
            raise UnifiedRetrievalError("video has several audio streams; pass audio_stream_index")
        media_type = _VIDEO_SUFFIXES[suffix]
        with _inventory(settings) as connection:
            latest = _latest(connection, source_ref)
            unchanged = (
                latest is not None
                and latest["event"] == "import"
                and latest["sha256"] == sha256
                and latest["stream_index"] == selected
            )
            if not unchanged:
                connection.execute(
                    "INSERT INTO events (source_ref, event, sha256, size_bytes, media_type,"
                    " modality, stream_index, selection_method, stream_count, recorded_at)"
                    " VALUES (?, 'import', ?, ?, ?, 'video', ?, ?, ?, ?)",
                    (
                        source_ref,
                        sha256,
                        len(payload),
                        media_type,
                        selected,
                        method,
                        probe.audio_streams,
                        datetime.now(UTC).isoformat(),
                    ),
                )
    return {
        "source_ref": source_ref,
        "imported": not unchanged,
        "sha256": sha256,
        "modality": "video",
        "audio_stream_index": selected,
        "duration_ms": probe.duration_ms,
    }


def _verify_image(payload: bytes) -> None:
    import io

    from PIL import Image, UnidentifiedImageError

    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 64 * 1024 * 1024:
                raise UnifiedRetrievalError("image exceeds the pixel limit")
            image.verify()
    except (UnidentifiedImageError, OSError, SyntaxError, ValueError):
        raise UnifiedRetrievalError("image content is invalid") from None


def _blob_path(store: Path, sha256: str) -> Path:
    return store / "sha256" / sha256[:2] / sha256 / "blob"


def _store_blob(store: Path, sha256: str, payload: bytes) -> None:
    """Write once: an existing blob must already hold exactly these bytes."""
    target = _blob_path(store, sha256)
    for directory in (store, target.parent.parent, target.parent):
        _private_directory(directory)
    if target.exists():
        if _file_sha256(target) != sha256:
            raise UnifiedRetrievalError("content-addressed evidence store is corrupt")
        return
    temporary = target.parent / f".tmp-{secrets.token_hex(8)}"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400)
    try:
        os.write(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, target)


def _read_blob(settings: UnifiedSettings, sha256: str) -> bytes:
    payload = _blob_path(settings.index_root / "evidence", sha256).read_bytes()
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise UnifiedRetrievalError("preserved source evidence changed")
    return payload


def forget_media(config: WorkspaceHostConfiguration, source_ref: str) -> dict[str, Any]:
    """Retire a live source for the next build; repeated forget is a no-op."""
    settings = _writable(config)
    if not isinstance(source_ref, str) or len(source_ref) > 4096:
        raise UnifiedRetrievalError("source_ref must be a bounded string")
    with _locked(settings), _inventory(settings) as connection:
        latest = _latest(connection, source_ref)
        if latest is None or latest["event"] != "import":
            return {"source_ref": source_ref, "forgotten": False}
        connection.execute(
            "INSERT INTO events (source_ref, event, recorded_at) VALUES (?, 'forget', ?)",
            (source_ref, datetime.now(UTC).isoformat()),
        )
    return {"source_ref": source_ref, "forgotten": True}


def _models(settings: UnifiedSettings, models: ModelFactory | None) -> ModelFactory:
    if models is not None:
        return models
    if settings.model_config is None:
        raise UnifiedRetrievalError("model_config is required to encode with a shared model")
    from capicove.model.configuration import load_configuration
    from capicove.model.factory import ModelFactory

    return ModelFactory(load_configuration(settings.model_config))


def _space_payload(space: EmbeddingSpaceRef) -> dict[str, Any]:
    return {
        "space_id": space.space_id,
        "provider_id": space.provider_id,
        "model_revision": space.model_revision,
        "preprocessing_version": space.preprocessing_version,
        "dimension": space.dimension,
        "normalization": space.normalization.value,
        "distance_metric": space.distance_metric.value,
        "query_modalities": sorted(item.value for item in space.query_modalities),
        "target_modalities": sorted(item.value for item in space.target_modalities),
        "schema_version": space.schema_version,
    }


def _canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def _vector_blob(vector: Sequence[float]) -> bytes:
    import numpy as np

    return np.asarray(vector, dtype="<f4").tobytes()


_UNIT_COLUMNS = (
    "ordinal INTEGER NOT NULL UNIQUE, unit_id TEXT PRIMARY KEY, evidence_id TEXT NOT NULL UNIQUE,"
    " target_type TEXT NOT NULL, source_ref TEXT NOT NULL, source_sha256 TEXT NOT NULL,"
    " media_type TEXT NOT NULL, start_ms INTEGER, end_ms INTEGER, start_char INTEGER,"
    " end_char INTEGER, text TEXT, receipt TEXT NOT NULL, receipt_sha256 TEXT NOT NULL,"
    " vector BLOB NOT NULL, vector_sha256 TEXT NOT NULL"
)
_UNIT_FIELDS = (
    "unit_id",
    "evidence_id",
    "target_type",
    "source_ref",
    "source_sha256",
    "media_type",
    "start_ms",
    "end_ms",
    "start_char",
    "end_char",
    "text",
    "receipt",
    "receipt_sha256",
    "vector",
    "vector_sha256",
)


def build_generation(
    config: WorkspaceHostConfiguration, *, models: ModelFactory | None = None
) -> dict[str, Any]:
    """Build, validate and publish one immutable generation; never activate it."""
    settings = _writable(config)
    factory = _models(settings, models)
    try:
        return _build(settings, factory)
    finally:
        if models is None:
            factory.close()


def _build(settings: UnifiedSettings, factory: ModelFactory) -> dict[str, Any]:
    with _locked(settings):
        _private_directory(settings.staging_root)
        generations = settings.index_root / "generations"
        _private_directory(generations)
        for stale in generations.glob(".staging-*"):
            shutil.rmtree(stale, ignore_errors=True)
        with _inventory(settings) as connection:
            live = _live(connection)
        inventory_digest = hashlib.sha256(
            _canonical(
                [
                    [row["source_ref"], row["sha256"], row["stream_index"], row["media_type"]]
                    for row in live
                ]
            )
        ).hexdigest()
        staging = generations / f".staging-{secrets.token_hex(8)}"
        staging.mkdir(mode=0o700)
        try:
            with factory.shared_space_encoder(settings.model_binding) as encoder:
                space = encoder.space
                if not settings.profile.targets <= space.target_modalities:
                    raise UnifiedRetrievalError("model space does not support the profile targets")
                units = sorted(
                    _audio_units(settings, [r for r in live if r["modality"] == "audio"], encoder)
                    + _text_units(settings, [r for r in live if r["modality"] == "text"], encoder)
                    + _image_units(settings, [r for r in live if r["modality"] == "image"], encoder)
                    + _video_units(
                        settings, [r for r in live if r["modality"] == "video"], encoder
                    ),
                    key=lambda unit: unit["unit_id"],
                )
            if len(units) > MAX_UNITS:
                raise UnifiedRetrievalError("searchable units exceed the preview limit")
            unit_records = [
                [u["unit_id"], u["receipt_sha256"], u["vector_sha256"], u["source_ref"]]
                for u in units
            ]
            manifest_body = {
                "format": FORMAT_VERSION,
                "workspace_id": settings.workspace_id,
                "corpus_ref": settings.corpus_ref,
                "profile": settings.profile.profile_id,
                "inventory_sha256": inventory_digest,
                "unit_manifest_sha256": hashlib.sha256(_canonical(unit_records)).hexdigest(),
                "unit_count": len(units),
                "space": _space_payload(space),
                "preprocessing": _preprocessing_identity(),
            }
            generation_id = "ug-" + hashlib.sha256(_canonical(manifest_body)).hexdigest()
            for unit in units:
                unit["evidence_id"] = (
                    "mev-"
                    + hashlib.sha256(
                        _canonical(
                            [
                                settings.workspace_id,
                                generation_id,
                                unit["unit_id"],
                                unit["receipt_sha256"],
                            ]
                        )
                    ).hexdigest()
                )
            target = generations / generation_id
            if target.exists():
                _validate_generation(target, generation_id)
                shutil.rmtree(staging, ignore_errors=True)
                return {"generation_id": generation_id, "units": len(units), "reused": True}
            database = staging / "units.sqlite3"
            with closing(sqlite3.connect(database)) as db:
                db.execute(f"CREATE TABLE units ({_UNIT_COLUMNS})")
                db.execute(
                    "CREATE VIRTUAL TABLE lexical USING fts5("
                    "unit_id UNINDEXED, body, tokenize='unicode61')"
                )
                placeholders = ", ".join("?" for _ in range(len(_UNIT_FIELDS) + 1))
                db.executemany(
                    f"INSERT INTO units VALUES ({placeholders})",
                    [
                        (ordinal, *(u[name] for name in _UNIT_FIELDS))
                        for ordinal, u in enumerate(units)
                    ],
                )
                db.executemany(
                    "INSERT INTO lexical (unit_id, body) VALUES (?, ?)",
                    [(u["unit_id"], u["text"]) for u in units if u["text"] is not None],
                )
                db.commit()
            manifest = {
                **manifest_body,
                "generation_id": generation_id,
                "units_sha256": _file_sha256(database),
                "created_at": datetime.now(UTC).isoformat(),
            }
            (staging / "manifest.json").write_bytes(_canonical(manifest))
            for item in staging.iterdir():
                os.chmod(item, 0o400)
            _validate_generation(staging, generation_id)
            os.chmod(staging, 0o500)
            os.rename(staging, target)
            _fsync_directory(generations)
        except BaseException:
            if staging.exists():
                os.chmod(staging, 0o700)
                shutil.rmtree(staging, ignore_errors=True)
            raise
    return {"generation_id": generation_id, "units": len(units), "reused": False}


def _preprocessing_identity() -> dict[str, Any]:
    from capicove.processing.audio.retrieval import AudioWindowPolicy

    return {
        "audio_window_sha256": AudioWindowPolicy().fingerprint.sha256,
        "text_chunking": CHUNKING_VERSION,
        "text_document_title": "none",
        "lexical": "fts5-unicode61-v1",
        "image": "whole-image-v1",
        "keyframes": KEYFRAME_POLICY,
        "fusion": f"weighted-rrf-k{RRF_K}-dense1-lexical1",
    }


def _encode_rows(
    settings: UnifiedSettings,
    encoder: Any,
    items: list[tuple[str, Any]],
    encode: Any,
) -> list[bytes]:
    """Encode in provider-sized batches, reusing cached vectors per receipt digest."""
    cache = _VectorCache(settings.index_root / "vectors.sqlite3", encoder.space)
    batch = max(1, int(getattr(encoder, "max_batch", 8)))
    try:
        vectors: dict[int, bytes] = {}
        pending: list[int] = []
        for index, (digest, _) in enumerate(items):
            cached = cache.get(digest)
            if cached is None:
                pending.append(index)
            else:
                vectors[index] = cached
        deadline = _deadline()
        for start in range(0, len(pending), batch):
            chunk = pending[start : start + batch]
            encoded = encode([items[index][1] for index in chunk], deadline=deadline)
            for index, vector in zip(chunk, encoded, strict=True):
                vectors[index] = _vector_blob(vector)
                cache.put(items[index][0], vectors[index])
        return [vectors[index] for index in range(len(items))]
    finally:
        cache.close()


def _unit(
    prefix: str,
    target: TargetType,
    row: sqlite3.Row,
    receipt: dict[str, Any],
    space: EmbeddingSpaceRef,
    vector: bytes,
    **fields: Any,
) -> dict[str, Any]:
    receipt_sha256 = hashlib.sha256(_canonical(receipt)).hexdigest()
    return {
        "unit_id": prefix
        + hashlib.sha256(_canonical([receipt_sha256, space.space_id])).hexdigest()[:40],
        "target_type": target.value,
        "source_ref": row["source_ref"],
        "source_sha256": row["sha256"],
        "media_type": row["media_type"],
        "start_ms": None,
        "end_ms": None,
        "start_char": None,
        "end_char": None,
        "text": None,
        "receipt": json.dumps(receipt, sort_keys=True),
        "receipt_sha256": receipt_sha256,
        "vector": vector,
        "vector_sha256": hashlib.sha256(vector).hexdigest(),
        **fields,
    }


def _text_units(
    settings: UnifiedSettings, live: list[sqlite3.Row], encoder: Any
) -> list[dict[str, Any]]:
    from capicove.model.multimodal_embedding import TextDocument
    from capicove.runtime.local_retrieval import chunk_markdown

    chunks: list[tuple[sqlite3.Row, dict[str, Any], str]] = []
    for row in live:
        text = _read_blob(settings, row["sha256"]).decode("utf-8")
        for start, end in chunk_markdown(text):
            body = text[start:end]
            receipt = {
                "schema": "capicove.text-chunk-receipt.v1",
                "source_sha256": row["sha256"],
                "start_char": start,
                "end_char": end,
                "text_sha256": hashlib.sha256(body.encode()).hexdigest(),
                "chunking": CHUNKING_VERSION,
            }
            chunks.append((row, receipt, body))
    vectors = _encode_rows(
        settings,
        encoder,
        [
            (hashlib.sha256(_canonical(receipt)).hexdigest(), TextDocument(body))
            for _, receipt, body in chunks
        ],
        encoder.encode_text_documents,
    )
    return [
        _unit(
            "tu-",
            TargetType.TEXT_CHUNK,
            row,
            receipt,
            encoder.space,
            vector,
            start_char=receipt["start_char"],
            end_char=receipt["end_char"],
            text=body,
        )
        for (row, receipt, body), vector in zip(chunks, vectors, strict=True)
    ]


def _image_units(
    settings: UnifiedSettings, live: list[sqlite3.Row], encoder: Any
) -> list[dict[str, Any]]:
    receipts = [
        {
            "schema": "capicove.image-receipt.v1",
            "source_sha256": row["sha256"],
            "media_type": row["media_type"],
            "representation": "whole-image-v1",
        }
        for row in live
    ]
    vectors = _encode_rows(
        settings,
        encoder,
        [
            (hashlib.sha256(_canonical(receipt)).hexdigest(), _read_blob(settings, row["sha256"]))
            for row, receipt in zip(live, receipts, strict=True)
        ],
        encoder.encode_images,
    )
    return [
        _unit("iu-", TargetType.IMAGE, row, receipt, encoder.space, vector)
        for row, receipt, vector in zip(live, receipts, vectors, strict=True)
    ]


def _audio_units(
    settings: UnifiedSettings,
    live: list[sqlite3.Row],
    encoder: Any,
    *,
    container: str = "audio",
) -> list[dict[str, Any]]:
    from capicove.model.multimodal_embedding import PcmAudio
    from capicove.processing.audio.models import AudioEvidence, AudioStreamSelection

    preparer = _audio_preparer(settings)
    cache = _VectorCache(settings.index_root / "vectors.sqlite3", encoder.space)
    batch = max(1, int(getattr(encoder, "max_batch", 8)))
    units: list[dict[str, Any]] = []
    store = settings.index_root / "evidence"
    try:
        for row in live:
            sha = row["sha256"]
            evidence = AudioEvidence(
                evidence_id=f"sha256:{sha}",
                sha256=sha,
                size_bytes=row["size_bytes"],
                media_type=row["media_type"],
                source_name=Path(row["source_ref"]).name or "source",
                stored_path=store / "sha256" / sha[:2] / sha / "blob",
            )
            selection = AudioStreamSelection(
                selected_audio_stream_index=row["stream_index"],
                selection_method=row["selection_method"],
                preflight_audio_stream_count=row["stream_count"],
            )
            deadline = _deadline()
            with preparer.prepare_preserved(
                evidence,
                selection,
                _MEDIA_SUFFIXES[row["media_type"]],
                settings.staging_root,
                deadline=deadline,
                container=container,
            ) as prepared:
                pending: list[int] = []
                vectors: dict[int, bytes] = {}
                for index, receipt in enumerate(prepared.receipts):
                    cached = cache.get(receipt.digest)
                    if cached is None:
                        pending.append(index)
                    else:
                        vectors[index] = cached
                for start in range(0, len(pending), batch):
                    chunk = pending[start : start + batch]
                    encoded = encoder.encode_audio(
                        [
                            PcmAudio(prepared.window_pcm(index), prepared.policy.sample_rate_hz)
                            for index in chunk
                        ],
                        deadline=deadline,
                    )
                    for index, vector in zip(chunk, encoded, strict=True):
                        vectors[index] = _vector_blob(vector)
                        cache.put(prepared.receipts[index].digest, vectors[index])
                for index, receipt in enumerate(prepared.receipts):
                    span = receipt.time_range
                    blob = vectors[index]
                    units.append(
                        {
                            "unit_id": "au-"
                            + hashlib.sha256(
                                _canonical([receipt.digest, encoder.space.space_id])
                            ).hexdigest()[:40],
                            "target_type": TargetType.AUDIO_SEGMENT.value,
                            "source_ref": row["source_ref"],
                            "source_sha256": sha,
                            "media_type": row["media_type"],
                            "start_ms": span.start_ms,
                            "end_ms": span.end_ms,
                            "start_char": None,
                            "end_char": None,
                            "text": None,
                            "receipt": json.dumps(receipt.to_payload(), sort_keys=True),
                            "receipt_sha256": receipt.digest,
                            "vector": blob,
                            "vector_sha256": hashlib.sha256(blob).hexdigest(),
                        }
                    )
    finally:
        cache.close()
    return sorted(units, key=lambda unit: unit["unit_id"])


def _video_units(
    settings: UnifiedSettings, live: list[sqlite3.Row], encoder: Any
) -> list[dict[str, Any]]:
    """Keyframes as image-space units plus soundtrack windows as audio units."""
    from capicove.processing.video.retrieval._adapters.sampled_frames import (
        VIDEO_DEMUXERS,
        FrameSamplingError,
    )

    sampler = _frame_sampler(settings)
    store = settings.index_root / "evidence"
    units: list[dict[str, Any]] = []
    for row in live:
        sha = row["sha256"]
        blob = _blob_path(store, sha)
        if _file_sha256(blob) != sha:
            raise UnifiedRetrievalError("preserved source evidence changed")
        demuxer = VIDEO_DEMUXERS[_MEDIA_SUFFIXES[row["media_type"]]]
        try:
            probe = sampler.probe(blob, demuxer)
            timestamps = list(range(0, probe.duration_ms, KEYFRAME_INTERVAL_MS))
            if len(timestamps) > MAX_KEYFRAMES_PER_VIDEO:
                raise UnifiedRetrievalError("video exceeds the keyframe budget")
            frames = [
                sampler.frame(blob, demuxer, timestamp, KEYFRAME_MAX_SIDE)
                for timestamp in timestamps
            ]
        except FrameSamplingError as error:
            raise UnifiedRetrievalError(f"video frames could not be sampled: {error}") from None
        receipts = [
            {
                "schema": "capicove.keyframe-receipt.v1",
                "source_sha256": sha,
                "media_type": row["media_type"],
                "policy": KEYFRAME_POLICY,
                "timestamp_ms": timestamp,
                "frame_sha256": hashlib.sha256(frame).hexdigest(),
            }
            for timestamp, frame in zip(timestamps, frames, strict=True)
        ]
        vectors = _encode_rows(
            settings,
            encoder,
            [
                (hashlib.sha256(_canonical(receipt)).hexdigest(), frame)
                for receipt, frame in zip(receipts, frames, strict=True)
            ],
            encoder.encode_images,
        )
        for receipt, vector in zip(receipts, vectors, strict=True):
            units.append(
                _unit(
                    "ku-",
                    TargetType.KEYFRAME,
                    row,
                    receipt,
                    encoder.space,
                    vector,
                    start_ms=receipt["timestamp_ms"],
                )
            )
    soundtracks = [row for row in live if row["stream_index"] is not None]
    return units + _audio_units(settings, soundtracks, encoder, container="video")


class _VectorCache:
    """Reuse vectors for an identical receipt within one complete embedding space."""

    def __init__(self, path: Path, space: EmbeddingSpaceRef) -> None:
        self._space = hashlib.sha256(_canonical(_space_payload(space))).hexdigest()
        self._connection = sqlite3.connect(path)
        self._connection.execute(
            "CREATE TABLE IF NOT EXISTS vectors (space TEXT NOT NULL, receipt TEXT NOT NULL,"
            " vector BLOB NOT NULL, sha256 TEXT NOT NULL, PRIMARY KEY (space, receipt))"
        )

    def get(self, receipt: str) -> bytes | None:
        row = self._connection.execute(
            "SELECT vector, sha256 FROM vectors WHERE space = ? AND receipt = ?",
            (self._space, receipt),
        ).fetchone()
        if row is None or hashlib.sha256(row[0]).hexdigest() != row[1]:
            return None
        return bytes(row[0])

    def put(self, receipt: str, vector: bytes) -> None:
        self._connection.execute(
            "INSERT OR REPLACE INTO vectors VALUES (?, ?, ?, ?)",
            (self._space, receipt, vector, hashlib.sha256(vector).hexdigest()),
        )
        self._connection.commit()

    def close(self) -> None:
        self._connection.close()


def _file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _validate_generation(directory: Path, generation_id: str) -> dict[str, Any]:
    """Verify the manifest identity and database digest before any read."""
    try:
        manifest = json.loads((directory / "manifest.json").read_bytes())
    except (OSError, ValueError):
        raise UnifiedRetrievalError("index_corrupt") from None
    body = {
        key: value
        for key, value in manifest.items()
        if key not in {"generation_id", "units_sha256", "created_at"}
    }
    if (
        not isinstance(manifest, dict)
        or manifest.get("format") != FORMAT_VERSION
        or manifest.get("generation_id") != generation_id
        or "ug-" + hashlib.sha256(_canonical(body)).hexdigest() != generation_id
        or _file_sha256(directory / "units.sqlite3") != manifest.get("units_sha256")
    ):
        raise UnifiedRetrievalError("index_corrupt")
    return manifest


def activate_generation(config: WorkspaceHostConfiguration, generation_id: str) -> dict[str, Any]:
    """Atomically point the active alias at one validated, published generation."""
    settings = _writable(config)
    if not isinstance(generation_id, str) or _GENERATION.fullmatch(generation_id) is None:
        raise UnifiedRetrievalError("invalid generation_id")
    with _locked(settings):
        directory = settings.index_root / "generations" / generation_id
        if not directory.is_dir():
            raise UnifiedRetrievalError("generation is not published")
        manifest = _validate_generation(directory, generation_id)
        if (
            manifest["workspace_id"] != settings.workspace_id
            or manifest["corpus_ref"] != settings.corpus_ref
            or manifest["profile"] != settings.profile.profile_id
        ):
            raise UnifiedRetrievalError("generation belongs to another workspace or profile")
        pointer = settings.index_root / f".active-{secrets.token_hex(8)}"
        payload = _canonical(
            {
                "generation_id": generation_id,
                "manifest_sha256": _file_sha256(directory / "manifest.json"),
            }
        )
        descriptor = os.open(pointer, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.replace(pointer, settings.index_root / "active.json")
        _fsync_directory(settings.index_root)
    return {"generation_id": generation_id, "active": True}


@dataclass(frozen=True, slots=True)
class _Lease:
    generation_id: str
    manifest: Mapping[str, Any]
    connection: sqlite3.Connection


@contextmanager
def _lease(settings: UnifiedSettings) -> Iterator[_Lease]:
    """Open the active generation read-only after verifying its pinned digests."""
    try:
        pointer = json.loads((settings.index_root / "active.json").read_bytes())
        generation_id = pointer["generation_id"]
    except (OSError, ValueError, KeyError, TypeError):
        raise UnifiedRetrievalError("index_generation_unavailable") from None
    if not isinstance(generation_id, str) or _GENERATION.fullmatch(generation_id) is None:
        raise UnifiedRetrievalError("index_corrupt")
    directory = settings.index_root / "generations" / generation_id
    try:
        if _file_sha256(directory / "manifest.json") != pointer.get("manifest_sha256"):
            raise UnifiedRetrievalError("index_corrupt")
        manifest = _validate_generation(directory, generation_id)
    except OSError:
        raise UnifiedRetrievalError("index_corrupt") from None
    if manifest["workspace_id"] != settings.workspace_id or manifest["corpus_ref"] != (
        settings.corpus_ref
    ):
        raise UnifiedRetrievalError("index_corrupt")
    connection = sqlite3.connect(
        f"file:{directory / 'units.sqlite3'}?mode=ro&immutable=1", uri=True
    )
    connection.row_factory = sqlite3.Row
    try:
        yield _Lease(generation_id, manifest, connection)
    finally:
        connection.close()


class UnifiedRetrievalService:
    """Search and Evidence over the active shared-space generation."""

    def __init__(self, settings: UnifiedSettings, models: ModelFactory | None) -> None:
        self._settings = settings
        self._models = models
        self._owned: ModelFactory | None = None

    def _factory(self) -> ModelFactory:
        """Keep one resident model factory for this service's lifetime."""
        if self._models is not None:
            return self._models
        if self._owned is None:
            self._owned = _models(self._settings, None)
        return self._owned

    def close(self) -> None:
        """Release an owned resident model and its native child process."""
        owned, self._owned = self._owned, None
        if owned is not None:
            owned.close()

    def search(self, request: SearchRequest) -> SearchResult:
        """Return canonical results; every failure maps to a stable failure code."""
        query_id = f"query-{uuid.uuid4().hex}"
        try:
            return self._search(request, query_id)
        except _SearchFailure as failure:
            return SearchResult(SearchStatus.FAILED, query_id, failure_code=failure.code)
        except UnifiedRetrievalError as error:
            code = {
                "index_generation_unavailable": SearchFailureCode.INDEX_GENERATION_UNAVAILABLE,
                "index_corrupt": SearchFailureCode.INDEX_CORRUPT,
            }.get(str(error), SearchFailureCode.INTERNAL)
            return SearchResult(SearchStatus.FAILED, query_id, failure_code=code)
        except sqlite3.DatabaseError:
            return SearchResult(
                SearchStatus.FAILED, query_id, failure_code=SearchFailureCode.INDEX_CORRUPT
            )

    def _search(self, request: SearchRequest, query_id: str) -> SearchResult:
        from capicove.model.multimodal_embedding import MultimodalEmbeddingError

        profile = self._settings.profile
        if (
            not isinstance(request, SearchRequest)
            or request.retrieval_profile_ref != profile.profile_id
            or request.query.type not in profile.queries
            or not request.targets <= profile.targets
            or request.cursor is not None
        ):
            raise _SearchFailure(SearchFailureCode.INVALID_QUERY)
        clip = (
            self._query_clip(request.query) if isinstance(request.query, AudioClipQuery) else None
        )
        image = self._query_image(request.query) if isinstance(request.query, ImageQuery) else None
        with _lease(self._settings) as lease:
            rows = lease.connection.execute(
                "SELECT ordinal, unit_id, evidence_id, target_type, source_ref, source_sha256,"
                " media_type, start_ms, end_ms, start_char, end_char, text, vector, vector_sha256"
                " FROM units ORDER BY ordinal"
            ).fetchall()
            eligible = [row for row in rows if _eligible(row, request, self._settings)]
            if not eligible:
                return SearchResult(
                    SearchStatus.SUCCEEDED,
                    query_id,
                    warning_codes=(SearchWarningCode.RETRIEVAL_PREVIEW,),
                )
            deadline = datetime.now(UTC) + timedelta(seconds=120)
            try:
                with self._factory().shared_space_encoder(self._settings.model_binding) as encoder:
                    if _space_payload(encoder.space) != lease.manifest["space"]:
                        raise _SearchFailure(SearchFailureCode.EMBEDDING_SPACE_INCOMPATIBLE)
                    if isinstance(request.query, TextQuery):
                        vector = encoder.encode_text_queries(
                            [request.query.text], deadline=deadline
                        )[0]
                    elif image is not None:
                        vector = encoder.encode_images([image], deadline=deadline)[0]
                    else:
                        assert clip is not None
                        vector = encoder.encode_audio([clip], deadline=deadline)[0]
                    space = encoder.space
            except MultimodalEmbeddingError as error:
                raise _SearchFailure(
                    SearchFailureCode.DEADLINE_EXCEEDED
                    if error.code == "deadline_exceeded"
                    else SearchFailureCode.PROVIDER_UNAVAILABLE
                ) from None
            dense = _dense_lane(eligible, vector, space.dimension)
            lexical: list[tuple[str, float]] = []
            if isinstance(request.query, TextQuery) and TargetType.TEXT_CHUNK in request.targets:
                lexical = _lexical_lane(lease.connection, request.query.text, eligible)
            fused = _fuse({"shared_dense": dense, "lexical": lexical})
            by_id = {row["unit_id"]: row for row in eligible}
            hits = tuple(
                self._hit(request, by_id[unit_id], rank, observations, lease, space)
                for rank, (unit_id, observations) in enumerate(fused[: request.top_k], start=1)
            )
        return SearchResult(
            SearchStatus.SUCCEEDED,
            query_id,
            warning_codes=(SearchWarningCode.RETRIEVAL_PREVIEW,),
            hits=hits,
        )

    def _hit(
        self,
        request: SearchRequest,
        row: sqlite3.Row,
        rank: int,
        observations: dict[str, tuple[int, float]],
        lease: _Lease,
        space: EmbeddingSpaceRef,
    ) -> CompactHit:
        target = TargetType(row["target_type"])
        fields: dict[str, Any] = {}
        if target is TargetType.AUDIO_SEGMENT:
            fields["time_range"] = TemporalRange(row["start_ms"], row["end_ms"])
        elif target is TargetType.KEYFRAME:
            fields["timestamp_ms"] = row["start_ms"]
        elif target is TargetType.TEXT_CHUNK:
            fields["text_locator"] = TextChunkLocator(
                TextOrigin.DOCUMENT, CharacterRange(row["start_char"], row["end_char"])
            )
            fields["snippet"] = preview_snippet(row["text"])
        compact = CompactHit(row["unit_id"], target, 1.0 / rank, row["evidence_id"], **fields)
        if request.detail_level is not DetailLevel.FULL:
            return compact
        dense_kind = {
            TargetType.AUDIO_SEGMENT: MatchKind.AUDIO,
            TargetType.IMAGE: MatchKind.VISUAL,
            TargetType.KEYFRAME: MatchKind.VISUAL,
            TargetType.TEXT_CHUNK: MatchKind.SEMANTIC,
        }[target]
        kinds = tuple(
            MatchKind.LEXICAL if lane == "lexical" else dense_kind for lane in observations
        )
        lanes = tuple(
            LaneScore(
                lane,
                score,
                min(1.0, max(0.0, (score + 1.0) / 2.0))
                if lane == "shared_dense"
                else 1.0 / lane_rank,
                "cosine" if lane == "shared_dense" else "fts5_bm25_negated",
                "uncalibrated",
                lane_rank,
            )
            for lane, (lane_rank, score) in observations.items()
        )
        return FullHit(
            compact.id,
            compact.target_type,
            compact.brief_score,
            compact.evidence_id,
            searchable_unit_ref=row["unit_id"],
            match_kinds=kinds,
            lane_scores=lanes,
            embedding_spaces=(space,),
            index_generation_ids=(lease.generation_id,),
            provider_ids=(space.provider_id,),
            warning_codes=(SearchWarningCode.RETRIEVAL_PREVIEW.value,),
            **fields,
        )

    def _query_image(self, query: ImageQuery) -> bytes:
        """Return verified image bytes from an authorized path or bounded inline data."""
        source = query.source
        if isinstance(source, LocalPathSource):
            path = self._authorized_query_path(source)
            if path.suffix.lower() not in _IMAGE_SUFFIXES:
                raise _SearchFailure(SearchFailureCode.INVALID_MEDIA)
            payload = path.read_bytes()
            media_type = _IMAGE_SUFFIXES[path.suffix.lower()]
        elif isinstance(source, InlineBytesSource):
            payload, media_type = source.data, source.media_type
            if len(payload) > MAX_INLINE_QUERY_BYTES:
                raise _SearchFailure(SearchFailureCode.INPUT_TOO_LARGE)
        else:
            raise _SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)
        if not any(
            payload.startswith(sig) and kind == media_type for sig, kind in _IMAGE_SIGNATURES
        ):
            raise _SearchFailure(SearchFailureCode.INVALID_MEDIA)
        try:
            _verify_image(payload)
        except UnifiedRetrievalError:
            raise _SearchFailure(SearchFailureCode.INVALID_MEDIA) from None
        return payload

    def _authorized_query_path(self, source: LocalPathSource) -> Path:
        path = Path(source.path)
        for alias, root in self._settings.source_roots.items():
            if path.is_absolute() and path.is_relative_to(root):
                try:
                    return _authorized_file(
                        self._settings, alias, path.relative_to(root).as_posix()
                    )
                except UnifiedRetrievalError:
                    break
        raise _SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)

    def _query_clip(self, query: AudioClipQuery) -> Any:
        """Decode only the query source, bounded to one configured window."""
        from capicove.model.multimodal_embedding import PcmAudio
        from capicove.processing.audio.retrieval import AudioWindowError, AudioWindowPolicy

        policy = AudioWindowPolicy()
        workspace = self._settings.staging_root / f".query-{secrets.token_hex(8)}"
        try:
            _private_directory(self._settings.staging_root)
            workspace.mkdir(mode=0o700)
            source = self._query_source(query, workspace)
            store, staging = workspace / "store", workspace / "pcm"
            store.mkdir(mode=0o700)
            staging.mkdir(mode=0o700)
            try:
                prepared = _audio_preparer(self._settings).prepare(
                    source, store, staging, stream_index=query.stream_index, deadline=_deadline(5)
                )
            except AudioWindowError:
                raise _SearchFailure(SearchFailureCode.INVALID_MEDIA) from None
            with prepared:
                rate = policy.sample_rate_hz
                if query.time_range is None:
                    start, end = 0, prepared.total_frames
                else:
                    start = query.time_range.start_ms * rate // 1000
                    end = -(-query.time_range.end_ms * rate // 1000)
                if (
                    end > prepared.total_frames
                    or end - start > policy.window_frames
                    or end <= start
                ):
                    raise _SearchFailure(SearchFailureCode.INPUT_TOO_LARGE)
                return PcmAudio(_read_frames(prepared, start, end), rate)
        finally:
            shutil.rmtree(workspace, ignore_errors=True)

    def _query_source(self, query: AudioClipQuery, workspace: Path) -> Path:
        source = query.source
        if isinstance(source, LocalPathSource):
            path = self._authorized_query_path(source)
            if path.suffix.lower() not in _AUDIO_SUFFIXES:
                raise _SearchFailure(SearchFailureCode.INVALID_MEDIA)
            return path
        if isinstance(source, InlineBytesSource):
            suffix = _MEDIA_SUFFIXES.get(source.media_type)
            if suffix not in _AUDIO_SUFFIXES or len(source.data) > MAX_INLINE_QUERY_BYTES:
                raise _SearchFailure(SearchFailureCode.INVALID_MEDIA)
            path = workspace / f"query{suffix}"
            path.write_bytes(source.data)
            return path
        raise _SearchFailure(SearchFailureCode.SOURCE_NOT_AUTHORIZED)

    def read_evidence(
        self, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, Any]:
        """Reauthorize, then return bounded metadata and lineage for one unit."""
        if context.workspace_id != self._settings.workspace_id:
            raise PermissionError("evidence belongs to another workspace")
        if not isinstance(evidence_id, str) or _EVIDENCE.fullmatch(evidence_id) is None:
            raise LookupError("evidence is unavailable")
        if not sections or set(sections) - {"metadata", "lineage"}:
            raise LookupError("unsupported evidence section")
        try:
            with _lease(self._settings) as lease:
                row = lease.connection.execute(
                    "SELECT * FROM units WHERE evidence_id = ?", (evidence_id,)
                ).fetchone()
                manifest = dict(lease.manifest)
        except (UnifiedRetrievalError, sqlite3.DatabaseError):
            raise LookupError("no consistent active generation") from None
        if row is None:
            raise LookupError("evidence is unavailable")
        receipt = json.loads(row["receipt"])
        if hashlib.sha256(_canonical(receipt)).hexdigest() != row["receipt_sha256"]:
            raise LookupError("evidence receipt changed")
        try:
            source = _read_blob(self._settings, row["source_sha256"])
        except (OSError, UnifiedRetrievalError):
            raise LookupError("source evidence is unavailable or changed") from None
        target = TargetType(row["target_type"])
        result: dict[str, Any] = {}
        if "metadata" in sections:
            metadata: dict[str, Any] = {
                "kind": row["target_type"],
                "source_ref": row["source_ref"],
                "media_type": row["media_type"],
                "source_sha256": row["source_sha256"],
                "corpus_ref": self._settings.corpus_ref,
                "generation_id": manifest["generation_id"],
            }
            if target is TargetType.AUDIO_SEGMENT:
                metadata.update(
                    {
                        "audio_stream_index": receipt["audio_stream_index"],
                        "selection_method": receipt["selection_method"],
                        "sample_rate_hz": receipt["sample_rate_hz"],
                        "start_frame": receipt["start_frame"],
                        "end_frame": receipt["end_frame"],
                        "start_ms": row["start_ms"],
                        "end_ms": row["end_ms"],
                        "pcm_sha256": receipt["pcm_sha256"],
                    }
                )
            elif target is TargetType.TEXT_CHUNK:
                text = source.decode("utf-8")[row["start_char"] : row["end_char"]]
                if hashlib.sha256(text.encode()).hexdigest() != receipt["text_sha256"]:
                    raise LookupError("text evidence changed")
                metadata.update(
                    {"start_char": row["start_char"], "end_char": row["end_char"], "text": text}
                )
            elif target is TargetType.KEYFRAME:
                metadata.update(
                    {
                        "timestamp_ms": row["start_ms"],
                        "frame_sha256": receipt["frame_sha256"],
                        "keyframe_policy": receipt["policy"],
                        "size_bytes": len(source),
                    }
                )
            else:
                metadata["size_bytes"] = len(source)
            result["metadata"] = metadata
        if "lineage" in sections:
            result["lineage"] = {
                "generation_id": manifest["generation_id"],
                "format": manifest["format"],
                "unit_manifest_sha256": manifest["unit_manifest_sha256"],
                "receipt": receipt,
                "receipt_sha256": row["receipt_sha256"],
                "vector_sha256": row["vector_sha256"],
                "space_id": manifest["space"]["space_id"],
                "model_revision": manifest["space"]["model_revision"],
            }
        return result

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]:
        """Adapt to the Evidence Tool reader protocol."""
        return self.read_evidence(context, evidence_id, sections)


class _SearchFailure(Exception):
    def __init__(self, code: SearchFailureCode) -> None:
        super().__init__(code.value)
        self.code = code


def _dense_lane(
    eligible: list[sqlite3.Row], vector: Sequence[float], dimension: int
) -> list[tuple[str, float]]:
    """Exact cosine over eligible normalized vectors, deepest CANDIDATE_DEPTH, unit-ID ties."""
    import numpy as np

    if any(hashlib.sha256(row["vector"]).hexdigest() != row["vector_sha256"] for row in eligible):
        raise UnifiedRetrievalError("index_corrupt")
    matrix = np.frombuffer(b"".join(row["vector"] for row in eligible), dtype="<f4")
    if matrix.size != len(eligible) * dimension:
        raise UnifiedRetrievalError("index_corrupt")
    scores = matrix.reshape(len(eligible), dimension) @ np.asarray(vector, dtype=np.float32)
    ranked = sorted(
        ((row["unit_id"], float(score)) for row, score in zip(eligible, scores, strict=True)),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[:CANDIDATE_DEPTH]


def _lexical_lane(
    connection: sqlite3.Connection, text: str, eligible: list[sqlite3.Row]
) -> list[tuple[str, float]]:
    """FTS5 bm25 over eligible text chunks; tokens are quoted so input is never syntax."""
    tokens = re.findall(r"\w+", text)[:64]
    if not tokens:
        return []
    allowed = {row["unit_id"] for row in eligible if row["text"] is not None}
    expression = " OR ".join('"' + token.replace('"', '""') + '"' for token in tokens)
    rows = connection.execute(
        "SELECT unit_id, bm25(lexical) AS score FROM lexical WHERE lexical MATCH ?"
        " ORDER BY score, unit_id LIMIT ?",
        (expression, MAX_UNITS),
    ).fetchall()
    ranked = [(row["unit_id"], -float(row["score"])) for row in rows if row["unit_id"] in allowed]
    return ranked[:CANDIDATE_DEPTH]


def _fuse(
    lanes: Mapping[str, list[tuple[str, float]]],
) -> list[tuple[str, dict[str, tuple[int, float]]]]:
    """Weighted RRF (all weights 1.0, k=60) over ranks; raw scores never cross lanes."""
    scores: dict[str, float] = {}
    observations: dict[str, dict[str, tuple[int, float]]] = {}
    for lane, ranked in lanes.items():
        for rank, (unit_id, raw) in enumerate(ranked, start=1):
            scores[unit_id] = scores.get(unit_id, 0.0) + 1.0 / (RRF_K + rank)
            observations.setdefault(unit_id, {})[lane] = (rank, raw)
    ordered = sorted(scores, key=lambda unit_id: (-scores[unit_id], unit_id))
    return [(unit_id, observations[unit_id]) for unit_id in ordered]


def _read_frames(prepared: Any, start: int, end: int) -> bytes:
    """Assemble an exact frame range from verified overlapping windows."""
    output = bytearray()
    position = start
    for index, window in enumerate(prepared.windows):
        if window.end_frame <= position or window.start_frame > position:
            continue
        stop = min(end, window.end_frame)
        payload = prepared.window_pcm(index)
        output += payload[(position - window.start_frame) * 2 : (stop - window.start_frame) * 2]
        position = stop
        if position >= end:
            break
    if position != end:
        raise _SearchFailure(SearchFailureCode.INVALID_MEDIA)
    return bytes(output)


def _eligible(row: sqlite3.Row, request: SearchRequest, settings: UnifiedSettings) -> bool:
    """Apply every filter before ranking so excluded neighbors cannot hide results."""
    filters: SearchFilters = request.filters
    target = TargetType(row["target_type"])
    if target not in request.targets:
        return False
    if filters.corpus_refs and settings.corpus_ref not in filters.corpus_refs:
        return False
    if filters.artifact_refs and f"sha256:{row['source_sha256']}" not in filters.artifact_refs:
        return False
    if filters.media_types and row["media_type"] not in filters.media_types:
        return False
    if filters.text_origins and (
        target is not TargetType.TEXT_CHUNK or TextOrigin.DOCUMENT not in filters.text_origins
    ):
        return False
    if filters.time_range is not None:
        if target is TargetType.KEYFRAME:
            return bool(filters.time_range.start_ms <= row["start_ms"] < filters.time_range.end_ms)
        if row["start_ms"] is None or row["end_ms"] is None:
            return False
        if not (
            row["start_ms"] < filters.time_range.end_ms
            and filters.time_range.start_ms < row["end_ms"]
        ):
            return False
    return True


@dataclass(frozen=True, slots=True)
class UnifiedRetrievalRuntime:
    """Expose startup-bound canonical Search and Evidence handlers."""

    service: UnifiedRetrievalService
    search_handler: Any
    evidence_handler: Any

    def search(self, request: SearchRequest) -> SearchResult:
        """Delegate to the shared service."""
        return self.service.search(request)

    def read_evidence(
        self, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, Any]:
        """Delegate to the shared service."""
        return self.service.read_evidence(context, evidence_id, sections)

    def close(self) -> None:
        """Release resident model resources owned by the service."""
        self.service.close()


def open_unified_retrieval(
    config: WorkspaceHostConfiguration, *, models: ModelFactory | None = None
) -> UnifiedRetrievalRuntime:
    """Compose read-only handlers; this never creates storage or builds an index."""
    from capicove.tools.evidence import EvidenceToolHandler
    from capicove.tools.multimodal_search import (
        CAPICOVE_MULTIMODAL_EVIDENCE_SPEC,
        MultimodalSearchToolHandler,
    )

    settings = unified_settings(config)
    service = UnifiedRetrievalService(settings, models)
    return UnifiedRetrievalRuntime(
        service,
        MultimodalSearchToolHandler(service, profile=settings.profile.profile_id),
        EvidenceToolHandler(service, spec=CAPICOVE_MULTIMODAL_EVIDENCE_SPEC),
    )


__all__ = [
    "FORMAT_VERSION",
    "PROFILES",
    "UnifiedRetrievalError",
    "UnifiedRetrievalRuntime",
    "UnifiedRetrievalService",
    "UnifiedSettings",
    "activate_generation",
    "build_generation",
    "forget_media",
    "import_media",
    "open_unified_retrieval",
    "unified_settings",
]
