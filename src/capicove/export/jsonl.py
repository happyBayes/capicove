"""One-pass bounded JSONL export of complete observable Agent Episodes."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from collections.abc import Callable
from datetime import datetime
from importlib import resources
from pathlib import Path, PurePosixPath
from typing import BinaryIO

from capicove.export._state import ExportAggregation, ExportStateError
from capicove.export.media import ExportMediaResolver, TraceMediaCopyError, copy_media_reference
from capicove.export.models import (
    MediaMode,
    TraceExportManifest,
    TraceExportOptions,
    TraceExportShard,
    encode_trace_export_manifest,
)
from capicove.traces import TraceReader
from capicove.traces.codec import canonical_json_bytes, canonical_record_bytes


_SCHEMA_NAME = "capicove-agent-trace-v1.schema.json"
_SCHEMA_PATH = PurePosixPath("schemas") / _SCHEMA_NAME
DEFAULT_TRACE_EXPORT_OPTIONS = TraceExportOptions()


class TraceExportError(RuntimeError):
    """Report a stable local trace export failure."""


class TraceExportExistsError(TraceExportError):
    """Report refusal to overwrite an existing export path."""


class TraceExportRecordTooLargeError(TraceExportError):
    """Report a record larger than the configured bounded line size."""


class TraceExporter:
    """Stream one fixed complete-Episode snapshot into an atomic export package."""

    def __init__(
        self,
        reader: TraceReader,
        *,
        clock: Callable[[], datetime],
        export_id_factory: Callable[[], str],
        media_resolver: ExportMediaResolver | None = None,
    ) -> None:
        if not callable(clock):
            raise ValueError("clock must be callable")
        if not callable(export_id_factory):
            raise ValueError("export_id_factory must be callable")
        self._reader = reader
        self._clock = clock
        self._export_id_factory = export_id_factory
        self._media_resolver = media_resolver

    def export(
        self,
        output_directory: Path,
        /,
        *,
        options: TraceExportOptions = DEFAULT_TRACE_EXPORT_OPTIONS,
    ) -> TraceExportManifest:
        """Publish one new portable package or leave no final output on failure."""
        if not isinstance(output_directory, Path) or not output_directory.is_absolute():
            raise ValueError("output_directory must be absolute")
        if not isinstance(options, TraceExportOptions):
            raise ValueError("options must be TraceExportOptions")
        if options.media_mode is MediaMode.COPY and self._media_resolver is None:
            raise TraceExportError("copy media mode requires an export media resolver")
        if output_directory.exists():
            raise TraceExportExistsError("trace export output already exists")
        parent = output_directory.parent
        if not parent.is_dir():
            raise TraceExportError("trace export parent directory does not exist")
        temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.tmp-", dir=parent))
        temporary.chmod(0o700)
        state_path = temporary / "_export-state.sqlite3"
        state: ExportAggregation | None = None
        writer: _ShardWriter | None = None
        try:
            created_at = self._clock()
            if not isinstance(created_at, datetime) or created_at.utcoffset() is None:
                raise TraceExportError("export clock must return a timezone-aware datetime")
            export_id = self._export_id_factory()
            if not isinstance(export_id, str) or not export_id.strip():
                raise TraceExportError("export ID must not be blank")
            schema_bytes = (
                resources.files("capicove.export.schemas").joinpath(_SCHEMA_NAME).read_bytes()
            )
            schema_file = temporary / _SCHEMA_PATH
            schema_file.parent.mkdir()
            _write_synced(schema_file, schema_bytes)
            schema_sha256 = hashlib.sha256(schema_bytes).hexdigest()

            state = ExportAggregation(state_path)
            shards: list[TraceExportShard] = []
            prior_key: tuple[str, int] | None = None
            record_count = 0
            with self._reader.open_snapshot(options.query) as snapshot:
                summary = snapshot.summary
                for record in snapshot.iter_records():
                    key = (record.episode_id, record.sequence)
                    if prior_key is not None and key <= prior_key:
                        raise TraceExportError(
                            "trace snapshot records must be ordered by episode_id and sequence"
                        )
                    prior_key = key
                    encoded = canonical_record_bytes(record)
                    if len(encoded) > options.max_record_bytes:
                        raise TraceExportRecordTooLargeError(
                            "trace record exceeds the configured byte limit"
                        )
                    line_size = len(encoded) + 1
                    if writer is not None and writer.should_rotate(line_size, options):
                        shards.append(writer.close())
                        writer = None
                    if writer is None:
                        writer = _ShardWriter(temporary, len(shards))
                    writer.write(encoded)
                    if options.media_mode is MediaMode.COPY:
                        for reference in record.media:
                            if self._media_resolver is None:
                                raise TraceExportError("copy media mode requires a resolver")
                            source = self._media_resolver.resolve(reference)
                            if source is None:
                                raise TraceExportError("trace media could not be resolved locally")
                            try:
                                copy_media_reference(reference, source, temporary)
                            except TraceMediaCopyError as error:
                                raise TraceExportError(
                                    f"trace media copy failed: {error.code}"
                                ) from error
                    state.observe(
                        record,
                        assume_media_available=options.media_mode is MediaMode.COPY,
                    )
                    record_count += 1
                if writer is not None:
                    shards.append(writer.close())
                    writer = None

                media_count, media_size_bytes, missing_media_count = state.media_summary()
                manifest = TraceExportManifest(
                    format_version=1,
                    schema_version=1,
                    export_id=export_id,
                    created_at=created_at,
                    options=options,
                    shards=tuple(shards),
                    record_count=record_count,
                    record_type_counts=state.counts("record_type"),
                    modality_counts=state.counts("modality"),
                    tool_status_counts=state.counts("tool_status"),
                    episode_status_counts=state.counts("episode_status"),
                    included_episode_count=summary.selected_episode_count,
                    incomplete_episode_count=summary.selected_incomplete_episode_count,
                    omitted_restricted_episode_count=summary.omitted_restricted_episode_count,
                    media_count=media_count,
                    media_size_bytes=media_size_bytes,
                    missing_media_count=missing_media_count,
                    media_layout="sha256-v1",
                    schema_path=_SCHEMA_PATH,
                    schema_sha256=schema_sha256,
                )
            state.close()
            state = None
            state_path.unlink()
            manifest_bytes = canonical_json_bytes(encode_trace_export_manifest(manifest)) + b"\n"
            _write_synced(temporary / "manifest.json", manifest_bytes)
            _sync_directory(schema_file.parent)
            _sync_directory(temporary)
            if output_directory.exists():
                raise TraceExportExistsError("trace export output already exists")
            os.rename(temporary, output_directory)
            _sync_directory(parent)
            return manifest
        except (TraceExportError, ExportStateError):
            raise
        except OSError as error:
            raise TraceExportError("trace export filesystem operation failed") from error
        except Exception:
            raise
        finally:
            if writer is not None:
                writer.abort()
            if state is not None:
                state.close()
            if temporary.exists():
                shutil.rmtree(temporary)


class _ShardWriter:
    """Write and hash one shard without rereading it."""

    def __init__(self, root: Path, index: int) -> None:
        self.relative_path = PurePosixPath(f"records-{index:05d}.jsonl")
        self._file: BinaryIO = (root / self.relative_path).open("xb")
        self._hasher = hashlib.sha256()
        self.record_count = 0
        self.size_bytes = 0
        self._closed = False

    def should_rotate(self, next_line_size: int, options: TraceExportOptions) -> bool:
        """Return whether the next line belongs in a new shard."""
        return self.record_count > 0 and (
            self.record_count >= options.max_records_per_shard
            or self.size_bytes + next_line_size > options.max_bytes_per_shard
        )

    def write(self, encoded_record: bytes) -> None:
        """Append exactly one canonical JSON line."""
        line = encoded_record + b"\n"
        self._file.write(line)
        self._hasher.update(line)
        self.record_count += 1
        self.size_bytes += len(line)

    def close(self) -> TraceExportShard:
        """Flush and return immutable shard metadata."""
        if self._closed:
            raise TraceExportError("trace shard is already closed")
        self._file.flush()
        os.fsync(self._file.fileno())
        self._file.close()
        self._closed = True
        return TraceExportShard(
            path=self.relative_path,
            record_count=self.record_count,
            size_bytes=self.size_bytes,
            sha256=self._hasher.hexdigest(),
        )

    def abort(self) -> None:
        """Close an unpublished shard after export failure."""
        if not self._closed:
            self._file.close()
            self._closed = True


def _write_synced(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _sync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
