"""Capability-owned, metadata-only reads for configured operator trace stores."""

from __future__ import annotations

import hashlib
import os
import re
import stat
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Self

from capicove.export import (
    MediaMode,
    TraceExporter,
    TraceExportExistsError,
    TraceExportManifest,
    TraceExportOptions,
    WorkspaceExportMediaResolver,
    inspect_trace_export,
)
from capicove.export.validation import TraceValidationReport
from capicove.traces.events import EpisodeStarted
from capicove.traces.models import TraceRecord
from capicove.traces.ports import (
    DEFAULT_TRACE_QUERY,
    TraceQuery,
    TraceReader,
    TraceReadSnapshot,
    TraceSelectionSummary,
)


@dataclass(frozen=True, slots=True)
class TraceStoreInspection:
    """Describe a bounded trace-store snapshot without record content."""

    store_alias: str
    episode_count: int
    incomplete_episode_count: int
    omitted_restricted_episode_count: int


@dataclass(frozen=True, slots=True)
class TraceValidationFinding:
    """Expose only a validation finding's machine-readable classification."""

    severity: str
    code: str


@dataclass(frozen=True, slots=True)
class TraceExportInspection:
    """Describe a configured export without record IDs, text, or paths."""

    export_id: str
    valid: bool
    records_checked: int
    episodes_checked: int
    media_checked: int
    findings: tuple[TraceValidationFinding, ...]
    findings_truncated: bool


@dataclass(frozen=True, slots=True)
class TraceOperatorExportResult:
    """Expose fixed-size, content-free metadata for one operator export."""

    export_id: str
    included_episode_count: int
    incomplete_episode_count: int
    omitted_restricted_episode_count: int
    record_count: int
    shard_count: int
    media_count: int
    missing_media_count: int
    media_mode: str


def _operator_export_result(manifest: TraceExportManifest) -> TraceOperatorExportResult:
    """Project a complete on-disk receipt to a bounded operator response."""
    return TraceOperatorExportResult(
        export_id=manifest.export_id,
        included_episode_count=manifest.included_episode_count,
        incomplete_episode_count=manifest.incomplete_episode_count,
        omitted_restricted_episode_count=manifest.omitted_restricted_episode_count,
        record_count=manifest.record_count,
        shard_count=len(manifest.shards),
        media_count=manifest.media_count,
        missing_media_count=manifest.missing_media_count,
        media_mode=manifest.options.media_mode.value,
    )


class _WorkspaceBoundTraceSnapshot:
    """Reject any Episode outside the operator's startup workspace scope."""

    def __init__(self, snapshot: TraceReadSnapshot, workspace_id: str) -> None:
        self._snapshot = snapshot
        self._workspace_id = workspace_id

    @property
    def summary(self) -> TraceSelectionSummary:
        """Return the underlying fixed selection summary."""
        return self._snapshot.summary

    def iter_records(self) -> Iterator[TraceRecord]:
        """Yield only records whose Episode starts in the configured workspace."""
        current_episode: str | None = None
        for record in self._snapshot.iter_records():
            if record.episode_id != current_episode:
                if (
                    record.sequence != 0
                    or not isinstance(record.payload, EpisodeStarted)
                    or record.payload.workspace_id != self._workspace_id
                ):
                    raise ValueError("trace export workspace is unavailable")
                current_episode = record.episode_id
            yield record

    def __enter__(self) -> Self:
        self._snapshot.__enter__()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self._snapshot.__exit__(exc_type, exc_value, traceback)


class _WorkspaceBoundTraceReader:
    """Apply one immutable workspace scope to every exported snapshot."""

    def __init__(self, reader: TraceReader, workspace_id: str) -> None:
        self._reader = reader
        self._workspace_id = workspace_id

    def open_snapshot(self, query: TraceQuery = DEFAULT_TRACE_QUERY, /) -> TraceReadSnapshot:
        """Open one fixed snapshot with a workspace guard on its records."""
        return _WorkspaceBoundTraceSnapshot(self._reader.open_snapshot(query), self._workspace_id)


class TraceOperatorReader:
    """Read only startup-configured trace stores and exports by opaque alias."""

    def __init__(
        self,
        *,
        stores: Mapping[str, TraceReader],
        exports: Mapping[str, Callable[[bool], TraceValidationReport]] | None = None,
    ) -> None:
        selected_exports = {} if exports is None else exports
        if (
            not isinstance(stores, Mapping)
            or not isinstance(selected_exports, Mapping)
            or len(stores) + len(selected_exports) > 32
            or any(
                not isinstance(alias, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", alias) is None
                for alias in (*stores, *selected_exports)
            )
        ):
            raise ValueError("trace resources must have at most 32 bounded aliases")
        if stores.keys() & selected_exports.keys():
            raise ValueError("trace stores and exports require distinct aliases")
        if any(not callable(validator) for validator in selected_exports.values()):
            raise ValueError("trace exports require configured validators")
        self._stores = MappingProxyType(dict(stores))
        self._exports = MappingProxyType(dict(selected_exports))

    def inspect_store(self, alias: str) -> TraceStoreInspection:
        """Return counts from one fixed snapshot, excluding restricted Episodes."""
        try:
            reader = self._stores[alias]
        except (KeyError, TypeError) as error:
            raise KeyError("trace store is unavailable") from error
        try:
            with reader.open_snapshot(TraceQuery(include_incomplete=True)) as snapshot:
                summary = snapshot.summary
                return TraceStoreInspection(
                    store_alias=alias,
                    episode_count=summary.selected_episode_count,
                    incomplete_episode_count=summary.selected_incomplete_episode_count,
                    omitted_restricted_episode_count=summary.omitted_restricted_episode_count,
                )
        except Exception:
            raise ValueError("trace store inspection failed") from None

    def validate_export(self, export_id: str, *, verify_media: bool) -> TraceExportInspection:
        """Validate one configured export and return content-free bounded findings."""
        if type(verify_media) is not bool:
            raise ValueError("verify_media must be a boolean")
        try:
            validator = self._exports[export_id]
        except (KeyError, TypeError) as error:
            raise KeyError("trace export is unavailable") from error
        try:
            report = validator(verify_media)
        except Exception:
            raise ValueError("trace export validation failed") from None
        if not isinstance(report, TraceValidationReport):
            raise ValueError("trace export validator returned an invalid report")
        selected_issues = report.issues[:100]
        if any(
            re.fullmatch(r"[A-Z][A-Z0-9_]{0,63}", issue.code) is None for issue in selected_issues
        ):
            raise ValueError("trace export validation failed")
        findings = tuple(
            TraceValidationFinding(issue.severity.value, issue.code) for issue in selected_issues
        )
        return TraceExportInspection(
            export_id=export_id,
            valid=report.valid,
            records_checked=report.records_checked,
            episodes_checked=report.episodes_checked,
            media_checked=report.media_checked,
            findings=findings,
            findings_truncated=len(report.issues) > len(findings),
        )

    def inspect_export(self, export_id: str) -> TraceExportInspection:
        """Inspect one configured export without eagerly hashing copied media."""
        return self.validate_export(export_id, verify_media=False)


class TraceOperatorExporter:
    """Export only into configured private roots under explicit admin policy."""

    def __init__(
        self,
        *,
        reader: TraceReader,
        workspace_id: str,
        output_roots: Mapping[str, Path],
        media_root: Path | None = None,
        administrative: bool = False,
        permissions: frozenset[str] = frozenset(),
    ) -> None:
        if (
            not isinstance(workspace_id, str)
            or not workspace_id.strip()
            or len(workspace_id.encode("utf-8")) > 128
        ):
            raise ValueError("trace workspace identity is required")
        if (
            not isinstance(output_roots, Mapping)
            or not output_roots
            or len(output_roots) > 16
            or any(
                not isinstance(alias, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,63}", alias) is None
                for alias in output_roots
            )
        ):
            raise ValueError("trace output roots require bounded aliases")
        root_identities: dict[str, tuple[int, int]] = {}
        for alias, root in output_roots.items():
            _require_private_export_root(root)
            if os.name == "posix":
                descriptor = _open_private_export_root_posix(root)
                try:
                    metadata = os.fstat(descriptor)
                    root_identities[alias] = (metadata.st_dev, metadata.st_ino)
                finally:
                    os.close(descriptor)
        if media_root is not None:
            _require_safe_media_root(media_root)
        if type(administrative) is not bool:
            raise ValueError("administrative must be a boolean")
        if not isinstance(permissions, frozenset) or any(
            not isinstance(permission, str) for permission in permissions
        ):
            raise ValueError("trace permissions must be a set of strings")
        self._reader = reader
        self._workspace_id = workspace_id
        self._output_roots = MappingProxyType(dict(output_roots))
        self._output_root_identities = MappingProxyType(root_identities)
        self._media_root = media_root
        self._administrative = administrative
        self._permissions = frozenset(permissions)

    def export(
        self,
        *,
        output_root_alias: str,
        query: TraceQuery,
        idempotency_key: str,
        media_mode: MediaMode = MediaMode.REFERENCE,
    ) -> TraceOperatorExportResult:
        """Publish or verify one scope-bound, idempotent trace package."""
        try:
            root = self._output_roots[output_root_alias]
        except (KeyError, TypeError) as error:
            raise KeyError("trace output root is unavailable") from error
        if not isinstance(query, TraceQuery) or not isinstance(media_mode, MediaMode):
            raise ValueError("trace export request is invalid")
        if (
            not isinstance(idempotency_key, str)
            or not idempotency_key.strip()
            or len(idempotency_key) > 128
        ):
            raise ValueError("trace export idempotency key is invalid")
        if "trace:export" not in self._permissions:
            raise PermissionError("trace export is unavailable")
        if query.include_restricted and not (
            self._administrative and "trace:export_restricted" in self._permissions
        ):
            raise PermissionError("restricted trace export is unavailable")
        if media_mode is MediaMode.COPY and not (
            self._administrative
            and "trace:copy_media" in self._permissions
            and self._media_root is not None
        ):
            raise PermissionError("trace media copy is unavailable")
        if media_mode is MediaMode.COPY and self._media_root is not None:
            _require_safe_media_root(self._media_root)
        _require_private_export_root(root)
        if os.name == "posix":
            descriptor = _open_private_export_root_posix(root)
            try:
                metadata = os.fstat(descriptor)
                identity = (metadata.st_dev, metadata.st_ino)
                if identity != self._output_root_identities[output_root_alias]:
                    raise ValueError("trace export failed")
                bound_root = Path(f"/dev/fd/{descriptor}")
                if not bound_root.is_dir():
                    raise ValueError("trace export failed")
                manifest = self._export_to_root(
                    bound_root, output_root_alias, query, idempotency_key, media_mode
                )
                _require_current_export_root_posix(root, identity)
                return _operator_export_result(manifest)
            finally:
                os.close(descriptor)
        return _operator_export_result(
            self._export_to_root(root, output_root_alias, query, idempotency_key, media_mode)
        )

    def _export_to_root(
        self,
        root: Path,
        output_root_alias: str,
        query: TraceQuery,
        idempotency_key: str,
        media_mode: MediaMode,
    ) -> TraceExportManifest:
        """Export through one already-authorized output root."""
        identity = hashlib.sha256(
            (self._workspace_id + "\0" + output_root_alias + "\0" + idempotency_key).encode()
        ).hexdigest()[:32]
        export_id = f"export:{identity}"
        destination = root / f"trace-{identity}"
        inspection_destination = self._output_roots[output_root_alias] / destination.name
        options = TraceExportOptions(query=query, media_mode=media_mode)
        if destination.exists() or destination.is_symlink():
            if os.name == "posix":
                _require_current_export_root_posix(
                    self._output_roots[output_root_alias],
                    self._output_root_identities[output_root_alias],
                )
            return _verify_operator_export(
                inspection_destination, export_id, options, self._workspace_id
            )
        try:
            resolver = (
                WorkspaceExportMediaResolver(self._media_root)
                if media_mode is MediaMode.COPY and self._media_root is not None
                else None
            )
            return TraceExporter(
                _WorkspaceBoundTraceReader(self._reader, self._workspace_id),
                clock=lambda: datetime.now(UTC),
                export_id_factory=lambda: export_id,
                media_resolver=resolver,
            ).export(destination, options=options)
        except TraceExportExistsError:
            if os.name == "posix":
                _require_current_export_root_posix(
                    self._output_roots[output_root_alias],
                    self._output_root_identities[output_root_alias],
                )
            return _verify_operator_export(
                inspection_destination, export_id, options, self._workspace_id
            )
        except Exception:
            raise ValueError("trace export failed") from None


def _open_private_export_root_posix(root: Path) -> int:
    """Open every ancestor without following links and pin the private root."""
    if not hasattr(os, "O_DIRECTORY") or not hasattr(os, "O_NOFOLLOW"):
        raise ValueError("trace export failed")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    directory_fd: int | None = None
    try:
        directory_fd = os.open(root.anchor, flags)
        for component in root.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        metadata = os.fstat(directory_fd)
        if (
            not stat.S_ISDIR(metadata.st_mode)
            or metadata.st_uid != os.geteuid()
            or metadata.st_mode & 0o077
        ):
            raise ValueError("trace export failed")
        return directory_fd
    except OSError:
        if directory_fd is not None:
            os.close(directory_fd)
        raise ValueError("trace export failed") from None
    except Exception:
        if directory_fd is not None:
            os.close(directory_fd)
        raise


def _require_current_export_root_posix(root: Path, identity: tuple[int, int]) -> None:
    """Reject a configured alias that changed while its export was running."""
    descriptor = _open_private_export_root_posix(root)
    try:
        metadata = os.fstat(descriptor)
        if (metadata.st_dev, metadata.st_ino) != identity:
            raise ValueError("trace export failed")
    finally:
        os.close(descriptor)


def _verify_operator_export(
    destination: Path, export_id: str, options: TraceExportOptions, workspace_id: str
) -> TraceExportManifest:
    """Return only a matching, fully valid prior idempotent export."""
    try:
        manifest, report = inspect_trace_export(
            destination,
            verify_media=options.media_mode is MediaMode.COPY,
            required_workspace_id=workspace_id,
        )
    except (OSError, ValueError) as error:
        raise ValueError("trace export receipt is invalid") from error
    if not report.valid:
        raise ValueError("trace export receipt is invalid")
    if manifest.export_id != export_id or manifest.options != options:
        raise ValueError("idempotency_conflict")
    return manifest


def _require_private_export_root(root: Path) -> None:
    """Fail closed on an unowned or externally writable output directory."""
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("trace output root must be absolute")
    if os.name == "nt":
        from capicove._windows_files import validate_windows_path

        validate_windows_path(
            root, directory=True, private_acl=True, require_private_inheritance=True
        )
        return
    if (
        os.name != "posix"
        or root.is_symlink()
        or any(parent.is_symlink() for parent in root.parents)
    ):
        raise ValueError("trace output root is unsafe")
    metadata = root.stat()
    if (
        not stat.S_ISDIR(metadata.st_mode)
        or metadata.st_uid != os.geteuid()
        or metadata.st_mode & 0o077
    ):
        raise ValueError("trace output root must be owner-private")


def _require_safe_media_root(root: Path) -> None:
    """Reject linked or unavailable configured media roots before copy."""
    if not isinstance(root, Path) or not root.is_absolute():
        raise ValueError("trace media root must be absolute")
    try:
        if os.name == "nt":
            from capicove._windows_files import validate_windows_path

            validate_windows_path(root, directory=True)
            return
        if (
            os.name != "posix"
            or root.is_symlink()
            or any(parent.is_symlink() for parent in root.parents)
        ):
            raise ValueError("trace media root is unsafe")
        if not stat.S_ISDIR(root.lstat().st_mode):
            raise ValueError("trace media root is unsafe")
    except OSError as error:
        raise ValueError("trace media root is unsafe") from error
