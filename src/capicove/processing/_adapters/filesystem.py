"""Private verified local publication of immutable Parquet datasets."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

from capicove.contracts.data import (
    DataFormat,
    DatasetSchema,
    DatasetSnapshotRef,
)


class DatasetPublicationError(RuntimeError):
    """Report a bounded local snapshot-publication failure."""


@dataclass(frozen=True, slots=True)
class PublicationRequest:
    """Describe one staged Parquet dataset awaiting publication."""

    run_id: str
    dataset_id: str
    version: int
    staging_path: Path
    schema: DatasetSchema
    processing_fingerprint: str
    parent_snapshot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if re.fullmatch(r"run:[A-Za-z0-9][A-Za-z0-9._:-]*", self.run_id) is None:
            raise ValueError("run_id must be canonical")
        if re.fullmatch(r"dataset:[A-Za-z0-9][A-Za-z0-9._-]*", self.dataset_id) is None:
            raise ValueError("dataset_id must be canonical")
        if isinstance(self.version, bool) or not isinstance(self.version, int) or self.version <= 0:
            raise ValueError("version must be a positive integer")
        if not isinstance(self.staging_path, Path) or not self.staging_path.is_absolute():
            raise ValueError("staging_path must be absolute")
        if not isinstance(self.schema, DatasetSchema):
            raise TypeError("schema must be a DatasetSchema")
        if re.fullmatch(r"[0-9a-f]{64}", self.processing_fingerprint) is None:
            raise ValueError("processing_fingerprint must be a lowercase SHA-256 hash")


@dataclass(frozen=True, slots=True)
class PendingPublication:
    """Identify one filesystem publication awaiting metadata reconciliation."""

    run_id: str
    snapshot_id: str | None
    manifest_sha256: str | None
    destination: Path
    published: bool

    def __post_init__(self) -> None:
        if re.fullmatch(r"run:[A-Za-z0-9][A-Za-z0-9._:-]*", self.run_id) is None:
            raise ValueError("run_id must be canonical")
        if (
            self.snapshot_id is not None
            and re.fullmatch(r"snapshot:[0-9a-f]{64}", self.snapshot_id) is None
        ):
            raise ValueError("snapshot_id must be canonical")
        if (
            self.manifest_sha256 is not None
            and re.fullmatch(r"[0-9a-f]{64}", self.manifest_sha256) is None
        ):
            raise ValueError("manifest_sha256 must be canonical")
        if self.published and (self.snapshot_id is None or self.manifest_sha256 is None):
            raise ValueError("published pending identity must be complete")
        if not self.destination.is_absolute():
            raise ValueError("destination must be absolute")


class LocalDatasetPublisher:
    """Stage and atomically publish exact local Parquet files."""

    def __init__(self, root: Path) -> None:
        if not isinstance(root, Path) or not root.is_absolute():
            raise ValueError("root must be an absolute Path")
        self._root = root.resolve()
        self._staging_root = self._root / ".staging"
        if self._staging_root.is_symlink():
            raise DatasetPublicationError("staging root must not be a symbolic link")
        self._staging_root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if self._staging_root.is_symlink() or self._staging_root.resolve() != self._staging_root:
            raise DatasetPublicationError("staging root must remain inside the publisher root")

    def stage(self, run_id: str, /) -> Path:
        """Create one private empty run staging directory."""
        if re.fullmatch(r"run:[A-Za-z0-9][A-Za-z0-9._:-]*", run_id) is None:
            raise ValueError("run_id must be canonical")
        name = f"run-{run_id.encode('utf-8').hex()}"
        path = self._staging_root / name
        if os.path.lexists(path):
            raise DatasetPublicationError("run staging directory already exists")
        temporary = Path(tempfile.mkdtemp(prefix=f".{name}.", dir=self._staging_root))
        try:
            marker_bytes = (
                json.dumps(
                    {"run_id": run_id, "state": "staged"},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            _write_exclusive(temporary / ".capicove-pending.json", marker_bytes)
            _fsync_directory(temporary)
            if os.path.lexists(path):
                raise DatasetPublicationError("run staging directory already exists")
            os.rename(temporary, path)
            _fsync_directory(self._staging_root)
        except DatasetPublicationError:
            _remove_private_staging(temporary, self._staging_root)
            raise
        except OSError as error:
            _remove_private_staging(temporary, self._staging_root)
            raise DatasetPublicationError("run staging creation failed") from error
        return path.resolve()

    def discard(self, staging_path: Path, /) -> None:
        """Remove only an unpublished directory owned by this publisher."""
        _remove_private_staging(staging_path, self._staging_root)

    def retract(self, snapshot: DatasetSnapshotRef, /) -> None:
        """Remove one exact snapshot when its metadata transaction fails."""
        destination = self._snapshot_destination(snapshot)
        _remove_verified_destination(destination, snapshot.manifest_sha256)

    def validate(
        self, snapshot: DatasetSnapshotRef, /, *, validator: Callable[[Path], bool]
    ) -> None:
        """Verify the complete manifest and safe part inventory before reading."""
        destination = self._snapshot_destination(snapshot)
        for ancestor in (destination, *destination.parents):
            if ancestor.is_symlink():
                raise DatasetPublicationError("snapshot path contains a symbolic link")
        _verify_manifest(destination, snapshot.manifest_sha256)
        for path in destination.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise DatasetPublicationError("snapshot contains an unsafe path")
        entries = self._part_entries(destination)
        expected = {
            "data_format": DataFormat.PARQUET.value,
            "dataset_id": snapshot.dataset_id,
            "files": entries,
            "parent_snapshot_ids": list(snapshot.parent_snapshot_ids),
            "processing_fingerprint": snapshot.processing_fingerprint,
            "schema_sha256": snapshot.schema.sha256,
            "version": snapshot.version,
        }
        manifest = json.loads((destination / "manifest.json").read_bytes())
        if manifest != expected or snapshot.data_format is not DataFormat.PARQUET:
            raise DatasetPublicationError("snapshot manifest or part inventory changed")
        if snapshot.snapshot_id != f"snapshot:{snapshot.manifest_sha256}":
            raise DatasetPublicationError("snapshot identity does not match manifest")
        if not validator(destination):
            raise DatasetPublicationError("snapshot validator rejected committed output")

    def finalize(self, run_id: str, snapshot: DatasetSnapshotRef, /) -> None:
        """Clear the recovery marker for one durably committed snapshot."""
        destination = self._snapshot_destination(snapshot)
        _verify_manifest(destination, snapshot.manifest_sha256)
        marker = destination / ".capicove-pending.json"
        value = _read_pending_marker(marker)
        if (
            value.get("snapshot_id") != snapshot.snapshot_id
            or value.get("run_id") != run_id
            or value.get("manifest_sha256") != snapshot.manifest_sha256
            or set(value) != {"manifest_sha256", "run_id", "snapshot_id"}
        ):
            raise DatasetPublicationError("pending publication identity changed")
        marker.unlink()
        _fsync_directory(destination)

    def pending_publications(self) -> tuple[PendingPublication, ...]:
        """Return exact pending publications for startup reconciliation."""
        pending: list[PendingPublication] = []
        for destination in sorted(self._staging_root.iterdir()):
            if destination.is_symlink() or not destination.is_dir():
                raise DatasetPublicationError("staging recovery path is invalid")
            marker = destination / ".capicove-pending.json"
            derived_run_id = _run_id_from_staging_name(destination.name)
            try:
                value = _read_pending_marker(marker)
                marker_run_id = str(value["run_id"])
                if derived_run_id is not None and marker_run_id != derived_run_id:
                    raise DatasetPublicationError("staging recovery identity changed")
                run_id = marker_run_id
            except (DatasetPublicationError, KeyError, TypeError, ValueError) as error:
                if derived_run_id is None:
                    raise DatasetPublicationError(
                        "staging recovery identity is unavailable"
                    ) from error
                run_id = derived_run_id
            try:
                pending.append(
                    PendingPublication(
                        run_id=run_id,
                        snapshot_id=None,
                        manifest_sha256=None,
                        destination=destination,
                        published=False,
                    )
                )
            except ValueError as error:
                raise DatasetPublicationError("staging recovery identity is invalid") from error
        for marker in sorted(self._root.glob("*/v*/.capicove-pending.json")):
            if (
                marker.is_symlink()
                or marker.parent.is_symlink()
                or marker.parent.parent.is_symlink()
            ):
                raise DatasetPublicationError("pending publication path contains a symbolic link")
            value = _read_pending_marker(marker)
            try:
                pending.append(
                    PendingPublication(
                        run_id=str(value["run_id"]),
                        snapshot_id=(str(value["snapshot_id"]) if "snapshot_id" in value else None),
                        manifest_sha256=(
                            str(value["manifest_sha256"]) if "manifest_sha256" in value else None
                        ),
                        destination=marker.parent,
                        published=True,
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise DatasetPublicationError("pending publication marker is invalid") from error
        return tuple(pending)

    def retract_pending(self, pending: PendingPublication, /) -> None:
        """Remove one validated pending publication during recovery."""
        if not pending.published:
            if pending.destination.parent != self._staging_root or pending.destination.is_symlink():
                raise DatasetPublicationError("pending staging path is unsafe")
            shutil.rmtree(pending.destination)
            _fsync_directory(self._staging_root)
            return
        expected = self._root / pending.destination.parent.name / pending.destination.name
        if pending.destination != expected:
            raise DatasetPublicationError("pending publication is outside the publisher root")
        if pending.manifest_sha256 is None:  # pragma: no cover - dataclass invariant
            raise DatasetPublicationError("pending publication identity is incomplete")
        _remove_verified_destination(pending.destination, pending.manifest_sha256)

    def finalize_pending(
        self,
        pending: PendingPublication,
        snapshot: DatasetSnapshotRef,
        /,
    ) -> None:
        """Clear one validated recovery marker after durable success."""
        if not pending.published or pending.snapshot_id is None:
            raise DatasetPublicationError("staged publication cannot be finalized")
        destination = self._snapshot_destination(snapshot)
        if (
            pending.destination != destination
            or pending.snapshot_id != snapshot.snapshot_id
            or pending.manifest_sha256 != snapshot.manifest_sha256
        ):
            raise DatasetPublicationError("pending publication identity changed")
        _verify_manifest(destination, snapshot.manifest_sha256)
        marker = pending.destination / ".capicove-pending.json"
        value = _read_pending_marker(marker)
        if value != {
            "manifest_sha256": pending.manifest_sha256,
            "run_id": pending.run_id,
            "snapshot_id": pending.snapshot_id,
        }:
            raise DatasetPublicationError("pending publication identity changed")
        marker.unlink()
        _fsync_directory(pending.destination)

    def publish(
        self,
        request: PublicationRequest,
        /,
        *,
        validator: Callable[[Path], bool],
    ) -> DatasetSnapshotRef:
        """Verify, manifest, and atomically commit a staged Parquet dataset."""
        staging = request.staging_path
        if staging.parent.resolve() != self._staging_root.resolve():
            raise DatasetPublicationError("staging path is outside the publisher root")
        try:
            staged_value = _read_pending_marker(staging / ".capicove-pending.json")
            if staged_value != {"run_id": request.run_id, "state": "staged"}:
                raise DatasetPublicationError("staging identity does not match the run")
            entries = self._part_entries(staging)
            if not validator(staging):
                raise DatasetPublicationError("dataset validator rejected staged output")
            manifest_value = {
                "data_format": DataFormat.PARQUET.value,
                "dataset_id": request.dataset_id,
                "files": entries,
                "parent_snapshot_ids": list(request.parent_snapshot_ids),
                "processing_fingerprint": request.processing_fingerprint,
                "schema_sha256": request.schema.sha256,
                "version": request.version,
            }
            manifest_bytes = (
                json.dumps(
                    manifest_value,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
            manifest_path = staging / "manifest.json"
            _write_exclusive(manifest_path, manifest_bytes)

            destination = self._destination(request.dataset_id, request.version)
            if destination.exists():
                raise DatasetPublicationError("dataset version is already committed")
            snapshot = DatasetSnapshotRef(
                dataset_id=request.dataset_id,
                version=request.version,
                snapshot_id=f"snapshot:{manifest_sha256}",
                uri=destination.as_uri(),
                data_format=DataFormat.PARQUET,
                schema=request.schema,
                manifest_sha256=manifest_sha256,
                processing_fingerprint=request.processing_fingerprint,
                parent_snapshot_ids=request.parent_snapshot_ids,
            )
            pending_bytes = (
                json.dumps(
                    {
                        "manifest_sha256": manifest_sha256,
                        "run_id": request.run_id,
                        "snapshot_id": snapshot.snapshot_id,
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode("utf-8")
                + b"\n"
            )
            _replace_file(staging / ".capicove-pending.json", pending_bytes)
            _fsync_directory(staging)
            os.replace(staging, destination)
            _fsync_directory(destination.parent)
            return snapshot
        except DatasetPublicationError:
            _remove_private_staging(staging, self._staging_root)
            raise
        except OSError as error:
            _remove_private_staging(staging, self._staging_root)
            raise DatasetPublicationError("dataset publication failed") from error

    def _destination(self, dataset_id: str, version: int) -> Path:
        dataset_name = dataset_id.removeprefix("dataset:")
        dataset_root = self._root / dataset_name
        if dataset_root.is_symlink():
            raise DatasetPublicationError("dataset destination must not be a symbolic link")
        dataset_root.mkdir(parents=False, exist_ok=True, mode=0o700)
        if dataset_root.is_symlink() or dataset_root.resolve() != dataset_root:
            raise DatasetPublicationError("dataset destination must not be a symbolic link")
        destination = dataset_root / f"v{version}"
        if destination.is_symlink():
            raise DatasetPublicationError("dataset version must not be a symbolic link")
        return destination

    def _snapshot_destination(self, snapshot: DatasetSnapshotRef) -> Path:
        dataset_name = snapshot.dataset_id.removeprefix("dataset:")
        destination = self._root / dataset_name / f"v{snapshot.version}"
        parsed = urlsplit(snapshot.uri)
        declared = Path(unquote(parsed.path))
        if parsed.scheme != "file" or parsed.netloc or declared != destination:
            raise DatasetPublicationError("snapshot is outside the publisher root")
        if destination.is_symlink() or destination.parent.is_symlink():
            raise DatasetPublicationError("snapshot path contains a symbolic link")
        return destination

    def _part_entries(self, staging: Path) -> list[dict[str, object]]:
        if not staging.is_dir() or staging.is_symlink():
            raise DatasetPublicationError("staging path must be a regular directory")
        entries: list[dict[str, object]] = []
        for path in sorted(staging.rglob("part-*.parquet")):
            observed = path.lstat()
            if not stat.S_ISREG(observed.st_mode):
                raise DatasetPublicationError("dataset parts must be regular files")
            relative = path.relative_to(staging)
            pure = PurePosixPath(relative.as_posix())
            if pure.is_absolute() or ".." in pure.parts:
                raise DatasetPublicationError("dataset part path escapes staging")
            if observed.st_size <= 0:
                raise DatasetPublicationError("dataset parts must not be empty")
            entries.append(
                {
                    "path": pure.as_posix(),
                    "sha256": _file_sha256(path),
                    "size_bytes": observed.st_size,
                }
            )
        if not entries:
            raise DatasetPublicationError("staging contains no Parquet parts")
        return entries


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _write_exclusive(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)


def _replace_file(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(".tmp")
    _write_exclusive(temporary, payload)
    try:
        os.replace(temporary, path)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise


def _run_id_from_staging_name(name: str) -> str | None:
    match = re.fullmatch(r"run-([0-9a-f]+)", name)
    if match is None:
        match = re.fullmatch(r"\.run-([0-9a-f]+)\.[A-Za-z0-9_-]+", name)
    if match is None:
        return None
    try:
        return bytes.fromhex(match.group(1)).decode("utf-8")
    except (UnicodeDecodeError, ValueError):
        return None


def _read_pending_marker(path: Path) -> dict[str, object]:
    if not path.is_file() or path.is_symlink() or path.stat().st_size > 16 * 1024:
        raise DatasetPublicationError("pending publication marker is invalid")
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DatasetPublicationError("pending publication marker is invalid") from error
    if not isinstance(value, dict):
        raise DatasetPublicationError("pending publication marker is invalid")
    return value


def _remove_verified_destination(destination: Path, manifest_sha256: str) -> None:
    if destination.is_symlink() or destination.parent.is_symlink():
        raise DatasetPublicationError("snapshot path contains a symbolic link")
    _verify_manifest(destination, manifest_sha256)
    shutil.rmtree(destination)
    _fsync_directory(destination.parent)


def _verify_manifest(destination: Path, manifest_sha256: str) -> None:
    manifest = destination / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise DatasetPublicationError("snapshot manifest is unavailable")
    if hashlib.sha256(manifest.read_bytes()).hexdigest() != manifest_sha256:
        raise DatasetPublicationError("snapshot manifest identity changed")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _remove_private_staging(path: Path, staging_root: Path) -> None:
    try:
        if path.parent.resolve() == staging_root.resolve():
            shutil.rmtree(path, ignore_errors=True)
    except OSError:
        pass
