"""Append-only local receipt storage for accepted chunk results."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import stat
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any
from uuid import uuid4

from capicove.books.chunk_models import digest
from capicove.books.chunk_result_submission import canonical_json, validate_submission
from capicove.books.workspace import (
    _atomic_write,
    _reject_symlinks,
    _verify_pinned_parent_posix,
)


def _read_regular_file_posix(path: Path, *, max_bytes: int) -> bytes:
    """Read a bounded ledger file through pinned no-follow POSIX handles."""
    if not path.is_absolute() or not all(
        hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")
    ):
        raise ValueError("secure ledger file access is unavailable")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    file_flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:-1]:
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        file_fd = os.open(path.name, file_flags, dir_fd=directory_fd)
    finally:
        os.close(directory_fd)
    with os.fdopen(file_fd, "rb") as source:
        before = os.fstat(source.fileno())
        if not stat.S_ISREG(before.st_mode) or before.st_size > max_bytes:
            raise ValueError("invalid ledger file")
        content = source.read(max_bytes + 1)
        after = os.fstat(source.fileno())
        if len(content) > max_bytes or (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
            before.st_ctime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
            after.st_ctime_ns,
        ):
            raise ValueError("invalid ledger file")
        return content


def _discard_uncommitted_staging(records_root: Path) -> None:
    """Discard only interrupted result staging directories before a retry."""
    if os.name != "posix":
        _reject_symlinks(records_root)
        if records_root.exists():
            for entry in records_root.iterdir():
                if entry.name.startswith(".result-"):
                    if entry.is_symlink() or not entry.is_dir():
                        raise ValueError("invalid chunk result staging")
                    shutil.rmtree(entry)
        return
    if not records_root.is_absolute() or not all(
        hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")
    ):
        raise ValueError("secure chunk result staging access is unavailable")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(records_root.anchor, flags)
    try:
        for component in records_root.parts[1:]:
            try:
                next_fd = os.open(component, flags, dir_fd=directory_fd)
            except FileNotFoundError:
                return
            os.close(directory_fd)
            directory_fd = next_fd
        for name in os.listdir(directory_fd):
            if not name.startswith(".result-"):
                continue
            mode = os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode
            if not stat.S_ISDIR(mode) or not shutil.rmtree.avoids_symlink_attacks:
                raise ValueError("invalid chunk result staging")
            shutil.rmtree(name, dir_fd=directory_fd)
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _open_records_root_posix(records_root: Path) -> int:
    """Create and pin one Book-owned records directory without following links."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(records_root.anchor, flags)
    try:
        for component in records_root.parts[1:]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        return directory_fd
    except BaseException:
        os.close(directory_fd)
        raise


def _write_staged_file_posix(staging_fd: int, name: str, content: bytes) -> None:
    """Write one new result file inside the pinned private staging directory."""
    descriptor = os.open(
        name,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
        0o600,
        dir_fd=staging_fd,
    )
    with os.fdopen(descriptor, "wb") as output:
        output.write(content)
        output.flush()
        os.fsync(output.fileno())


def _valid_receipt_key(value: object) -> bool:
    """Reject malformed UTF-8 keys while reading an untrusted receipt index."""
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return len(value.encode("utf-8")) <= 128
    except UnicodeError:
        return False


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Reject ambiguous keys in a persisted receipt index."""
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate receipt index key")
        result[key] = value
    return result


class ChunkResultLedger:
    """Persist verified result versions in one immutable chunk run."""

    def __init__(self, run_directory: Path, run_id: str, *, book_id: str) -> None:
        self.root = run_directory / "chunk-results" / "v1"
        self.run_id = run_id
        self.book_id = book_id

    def _read_receipts(self) -> list[dict[str, Any]]:
        versions_root = self.root.parent
        _reject_symlinks(versions_root)
        if versions_root.exists():
            if not versions_root.is_dir() or any(
                entry.name != "v1" for entry in versions_root.iterdir()
            ):
                raise ValueError("unknown chunk result version")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("invalid chunk result version directory")
        path = self.root / "receipts.json"
        _reject_symlinks(path)
        if not path.exists():
            return []
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise ValueError("invalid result receipt index")
        try:
            raw = (
                _read_regular_file_posix(path, max_bytes=16 * 1024 * 1024)
                if os.name == "posix"
                else path.read_bytes()
            )
            envelope = json.loads(raw, object_pairs_hook=_unique_json_object)
        except (OSError, ValueError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid result receipt index") from error
        if (
            not isinstance(envelope, dict)
            or set(envelope) != {"schema_version", "records"}
            or type(envelope["schema_version"]) is not int
            or envelope["schema_version"] != 1
            or not isinstance(envelope["records"], list)
        ):
            raise ValueError("invalid result receipt index")
        receipts = envelope["records"]
        versions: dict[str, int] = {}
        prior_ids: dict[str, str] = {}
        keys: set[tuple[str, str]] = set()
        for sequence, receipt in enumerate(receipts, 1):
            if (
                not isinstance(receipt, dict)
                or set(receipt)
                != {"sequence", "task_id", "version", "result_id", "idempotency_key", "sha256"}
                or type(receipt["sequence"]) is not int
                or receipt["sequence"] != sequence
                or type(receipt["version"]) is not int
                or not isinstance(receipt["task_id"], str)
                or re.fullmatch(r"chunk-task:[0-9a-f]{24}", receipt["task_id"]) is None
                or not isinstance(receipt["result_id"], str)
                or re.fullmatch(r"chunk-result:[0-9a-f]{24}", receipt["result_id"]) is None
                or not _valid_receipt_key(receipt["idempotency_key"])
                or not isinstance(receipt["sha256"], str)
                or re.fullmatch(r"[0-9a-f]{64}", receipt["sha256"]) is None
            ):
                raise ValueError("invalid result receipt index")
            task_id = receipt["task_id"]
            key = (task_id, receipt["idempotency_key"])
            if receipt["version"] != versions.get(task_id, 0) + 1 or key in keys:
                raise ValueError("invalid result receipt index")
            record = self._read_record(receipt)
            if record["supersedes_result_id"] != prior_ids.get(task_id):
                raise ValueError("invalid result receipt index")
            versions[task_id] = receipt["version"]
            prior_ids[task_id] = receipt["result_id"]
            keys.add(key)
        return receipts

    def _read_record(self, receipt: dict[str, Any]) -> dict[str, Any]:
        name = f"{receipt['sequence']:08d}"
        directory = self.root / "records" / name
        path = directory / "record.json"
        sidecar = directory / "sha256"
        _reject_symlinks(path)
        _reject_symlinks(sidecar)
        if not path.is_file() or not sidecar.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("invalid chunk result record")
        try:
            raw = (
                _read_regular_file_posix(path, max_bytes=1024 * 1024)
                if os.name == "posix"
                else path.read_bytes()
            )
        except (OSError, ValueError) as error:
            raise ValueError("invalid chunk result record") from error
        actual = hashlib.sha256(raw).hexdigest()
        try:
            sidecar_hash = (
                _read_regular_file_posix(sidecar, max_bytes=64).decode("ascii")
                if os.name == "posix"
                else sidecar.read_text(encoding="ascii")
            )
        except (OSError, UnicodeError, ValueError) as error:
            raise ValueError("invalid chunk result record") from error
        if actual != sidecar_hash or actual != receipt["sha256"]:
            raise ValueError("chunk result hash mismatch")
        try:
            record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("invalid chunk result record") from error
        try:
            canonical = canonical_json(record)
        except (TypeError, UnicodeError, ValueError) as error:
            raise ValueError("invalid chunk result record") from error
        if (
            not isinstance(record, dict)
            or set(record)
            != {
                "sequence",
                "task_id",
                "version",
                "result_id",
                "supersedes_result_id",
                "submission",
            }
            or any(
                record[key] != receipt[key]
                for key in ("sequence", "task_id", "version", "result_id")
            )
            or canonical != raw
        ):
            raise ValueError("invalid chunk result record")
        submission = validate_submission(record["submission"])
        if submission["idempotency_key"] != receipt["idempotency_key"]:
            raise ValueError("invalid result receipt index")
        if (
            submission["book_id"] != self.book_id
            or submission["run_id"] != self.run_id
            or submission["task_id"] != record["task_id"]
            or (
                record["supersedes_result_id"] is not None
                and (
                    not isinstance(record["supersedes_result_id"], str)
                    or re.fullmatch(r"chunk-result:[0-9a-f]{24}", record["supersedes_result_id"])
                    is None
                )
            )
            or record["result_id"]
            != "chunk-result:"
            + digest([self.run_id, record["task_id"], record["version"], submission])[:24]
        ):
            raise ValueError("invalid chunk result identity")
        return record

    def _recover_complete_orphan(
        self,
        receipts: list[dict[str, Any]],
        validate_orphan: Callable[[dict[str, Any], dict[str, dict[str, Any]]], None],
    ) -> list[dict[str, Any]]:
        """Index one complete unreceipted record before accepting a new version."""
        sequence = len(receipts) + 1
        directory = self.root / "records" / f"{sequence:08d}"
        _reject_symlinks(directory)
        if not directory.exists():
            return receipts
        path = directory / "record.json"
        sidecar = directory / "sha256"
        _reject_symlinks(path)
        _reject_symlinks(sidecar)
        if not path.is_file() or not sidecar.is_file() or path.stat().st_size > 1024 * 1024:
            raise ValueError("unrecovered chunk result record")
        try:
            raw = (
                _read_regular_file_posix(path, max_bytes=1024 * 1024)
                if os.name == "posix"
                else path.read_bytes()
            )
        except (OSError, ValueError) as error:
            raise ValueError("unrecovered chunk result record") from error
        try:
            record = json.loads(raw)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("unrecovered chunk result record") from error
        if not isinstance(record, dict) or set(record) != {
            "sequence",
            "task_id",
            "version",
            "result_id",
            "supersedes_result_id",
            "submission",
        }:
            raise ValueError("unrecovered chunk result record")
        submission = validate_submission(record["submission"])
        task_id = submission["task_id"]
        version = 1 + sum(receipt["task_id"] == task_id for receipt in receipts)
        receipt = {
            "sequence": sequence,
            "task_id": task_id,
            "version": version,
            "result_id": record["result_id"],
            "idempotency_key": submission["idempotency_key"],
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        if (
            record["sequence"] != sequence
            or type(record["version"]) is not int
            or record["version"] != version
            or any(
                old["task_id"] == task_id
                and old["idempotency_key"] == submission["idempotency_key"]
                for old in receipts
            )
        ):
            raise ValueError("unrecovered chunk result record")
        recovered_record = self._read_record(receipt)
        previous = next(
            (old["result_id"] for old in reversed(receipts) if old["task_id"] == task_id), None
        )
        if recovered_record["supersedes_result_id"] != previous:
            raise ValueError("unrecovered chunk result record")
        current = {old["task_id"]: self._read_record(old) for old in receipts}
        try:
            validate_orphan(submission, current)
        except ValueError as error:
            raise ValueError("unrecovered chunk result record") from error
        recovered = [*receipts, receipt]
        _atomic_write(
            self.root / "receipts.json",
            canonical_json({"schema_version": 1, "records": recovered}),
        )
        return recovered

    def current_results(self) -> dict[str, dict[str, Any]]:
        """Read and verify the newest accepted version for each task."""
        current: dict[str, dict[str, Any]] = {}
        for receipt in self._read_receipts():
            current[receipt["task_id"]] = self._read_record(receipt)
        return current

    def find_idempotent(self, submission: dict[str, Any]) -> dict[str, Any] | None:
        """Return an already accepted exact retry, including a stale version."""
        for receipt in self._read_receipts():
            if (
                receipt["task_id"] == submission["task_id"]
                and receipt["idempotency_key"] == submission["idempotency_key"]
            ):
                if self._read_record(receipt)["submission"] != submission:
                    raise ValueError("idempotency_conflict")
                return {
                    "task_id": receipt["task_id"],
                    "result_id": receipt["result_id"],
                    "version": receipt["version"],
                }
        return None

    def recover_pending(
        self,
        *,
        validate_orphan: Callable[[dict[str, Any], dict[str, dict[str, Any]]], None],
    ) -> list[dict[str, Any]]:
        """Recover a verified orphan before a new submission reads dependencies."""
        _discard_uncommitted_staging(self.root / "records")
        return self._recover_complete_orphan(self._read_receipts(), validate_orphan)

    def append(
        self,
        submission: dict[str, Any],
        *,
        validate_orphan: Callable[[dict[str, Any], dict[str, dict[str, Any]]], None],
    ) -> dict[str, Any]:
        """Append one accepted version or return an exact idempotent retry."""
        receipts = self.recover_pending(validate_orphan=validate_orphan)
        task_id = submission["task_id"]
        key = submission["idempotency_key"]
        for receipt in receipts:
            if receipt["task_id"] == task_id and receipt["idempotency_key"] == key:
                if self._read_record(receipt)["submission"] != submission:
                    raise ValueError("idempotency_conflict")
                return {
                    "task_id": task_id,
                    "result_id": receipt["result_id"],
                    "version": receipt["version"],
                }
        version = 1 + sum(receipt["task_id"] == task_id for receipt in receipts)
        sequence = len(receipts) + 1
        previous = next(
            (
                receipt["result_id"]
                for receipt in reversed(receipts)
                if receipt["task_id"] == task_id
            ),
            None,
        )
        result_id = (
            "chunk-result:" + digest([submission["run_id"], task_id, version, submission])[:24]
        )
        record = {
            "sequence": sequence,
            "task_id": task_id,
            "version": version,
            "result_id": result_id,
            "supersedes_result_id": previous,
            "submission": submission,
        }
        raw = canonical_json(record)
        receipt = {
            "sequence": sequence,
            "task_id": task_id,
            "version": version,
            "result_id": result_id,
            "idempotency_key": key,
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        records_root = self.root / "records"
        record_directory = records_root / f"{sequence:08d}"
        _reject_symlinks(record_directory)
        if record_directory.exists():
            if self._read_record(receipt) != record:
                raise ValueError("unrecovered chunk result record")
        elif os.name == "posix":
            if not shutil.rmtree.avoids_symlink_attacks:
                raise ValueError("secure chunk result staging cleanup is unavailable")
            records_fd = _open_records_root_posix(records_root)
            staging_name = ".result-" + uuid4().hex
            try:
                os.mkdir(staging_name, mode=0o700, dir_fd=records_fd)
                try:
                    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
                    staging_fd = os.open(staging_name, flags, dir_fd=records_fd)
                    try:
                        _write_staged_file_posix(staging_fd, "record.json", raw)
                        _write_staged_file_posix(
                            staging_fd, "sha256", receipt["sha256"].encode("ascii")
                        )
                        os.fsync(staging_fd)
                        _verify_pinned_parent_posix(records_root, records_fd, flags)
                        os.rename(
                            staging_name,
                            record_directory.name,
                            src_dir_fd=records_fd,
                            dst_dir_fd=records_fd,
                        )
                        installed_fd = os.open(record_directory.name, flags, dir_fd=records_fd)
                        try:
                            staged = os.fstat(staging_fd)
                            installed = os.fstat(installed_fd)
                            if (staged.st_dev, staged.st_ino) != (
                                installed.st_dev,
                                installed.st_ino,
                            ):
                                raise ValueError("result staging changed during installation")
                        finally:
                            os.close(installed_fd)
                        os.fsync(records_fd)
                        _verify_pinned_parent_posix(records_root, records_fd, flags)
                    finally:
                        os.close(staging_fd)
                finally:
                    try:
                        shutil.rmtree(staging_name, dir_fd=records_fd)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(records_fd)
        else:
            records_root.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(prefix=".result-", dir=records_root))
            try:
                _atomic_write(staging / "record.json", raw)
                _atomic_write(staging / "sha256", receipt["sha256"].encode("ascii"))
                staging.replace(record_directory)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)
        if self._read_record(receipt) != record:
            raise ValueError("invalid chunk result record")
        _atomic_write(
            self.root / "receipts.json",
            canonical_json({"schema_version": 1, "records": [*receipts, receipt]}),
        )
        return {"task_id": task_id, "result_id": result_id, "version": version}
