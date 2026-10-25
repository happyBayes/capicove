"""Safe immutable-file workspace for resumable book processing."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from capicove.books.codec import (
    book_structure_from_json,
    book_structure_to_json,
    structure_review_submission_from_json,
    structure_review_submission_to_json,
    structure_review_task_from_json,
    structure_review_task_to_json,
    summary_submission_from_json,
    summary_submission_to_json,
    summary_task_from_json,
    summary_task_to_json,
)
from capicove.books.models import BookStructure, StructureReviewTask
from capicove.books.review import StructureReviewSubmission
from capicove.books.tasks import SummarySubmission, SummaryTask
from capicove.contracts import DocumentParseResult
from capicove.documents import document_parse_result_from_json, document_parse_result_to_json


_WORKSPACE_VERSION = 2
_PREPARED_ARTIFACT_NAMES = frozenset(
    {"representation.json", "structure.json", "tasks.jsonl", "structure-tasks.jsonl"}
)
_PREPARATION_POLICY_NAME = "preparation-policy.json"
_REVIEWED_ARTIFACT_NAMES = frozenset({"reviewed-structure.json", "reviewed-tasks.jsonl"})
_REVIEWED_MANIFEST_NAME = "reviewed-manifest.json"


@dataclass(frozen=True, slots=True)
class SubmissionVersion:
    """Identify one append-only accepted submission version."""

    task_id: str
    version: int
    previous_version: int | None
    submission: SummarySubmission


@dataclass(frozen=True, slots=True)
class StructureReviewVersion:
    """Identify one append-only structure review submission version."""

    task_id: str
    version: int
    previous_version: int | None
    submission: StructureReviewSubmission


class LocalBookWorkspace:
    """Persist confidential book evidence and Agent exchanges locally."""

    def __init__(self, root: Path) -> None:
        self.root = root.expanduser().resolve()

    @classmethod
    def create(cls, root: Path) -> LocalBookWorkspace:
        """Create a new workspace without overwriting existing content."""
        resolved = root.expanduser().resolve()
        if resolved.exists() and any(resolved.iterdir()):
            raise ValueError("book workspace destination is not empty")
        resolved.mkdir(parents=True, exist_ok=True)
        (resolved / "raw").mkdir()
        (resolved / "books").mkdir()
        _atomic_write(
            resolved / "workspace.json",
            _json_bytes({"schema_version": _WORKSPACE_VERSION}),
        )
        workspace = cls(resolved)
        workspace._validate()
        return workspace

    @classmethod
    def open(cls, root: Path) -> LocalBookWorkspace:
        """Open and validate an existing workspace."""
        workspace = cls(root)
        workspace._validate()
        return workspace

    def preserve_source(self, source: Path, *, expected_sha256: str | None = None) -> Path:
        """Copy a regular PDF into content-addressed immutable storage."""
        if source.is_symlink() or not source.is_file():
            raise ValueError("book source must be a regular file")
        raw = source.read_bytes()
        digest = hashlib.sha256(raw).hexdigest()
        if expected_sha256 is not None and digest != expected_sha256:
            raise ValueError("book source changed while it was being prepared")
        target = self.root / "raw" / f"{digest}.pdf"
        _reject_symlinks(target)
        if target.exists():
            if target.is_symlink() or target.read_bytes() != raw:
                raise ValueError("content-addressed source bytes disagree")
            return target
        _atomic_write(target, raw)
        return target

    def has_book(self, book_id: str) -> bool:
        """Return whether one prepared book directory exists."""
        path = self._book_path(book_id) / "manifest.json"
        _reject_symlinks(path)
        return path.is_file()

    def save_prepared(
        self,
        book_id: str,
        result: DocumentParseResult,
        structure: BookStructure,
        tasks: tuple[SummaryTask, ...],
        review_tasks: tuple[StructureReviewTask, ...],
        *,
        evidence_only: bool = False,
    ) -> bool:
        """Atomically persist one immutable prepared book, idempotently."""
        directory = self._book_path(book_id)
        artifacts = _prepared_artifacts(
            result,
            structure,
            tasks,
            review_tasks,
            evidence_only=evidence_only,
        )
        payloads = {
            "manifest.json": _prepared_manifest(
                book_id,
                result,
                review_tasks,
                artifacts,
            ),
            **artifacts,
        }
        if directory.exists():
            manifest = self.read_manifest(book_id)
            policy_path = directory / _PREPARATION_POLICY_NAME
            _reject_symlinks(policy_path)
            if policy_path.exists() is not (_PREPARATION_POLICY_NAME in artifacts):
                raise ValueError("prepared book identity conflicts with durable content")
            for name, payload in artifacts.items():
                path = directory / name
                _reject_symlinks(path)
                if not path.is_file() or path.read_bytes() != payload:
                    raise ValueError("prepared book identity conflicts with durable content")
            if _json_bytes(manifest) != payloads["manifest.json"]:
                raise ValueError("prepared book identity conflicts with durable content")
            return False
        staging = Path(tempfile.mkdtemp(prefix=".book-", dir=self.root / "books"))
        try:
            (staging / "submissions").mkdir()
            for name, payload in payloads.items():
                _atomic_write(staging / name, payload)
            staging.replace(directory)
        finally:
            if staging.exists():
                import shutil

                shutil.rmtree(staging)
        return True

    def read_manifest(self, book_id: str) -> dict[str, object]:
        """Read one closed book manifest."""
        if _read_json(self.root / "workspace.json") != {"schema_version": _WORKSPACE_VERSION}:
            raise ValueError(
                "book workspace manifest is invalid; legacy caches must be regenerated"
            )
        value = _read_json(self._book_path(book_id) / "manifest.json")
        current = {
            "schema_version",
            "book_id",
            "source_sha256",
            "representation_sha256",
            "review_task_count",
            "artifact_sha256",
        }
        if (
            set(value) != current
            or type(value["schema_version"]) is not int
            or value["schema_version"] != 2
            or not _valid_manifest_base(value, book_id)
            or not _valid_artifact_hashes(value["artifact_sha256"])
        ):
            raise ValueError("book manifest is invalid")
        return value

    def is_evidence_only(self, book_id: str) -> bool:
        """Read the opt-in preparation policy; legacy books have no policy file."""
        manifest = self.read_manifest(book_id)
        path = self._book_path(book_id) / "preparation-policy.json"
        _reject_symlinks(path)
        hashes = manifest.get("artifact_sha256")
        if isinstance(hashes, Mapping):
            declared = _PREPARATION_POLICY_NAME in hashes
            if path.exists() is not declared:
                raise ValueError("preparation policy does not match its manifest")
            if not declared:
                return False
            payload = self._read_prepared_artifact(book_id, _PREPARATION_POLICY_NAME)
            policy = _json_object_from_bytes(payload, _PREPARATION_POLICY_NAME)
            if len(payload) > 1024 or policy != {
                "schema_version": 1,
                "evidence_only": True,
            }:
                raise ValueError("invalid preparation policy")
            return True
        if not path.exists():
            return False
        if (
            not path.is_file()
            or path.stat().st_size > 1024
            or _read_json(path) != {"schema_version": 1, "evidence_only": True}
        ):
            raise ValueError("invalid preparation policy")
        return True

    def read_result(self, book_id: str) -> DocumentParseResult:
        """Read immutable parsed document evidence."""
        payload = self._read_prepared_artifact(book_id, "representation.json")
        return document_parse_result_from_json(
            _json_object_from_bytes(payload, "representation.json")
        )

    def read_structure(self, book_id: str, *, original: bool = False) -> BookStructure:
        """Read immutable deterministic book structure."""
        directory = self._book_path(book_id)
        path = directory / "reviewed-structure.json"
        if original or not self.has_reviewed(book_id):
            payload = self._read_prepared_artifact(book_id, "structure.json")
            return book_structure_from_json(_json_object_from_bytes(payload, "structure.json"))
        payload = self._read_reviewed_artifact(book_id, path.name)
        return book_structure_from_json(_json_object_from_bytes(payload, path.name))

    def read_tasks(self, book_id: str) -> tuple[SummaryTask, ...]:
        """Read immutable summary tasks in bottom-up order."""
        directory = self._book_path(book_id)
        path = directory / "reviewed-tasks.jsonl"
        if not self.has_reviewed(book_id):
            payload = self._read_prepared_artifact(book_id, "tasks.jsonl")
            values = _jsonl_from_bytes(payload, "tasks.jsonl")
        else:
            payload = self._read_reviewed_artifact(book_id, path.name)
            values = _jsonl_from_bytes(payload, path.name)
        return tuple(summary_task_from_json(value) for value in values)

    def read_structure_tasks(self, book_id: str) -> tuple[StructureReviewTask, ...]:
        """Read immutable bounded low-confidence structure tasks."""
        payload = self._read_prepared_artifact(book_id, "structure-tasks.jsonl")
        return tuple(
            structure_review_task_from_json(value)
            for value in _jsonl_from_bytes(payload, "structure-tasks.jsonl")
        )

    def _read_prepared_artifact(self, book_id: str, name: str) -> bytes:
        """Read one immutable artifact and verify schema-v2 byte identity."""
        if name not in {*_PREPARED_ARTIFACT_NAMES, _PREPARATION_POLICY_NAME}:
            raise ValueError("unsupported prepared artifact")
        manifest = self.read_manifest(book_id)
        hashes = manifest.get("artifact_sha256")
        if not isinstance(hashes, Mapping):
            raise ValueError("legacy book requires verified preparation migration")
        expected = hashes.get(name)
        if not isinstance(expected, str):
            raise ValueError(f"prepared artifact is not declared: {name}")
        path = self._book_path(book_id) / name
        _reject_symlinks(path)
        try:
            if not path.is_file():
                raise ValueError(f"prepared artifact is missing: {name}")
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"prepared artifact cannot be read: {name}") from error
        if hashlib.sha256(payload).hexdigest() != expected:
            raise ValueError(f"prepared artifact does not match its manifest: {name}")
        return payload

    def _read_reviewed_artifact(self, book_id: str, name: str) -> bytes:
        """Read one reviewed projection and verify it when a manifest is present."""
        if name not in _REVIEWED_ARTIFACT_NAMES:
            raise ValueError("unsupported reviewed artifact")
        directory = self._book_path(book_id)
        path = directory / name
        _reject_symlinks(path)
        try:
            if not path.is_file():
                raise ValueError(f"reviewed artifact is missing: {name}")
            payload = path.read_bytes()
        except OSError as error:
            raise ValueError(f"reviewed artifact cannot be read: {name}") from error
        manifest_path = directory / _REVIEWED_MANIFEST_NAME
        _reject_symlinks(manifest_path)
        if not manifest_path.is_file():
            raise ValueError("reviewed projection requires verified migration")
        manifest = _read_json(manifest_path)
        hashes = _reviewed_artifact_hashes(manifest, book_id)
        if hashlib.sha256(payload).hexdigest() != hashes[name]:
            raise ValueError(f"reviewed artifact does not match its manifest: {name}")
        return payload

    def append_submission(self, book_id: str, submission: SummarySubmission) -> SubmissionVersion:
        """Append a validated submission version without rewriting history."""
        task_key = _task_key(submission.task_id)
        directory = self._book_path(book_id) / "submissions" / task_key
        _reject_symlinks(directory)
        directory.mkdir(exist_ok=True)
        existing = sorted(directory.glob("*.json"))
        version = len(existing) + 1
        previous = None if version == 1 else version - 1
        record = {
            "version": version,
            "previous_version": previous,
            "submission": summary_submission_to_json(submission),
        }
        _atomic_write(directory / f"{version:06d}.json", _json_bytes(record))
        return SubmissionVersion(submission.task_id, version, previous, submission)

    def append_structure_submission(
        self, book_id: str, submission: StructureReviewSubmission
    ) -> StructureReviewVersion:
        """Append one bounded structure decision without rewriting history."""
        directory = (
            self._book_path(book_id)
            / "structure-submissions"
            / submission.task_id.replace(":", "-")
        )
        _reject_symlinks(directory)
        directory.mkdir(parents=True, exist_ok=True)
        existing = sorted(directory.glob("*.json"))
        version = len(existing) + 1
        previous = None if version == 1 else version - 1
        record = {
            "version": version,
            "previous_version": previous,
            "submission": structure_review_submission_to_json(submission),
        }
        _atomic_write(directory / f"{version:06d}.json", _json_bytes(record))
        return StructureReviewVersion(submission.task_id, version, previous, submission)

    def latest_structure_submissions(self, book_id: str) -> dict[str, StructureReviewVersion]:
        """Read the latest decision for every submitted structure task."""
        root = self._book_path(book_id) / "structure-submissions"
        _reject_symlinks(root)
        values: dict[str, StructureReviewVersion] = {}
        if not root.is_dir():
            return values
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            _reject_symlinks(directory)
            versions = sorted(directory.glob("*.json"))
            if not versions:
                continue
            record = _read_version_record(versions[-1])
            submission = structure_review_submission_from_json(record["submission"])
            version, previous = _version_numbers(record)
            values[submission.task_id] = StructureReviewVersion(
                submission.task_id, version, previous, submission
            )
        return values

    def save_reviewed(
        self,
        book_id: str,
        structure: BookStructure,
        tasks: tuple[SummaryTask, ...],
    ) -> None:
        """Create immutable reviewed structure and regenerated summary tasks."""
        directory = self._book_path(book_id)
        artifacts = {
            "reviewed-structure.json": _json_bytes(book_structure_to_json(structure)),
            "reviewed-tasks.jsonl": _jsonl_bytes(summary_task_to_json(task) for task in tasks),
        }
        manifest = _json_bytes(
            {
                "schema_version": 1,
                "book_id": book_id,
                "artifact_sha256": {
                    name: hashlib.sha256(payload).hexdigest()
                    for name, payload in sorted(artifacts.items())
                },
            }
        )
        manifest_path = directory / _REVIEWED_MANIFEST_NAME
        _reject_symlinks(manifest_path)
        if manifest_path.exists():
            if not self.has_reviewed(book_id) or manifest_path.read_bytes() != manifest:
                raise ValueError("reviewed structure already finalized differently")
            for name, payload in artifacts.items():
                if (directory / name).read_bytes() != payload:
                    raise ValueError("reviewed structure already finalized differently")
            return
        for name, payload in artifacts.items():
            path = directory / name
            _reject_symlinks(path)
            if path.exists():
                if not path.is_file():
                    raise ValueError("reviewed structure already finalized differently")
                if path.read_bytes() != payload:
                    raise ValueError("reviewed structure already finalized differently")
                continue
            _atomic_write(path, payload)
        _atomic_write(manifest_path, manifest)

    def has_reviewed(self, book_id: str) -> bool:
        """Expose a reviewed projection only after both outputs are present."""
        directory = self._book_path(book_id)
        paths = tuple(directory / name for name in sorted(_REVIEWED_ARTIFACT_NAMES))
        for path in paths:
            _reject_symlinks(path)
        manifest_path = directory / _REVIEWED_MANIFEST_NAME
        _reject_symlinks(manifest_path)
        present = tuple(path.is_file() for path in paths)
        if manifest_path.exists():
            if not manifest_path.is_file() or not all(present):
                raise ValueError("reviewed projection is incomplete")
            manifest = _read_json(manifest_path)
            hashes = _reviewed_artifact_hashes(manifest, book_id)
            for path in paths:
                if hashlib.sha256(path.read_bytes()).hexdigest() != hashes[path.name]:
                    raise ValueError(f"reviewed artifact does not match its manifest: {path.name}")
            return True
        if all(present):
            raise ValueError("reviewed projection requires verified migration")
        return False

    def has_unverified_reviewed_projection(self, book_id: str) -> bool:
        """Detect a complete legacy pair only for deterministic migration."""
        directory = self._book_path(book_id)
        artifacts = tuple(directory / name for name in sorted(_REVIEWED_ARTIFACT_NAMES))
        manifest = directory / _REVIEWED_MANIFEST_NAME
        for path in (*artifacts, manifest):
            _reject_symlinks(path)
        return not manifest.exists() and all(path.is_file() for path in artifacts)

    def reviewed_projection_is_verified(self, book_id: str) -> bool:
        """Return whether a complete reviewed projection has a valid byte manifest."""
        if not self.has_reviewed(book_id):
            return False
        manifest = self._book_path(book_id) / _REVIEWED_MANIFEST_NAME
        _reject_symlinks(manifest)
        return manifest.is_file()

    def latest_submissions(self, book_id: str) -> dict[str, SubmissionVersion]:
        """Read the latest accepted version for every completed task."""
        root = self._book_path(book_id) / "submissions"
        _reject_symlinks(root)
        values: dict[str, SubmissionVersion] = {}
        if not root.is_dir():
            return values
        for directory in sorted(path for path in root.iterdir() if path.is_dir()):
            _reject_symlinks(directory)
            versions = sorted(directory.glob("*.json"))
            if not versions:
                continue
            record = _read_version_record(versions[-1])
            submission = summary_submission_from_json(record["submission"])
            version, previous = _version_numbers(record)
            values[submission.task_id] = SubmissionVersion(
                submission.task_id, version, previous, submission
            )
        return values

    def _book_path(self, book_id: str) -> Path:
        if re.fullmatch(r"book:[0-9a-f]{24}", book_id) is None:
            raise ValueError("book_id must be canonical")
        path = self.root / "books" / book_id.replace(":", "-")
        _reject_symlinks(path)
        return path

    def _validate(self) -> None:
        if self.root.is_symlink() or not self.root.is_dir():
            raise ValueError("book workspace root must be a real directory")
        for name in ("raw", "books"):
            path = self.root / name
            if path.is_symlink():
                raise ValueError(f"managed root is a symbolic link: {name}")
            if not path.is_dir():
                raise ValueError(f"managed root is missing: {name}")
        manifest = _read_json(self.root / "workspace.json")
        if manifest != {"schema_version": _WORKSPACE_VERSION}:
            raise ValueError("book workspace manifest is invalid")


def _prepared_artifacts(
    result: DocumentParseResult,
    structure: BookStructure,
    tasks: tuple[SummaryTask, ...],
    review_tasks: tuple[StructureReviewTask, ...],
    *,
    evidence_only: bool,
) -> dict[str, bytes]:
    artifacts = {
        "representation.json": _json_bytes(document_parse_result_to_json(result)),
        "structure.json": _json_bytes(book_structure_to_json(structure)),
        "tasks.jsonl": _jsonl_bytes(summary_task_to_json(task) for task in tasks),
        "structure-tasks.jsonl": _jsonl_bytes(
            structure_review_task_to_json(task) for task in review_tasks
        ),
    }
    if evidence_only:
        artifacts[_PREPARATION_POLICY_NAME] = _json_bytes(
            {"schema_version": 1, "evidence_only": True}
        )
    return artifacts


def _prepared_manifest(
    book_id: str,
    result: DocumentParseResult,
    review_tasks: tuple[StructureReviewTask, ...],
    artifacts: Mapping[str, bytes],
) -> bytes:
    return _json_bytes(
        {
            "schema_version": 2,
            "book_id": book_id,
            "source_sha256": result.source_sha256,
            "representation_sha256": result.representation_sha256,
            "review_task_count": len(review_tasks),
            "artifact_sha256": {
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(artifacts.items())
            },
        }
    )


def _valid_artifact_hashes(value: object) -> bool:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) or not isinstance(digest, str) for key, digest in value.items()
    ):
        return False
    names = frozenset(value)
    if names not in (
        _PREPARED_ARTIFACT_NAMES,
        _PREPARED_ARTIFACT_NAMES | {_PREPARATION_POLICY_NAME},
    ):
        return False
    return all(re.fullmatch(r"[0-9a-f]{64}", digest) is not None for digest in value.values())


def _valid_manifest_base(value: Mapping[str, object], book_id: str) -> bool:
    review_count = value.get("review_task_count")
    return bool(
        value.get("book_id") == book_id
        and isinstance(value.get("source_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(value["source_sha256"]))
        and isinstance(value.get("representation_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", str(value["representation_sha256"]))
        and type(review_count) is int
        and review_count >= 0
    )


def _reviewed_artifact_hashes(value: Mapping[str, object], book_id: str) -> dict[str, str]:
    hashes = value.get("artifact_sha256")
    if (
        set(value) != {"schema_version", "book_id", "artifact_sha256"}
        or type(value.get("schema_version")) is not int
        or value.get("schema_version") != 1
        or value.get("book_id") != book_id
        or not isinstance(hashes, Mapping)
        or set(hashes) != _REVIEWED_ARTIFACT_NAMES
        or any(
            not isinstance(name, str)
            or not isinstance(digest, str)
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            for name, digest in hashes.items()
        )
    ):
        raise ValueError("reviewed manifest is invalid")
    return {str(name): str(digest) for name, digest in hashes.items()}


def _reject_symlinks(path: Path) -> None:
    """Reject redirects inside managed paths before reading or creating files."""
    if any(component.is_symlink() for component in (path, *path.parents)):
        raise ValueError("managed book path must not contain a symbolic link")


def _task_key(task_id: str) -> str:
    if re.fullmatch(r"summary:[0-9a-f]{24}", task_id) is None:
        raise ValueError("task_id must be canonical")
    return task_id.replace(":", "-")


def _read_json(path: Path) -> dict[str, object]:
    value, _ = _read_json_snapshot(path)
    return value


def _read_json_snapshot(path: Path) -> tuple[dict[str, object], bytes]:
    _reject_symlinks(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"durable JSON is invalid: {path.name}") from error
    return _json_object_from_bytes(payload, path.name), payload


def _json_object_from_bytes(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"durable JSON is invalid: {name}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"durable JSON must be an object: {name}")
    return value


def _read_jsonl(path: Path) -> tuple[dict[str, object], ...]:
    _reject_symlinks(path)
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise ValueError(f"durable JSONL is invalid: {path.name}") from error
    return _jsonl_from_bytes(payload, path.name)


def _jsonl_from_bytes(payload: bytes, name: str) -> tuple[dict[str, object], ...]:
    values: list[dict[str, object]] = []
    try:
        lines = payload.decode("utf-8").splitlines()
    except UnicodeError as error:
        raise ValueError(f"durable JSONL is invalid: {name}") from error
    for number, line in enumerate(lines, 1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"durable JSONL line {number} is invalid") from error
        if not isinstance(value, dict):
            raise ValueError(f"durable JSONL line {number} must be an object")
        values.append(value)
    return tuple(values)


def _read_version_record(path: Path) -> dict[str, object]:
    record = _read_json(path)
    if set(record) != {"version", "previous_version", "submission"}:
        raise ValueError("submission version record is invalid")
    return record


def _version_numbers(record: dict[str, object]) -> tuple[int, int | None]:
    version = record["version"]
    previous = record["previous_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise ValueError("submission version is invalid")
    if previous is not None and (isinstance(previous, bool) or not isinstance(previous, int)):
        raise ValueError("previous submission version is invalid")
    return version, previous


def _json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _jsonl_bytes(values: Iterable[object]) -> bytes:
    return b"".join(_json_bytes(value) for value in values)


def _atomic_write(path: Path, content: bytes) -> None:
    _reject_symlinks(path)
    if os.name == "posix":
        _atomic_write_posix(path, content)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _atomic_write_posix(path: Path, content: bytes) -> None:
    """Replace a Book file relative to a pinned, no-follow parent."""
    if not path.is_absolute() or not all(
        hasattr(os, name) for name in ("O_DIRECTORY", "O_NOFOLLOW")
    ):
        raise ValueError("secure book file write is unavailable")
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    directory_fd = os.open(path.anchor, directory_flags)
    try:
        for component in path.parts[1:-1]:
            try:
                os.mkdir(component, mode=0o700, dir_fd=directory_fd)
            except FileExistsError:
                pass
            next_fd = os.open(component, directory_flags, dir_fd=directory_fd)
            os.close(directory_fd)
            directory_fd = next_fd
        temporary = f".{path.name}.{uuid4().hex}"
        temporary_fd = os.open(
            temporary,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW,
            0o600,
            dir_fd=directory_fd,
        )
        try:
            with os.fdopen(temporary_fd, "wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path.name, src_dir_fd=directory_fd, dst_dir_fd=directory_fd)
            os.fsync(directory_fd)
            _verify_pinned_parent_posix(path.parent, directory_fd, directory_flags)
        finally:
            try:
                os.unlink(temporary, dir_fd=directory_fd)
            except FileNotFoundError:
                pass
    finally:
        os.close(directory_fd)


def _verify_pinned_parent_posix(parent: Path, pinned_fd: int, flags: int) -> None:
    """Reject success if the selected Book parent no longer names the pinned directory."""
    current_fd = os.open(parent.anchor, flags)
    try:
        for component in parent.parts[1:]:
            next_fd = os.open(component, flags, dir_fd=current_fd)
            os.close(current_fd)
            current_fd = next_fd
        pinned = os.fstat(pinned_fd)
        current = os.fstat(current_fd)
        if (pinned.st_dev, pinned.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("managed book path changed during write")
    finally:
        os.close(current_fd)


__all__ = ["LocalBookWorkspace", "StructureReviewVersion", "SubmissionVersion"]
