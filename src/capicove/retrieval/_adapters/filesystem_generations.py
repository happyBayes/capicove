"""Descriptor-safe filesystem publication for immutable personal generations."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import tempfile
from collections import defaultdict
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath


try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised only on non-POSIX hosts
    _fcntl = None  # type: ignore[assignment]

from capicove.contracts.retrieval import IndexGeneration
from capicove.retrieval.image_generations import (
    MAX_GENERATION_MANIFEST_BYTES,
    GenerationDirectoryInspection,
    GenerationDirectoryState,
    GenerationPublicationError,
    GenerationSetManifest,
    GenerationValidationError,
    LaneGeneration,
    PayloadFileRef,
    PayloadOwner,
    PayloadValidationKey,
    PayloadValidator,
    PublishedGenerationSet,
    SealedFileIdentity,
    StagedGenerationSetManifest,
    StagedUnitCatalog,
    UnitCatalogManifest,
    ValidatedGenerationSet,
    ValidationReport,
    decode_generation_set_manifest,
    encode_generation_set_manifest,
    encode_validation_report,
    staged_generation_identity_sha256,
    validate_staged_generation_set_manifest,
)


_PRIVATE_DIRECTORY_MODE = 0o700
_PRIVATE_FILE_MODE = 0o600
_MANIFEST_NAME = PurePosixPath("manifest.json")
_SAFE_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_READ_CHUNK_BYTES = 1024 * 1024


class FilesystemGenerationStore:
    """Validate and atomically publish immutable generation directories.

    Publication is intentionally inert: this class has no activation, alias, or
    current-generation method.  Lifecycle state belongs to a separate catalog.
    """

    def __init__(self, root: Path) -> None:
        self.root = Path(os.path.abspath(os.fspath(root)))
        self.staging_root = self.root / ".staging"
        self.lock_root = self.root / ".locks"
        for directory in (self.root, self.staging_root, self.lock_root):
            _ensure_private_directory(directory)

    def create_staging(self, build_id: str) -> Path:
        """Create one unique owner-only staging directory on the publication volume."""
        _require_safe_component("build_id", build_id)
        path = Path(tempfile.mkdtemp(prefix=f"{build_id}-", dir=self.staging_root))
        path.chmod(_PRIVATE_DIRECTORY_MODE)
        _fsync_directory(self.staging_root)
        return path

    def validate(
        self,
        staging_directory: Path,
        staged_manifest: StagedGenerationSetManifest,
        *,
        native_validators: Mapping[PayloadValidationKey, PayloadValidator],
        deadline: datetime,
    ) -> ValidatedGenerationSet:
        """Validate closed staged bytes and write their one final canonical manifest."""
        _check_deadline(deadline, GenerationValidationError)
        if not isinstance(staged_manifest, StagedGenerationSetManifest):
            raise TypeError("staged_manifest must be a StagedGenerationSetManifest")
        try:
            validate_staged_generation_set_manifest(staged_manifest)
        except (TypeError, ValueError) as error:
            raise GenerationValidationError(str(error)) from error
        staging = self._require_staging_directory(staging_directory)
        self._reject_reserved_payload_paths(staged_manifest.payloads)
        expected_paths = {payload.relative_path for payload in staged_manifest.payloads}
        self._require_exact_files(staging, expected_paths, GenerationValidationError)

        initial_seals: dict[PurePosixPath, SealedFileIdentity] = {}
        for payload in staged_manifest.payloads:
            _check_deadline(deadline, GenerationValidationError)
            seal = _seal_file(staging, payload.relative_path, fsync_file=True)
            if seal.size_bytes != payload.size_bytes:
                raise GenerationValidationError(
                    f"payload size changed for role {payload.role}: "
                    f"expected {payload.size_bytes}, observed {seal.size_bytes}"
                )
            if seal.sha256 != payload.sha256:
                raise GenerationValidationError(
                    f"payload checksum mismatch for role {payload.role}"
                )
            initial_seals[payload.relative_path] = seal

        owner_payloads = _owner_payloads(staged_manifest)
        expected_keys = {
            PayloadValidationKey(payload.role, owner_format)
            for owner, owner_format, payload in owner_payloads
        }
        actual_keys = set(native_validators)
        missing_validators = sorted(
            expected_keys - actual_keys,
            key=lambda item: (item.index_format, item.role),
        )
        extra_validators = sorted(
            actual_keys - expected_keys,
            key=lambda item: (item.index_format, item.role),
        )
        if missing_validators or extra_validators:
            raise GenerationValidationError(
                "native validator mapping must exactly match payload role/index format pairs; "
                f"missing={missing_validators}, extra={extra_validators}"
            )

        observations: dict[str, list[ValidationReport]] = defaultdict(list)
        owner_by_id: dict[str, PayloadOwner] = {}
        payload_digests_by_owner: dict[str, set[str]] = defaultdict(set)
        for owner, owner_format, payload in owner_payloads:
            _check_deadline(deadline, GenerationValidationError)
            owner_kind, owner_id = _owner_identity(owner)
            owner_by_id[owner_id] = owner
            payload_digests_by_owner[owner_id].add(payload.sha256)
            key = PayloadValidationKey(payload.role, owner_format)
            validator = native_validators[key]
            report = validator.validate(staging, payload, owner, deadline=deadline)
            if not isinstance(report, ValidationReport):
                raise GenerationValidationError("native validator returned an invalid report")
            if report.validator_id != validator.validator_id:
                raise GenerationValidationError("native validator_id does not match its report")
            if report.owner_kind != owner_kind or report.owner_id != owner_id:
                raise GenerationValidationError("native validation report owner does not match")
            if payload.sha256 not in report.observed_payload_sha256s:
                raise GenerationValidationError(
                    "native validation report omitted the payload digest"
                )
            if not report.passed:
                raise GenerationValidationError(
                    f"native validation failed for payload role {payload.role}"
                )
            observations[owner_id].append(report)

        # Native readers are untrusted with respect to immutability.  Re-open every
        # file through no-follow descriptors before creating any final report.
        for relative_path, initial in initial_seals.items():
            observed = _seal_file(staging, relative_path, fsync_file=True)
            if observed != initial:
                raise GenerationValidationError(
                    f"staged payload changed during native validation: {relative_path}"
                )

        _check_deadline(deadline, GenerationValidationError)
        created_at = datetime.now(UTC)
        staged_digest = staged_generation_identity_sha256(staged_manifest)
        report_payloads: list[PayloadFileRef] = []
        report_refs: dict[str, str] = {}
        for owner_id in sorted(owner_by_id):
            owner = owner_by_id[owner_id]
            owner_kind, _ = _owner_identity(owner)
            owner_observations = observations[owner_id]
            validator_ids = sorted({report.validator_id for report in owner_observations})
            capability_codes = sorted(
                {
                    capability
                    for report in owner_observations
                    for capability in report.capability_codes
                }
            )
            payload_digests = sorted(payload_digests_by_owner[owner_id])
            owner_identity_digest = hashlib.sha256(f"{owner_kind}:{owner_id}".encode()).hexdigest()
            report_ref = f"validation:{owner_identity_digest[:32]}"
            combined_validator_id = ".".join(validator_ids)
            if len(combined_validator_id) > 255:
                combined_validator_id = (
                    "validators:" + hashlib.sha256(combined_validator_id.encode()).hexdigest()
                )
            report = ValidationReport(
                report_id=report_ref,
                owner_kind=owner_kind,
                owner_id=owner_id,
                validator_id=combined_validator_id,
                staged_identity_sha256=staged_digest,
                observed_payload_sha256s=tuple(payload_digests),
                capability_codes=tuple(capability_codes),
                passed=True,
                created_at=created_at,
            )
            report_bytes = encode_validation_report(report)
            if len(report_bytes) > MAX_GENERATION_MANIFEST_BYTES:
                raise GenerationValidationError("canonical validation report exceeds 1 MiB")
            report_path = PurePosixPath("validation") / (owner_identity_digest + ".json")
            _write_new_private_file(staging, report_path, report_bytes)
            report_payloads.append(
                PayloadFileRef(
                    relative_path=report_path,
                    sha256=hashlib.sha256(report_bytes).hexdigest(),
                    size_bytes=len(report_bytes),
                    role=report_ref,
                )
            )
            report_refs[owner_id] = report_ref

        try:
            final_manifest = _finalize_manifest(
                staged_manifest,
                report_payloads=tuple(report_payloads),
                report_refs=report_refs,
                created_at=created_at,
            )
            manifest_bytes = encode_generation_set_manifest(final_manifest)
        except (TypeError, ValueError) as error:
            raise GenerationValidationError(
                f"cannot finalize the immutable generation manifest: {error}"
            ) from error
        _write_new_private_file(staging, _MANIFEST_NAME, manifest_bytes)
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()

        final_paths = {
            *(payload.relative_path for payload in final_manifest.payloads),
            _MANIFEST_NAME,
        }
        self._require_exact_files(staging, final_paths, GenerationValidationError)
        sealed_files = tuple(
            _seal_file(staging, relative_path, fsync_file=True)
            for relative_path in sorted(final_paths, key=PurePosixPath.as_posix)
        )
        _fsync_tree_directories(staging)
        _fsync_directory(self.staging_root)
        return ValidatedGenerationSet(
            staging_directory=staging,
            manifest=final_manifest,
            manifest_bytes=manifest_bytes,
            manifest_sha256=manifest_sha256,
            sealed_files=sealed_files,
        )

    def publish(
        self,
        validated: ValidatedGenerationSet,
        *,
        deadline: datetime,
    ) -> PublishedGenerationSet:
        """Atomically rename a sealed set without replacing an existing directory."""
        _check_deadline(deadline, GenerationPublicationError)
        if not isinstance(validated, ValidatedGenerationSet):
            raise TypeError("validated must be a ValidatedGenerationSet")
        staging = self._require_staging_directory(
            validated.staging_directory,
            error_type=GenerationPublicationError,
        )
        activation_set_id = validated.manifest.activation_set_id
        _require_safe_component("activation_set_id", activation_set_id)
        destination = self.root / activation_set_id

        with self._publication_lock(activation_set_id):
            _check_deadline(deadline, GenerationPublicationError)
            if _path_exists_without_following(destination):
                raise GenerationPublicationError(
                    f"generation {activation_set_id} is already published"
                )
            if (
                staging.stat(follow_symlinks=False).st_dev
                != self.root.stat(follow_symlinks=False).st_dev
            ):
                raise GenerationPublicationError(
                    "staging and publication root must be on the same filesystem"
                )
            expected_seals = {seal.relative_path: seal for seal in validated.sealed_files}
            self._require_exact_files(
                staging,
                set(expected_seals),
                GenerationPublicationError,
            )
            for relative_path, expected in expected_seals.items():
                try:
                    observed = _seal_file(staging, relative_path, fsync_file=True)
                except GenerationValidationError as error:
                    raise GenerationPublicationError(
                        f"sealed file changed before publication: {relative_path}: {error}"
                    ) from error
                if observed != expected:
                    raise GenerationPublicationError(
                        f"sealed file changed before publication: {relative_path}"
                    )
            manifest_seal = expected_seals.get(_MANIFEST_NAME)
            if manifest_seal is None or manifest_seal.sha256 != validated.manifest_sha256:
                raise GenerationPublicationError(
                    "sealed manifest does not match the validation token"
                )
            try:
                manifest_bytes, _ = _read_file(staging, _MANIFEST_NAME)
            except GenerationValidationError as error:
                raise GenerationPublicationError("sealed manifest cannot be re-opened") from error
            if manifest_bytes != validated.manifest_bytes:
                raise GenerationPublicationError("sealed manifest bytes changed")

            try:
                _fsync_tree_directories(staging)
            except (OSError, GenerationValidationError) as error:
                raise GenerationPublicationError(
                    f"cannot fsync the sealed generation tree: {error}"
                ) from error
            try:
                os.rename(staging, destination)
            except OSError as error:
                raise GenerationPublicationError(
                    f"atomic no-overwrite rename failed: {error}"
                ) from error
            try:
                _fsync_directory(self.root)
            except (OSError, GenerationValidationError) as error:
                raise GenerationPublicationError(
                    f"published directory rename was not durably fsynced: {error}"
                ) from error

        return PublishedGenerationSet(
            root=destination,
            manifest=validated.manifest,
            manifest_sha256=validated.manifest_sha256,
        )

    def open_published(
        self,
        activation_set_id: str,
        *,
        deadline: datetime,
        expected_manifest_sha256: str | None = None,
    ) -> PublishedGenerationSet:
        """Re-open and revalidate one immutable generation without activating it."""
        _check_deadline(deadline, GenerationValidationError)
        _require_safe_component("activation_set_id", activation_set_id)
        root = self.root / activation_set_id
        published = self._open_directory(root, deadline=deadline)
        if published.manifest.activation_set_id != activation_set_id:
            raise GenerationValidationError(
                "published directory name does not match manifest activation_set_id"
            )
        if (
            expected_manifest_sha256 is not None
            and published.manifest_sha256 != expected_manifest_sha256
        ):
            raise GenerationValidationError("published manifest checksum does not match expected")
        return published

    def scan_published(
        self,
        *,
        deadline: datetime,
    ) -> tuple[GenerationDirectoryInspection, ...]:
        """Identify valid generations, corrupt entries, and inert renamed orphans."""
        _check_deadline(deadline, GenerationValidationError)
        inspections: list[GenerationDirectoryInspection] = []
        for entry in sorted(self.root.iterdir(), key=lambda item: item.name):
            if entry.name.startswith("."):
                continue
            _check_deadline(deadline, GenerationValidationError)
            try:
                entry_stat = entry.lstat()
                if not stat.S_ISDIR(entry_stat.st_mode) or stat.S_ISLNK(entry_stat.st_mode):
                    raise GenerationValidationError("publication entry is not a regular directory")
                published = self._open_directory(entry, deadline=deadline)
                state = (
                    GenerationDirectoryState.PUBLISHED
                    if entry.name == published.manifest.activation_set_id
                    else GenerationDirectoryState.ORPHANED
                )
                reason = (
                    "manifest and directory name match"
                    if state is GenerationDirectoryState.PUBLISHED
                    else "manifest activation_set_id differs from renamed directory"
                )
                inspections.append(
                    GenerationDirectoryInspection(
                        root=entry,
                        state=state,
                        activation_set_id=published.manifest.activation_set_id,
                        manifest_sha256=published.manifest_sha256,
                        reason=reason,
                    )
                )
            except (OSError, GenerationValidationError, ValueError) as error:
                inspections.append(
                    GenerationDirectoryInspection(
                        root=entry,
                        state=GenerationDirectoryState.CORRUPT,
                        activation_set_id=None,
                        manifest_sha256=None,
                        reason=str(error),
                    )
                )
        return tuple(inspections)

    def _open_directory(self, root: Path, *, deadline: datetime) -> PublishedGenerationSet:
        if not _path_exists_without_following(root):
            raise GenerationValidationError(f"published generation does not exist: {root.name}")
        _require_private_directory(root, GenerationValidationError)
        manifest_bytes, manifest_seal = _read_file(
            root,
            _MANIFEST_NAME,
            maximum_bytes=1_048_576,
        )
        try:
            manifest = decode_generation_set_manifest(manifest_bytes)
        except (TypeError, ValueError) as error:
            raise GenerationValidationError(f"invalid published manifest: {error}") from error
        expected_paths = {
            *(payload.relative_path for payload in manifest.payloads),
            _MANIFEST_NAME,
        }
        self._require_exact_files(root, expected_paths, GenerationValidationError)
        for payload in manifest.payloads:
            _check_deadline(deadline, GenerationValidationError)
            observed = _seal_file(root, payload.relative_path)
            if observed.size_bytes != payload.size_bytes:
                raise GenerationValidationError(
                    f"published payload size mismatch for role {payload.role}"
                )
            if observed.sha256 != payload.sha256:
                raise GenerationValidationError(
                    f"published payload checksum mismatch for role {payload.role}"
                )
        return PublishedGenerationSet(
            root=root,
            manifest=manifest,
            manifest_sha256=manifest_seal.sha256,
        )

    def _require_staging_directory(
        self,
        staging_directory: Path,
        *,
        error_type: type[GenerationValidationError | GenerationPublicationError] = (
            GenerationValidationError
        ),
    ) -> Path:
        staging = Path(os.path.abspath(os.fspath(staging_directory)))
        if staging.parent != self.staging_root:
            raise error_type("staging directory must be a direct child of the private staging root")
        _require_private_directory(staging, error_type)
        return staging

    def _require_exact_files(
        self,
        root: Path,
        expected: set[PurePosixPath],
        error_type: type[GenerationValidationError | GenerationPublicationError],
    ) -> None:
        observed = _inventory_private_tree(root, error_type)
        missing = sorted(expected - observed, key=PurePosixPath.as_posix)
        extra = sorted(observed - expected, key=PurePosixPath.as_posix)
        if missing or extra:
            raise error_type(f"generation file set has missing={missing}, extra={extra}")

    def _reject_reserved_payload_paths(self, payloads: tuple[PayloadFileRef, ...]) -> None:
        for payload in payloads:
            if (
                payload.relative_path == _MANIFEST_NAME
                or payload.relative_path.parts[0] == "validation"
            ):
                raise GenerationValidationError(
                    "staged payload paths must not use manifest.json or the validation namespace"
                )

    @contextmanager
    def _publication_lock(self, activation_set_id: str) -> Iterator[None]:
        if _fcntl is None:
            raise GenerationPublicationError(
                "POSIX flock support is required for no-overwrite publication"
            )
        lock_name = hashlib.sha256(activation_set_id.encode()).hexdigest() + ".lock"
        lock_path = self.lock_root / lock_name
        flags = os.O_RDWR | os.O_CREAT
        if hasattr(os, "O_CLOEXEC"):
            flags |= os.O_CLOEXEC
        try:
            file_descriptor = os.open(lock_path, flags, _PRIVATE_FILE_MODE)
        except OSError as error:
            raise GenerationPublicationError(
                f"cannot open the publication lock: {error}"
            ) from error
        try:
            os.fchmod(file_descriptor, _PRIVATE_FILE_MODE)
            try:
                _fcntl.flock(file_descriptor, _fcntl.LOCK_EX)
            except OSError as error:
                raise GenerationPublicationError(
                    f"cannot acquire the publication lock: {error}"
                ) from error
            yield
        finally:
            try:
                _fcntl.flock(file_descriptor, _fcntl.LOCK_UN)
            except OSError:
                pass
            os.close(file_descriptor)


def _finalize_manifest(
    staged: StagedGenerationSetManifest,
    *,
    report_payloads: tuple[PayloadFileRef, ...],
    report_refs: Mapping[str, str],
    created_at: datetime,
) -> GenerationSetManifest:
    unit_catalog = UnitCatalogManifest(
        catalog_id=staged.unit_catalog.catalog_id,
        unit_count=staged.unit_catalog.unit_count,
        unit_manifest_sha256=staged.unit_catalog.unit_manifest_sha256,
        catalog_format=staged.unit_catalog.catalog_format,
        payload_ref=staged.unit_catalog.payload_ref,
        validation_report_ref=report_refs[staged.unit_catalog.catalog_id],
    )
    lanes = tuple(
        LaneGeneration(
            lane=lane.lane,
            generation=IndexGeneration(
                generation_id=lane.generation_id,
                activation_set_id=lane.activation_set_id,
                workspace_id=staged.workspace_id,
                corpus_ref=staged.corpus_ref,
                target_type=staged.target_type,
                unit_manifest_sha256=lane.unit_manifest_sha256,
                embedding_spaces=lane.embedding_spaces,
                hash_spaces=lane.hash_spaces,
                retriever_id=lane.retriever_id,
                index_format=lane.index_format,
                index_parameters_ref=lane.index_parameters_ref,
                shard_refs=lane.payload_refs,
                calibration_profile_ref=f"{lane.lane.value}-personal-lite-v1",
                build_processing_fingerprint=staged.processing_fingerprint,
                created_at=created_at,
                validation_report_ref=report_refs[lane.generation_id],
            ),
            target_bundle_manifest_sha256=lane.target_bundle_manifest_sha256,
        )
        for lane in staged.lanes
    )
    return GenerationSetManifest(
        activation_set_id=staged.activation_set_id,
        workspace_id=staged.workspace_id,
        corpus_ref=staged.corpus_ref,
        retrieval_profile_ref=staged.retrieval_profile_ref,
        target_type=staged.target_type,
        unit_manifest_sha256=staged.unit_manifest_sha256,
        frozen_unit_manifest_proof_sha256=staged.frozen_unit_manifest_proof_sha256,
        target_representation_manifest_sha256=staged.target_representation_manifest_sha256,
        processing_fingerprint=staged.processing_fingerprint,
        unit_catalog=unit_catalog,
        lanes=lanes,
        payloads=staged.payloads + report_payloads,
        created_at=created_at,
    )


def _owner_payloads(
    manifest: StagedGenerationSetManifest,
) -> list[tuple[PayloadOwner, str, PayloadFileRef]]:
    payload_by_role = {payload.role: payload for payload in manifest.payloads}
    result: list[tuple[PayloadOwner, str, PayloadFileRef]] = [
        (
            manifest.unit_catalog,
            manifest.unit_catalog.catalog_format,
            payload_by_role[manifest.unit_catalog.payload_ref],
        )
    ]
    for lane in manifest.lanes:
        for role in (lane.index_parameters_ref.payload_ref, *lane.payload_refs):
            result.append((lane, lane.index_format, payload_by_role[role]))
    return result


def _owner_identity(owner: PayloadOwner) -> tuple[str, str]:
    if isinstance(owner, StagedUnitCatalog):
        return "catalog", owner.catalog_id
    return "lane", owner.generation_id


def _inventory_private_tree(
    root: Path,
    error_type: type[GenerationValidationError | GenerationPublicationError],
) -> set[PurePosixPath]:
    _require_private_directory(root, error_type)
    result: set[PurePosixPath] = set()
    try:
        for current, directory_names, file_names in os.walk(root, followlinks=False):
            current_path = Path(current)
            for directory_name in directory_names:
                directory = current_path / directory_name
                info = directory.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
                    raise error_type(f"symlink or non-directory rejected: {directory}")
                _require_owner_only_mode(directory, info.st_mode, error_type)
            for file_name in file_names:
                file_path = current_path / file_name
                info = file_path.lstat()
                if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
                    raise error_type(f"symlink or non-regular payload rejected: {file_path}")
                _require_owner_only_mode(file_path, info.st_mode, error_type)
                relative = file_path.relative_to(root)
                result.add(PurePosixPath(*relative.parts))
    except OSError as error:
        raise error_type(f"cannot inspect generation tree: {error}") from error
    return result


def _seal_file(
    root: Path,
    relative_path: PurePosixPath,
    *,
    fsync_file: bool = False,
) -> SealedFileIdentity:
    _, seal = _read_file(root, relative_path, fsync_file=fsync_file, retain_bytes=False)
    return seal


def _read_file(
    root: Path,
    relative_path: PurePosixPath,
    *,
    maximum_bytes: int | None = None,
    fsync_file: bool = False,
    retain_bytes: bool = True,
) -> tuple[bytes, SealedFileIdentity]:
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise GenerationValidationError(
            "descriptor-relative O_NOFOLLOW and O_DIRECTORY support is required"
        )
    root_descriptor = os.open(root, os.O_RDONLY | directory_flag | nofollow)
    current_descriptor = root_descriptor
    owned_descriptors: list[int] = []
    try:
        for component in relative_path.parts[:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=current_descriptor,
            )
            owned_descriptors.append(next_descriptor)
            current_descriptor = next_descriptor
        file_descriptor = os.open(
            relative_path.parts[-1],
            os.O_RDONLY | nofollow,
            dir_fd=current_descriptor,
        )
        try:
            info = os.fstat(file_descriptor)
            if not stat.S_ISREG(info.st_mode):
                raise GenerationValidationError(f"payload is not a regular file: {relative_path}")
            _require_owner_only_mode(relative_path, info.st_mode, GenerationValidationError)
            if maximum_bytes is not None and info.st_size > maximum_bytes:
                raise GenerationValidationError(f"file is too large: {relative_path}")
            digest = hashlib.sha256()
            chunks: list[bytes] = []
            observed_size = 0
            while True:
                chunk = os.read(file_descriptor, _READ_CHUNK_BYTES)
                if not chunk:
                    break
                observed_size += len(chunk)
                if maximum_bytes is not None and observed_size > maximum_bytes:
                    raise GenerationValidationError(f"file is too large: {relative_path}")
                digest.update(chunk)
                if retain_bytes:
                    chunks.append(chunk)
            after = os.fstat(file_descriptor)
            if (
                after.st_dev != info.st_dev
                or after.st_ino != info.st_ino
                or after.st_size != info.st_size
                or after.st_mtime_ns != info.st_mtime_ns
            ):
                raise GenerationValidationError(f"file changed while being read: {relative_path}")
            if fsync_file:
                os.fsync(file_descriptor)
            seal = SealedFileIdentity(
                relative_path=relative_path,
                size_bytes=observed_size,
                sha256=digest.hexdigest(),
                device_id=after.st_dev,
                inode=after.st_ino,
                modified_time_ns=after.st_mtime_ns,
                mode=stat.S_IMODE(after.st_mode),
            )
            return b"".join(chunks), seal
        finally:
            os.close(file_descriptor)
    except OSError as error:
        raise GenerationValidationError(
            f"cannot safely open payload {relative_path}: {error}"
        ) from error
    finally:
        for descriptor in reversed(owned_descriptors):
            os.close(descriptor)
        os.close(root_descriptor)


def _write_new_private_file(root: Path, relative_path: PurePosixPath, content: bytes) -> None:
    parent = root.joinpath(*relative_path.parts[:-1])
    _ensure_private_directory(parent)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not nofollow or not directory_flag:
        raise GenerationValidationError(
            "descriptor-relative O_NOFOLLOW and O_DIRECTORY support is required"
        )
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= nofollow
    root_descriptor = os.open(root, os.O_RDONLY | directory_flag | nofollow)
    current_descriptor = root_descriptor
    owned_descriptors: list[int] = []
    try:
        for component in relative_path.parts[:-1]:
            next_descriptor = os.open(
                component,
                os.O_RDONLY | directory_flag | nofollow,
                dir_fd=current_descriptor,
            )
            owned_descriptors.append(next_descriptor)
            current_descriptor = next_descriptor
        descriptor = os.open(
            relative_path.parts[-1],
            flags,
            _PRIVATE_FILE_MODE,
            dir_fd=current_descriptor,
        )
        try:
            os.fchmod(descriptor, _PRIVATE_FILE_MODE)
            view = memoryview(content)
            offset = 0
            while offset < len(view):
                written = os.write(descriptor, view[offset:])
                if written <= 0:
                    raise GenerationValidationError(f"short write for {relative_path}")
                offset += written
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        os.fsync(current_descriptor)
    except OSError as error:
        raise GenerationValidationError(f"cannot write {relative_path}: {error}") from error
    finally:
        for directory_descriptor in reversed(owned_descriptors):
            os.close(directory_descriptor)
        os.close(root_descriptor)


def _ensure_private_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=_PRIVATE_DIRECTORY_MODE)
    info = path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise GenerationValidationError(f"directory must not be a symlink: {path}")
    path.chmod(_PRIVATE_DIRECTORY_MODE)


def _require_private_directory(
    path: Path,
    error_type: type[GenerationValidationError | GenerationPublicationError],
) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise error_type(f"generation directory is unavailable: {error}") from error
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise error_type(f"generation directory must be a real directory: {path}")
    _require_owner_only_mode(path, info.st_mode, error_type)


def _require_owner_only_mode(
    path: Path | PurePosixPath,
    mode: int,
    error_type: type[GenerationValidationError | GenerationPublicationError],
) -> None:
    if stat.S_IMODE(mode) & 0o077:
        raise error_type(f"owner-only permission required: {path}")


def _fsync_tree_directories(root: Path) -> None:
    directories = [Path(current) for current, _, _ in os.walk(root, followlinks=False)]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    directory_flag = getattr(os, "O_DIRECTORY", 0)
    if not directory_flag:
        raise GenerationValidationError("directory fsync support is required")
    descriptor = os.open(path, os.O_RDONLY | directory_flag)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_exists_without_following(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _require_safe_component(name: str, value: str) -> None:
    if not isinstance(value, str) or not _SAFE_COMPONENT_PATTERN.fullmatch(value):
        raise ValueError(f"{name} must be a safe filesystem component")


def _check_deadline(
    deadline: datetime,
    error_type: type[GenerationValidationError | GenerationPublicationError],
) -> None:
    if (
        not isinstance(deadline, datetime)
        or deadline.tzinfo is None
        or deadline.utcoffset() is None
    ):
        raise error_type("deadline must be timezone-aware")
    if datetime.now(UTC) >= deadline.astimezone(UTC):
        raise error_type("deadline exceeded")
