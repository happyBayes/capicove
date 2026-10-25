"""Lazy Ray execution adapter for pHash training-set deduplication."""

from __future__ import annotations

import importlib
import math
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from capicove.processing.image.deduplication.perceptual import (
    ImagePerceptualDeduplicator,
    ImageRecord,
    PerceptualDeduplicationResult,
    PerceptualHashProvider,
    PreparedImageRecord,
)
from capicove.processing.image.invocation import (
    ImageDeadlineExceeded,
    check_image_deadline,
    current_image_deadline,
    image_invocation_deadline,
    remaining_image_seconds,
)
from capicove.processing.image.io import account_image_source_dimensions
from capicove.processing.image.models import ImageSource


_CANCELLATION_WAIT_SECONDS = 5.0


class RayImagePerceptualDeduplicator:
    """Distribute pHash preparation for visual training-image deduplication.

    Ray changes only execution coordination. The result uses the same pHash,
    Hamming-distance, connected-component, and stable-winner semantics as the
    local implementation; it is not byte-exact deduplication.
    """

    def __init__(
        self,
        max_hamming_distance: int = 0,
        *,
        hasher: PerceptualHashProvider | None = None,
        timeout_seconds: float = 300.0,
        max_retries: int = 1,
    ) -> None:
        if not math.isfinite(timeout_seconds) or timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive and finite")
        if not isinstance(max_retries, int) or isinstance(max_retries, bool) or max_retries < 0:
            raise ValueError("max_retries must be a non-negative integer")
        self._deduplicator = ImagePerceptualDeduplicator(max_hamming_distance, hasher)
        self._hasher = self._deduplicator.hasher
        self._timeout_seconds = timeout_seconds
        self._max_retries = max_retries

    def group(self, records: tuple[ImageRecord, ...]) -> PerceptualDeduplicationResult:
        """Prepare records on an already-initialized Ray runtime and group locally."""
        record_ids = tuple(record.record_id for record in records)
        if len(set(record_ids)) != len(record_ids):
            raise ValueError("record_id values must be unique")
        check_image_deadline()
        try:
            ray = importlib.import_module("ray")
        except ModuleNotFoundError as error:
            raise RuntimeError("Ray is unavailable; install capicove[image-ray]") from error
        if not bool(ray.is_initialized()):
            raise RuntimeError("Ray must be explicitly initialized before image deduplication")
        materialized = tuple(_materialize_record(record) for record in records)
        for record in materialized:
            for source in record.images:
                account_image_source_dimensions(source)
        remote_function = ray.remote(_prepare_record_remote).options(max_retries=self._max_retries)
        references: list[object] = []
        deadline = current_image_deadline()
        try:
            for record in materialized:
                check_image_deadline()
                references.append(remote_function.remote(record, self._hasher, deadline))
            remote_results = cast(
                list[_RemotePreparation],
                ray.get(
                    references,
                    timeout=remaining_image_seconds(self._timeout_seconds),
                ),
            )
        except Exception as error:
            cleanup_error = _cancel_and_wait(ray, references)
            if cleanup_error is not None:
                raise cleanup_error from error
            if isinstance(error, TimeoutError) or "timeout" in type(error).__name__.casefold():
                raise ImageDeadlineExceeded(
                    "Ray image preparation exceeded its deadline"
                ) from error
            raise
        provider_version = _validate_remote_lineage(remote_results, self._hasher)
        result = self._deduplicator.group_prepared(tuple(item.prepared for item in remote_results))
        return replace(result, provider_version=provider_version)


@dataclass(frozen=True, slots=True)
class _RemotePreparation:
    prepared: PreparedImageRecord
    provider_id: str
    provider_version: str | None
    hash_version: str


def _prepare_record_remote(
    record: ImageRecord,
    hasher: PerceptualHashProvider,
    deadline: object,
) -> _RemotePreparation:
    if deadline is None:
        prepared = ImagePerceptualDeduplicator(hasher=hasher).prepare(record)
    else:
        with image_invocation_deadline(cast(datetime, deadline)):
            prepared = ImagePerceptualDeduplicator(hasher=hasher).prepare(record)
    return _RemotePreparation(
        prepared=prepared,
        provider_id=hasher.provider_id,
        provider_version=hasher.provider_version,
        hash_version=hasher.hash_version,
    )


def _cancel_and_wait(ray: object, references: list[object]) -> RuntimeError | None:
    if not references:
        return None
    failures: list[Exception] = []
    for reference in references:
        try:
            ray.cancel(reference, force=True)  # type: ignore[attr-defined]
        except Exception as error:
            failures.append(error)
    try:
        _, remaining = ray.wait(  # type: ignore[attr-defined]
            references,
            num_returns=len(references),
            timeout=_CANCELLATION_WAIT_SECONDS,
        )
        if remaining:
            failures.append(RuntimeError("Ray tasks remained active after forced cancellation"))
    except Exception as error:
        failures.append(error)
    if failures:
        return RuntimeError(f"Ray cancellation failed for {len(failures)} cleanup operation(s)")
    return None


def _validate_remote_lineage(
    results: list[_RemotePreparation],
    hasher: PerceptualHashProvider,
) -> str | None:
    if not results:
        return hasher.provider_version
    identities = {(item.provider_id, item.hash_version) for item in results}
    if identities != {(hasher.provider_id, hasher.hash_version)}:
        raise RuntimeError("Ray image hash workers returned inconsistent provider lineage")
    versions = {item.provider_version for item in results}
    if len(versions) != 1:
        raise RuntimeError("Ray image hash workers returned inconsistent provider versions")
    return versions.pop()


def _materialize_record(record: ImageRecord) -> ImageRecord:
    images: list[ImageSource] = []
    for source in record.images:
        data = source.data
        if data is None:
            if source.path is None:  # pragma: no cover - ImageSource validates this
                raise ValueError("image source has no readable representation")
            data = Path(source.path).read_bytes()
        images.append(ImageSource(data=data, media_type=source.media_type))
    return ImageRecord(record.record_id, tuple(images), record.text)
