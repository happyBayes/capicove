"""Complete video generation-set activation, leasing, and runtime composition."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from typing import Generic, Protocol, TypeVar

from capicove.retrieval._adapters.numpy_visual import (
    NumpyExactVisualConfiguration,
    NumpyExactVisualRetriever,
)
from capicove.retrieval._adapters.sqlite_video_generations import (
    SqliteVideoGenerationSetCatalog,
)
from capicove.retrieval._adapters.sqlite_video_units import (
    SqliteVideoUnitCatalogReader,
)
from capicove.retrieval.ports import TextIndexGenerationLifecycle
from capicove.retrieval.video_models import (
    PinnedVideoGenerationSet,
    VideoGenerationSetManifest,
    VideoGenerationSetRecord,
    VideoGenerationSetState,
    VideoGenerationTransitionError,
)
from capicove.retrieval.video_ports import (
    VideoGenerationSetCatalog,
    VisualCandidateRetriever,
    VisualIndexGenerationLifecycle,
    VisualIndexWriter,
)


GenerationSetValidator = Callable[[VideoGenerationSetRecord], None]


class ManagedVisualBackend(
    VisualCandidateRetriever, VisualIndexGenerationLifecycle, VisualIndexWriter, Protocol
):
    """Startup-owned visual roles with explicit resource cleanup."""

    def close(self) -> None: ...


BackendT = TypeVar("BackendT", bound=ManagedVisualBackend)


class VideoTextMemberValidator:
    """Validate physical text members without independently changing text aliases."""

    def __init__(
        self, *, asr: TextIndexGenerationLifecycle, ocr: TextIndexGenerationLifecycle | None = None
    ) -> None:
        self._asr = asr
        self._ocr = ocr

    def __call__(self, record: VideoGenerationSetRecord) -> None:
        for manifest, pin, lifecycle in (
            (record.manifest.asr_text, record.pin.asr_text, self._asr),
            (record.manifest.ocr_text, record.pin.ocr_text, self._ocr),
        ):
            if manifest is None and pin is None:
                continue
            if manifest is None or pin is None or lifecycle is None:
                raise VideoGenerationTransitionError("text member validator missing")
            report = lifecycle.validate(manifest)
            if (
                not report.valid
                or report.manifest_sha256 != pin.manifest_sha256
                or report.generation_id != pin.generation_id
                or report.physical_index_ref != pin.physical_index_ref
                or report.mapping_sha256 != manifest.mapping_sha256
                or report.expected_unit_count != manifest.expected_unit_count
            ):
                raise VideoGenerationTransitionError("physical text member failed validation")


class VideoGenerationSetService:
    """Coordinate validated complete-set registration, activation, and leasing."""

    def __init__(
        self,
        catalog: VideoGenerationSetCatalog,
        *,
        validator: GenerationSetValidator | None = None,
    ) -> None:
        self._catalog = catalog
        self._validator = validator

    def register(
        self,
        manifest: VideoGenerationSetManifest,
        pin: PinnedVideoGenerationSet,
    ) -> VideoGenerationSetRecord:
        """Register one complete set whose members have already validated."""
        record = VideoGenerationSetRecord(
            manifest=manifest,
            pin=pin,
            state=VideoGenerationSetState.READY,
            revision=0,
        )
        return self._catalog.register(record)

    def activate(self, set_id: str) -> VideoGenerationSetRecord:
        """Revalidate all configured physical members before atomic alias change."""
        record = self._catalog.get(set_id)
        if record is None:
            raise VideoGenerationTransitionError("video generation set does not exist")
        if record.state is VideoGenerationSetState.ACTIVE:
            if self._validator is not None:
                self._validator(record)
            return record
        if record.state not in {
            VideoGenerationSetState.READY,
            VideoGenerationSetState.SUPERSEDED,
        }:
            raise VideoGenerationTransitionError(
                f"cannot activate generation set in {record.state.value} state"
            )
        if self._validator is None:
            raise VideoGenerationTransitionError(
                "generation-set activation requires a physical member validator"
            )
        self._validator(record)
        return self._catalog.record_activation(record.pin)

    def resolve_active(self, *, workspace_ref: str, profile_ref: str) -> PinnedVideoGenerationSet:
        """Return the exact complete active pin or fail closed."""
        record = self._catalog.active(workspace_ref=workspace_ref, profile_ref=profile_ref)
        if record is None:
            raise VideoGenerationTransitionError(
                "no active video generation set for workspace and profile"
            )
        return record.pin

    @contextmanager
    def lease_active(
        self, *, workspace_ref: str, profile_ref: str
    ) -> Iterator[PinnedVideoGenerationSet]:
        """Retain one complete active pin until the query exits for any reason."""
        lease_id, pin = self._catalog.acquire_active_lease(
            workspace_ref=workspace_ref, profile_ref=profile_ref
        )
        try:
            yield pin
        finally:
            self._catalog.release_lease(lease_id)

    def can_evict(self, set_id: str) -> bool:
        """Return whether a superseded set has no live query lease."""
        operation = getattr(self._catalog, "can_evict", None)
        if operation is None:
            raise VideoGenerationTransitionError(
                "configured catalog does not expose eviction safety"
            )
        return bool(operation(set_id))


@dataclass(slots=True)
class VideoRuntimeComponents(Generic[BackendT]):
    """Hold reopened provider-neutral components for VideoSearchService composition."""

    generation_sets: VideoGenerationSetService
    visual_backend: BackendT
    _catalogs: OrderedDict[tuple[str, str], SqliteVideoUnitCatalogReader] = field(
        default_factory=OrderedDict, init=False, repr=False
    )
    _catalog_lock: Lock = field(default_factory=Lock, init=False, repr=False)

    def catalog_resolver(self, pin: PinnedVideoGenerationSet) -> SqliteVideoUnitCatalogReader:
        """Open and verify the immutable unit catalog named by a leased pin."""
        key = (pin.unit_catalog_ref, pin.unit_manifest_sha256)
        with self._catalog_lock:
            reader = self._catalogs.get(key)
            if reader is None:
                reader = SqliteVideoUnitCatalogReader(Path(pin.unit_catalog_ref))
                if reader.unit_manifest_sha256 != pin.unit_manifest_sha256:
                    raise VideoGenerationTransitionError(
                        "unit catalog does not match the leased generation set"
                    )
                self._catalogs[key] = reader
            reader.verify_unchanged()
            self._catalogs.move_to_end(key)
            while len(self._catalogs) > 2:
                self._catalogs.popitem(last=False)
            return reader

    def close(self) -> None:
        """Release opened provider resources."""
        self.visual_backend.close()
        with self._catalog_lock:
            self._catalogs.clear()


def open_exact_video_runtime(
    *,
    generation_database_path: Path,
    index_root: Path,
    validate_text_members: GenerationSetValidator | None = None,
) -> VideoRuntimeComponents[NumpyExactVisualRetriever]:
    """Reopen persisted catalogs and exact indexes for runtime composition."""
    backend = NumpyExactVisualRetriever(NumpyExactVisualConfiguration(index_root))
    return open_video_runtime(
        generation_database_path=generation_database_path,
        visual_backend=backend,
        validate_text_members=validate_text_members,
    )


def open_video_runtime(
    *,
    generation_database_path: Path,
    visual_backend: BackendT,
    validate_text_members: GenerationSetValidator | None = None,
) -> VideoRuntimeComponents[BackendT]:
    """Compose an explicitly provisioned visual provider without importing plugins."""
    catalog = SqliteVideoGenerationSetCatalog(generation_database_path)
    backend = visual_backend

    def validate(record: VideoGenerationSetRecord) -> None:
        if validate_text_members is None:
            raise VideoGenerationTransitionError(
                "activation requires an explicit text member validator"
            )
        validate_text_members(record)
        unit_reader = SqliteVideoUnitCatalogReader(Path(record.pin.unit_catalog_ref))
        if unit_reader.unit_manifest_sha256 != record.manifest.unit_manifest_sha256:
            raise VideoGenerationTransitionError(
                "unit catalog manifest does not match the generation set"
            )
        for manifest, pin in (
            (record.manifest.keyframe_visual, record.pin.keyframe_visual),
            (record.manifest.segment_visual, record.pin.segment_visual),
        ):
            report = backend.validate(manifest)
            if not report.valid:
                raise VideoGenerationTransitionError(
                    "visual index member is corrupt or unvalidated"
                )
            if backend.activate(manifest, report) != pin:
                raise VideoGenerationTransitionError(
                    "visual index pin does not match the generation set"
                )

    return VideoRuntimeComponents(
        generation_sets=VideoGenerationSetService(catalog, validator=validate),
        visual_backend=backend,
    )


ExactVideoRuntimeComponents = VideoRuntimeComponents[NumpyExactVisualRetriever]


__all__ = [
    "ExactVideoRuntimeComponents",
    "GenerationSetValidator",
    "VideoGenerationSetService",
    "VideoRuntimeComponents",
    "VideoTextMemberValidator",
    "open_exact_video_runtime",
    "open_video_runtime",
]
