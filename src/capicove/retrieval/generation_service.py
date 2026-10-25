"""Provider-neutral orchestration for durable text-index generations."""

from __future__ import annotations

from capicove.retrieval.generations import (
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationState,
    TextIndexGenerationTransitionError,
)
from capicove.retrieval.ports import (
    TextIndexGenerationCatalog,
    TextIndexGenerationLifecycle,
)


class TextIndexGenerationService:
    """Coordinate a derived provider index with Capicove's authoritative catalog."""

    def __init__(
        self,
        lifecycle: TextIndexGenerationLifecycle,
        catalog: TextIndexGenerationCatalog,
    ) -> None:
        self._lifecycle = lifecycle
        self._catalog = catalog

    def create(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexGenerationRecord:
        """Register a manifest and create its provider physical index."""
        self._require_retriever(manifest.retriever_id)
        record = self._catalog.register(manifest)
        if record.physical_index_ref is not None:
            return record
        if record.state is not TextIndexGenerationState.BUILDING:
            raise TextIndexGenerationTransitionError(
                "physical index creation requires a building generation"
            )
        physical_index_ref = self._lifecycle.create(manifest)
        return self._catalog.record_physical_index(
            manifest.generation_id,
            physical_index_ref,
        )

    def validate(self, generation_id: str) -> TextIndexGenerationRecord:
        """Validate one building provider index and durably close its build state."""
        record = self._require_record(generation_id)
        self._require_retriever(record.manifest.retriever_id)
        if record.validation is not None:
            return record
        if (
            record.state is not TextIndexGenerationState.BUILDING
            or record.physical_index_ref is None
        ):
            raise TextIndexGenerationTransitionError(
                "validation requires a building generation with a physical index"
            )
        report = self._lifecycle.validate(record.manifest)
        return self._catalog.record_validation(report)

    def activate(self, generation_id: str) -> TextIndexGenerationRecord:
        """Switch the provider alias, then commit the exact returned active pin."""
        record = self._require_record(generation_id)
        self._require_retriever(record.manifest.retriever_id)
        if record.state is TextIndexGenerationState.ACTIVE:
            return record
        if record.state not in {
            TextIndexGenerationState.READY,
            TextIndexGenerationState.SUPERSEDED,
        }:
            raise TextIndexGenerationTransitionError(
                f"cannot activate generation in {record.state.value} state"
            )
        if record.validation is None:
            raise TextIndexGenerationTransitionError("activation requires validation evidence")
        pinned = self._lifecycle.activate(record.manifest, record.validation)
        return self._catalog.record_activation(pinned)

    def reconcile_active(
        self,
        *,
        workspace_ref: str,
        retriever_id: str,
    ) -> TextIndexGenerationRecord:
        """Recover a completed provider alias switch using its exact resolved pin."""
        self._require_retriever(retriever_id)
        pinned = self._lifecycle.resolve_active()
        record = self._catalog.get(pinned.generation_id)
        if record is None:
            raise TextIndexGenerationTransitionError(
                "resolved active generation does not exist in the catalog"
            )
        if (
            record.manifest.workspace_ref != workspace_ref
            or record.manifest.retriever_id != retriever_id
        ):
            raise TextIndexGenerationTransitionError(
                "resolved active generation belongs to a different catalog stream"
            )
        if record.state not in {
            TextIndexGenerationState.READY,
            TextIndexGenerationState.ACTIVE,
            TextIndexGenerationState.SUPERSEDED,
        }:
            raise TextIndexGenerationTransitionError("resolved active generation is not validated")
        if (
            pinned.manifest_sha256 != record.manifest.manifest_sha256
            or pinned.physical_index_ref != record.physical_index_ref
        ):
            raise TextIndexGenerationTransitionError(
                "resolved active generation pin does not match catalog evidence"
            )
        return self._catalog.record_activation(pinned)

    def _require_record(self, generation_id: str) -> TextIndexGenerationRecord:
        record = self._catalog.get(generation_id)
        if record is None:
            raise TextIndexGenerationTransitionError("generation does not exist")
        return record

    def _require_retriever(self, retriever_id: str) -> None:
        if retriever_id != self._lifecycle.provider_id:
            raise TextIndexGenerationTransitionError(
                "generation retriever does not match the configured lifecycle provider"
            )


__all__ = ["TextIndexGenerationService"]
