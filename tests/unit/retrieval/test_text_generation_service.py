"""Tests for provider-neutral text index generation orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import pytest

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval import (
    PinnedTextIndexGeneration,
    TextIndexGenerationManifest,
    TextIndexGenerationService,
    TextIndexGenerationState,
    TextIndexGenerationTransitionError,
    TextIndexValidationReport,
)
from capicove.retrieval._adapters.sqlite_generations import (
    SqliteTextIndexGenerationCatalog,
)


_RETRIEVER_ID = "capicove.retriever.elasticsearch"


def _embedding_space() -> EmbeddingSpaceRef:
    return EmbeddingSpaceRef(
        space_id="text-dense-v1",
        provider_id="embedding-provider",
        model_revision="revision-1",
        preprocessing_version="text-v1",
        dimension=3,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.COSINE,
        query_modalities=frozenset({QueryModality.TEXT}),
        target_modalities=frozenset({TargetType.TEXT_CHUNK}),
    )


def _manifest(
    generation_id: str,
    *,
    workspace_ref: str = "workspace-1",
    content_sha256: str = "b" * 64,
) -> TextIndexGenerationManifest:
    return TextIndexGenerationManifest(
        generation_id=generation_id,
        workspace_ref=workspace_ref,
        corpus_refs=("corpus-1",),
        searchable_unit_manifest_sha256="a" * 64,
        content_sha256=content_sha256,
        mapping_sha256="c" * 64,
        embedding_spaces=(_embedding_space(),),
        retriever_id=_RETRIEVER_ID,
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=2,
    )


@dataclass
class StatefulGenerationLifecycle:
    """Model the observable provider state used by the orchestration service."""

    provider_id: str = _RETRIEVER_ID
    invalid_generation_ids: set[str] = field(default_factory=set)
    created_generation_ids: list[str] = field(default_factory=list)
    active_pin: PinnedTextIndexGeneration | None = None

    def mapping_sha256(
        self,
        embedding_spaces: tuple[EmbeddingSpaceRef, ...],
    ) -> str:
        del embedding_spaces
        return "c" * 64

    def create(self, manifest: TextIndexGenerationManifest) -> str:
        self.created_generation_ids.append(manifest.generation_id)
        return f"capicove-text-g-{manifest.generation_id}"

    def validate(
        self,
        manifest: TextIndexGenerationManifest,
    ) -> TextIndexValidationReport:
        invalid = manifest.generation_id in self.invalid_generation_ids
        return TextIndexValidationReport(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref=f"capicove-text-g-{manifest.generation_id}",
            mapping_sha256=manifest.mapping_sha256 if not invalid else "e" * 64,
            expected_unit_count=manifest.expected_unit_count,
            observed_unit_count=manifest.expected_unit_count,
            issue_codes=() if not invalid else ("mapping_mismatch",),
        )

    def activate(
        self,
        manifest: TextIndexGenerationManifest,
        validation: TextIndexValidationReport,
    ) -> PinnedTextIndexGeneration:
        assert validation.valid
        self.active_pin = PinnedTextIndexGeneration(
            generation_id=manifest.generation_id,
            manifest_sha256=manifest.manifest_sha256,
            physical_index_ref=validation.physical_index_ref,
        )
        return self.active_pin

    def resolve_active(self) -> PinnedTextIndexGeneration:
        if self.active_pin is None:
            raise RuntimeError("provider alias is unavailable")
        return self.active_pin


class FailOnceActivationCatalog:
    """Delegate real catalog behavior except for one interrupted activation commit."""

    def __init__(self, catalog: SqliteTextIndexGenerationCatalog) -> None:
        self._catalog = catalog
        self.fail_next_activation = True

    def register(self, manifest: TextIndexGenerationManifest):  # type: ignore[no-untyped-def]
        return self._catalog.register(manifest)

    def get(self, generation_id: str):  # type: ignore[no-untyped-def]
        return self._catalog.get(generation_id)

    def record_physical_index(  # type: ignore[no-untyped-def]
        self,
        generation_id: str,
        physical_index_ref: str,
    ):
        return self._catalog.record_physical_index(generation_id, physical_index_ref)

    def record_validation(self, report: TextIndexValidationReport):  # type: ignore[no-untyped-def]
        return self._catalog.record_validation(report)

    def record_activation(self, pinned: PinnedTextIndexGeneration):  # type: ignore[no-untyped-def]
        if self.fail_next_activation:
            self.fail_next_activation = False
            raise RuntimeError("simulated catalog interruption")
        return self._catalog.record_activation(pinned)

    def active(self, *, workspace_ref: str, retriever_id: str):  # type: ignore[no-untyped-def]
        return self._catalog.active(
            workspace_ref=workspace_ref,
            retriever_id=retriever_id,
        )

    def history(self, generation_id: str):  # type: ignore[no-untyped-def]
        return self._catalog.history(generation_id)


def _service(
    tmp_path: Path,
) -> tuple[
    TextIndexGenerationService,
    SqliteTextIndexGenerationCatalog,
    StatefulGenerationLifecycle,
]:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    lifecycle = StatefulGenerationLifecycle()
    return TextIndexGenerationService(lifecycle, catalog), catalog, lifecycle


def _make_ready(
    service: TextIndexGenerationService,
    manifest: TextIndexGenerationManifest,
) -> None:
    service.create(manifest)
    service.validate(manifest.generation_id)


def test_create_registers_before_provider_creation_and_retry_is_idempotent(
    tmp_path: Path,
) -> None:
    service, catalog, lifecycle = _service(tmp_path)
    manifest = _manifest("generation-1")

    created = service.create(manifest)
    retried = service.create(manifest)

    assert retried == created
    assert created.state is TextIndexGenerationState.BUILDING
    assert created.physical_index_ref == "capicove-text-g-generation-1"
    assert lifecycle.created_generation_ids == [manifest.generation_id]
    assert len(catalog.history(manifest.generation_id)) == 2


def test_validate_and_activate_persist_the_exact_provider_pin(tmp_path: Path) -> None:
    service, catalog, lifecycle = _service(tmp_path)
    manifest = _manifest("generation-1")
    service.create(manifest)

    ready = service.validate(manifest.generation_id)
    active = service.activate(manifest.generation_id)

    assert ready.state is TextIndexGenerationState.READY
    assert active.state is TextIndexGenerationState.ACTIVE
    assert lifecycle.resolve_active() == PinnedTextIndexGeneration(
        generation_id=manifest.generation_id,
        manifest_sha256=manifest.manifest_sha256,
        physical_index_ref=active.physical_index_ref,  # type: ignore[arg-type]
    )
    assert (
        catalog.active(
            workspace_ref=manifest.workspace_ref,
            retriever_id=manifest.retriever_id,
        )
        == active
    )


def test_failed_validation_closes_generation_and_prevents_provider_activation(
    tmp_path: Path,
) -> None:
    service, catalog, lifecycle = _service(tmp_path)
    manifest = _manifest("generation-corrupt")
    lifecycle.invalid_generation_ids.add(manifest.generation_id)
    service.create(manifest)

    corrupt = service.validate(manifest.generation_id)

    assert corrupt.state is TextIndexGenerationState.CORRUPT
    with pytest.raises(TextIndexGenerationTransitionError, match="cannot activate"):
        service.activate(manifest.generation_id)
    assert lifecycle.active_pin is None
    assert (
        catalog.active(
            workspace_ref=manifest.workspace_ref,
            retriever_id=manifest.retriever_id,
        )
        is None
    )


def test_activation_can_roll_back_to_a_superseded_generation(tmp_path: Path) -> None:
    service, catalog, _ = _service(tmp_path)
    first = _manifest("generation-1", content_sha256="1" * 64)
    second = _manifest("generation-2", content_sha256="2" * 64)
    _make_ready(service, first)
    _make_ready(service, second)
    service.activate(first.generation_id)
    service.activate(second.generation_id)

    rolled_back = service.activate(first.generation_id)

    assert rolled_back.state is TextIndexGenerationState.ACTIVE
    assert catalog.get(second.generation_id).state is TextIndexGenerationState.SUPERSEDED  # type: ignore[union-attr]


def test_reconcile_recovers_alias_switch_interrupted_before_catalog_commit(
    tmp_path: Path,
) -> None:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    interrupted_catalog = FailOnceActivationCatalog(catalog)
    lifecycle = StatefulGenerationLifecycle()
    service = TextIndexGenerationService(lifecycle, interrupted_catalog)
    manifest = _manifest("generation-1")
    _make_ready(service, manifest)

    with pytest.raises(RuntimeError, match="catalog interruption"):
        service.activate(manifest.generation_id)

    assert lifecycle.active_pin is not None
    assert catalog.get(manifest.generation_id).state is TextIndexGenerationState.READY  # type: ignore[union-attr]

    recovered = service.reconcile_active(
        workspace_ref=manifest.workspace_ref,
        retriever_id=manifest.retriever_id,
    )

    assert recovered.state is TextIndexGenerationState.ACTIVE


@pytest.mark.parametrize("case", ["unknown", "workspace", "physical", "corrupt"])
def test_reconcile_rejects_any_resolved_pin_without_exact_valid_catalog_evidence(
    tmp_path: Path,
    case: str,
) -> None:
    service, catalog, lifecycle = _service(tmp_path)
    manifest = _manifest("generation-1")
    _make_ready(service, manifest)
    pin = PinnedTextIndexGeneration(
        generation_id=manifest.generation_id,
        manifest_sha256=manifest.manifest_sha256,
        physical_index_ref=f"capicove-text-g-{manifest.generation_id}",
    )
    workspace_ref = manifest.workspace_ref
    if case == "unknown":
        pin = replace(pin, generation_id="unknown-generation")
    elif case == "workspace":
        workspace_ref = "different-workspace"
    elif case == "physical":
        pin = replace(pin, physical_index_ref="different-index")
    else:
        corrupt = _manifest("generation-corrupt")
        lifecycle.invalid_generation_ids.add(corrupt.generation_id)
        service.create(corrupt)
        service.validate(corrupt.generation_id)
        pin = PinnedTextIndexGeneration(
            generation_id=corrupt.generation_id,
            manifest_sha256=corrupt.manifest_sha256,
            physical_index_ref=f"capicove-text-g-{corrupt.generation_id}",
        )
    lifecycle.active_pin = pin

    with pytest.raises(TextIndexGenerationTransitionError):
        service.reconcile_active(
            workspace_ref=workspace_ref,
            retriever_id=manifest.retriever_id,
        )

    assert (
        catalog.active(
            workspace_ref=manifest.workspace_ref,
            retriever_id=manifest.retriever_id,
        )
        is None
    )
