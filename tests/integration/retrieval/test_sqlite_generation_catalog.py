"""Integration tests for the authoritative SQLite text-generation catalog."""

from __future__ import annotations

from dataclasses import replace
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
    TextIndexGenerationConflictError,
    TextIndexGenerationEventType,
    TextIndexGenerationManifest,
    TextIndexGenerationState,
    TextIndexGenerationTransitionError,
    TextIndexValidationReport,
)
from capicove.retrieval._adapters.sqlite_generations import (
    SqliteTextIndexGenerationCatalog,
)


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
        retriever_id="capicove.retriever.elasticsearch",
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=2,
    )


def _physical_ref(manifest: TextIndexGenerationManifest) -> str:
    return f"capicove-text-g-{manifest.generation_id}"


def _report(
    manifest: TextIndexGenerationManifest,
    *,
    valid: bool = True,
) -> TextIndexValidationReport:
    return TextIndexValidationReport(
        generation_id=manifest.generation_id,
        manifest_sha256=manifest.manifest_sha256,
        physical_index_ref=_physical_ref(manifest),
        mapping_sha256=manifest.mapping_sha256 if valid else "e" * 64,
        expected_unit_count=manifest.expected_unit_count,
        observed_unit_count=manifest.expected_unit_count,
        issue_codes=() if valid else ("mapping_mismatch",),
    )


def _ready(
    catalog: SqliteTextIndexGenerationCatalog,
    manifest: TextIndexGenerationManifest,
) -> None:
    catalog.register(manifest)
    catalog.record_physical_index(manifest.generation_id, _physical_ref(manifest))
    catalog.record_validation(_report(manifest))


def _pin(manifest: TextIndexGenerationManifest) -> PinnedTextIndexGeneration:
    return PinnedTextIndexGeneration(
        generation_id=manifest.generation_id,
        manifest_sha256=manifest.manifest_sha256,
        physical_index_ref=_physical_ref(manifest),
    )


def test_catalog_persists_projection_and_append_only_history(tmp_path: Path) -> None:
    database = tmp_path / "retrieval.sqlite3"
    manifest = _manifest("generation-1")
    catalog = SqliteTextIndexGenerationCatalog(database)

    registered = catalog.register(manifest)
    created = catalog.record_physical_index(manifest.generation_id, _physical_ref(manifest))
    ready = catalog.record_validation(_report(manifest))

    reopened = SqliteTextIndexGenerationCatalog(database)
    assert registered.revision == 0
    assert created.revision == 1
    assert reopened.get(manifest.generation_id) == ready
    assert ready.state is TextIndexGenerationState.READY
    assert [event.event_type for event in reopened.history(manifest.generation_id)] == [
        TextIndexGenerationEventType.REGISTERED,
        TextIndexGenerationEventType.PHYSICAL_INDEX_CREATED,
        TextIndexGenerationEventType.VALIDATION_SUCCEEDED,
    ]
    assert [event.record.revision for event in reopened.history(manifest.generation_id)] == [
        0,
        1,
        2,
    ]


def test_registration_is_idempotent_only_for_the_same_manifest(tmp_path: Path) -> None:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    manifest = _manifest("generation-1")

    first = catalog.register(manifest)
    second = catalog.register(manifest)

    assert second == first
    assert len(catalog.history(manifest.generation_id)) == 1
    with pytest.raises(TextIndexGenerationConflictError, match="different manifest"):
        catalog.register(replace(manifest, content_sha256="f" * 64))


def test_failed_validation_becomes_corrupt_and_cannot_activate(tmp_path: Path) -> None:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    manifest = _manifest("generation-corrupt")
    catalog.register(manifest)
    catalog.record_physical_index(manifest.generation_id, _physical_ref(manifest))

    corrupt = catalog.record_validation(_report(manifest, valid=False))

    assert corrupt.state is TextIndexGenerationState.CORRUPT
    assert catalog.history(manifest.generation_id)[-1].event_type is (
        TextIndexGenerationEventType.VALIDATION_FAILED
    )
    with pytest.raises(TextIndexGenerationTransitionError, match="cannot activate"):
        catalog.record_activation(_pin(manifest))


def test_activation_supersedes_and_can_roll_back_without_losing_history(
    tmp_path: Path,
) -> None:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    first = _manifest("generation-1", content_sha256="1" * 64)
    second = _manifest("generation-2", content_sha256="2" * 64)
    _ready(catalog, first)
    _ready(catalog, second)

    catalog.record_activation(_pin(first))
    active_second = catalog.record_activation(_pin(second))

    assert active_second.state is TextIndexGenerationState.ACTIVE
    assert catalog.get(first.generation_id).state is TextIndexGenerationState.SUPERSEDED  # type: ignore[union-attr]
    assert (
        catalog.active(
            workspace_ref=first.workspace_ref,
            retriever_id=first.retriever_id,
        )
        == active_second
    )

    active_first = catalog.record_activation(_pin(first))

    assert active_first.state is TextIndexGenerationState.ACTIVE
    assert catalog.get(second.generation_id).state is TextIndexGenerationState.SUPERSEDED  # type: ignore[union-attr]
    assert [event.event_type for event in catalog.history(first.generation_id)] == [
        TextIndexGenerationEventType.REGISTERED,
        TextIndexGenerationEventType.PHYSICAL_INDEX_CREATED,
        TextIndexGenerationEventType.VALIDATION_SUCCEEDED,
        TextIndexGenerationEventType.ACTIVATED,
        TextIndexGenerationEventType.SUPERSEDED,
        TextIndexGenerationEventType.ACTIVATED,
    ]


def test_active_generation_is_scoped_by_workspace_and_retriever(tmp_path: Path) -> None:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    first = _manifest("generation-1", workspace_ref="workspace-1")
    second = _manifest("generation-2", workspace_ref="workspace-2")
    _ready(catalog, first)
    _ready(catalog, second)

    active_first = catalog.record_activation(_pin(first))
    active_second = catalog.record_activation(_pin(second))

    assert (
        catalog.active(
            workspace_ref="workspace-1",
            retriever_id=first.retriever_id,
        )
        == active_first
    )
    assert (
        catalog.active(
            workspace_ref="workspace-2",
            retriever_id=second.retriever_id,
        )
        == active_second
    )


def test_catalog_rejects_mismatched_and_stale_transitions(tmp_path: Path) -> None:
    catalog = SqliteTextIndexGenerationCatalog(tmp_path / "retrieval.sqlite3")
    manifest = _manifest("generation-1")
    catalog.register(manifest)
    catalog.record_physical_index(manifest.generation_id, _physical_ref(manifest))

    with pytest.raises(TextIndexGenerationConflictError, match="physical index"):
        catalog.record_physical_index(manifest.generation_id, "different-index")
    with pytest.raises(TextIndexGenerationConflictError, match="manifest"):
        catalog.record_validation(replace(_report(manifest), manifest_sha256="f" * 64))
    with pytest.raises(TextIndexGenerationTransitionError, match="does not exist"):
        catalog.record_physical_index("unknown-generation", "unknown-index")


def test_catalog_requires_an_absolute_regular_database_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        SqliteTextIndexGenerationCatalog(Path("retrieval.sqlite3"))

    target = tmp_path / "target.sqlite3"
    target.touch()
    link = tmp_path / "linked.sqlite3"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        SqliteTextIndexGenerationCatalog(link)
