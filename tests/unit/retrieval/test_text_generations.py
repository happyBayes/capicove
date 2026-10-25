"""Tests for provider-neutral immutable text index generations."""

from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta, timezone
from typing import Any, cast

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
    TextIndexGenerationEvent,
    TextIndexGenerationEventType,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationState,
    TextIndexValidationReport,
)


def embedding_space() -> EmbeddingSpaceRef:
    """Return one canonical text embedding space."""
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


def manifest() -> TextIndexGenerationManifest:
    """Return one complete immutable generation manifest."""
    return TextIndexGenerationManifest(
        generation_id="generation-42",
        workspace_ref="workspace-1",
        corpus_refs=("corpus-z", "corpus-a"),
        searchable_unit_manifest_sha256="a" * 64,
        content_sha256="b" * 64,
        mapping_sha256="c" * 64,
        embedding_spaces=(embedding_space(),),
        retriever_id="capicove.retriever.elasticsearch",
        index_format_version="elasticsearch-text-v1",
        build_processing_fingerprint=ProcessingFingerprint("d" * 64),
        expected_unit_count=2,
    )


def test_manifest_normalizes_identity_order_and_has_canonical_digest() -> None:
    value = manifest()

    assert value.corpus_refs == ("corpus-a", "corpus-z")
    assert (
        value.manifest_sha256
        == (
            "178366d62dfd5aa1a485324cf84d345080aa14f15a287209206abbf0a3ae24ab"  # pragma: allowlist secret
        )
    )
    assert replace(value, content_sha256="e" * 64).manifest_sha256 != value.manifest_sha256


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    [
        ("searchable_unit_manifest_sha256", "A" * 64),
        ("content_sha256", "not-a-digest"),
        ("mapping_sha256", ""),
    ],
)
def test_manifest_rejects_noncanonical_digests(
    field_name: str,
    field_value: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        replace(manifest(), **cast(Any, {field_name: field_value}))


def test_manifest_rejects_duplicate_spaces_and_non_text_spaces() -> None:
    space = embedding_space()
    with pytest.raises(ValueError, match="duplicate embedding spaces"):
        replace(manifest(), embedding_spaces=(space, space))

    image_space = replace(
        space,
        target_modalities=frozenset({TargetType.IMAGE}),
    )
    with pytest.raises(ValueError, match="text queries and text chunks"):
        replace(manifest(), embedding_spaces=(image_space,))


@pytest.mark.parametrize("expected_unit_count", [-1, True, 1.5])
def test_manifest_rejects_invalid_expected_unit_counts(
    expected_unit_count: object,
) -> None:
    with pytest.raises(ValueError, match="expected_unit_count"):
        replace(manifest(), expected_unit_count=expected_unit_count)  # type: ignore[arg-type]


def test_validation_report_represents_ready_and_corrupt_results() -> None:
    value = manifest()
    ready = TextIndexValidationReport(
        generation_id=value.generation_id,
        manifest_sha256=value.manifest_sha256,
        physical_index_ref="capicove-text-g-79091da8c3ddbe5b32099e4c",
        mapping_sha256=value.mapping_sha256,
        expected_unit_count=2,
        observed_unit_count=2,
        issue_codes=(),
    )
    corrupt = replace(
        ready,
        observed_unit_count=1,
        issue_codes=("document_count_mismatch",),
    )

    assert ready.valid is True
    assert corrupt.valid is False


def test_validation_report_rejects_inconsistent_issue_state() -> None:
    value = manifest()
    with pytest.raises(ValueError, match="document_count_mismatch"):
        TextIndexValidationReport(
            generation_id=value.generation_id,
            manifest_sha256=value.manifest_sha256,
            physical_index_ref="capicove-text-g-79091da8c3ddbe5b32099e4c",
            mapping_sha256=value.mapping_sha256,
            expected_unit_count=2,
            observed_unit_count=1,
            issue_codes=(),
        )


def test_pinned_generation_requires_canonical_manifest_and_physical_identity() -> None:
    value = manifest()
    pinned = PinnedTextIndexGeneration(
        generation_id=value.generation_id,
        manifest_sha256=value.manifest_sha256,
        physical_index_ref="capicove-text-g-79091da8c3ddbe5b32099e4c",
    )

    assert pinned.generation_id == "generation-42"
    with pytest.raises(ValueError, match="manifest_sha256"):
        replace(pinned, manifest_sha256="not-a-digest")
    with pytest.raises(ValueError, match="physical_index_ref"):
        replace(pinned, physical_index_ref=" ")


def test_generation_record_enforces_lifecycle_evidence() -> None:
    value = manifest()
    report = TextIndexValidationReport(
        generation_id=value.generation_id,
        manifest_sha256=value.manifest_sha256,
        physical_index_ref="capicove-text-g-79091da8c3ddbe5b32099e4c",
        mapping_sha256=value.mapping_sha256,
        expected_unit_count=value.expected_unit_count,
        observed_unit_count=value.expected_unit_count,
        issue_codes=(),
    )

    building = TextIndexGenerationRecord(
        manifest=value,
        state=TextIndexGenerationState.BUILDING,
        revision=0,
    )
    ready = TextIndexGenerationRecord(
        manifest=value,
        state=TextIndexGenerationState.READY,
        physical_index_ref=report.physical_index_ref,
        validation=report,
        revision=2,
    )

    assert building.physical_index_ref is None
    assert ready.validation is report
    with pytest.raises(ValueError, match="validated generation"):
        replace(ready, validation=None)
    with pytest.raises(ValueError, match="corrupt generation"):
        replace(ready, state=TextIndexGenerationState.CORRUPT)


def test_generation_record_accepts_corrupt_validation_only_in_corrupt_state() -> None:
    value = manifest()
    report = TextIndexValidationReport(
        generation_id=value.generation_id,
        manifest_sha256=value.manifest_sha256,
        physical_index_ref="capicove-text-g-79091da8c3ddbe5b32099e4c",
        mapping_sha256="e" * 64,
        expected_unit_count=value.expected_unit_count,
        observed_unit_count=value.expected_unit_count,
        issue_codes=("mapping_mismatch",),
    )

    record = TextIndexGenerationRecord(
        manifest=value,
        state=TextIndexGenerationState.CORRUPT,
        physical_index_ref=report.physical_index_ref,
        validation=report,
        revision=2,
    )

    assert record.validation is report
    with pytest.raises(ValueError, match="ready generation"):
        replace(record, state=TextIndexGenerationState.READY)


def test_generation_event_requires_positive_sequence_and_utc_timestamp() -> None:
    record = TextIndexGenerationRecord(
        manifest=manifest(),
        state=TextIndexGenerationState.BUILDING,
        revision=0,
    )
    event = TextIndexGenerationEvent(
        sequence=1,
        event_type=TextIndexGenerationEventType.REGISTERED,
        record=record,
        recorded_at=datetime(2026, 8, 26, tzinfo=UTC),
    )

    assert event.record is record
    with pytest.raises(ValueError, match="sequence"):
        replace(event, sequence=0)
    with pytest.raises(ValueError, match="UTC"):
        replace(event, recorded_at=datetime(2026, 8, 26))
    with pytest.raises(ValueError, match="UTC"):
        replace(event, recorded_at=datetime(2026, 8, 26, tzinfo=timezone(timedelta(hours=1))))
