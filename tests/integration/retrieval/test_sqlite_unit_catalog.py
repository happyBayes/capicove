"""Integration tests for the immutable personal SQLite unit catalog."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest

from capicove.contracts.retrieval import (
    DistanceMetric,
    EmbeddingSpaceRef,
    IndexGeneration,
    IndexParametersRef,
    ProcessingFingerprint,
    QueryModality,
    SearchableUnit,
    SearchFilters,
    TargetType,
    VectorNormalization,
)
from capicove.retrieval._adapters import sqlite_unit_catalog
from capicove.retrieval._adapters.personal_numpy import (
    NumpyExactIndexReader,
    NumpyExactIndexWriter,
)
from capicove.retrieval._adapters.sqlite_unit_catalog import (
    AllOrdinals,
    ContentCatalogMatch,
    IndexedUnitRecord,
    SparseOrdinals,
    SqliteUnitCatalog,
    SqliteUnitCatalogDescriptor,
    SqliteUnitCatalogWriter,
)
from capicove.retrieval.search_models import (
    ContentCandidateProvenance,
    QueryVector,
    RetrievalLane,
)


FUTURE = datetime.now(UTC) + timedelta(hours=1)
PAST = datetime.now(UTC) - timedelta(seconds=1)


def _unit(index: int, *, source_hash: str | None = None) -> SearchableUnit:
    digest = source_hash or f"{index + 1:064x}"
    return SearchableUnit(
        unit_id=f"unit-{index}",
        unit_type=TargetType.IMAGE,
        source_hash=digest,
        artifact_ref=f"artifact-{index % 2}",
        evidence_ref=f"evidence-{index}",
        representation_refs=(f"representation-{index}",),
        processing_run_ref="processing-run-1",
        processing_version="image-v1",
        processing_fingerprint=ProcessingFingerprint("a" * 64),
    )


def _rows(count: int) -> tuple[IndexedUnitRecord, ...]:
    return tuple(
        IndexedUnitRecord(
            ordinal=index,
            unit=_unit(index),
            source_sha256=f"{index + 1:064x}",
            media_type="image/png" if index % 2 == 0 else "image/jpeg",
        )
        for index in range(count)
    )


def _build(
    rows: tuple[IndexedUnitRecord, ...],
    path: Path,
) -> SqliteUnitCatalogDescriptor:
    return SqliteUnitCatalogWriter(max_units=10_000).build(
        rows,
        path,
        activation_set_id="activation-1",
        corpus_ref="corpus-1",
        deadline=FUTURE,
    )


def test_catalog_build_uses_private_unpublished_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[bool, int]] = []
    original = SqliteUnitCatalogWriter._write_database

    def inspect_staging(path: Path, *args: Any, **kwargs: Any) -> None:
        observed.append((path.exists(), path.parent.stat().st_mode & 0o777))
        original(path, *args, **kwargs)

    monkeypatch.setattr(SqliteUnitCatalogWriter, "_write_database", staticmethod(inspect_staging))

    _build(_rows(1), tmp_path / "units.sqlite3")

    assert observed == [(False, 0o700)]


def test_catalog_reopens_read_only_filters_and_emits_exact_candidates(tmp_path: Path) -> None:
    rows = _rows(4)
    manifest = _build(rows, tmp_path / "units.sqlite3")

    with SqliteUnitCatalog.open_readonly(manifest, deadline=FUTURE) as catalog:
        assert catalog.is_query_only
        assert catalog.unit_count == 4
        assert catalog.activation_set_id == "activation-1"
        assert catalog.unit_manifest_sha256 == manifest.unit_manifest_sha256
        assert catalog.lookup_content_hash(
            rows[3].source_sha256,
            SearchFilters(),
            limit=20,
            deadline=FUTURE,
        ) == (ContentCatalogMatch("unit-3", 3),)
        candidates = catalog.lookup_content_candidates(
            rows[3].source_sha256,
            SearchFilters(),
            limit=20,
            deadline=FUTURE,
        )
        assert len(candidates) == 1
        assert candidates[0].lane is RetrievalLane.CONTENT
        assert candidates[0].exact_content
        assert isinstance(candidates[0].provenance, ContentCandidateProvenance)
        assert candidates[0].provenance.activation_set_id == "activation-1"
        assert candidates[0].provenance.unit_catalog_id == manifest.catalog_id

        all_ordinals = catalog.select_ordinals(SearchFilters(), deadline=FUTURE)
        assert isinstance(all_ordinals, AllOrdinals)
        assert all_ordinals.unit_count == 4
        assert all_ordinals.ordinals is None
        assert len(all_ordinals.filter_sha256) == 64
        selection = catalog.select_ordinals(
            SearchFilters(
                corpus_refs=frozenset({"corpus-1"}),
                artifact_refs=frozenset({"artifact-0"}),
                media_types=frozenset({"image/png"}),
            ),
            deadline=FUTURE,
        )
        assert isinstance(selection, SparseOrdinals)
        assert selection.ordinals == (0, 2)
        empty_selection = catalog.select_ordinals(
            SearchFilters(corpus_refs=frozenset({"other-corpus"})),
            deadline=FUTURE,
        )
        assert isinstance(empty_selection, SparseOrdinals)
        assert empty_selection.ordinals == ()
        assert [
            unit.unit_id for unit in catalog.get_many(("unit-2", "unit-0"), deadline=FUTURE)
        ] == ["unit-2", "unit-0"]
        assert catalog.unit_ids_for_ordinals((3, 1), deadline=FUTURE) == (
            "unit-3",
            "unit-1",
        )

        with pytest.raises(sqlite3.OperationalError):
            catalog._connection.execute(
                "UPDATE units SET media_type = 'image/gif' WHERE ordinal = 0"
            )

    assert not Path(f"{manifest.path}-wal").exists()
    assert not Path(f"{manifest.path}-shm").exists()
    with sqlite3.connect(f"file:{manifest.path}?mode=ro", uri=True) as connection:
        assert connection.execute("PRAGMA integrity_check").fetchone() == ("ok",)
        assert connection.execute("PRAGMA journal_mode").fetchone() != ("wal",)


def test_catalog_descriptor_safe_snapshot_fallback_is_read_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _rows(2)
    descriptor = _build(rows, tmp_path / "fallback.sqlite3")
    monkeypatch.setattr(sqlite_unit_catalog, "_opened_descriptor_uri", lambda _opened: None)

    with SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE) as catalog:
        assert catalog.is_query_only
        assert catalog.lookup_content_hash(
            rows[0].source_sha256,
            SearchFilters(),
            1,
            FUTURE,
        ) == (ContentCatalogMatch("unit-0", 0),)


def test_catalog_serves_serialized_queries_from_lane_worker_threads(tmp_path: Path) -> None:
    manifest = _build(_rows(4), tmp_path / "units.sqlite3")

    with SqliteUnitCatalog.open_readonly(manifest, deadline=FUTURE) as catalog:
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = [
                executor.submit(
                    catalog.unit_ids_for_ordinals,
                    (index,),
                    deadline=FUTURE,
                )
                for index in range(3)
            ]

        assert [future.result() for future in futures] == [
            ("unit-0",),
            ("unit-1",),
            ("unit-2",),
        ]


def test_catalog_validates_uniqueness_contiguity_hash_and_capacity_before_write(
    tmp_path: Path,
) -> None:
    writer = SqliteUnitCatalogWriter(max_units=10_000)
    destination = tmp_path / "invalid.sqlite3"

    with pytest.raises(ValueError, match="10,000"):
        writer.build(
            _rows(10_001),
            destination,
            activation_set_id="activation-1",
            corpus_ref="corpus-1",
            deadline=FUTURE,
        )
    assert not destination.exists()
    first, second = _rows(2)
    with pytest.raises(ValueError, match="contiguous"):
        writer.build(
            (first, replace(second, ordinal=2)),
            destination,
            activation_set_id="activation-1",
            corpus_ref="corpus-1",
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="source_sha256"):
        writer.build(
            (replace(first, source_sha256="f" * 64),),
            destination,
            activation_set_id="activation-1",
            corpus_ref="corpus-1",
            deadline=FUTURE,
        )
    with pytest.raises(ValueError, match="unit_id"):
        writer.build(
            (first, replace(second, unit=replace(second.unit, unit_id=first.unit.unit_id))),
            destination,
            activation_set_id="activation-1",
            corpus_ref="corpus-1",
            deadline=FUTURE,
        )
    assert not destination.exists()


@pytest.mark.slow
def test_catalog_accepts_exact_personal_capacity(tmp_path: Path) -> None:
    manifest = _build(_rows(10_000), tmp_path / "exact-capacity.sqlite3")

    assert manifest.unit_count == 10_000
    with SqliteUnitCatalog.open_readonly(manifest, deadline=FUTURE) as catalog:
        assert catalog.unit_count == 10_000


def test_catalog_fails_closed_for_digest_schema_integrity_and_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = _build(_rows(2), tmp_path / "units.sqlite3")

    contents = bytearray(manifest.path.read_bytes())
    contents[-1] ^= 0x01
    manifest.path.write_bytes(contents)

    def unexpected_connect(*_args: object, **_kwargs: object) -> sqlite3.Connection:
        raise AssertionError("sqlite3.connect must not run before checksum validation")

    with monkeypatch.context() as scoped:
        scoped.setattr(sqlite3, "connect", unexpected_connect)
        with pytest.raises(ValueError, match="checksum"):
            SqliteUnitCatalog.open_readonly(manifest, deadline=FUTURE)

    schema_manifest = _build(_rows(1), tmp_path / "schema.sqlite3")
    with sqlite3.connect(schema_manifest.path) as connection:
        connection.execute("PRAGMA user_version=999")
    schema_manifest = replace(
        schema_manifest,
        sha256=SqliteUnitCatalogWriter.digest_file(schema_manifest.path),
        size_bytes=schema_manifest.path.stat().st_size,
    )
    with pytest.raises(ValueError, match="schema"):
        SqliteUnitCatalog.open_readonly(schema_manifest, deadline=FUTURE)

    expired_destination = tmp_path / "expired.sqlite3"
    with pytest.raises(TimeoutError, match="deadline"):
        SqliteUnitCatalogWriter().build(
            _rows(1),
            expired_destination,
            activation_set_id="activation-1",
            corpus_ref="corpus-1",
            deadline=PAST,
        )
    assert not expired_destination.exists()


def test_catalog_rejects_payload_replacement_after_same_object_digest(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build(_rows(2), tmp_path / "catalog-original.sqlite3")
    replacement = tmp_path / "catalog-replacement.sqlite3"
    replacement.write_bytes(descriptor.path.read_bytes())
    with sqlite3.connect(replacement) as connection:
        connection.execute("UPDATE units SET media_type = media_type")
    assert replacement.stat().st_size == descriptor.size_bytes
    assert SqliteUnitCatalogWriter.digest_file(replacement) != descriptor.sha256
    original_connect = sqlite3.connect
    replaced = False

    def replace_then_connect(
        database: str | Path,
        *args: Any,
        **kwargs: Any,
    ) -> sqlite3.Connection:
        nonlocal replaced
        if not replaced:
            replacement.replace(descriptor.path)
            replaced = True
        return cast(sqlite3.Connection, original_connect(database, *args, **kwargs))

    monkeypatch.setattr(sqlite3, "connect", replace_then_connect)

    with pytest.raises(ValueError, match="checksum payload changed"):
        SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE)


def test_catalog_maps_cantopen_after_payload_replacement_to_integrity_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build(_rows(2), tmp_path / "catalog-cantopen.sqlite3")
    original_path_binding = sqlite_unit_catalog._verify_catalog_path_binding
    connect_attempted = False

    def fail_to_connect(_opened: Any) -> sqlite3.Connection:
        nonlocal connect_attempted
        connect_attempted = True
        raise sqlite3.OperationalError("unable to open database file")

    def fail_binding_after_connect(opened: Any) -> None:
        if connect_attempted:
            raise ValueError("catalog checksum payload changed")
        original_path_binding(opened)

    monkeypatch.setattr(sqlite_unit_catalog, "_connect_opened_catalog", fail_to_connect)
    monkeypatch.setattr(
        sqlite_unit_catalog,
        "_verify_catalog_path_binding",
        fail_binding_after_connect,
    )

    with pytest.raises(ValueError, match="checksum payload changed"):
        SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE)


def test_catalog_preserves_cantopen_when_verified_payload_is_unchanged(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build(_rows(2), tmp_path / "catalog-cantopen-unchanged.sqlite3")

    def fail_to_connect(_opened: Any) -> sqlite3.Connection:
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(sqlite_unit_catalog, "_connect_opened_catalog", fail_to_connect)

    with pytest.raises(sqlite3.OperationalError, match="unable to open database file"):
        SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE)


def test_catalog_open_cleanup_preserves_primary_failure_and_closes_every_resource(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    descriptor = _build(_rows(2), tmp_path / "catalog-open-cleanup.sqlite3")

    class FailingCloseConnection:
        row_factory: object | None = None
        close_calls = 0

        def execute(self, _statement: str) -> None:
            raise ValueError("primary catalog open failure")

        def close(self) -> None:
            self.close_calls += 1
            raise sqlite3.OperationalError("connection cleanup failure")

    class FailingClosePayload:
        close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            raise OSError("payload cleanup failure")

    connection = FailingCloseConnection()
    payload = FailingClosePayload()
    monkeypatch.setattr(
        sqlite_unit_catalog,
        "_open_verified_catalog_payload",
        lambda *_args, **_kwargs: payload,
    )
    monkeypatch.setattr(
        sqlite_unit_catalog,
        "_connect_opened_catalog",
        lambda _opened: connection,
    )

    with pytest.raises(ValueError, match="primary catalog open failure") as caught:
        SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE)

    assert connection.close_calls == 1
    assert payload.close_calls == 1
    assert caught.value.__notes__ == [
        "SQLite connection cleanup also failed: OperationalError",
        "catalog payload cleanup also failed: OSError",
    ]


def test_catalog_fails_closed_if_published_path_changes_after_open(tmp_path: Path) -> None:
    descriptor = _build(_rows(2), tmp_path / "catalog-open.sqlite3")
    replacement = tmp_path / "catalog-after-open.sqlite3"
    replacement.write_bytes(descriptor.path.read_bytes())
    with sqlite3.connect(replacement) as connection:
        connection.execute("UPDATE units SET media_type = media_type")
    catalog = SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE)
    replacement.replace(descriptor.path)
    try:
        with pytest.raises(ValueError, match="checksum payload changed"):
            catalog.lookup_content_hash(
                _rows(2)[0].source_sha256,
                SearchFilters(),
                1,
                FUTURE,
            )
    finally:
        catalog.close()


def test_catalog_close_retries_only_the_resource_that_failed_once() -> None:
    class FailOnceConnection:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1
            if self.close_calls == 1:
                raise sqlite3.OperationalError("injected connection close failure")

    class CountingPayload:
        def __init__(self) -> None:
            self.close_calls = 0

        def close(self) -> None:
            self.close_calls += 1

    connection = FailOnceConnection()
    payload = CountingPayload()
    catalog = SqliteUnitCatalog(
        cast(Any, object()),
        cast(Any, connection),
        cast(Any, payload),
    )

    with pytest.raises(sqlite3.OperationalError, match="injected"):
        catalog.close()

    assert connection.close_calls == 1
    assert payload.close_calls == 1
    with pytest.raises(RuntimeError, match="unit catalog is closed"):
        _ = catalog.is_query_only

    catalog.close()
    catalog.close()

    assert connection.close_calls == 2
    assert payload.close_calls == 1


def test_catalog_get_many_is_bounded_and_missing_ids_fail_closed(tmp_path: Path) -> None:
    manifest = _build(_rows(150), tmp_path / "units.sqlite3")
    with SqliteUnitCatalog.open_readonly(manifest, deadline=FUTURE) as catalog:
        with pytest.raises(ValueError, match="at most 100"):
            catalog.get_many(tuple(f"unit-{index}" for index in range(101)), deadline=FUTURE)
        with pytest.raises(KeyError, match="missing"):
            catalog.get_many(("unit-0", "missing"), deadline=FUTURE)


def test_catalog_selection_drives_exact_generation_bound_numpy_search(tmp_path: Path) -> None:
    descriptor = _build(_rows(2), tmp_path / "units.sqlite3")
    space = EmbeddingSpaceRef(
        space_id="image-instance-v1",
        provider_id="local-image-encoder",
        model_revision="revision-1",
        preprocessing_version="image-preprocess-v1",
        dimension=2,
        normalization=VectorNormalization.L2,
        distance_metric=DistanceMetric.INNER_PRODUCT,
        query_modalities=frozenset({QueryModality.IMAGE}),
        target_modalities=frozenset({TargetType.IMAGE}),
    )
    generation = IndexGeneration(
        generation_id="generation-instance-1",
        activation_set_id=descriptor.activation_set_id,
        workspace_id="workspace-1",
        corpus_ref=descriptor.corpus_ref,
        target_type=TargetType.IMAGE,
        unit_manifest_sha256=descriptor.unit_manifest_sha256,
        embedding_spaces=(space,),
        hash_spaces=(),
        retriever_id="capicove.personal-numpy",
        index_format="numpy-flat-f32-v1",
        index_parameters_ref=IndexParametersRef(
            parameter_schema="numpy-flat-f32-v1",
            payload_ref="parameters:instance-1",
            sha256="b" * 64,
        ),
        shard_refs=("payload:instance-1",),
        calibration_profile_ref="exact-v1",
        build_processing_fingerprint=ProcessingFingerprint("a" * 64),
        created_at=datetime(2026, 8, 27, tzinfo=UTC),
        validation_report_ref="validation:instance-1",
    )
    payload = NumpyExactIndexWriter().build_array(
        np.asarray(((1.0, 0.0), (0.0, 1.0)), dtype=np.float32),
        generation,
        tmp_path / "instance.npy",
        lane=RetrievalLane.INSTANCE,
        deadline=FUTURE,
    )

    with SqliteUnitCatalog.open_readonly(descriptor, deadline=FUTURE) as catalog:
        selection = catalog.select_ordinals(
            SearchFilters(media_types=frozenset({"image/png"})),
            deadline=FUTURE,
        )
        with NumpyExactIndexReader.open_readonly(
            generation,
            payload,
            ordinal_resolver=catalog,
            deadline=FUTURE,
        ) as reader:
            hits = reader.search(
                QueryVector(space, (1.0, 0.0)),
                selection,
                limit=2,
                deadline=FUTURE,
            )

    assert [(hit.unit_id, hit.unit_ordinal) for hit in hits] == [("unit-0", 0)]
