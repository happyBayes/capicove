"""Contract tests for immutable FAISS HNSW visual generations."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from capicove_retriever_faiss.adapter import (
    FaissOperationError,
    FaissVisualRetriever,
)
from capicove_retriever_faiss.config import FaissConfiguration
from fake_faiss import FakeFaiss
from helpers import entry, manifest

from capicove.contracts import SearchFilters, TargetType, TemporalRange
from capicove.retrieval.video_models import (
    PinnedVisualIndexGeneration,
    VisualIndexEntry,
    VisualIndexGenerationManifest,
)


def configuration(tmp_path: Path, **changes: int) -> FaissConfiguration:
    """Return one bounded local plugin configuration."""
    values = {
        "index_root": tmp_path.resolve(),
        "threads": 2,
        "maximum_keyframes": 10,
        "maximum_segments": 10,
        "hnsw_m": 32,
        "ef_construction": 200,
        "ef_search": 64,
        **changes,
    }
    return FaissConfiguration.from_mapping(values)


def built_backend(
    tmp_path: Path,
    entries: tuple[VisualIndexEntry, ...],
    *,
    fake: FakeFaiss | None = None,
) -> tuple[
    FaissVisualRetriever,
    VisualIndexGenerationManifest,
    PinnedVisualIndexGeneration,
]:
    """Build and activate one fake-native generation."""
    backend = FaissVisualRetriever(configuration(tmp_path), faiss_module=fake or FakeFaiss())
    generation = manifest(entries)
    backend.generation_lifecycle.create(generation)
    backend.index_writer.add(index_generation_id=generation.generation_id, entries=entries)
    report = backend.generation_lifecycle.validate(generation)
    pin = backend.generation_lifecycle.activate(generation, report)
    return backend, generation, pin


def test_builds_hnsw_reopens_and_returns_stable_tie_ranks(tmp_path: Path) -> None:
    entries = (
        entry("unit-b", (1.0, 0.0)),
        entry("unit-a", (1.0, 0.0)),
        entry("unit-c", (0.0, 1.0)),
    )
    fake = FakeFaiss()
    _, generation, pin = built_backend(tmp_path, entries, fake=fake)

    reopened = FaissVisualRetriever(configuration(tmp_path), faiss_module=fake)
    candidates = reopened.candidate_retriever.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=generation.embedding_space,
        filters=SearchFilters(),
        generation=pin,
        top_k=2,
    )

    assert fake.last_created is not None
    assert fake.last_created.hnsw.m == 32
    assert fake.last_created.hnsw.efConstruction == 200
    assert [item.unit_id for item in candidates] == ["unit-a", "unit-b"]
    assert [item.rank for item in candidates] == [1, 2]
    assert fake.thread_calls and set(fake.thread_calls) == {2}


def test_warm_search_reuses_verified_index_but_rejects_changed_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFaiss()
    backend, generation, pin = built_backend(tmp_path, (entry("unit-a", (1.0, 0.0)),), fake=fake)
    calls = 0
    original = fake.read_index

    def counted(path: str) -> object:
        nonlocal calls
        calls += 1
        return original(path)

    monkeypatch.setattr(fake, "read_index", counted)
    for _ in range(3):
        assert backend.retrieve(
            query_vector=(1.0, 0.0),
            embedding_space=generation.embedding_space,
            filters=SearchFilters(),
            generation=pin,
            top_k=1,
        )
    assert calls <= 2
    (Path(pin.physical_index_ref) / "index.faiss").write_bytes(b"corrupt")
    with pytest.raises(FaissOperationError):
        backend.retrieve(
            query_vector=(1.0, 0.0),
            embedding_space=generation.embedding_space,
            filters=SearchFilters(),
            generation=pin,
            top_k=1,
        )


def test_activation_preloads_index_before_online_query(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeFaiss()
    backend, generation, pin = built_backend(tmp_path, (entry("unit-a", (1.0, 0.0)),), fake=fake)

    def unexpected_open(path: str) -> object:
        raise AssertionError("online search must not cold-open an activated index")

    monkeypatch.setattr(fake, "read_index", unexpected_open)
    assert backend.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=generation.embedding_space,
        filters=SearchFilters(),
        generation=pin,
        top_k=1,
    )


def test_validation_reconstructs_bounded_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_faiss import FakeIndex

    source = tuple(entry(f"unit-{i}", (1.0, 0.0)) for i in range(600))
    backend = FaissVisualRetriever(
        configuration(tmp_path, maximum_keyframes=1000), faiss_module=FakeFaiss()
    )
    generation = manifest(source)
    backend.create(generation)
    backend.add(index_generation_id=generation.generation_id, entries=source)
    reconstruct = FakeIndex.reconstruct_n
    counts = []

    def bounded(self: FakeIndex, start: int, count: int) -> np.ndarray:
        counts.append(count)
        assert count <= 512
        return reconstruct(self, start, count)

    monkeypatch.setattr(FakeIndex, "reconstruct_n", bounded)
    assert backend.validate(generation).valid
    assert sum(counts) == 600


def test_applies_supported_post_filters(tmp_path: Path) -> None:
    entries = (
        entry("unit-a", (1.0, 0.0), artifact_ref="artifact-a", timestamp_ms=1_000),
        entry("unit-b", (0.8, 0.6), artifact_ref="artifact-b", timestamp_ms=4_000),
    )
    backend, generation, pin = built_backend(tmp_path, entries)

    candidates = backend.candidate_retriever.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=generation.embedding_space,
        filters=SearchFilters(
            artifact_refs=frozenset({"artifact-b"}),
            time_range=TemporalRange(3_500, 4_500),
        ),
        generation=pin,
        top_k=2,
    )

    assert [item.unit_id for item in candidates] == ["unit-b"]


def test_reopen_can_change_search_budget_without_mutating_frozen_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fake_faiss import FakeIndex

    seen: list[int] = []
    search = FakeIndex.search

    def record(self: FakeIndex, queries: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        seen.append(self.hnsw.efSearch)
        return search(self, queries, top_k)

    monkeypatch.setattr(FakeIndex, "search", record)
    fake = FakeFaiss()
    _, generation, pin = built_backend(tmp_path, (entry("unit-a", (1.0, 0.0)),), fake=fake)
    path = Path(pin.physical_index_ref) / "index.faiss"
    original = path.read_bytes()
    reopened = FaissVisualRetriever(configuration(tmp_path, ef_search=1024), faiss_module=fake)
    assert reopened.validate(generation).valid
    assert reopened.retrieve(
        query_vector=(1.0, 0.0),
        embedding_space=generation.embedding_space,
        filters=SearchFilters(),
        generation=pin,
        top_k=1,
    )
    assert path.read_bytes() == original
    assert seen == [1024]


def test_streaming_build_validates_count_and_content(tmp_path: Path) -> None:
    entries = (entry("unit-a", (1.0, 0.0)), entry("unit-b", (0.0, 1.0)))
    generation = manifest(entries)
    backend = FaissVisualRetriever(configuration(tmp_path), faiss_module=FakeFaiss())
    backend.create(generation)
    with pytest.raises(FaissOperationError):
        backend.add_stream(index_generation_id=generation.generation_id, entries=iter(entries[:1]))
    assert not (tmp_path / generation.generation_id / "manifest.json").exists()
    backend.add_stream(index_generation_id=generation.generation_id, entries=iter(entries))
    assert backend.validate(generation).valid


def test_rejects_capacity_duplicate_dimension_and_target_mismatches(tmp_path: Path) -> None:
    entries = (entry("unit-a", (1.0, 0.0)), entry("unit-b", (0.0, 1.0)))
    generation = manifest(entries)

    limited = FaissVisualRetriever(
        configuration(tmp_path / "capacity", maximum_keyframes=1),
        faiss_module=FakeFaiss(),
    )
    with pytest.raises(ValueError, match="capacity"):
        limited.create(generation)

    backend = FaissVisualRetriever(configuration(tmp_path / "invalid"), faiss_module=FakeFaiss())
    backend.create(generation)
    with pytest.raises(ValueError, match="duplicate"):
        backend.add(index_generation_id=generation.generation_id, entries=(entries[0], entries[0]))
    with pytest.raises(ValueError, match="dimension"):
        backend.add(
            index_generation_id=generation.generation_id,
            entries=(VisualIndexEntry(entries[0].unit, (1.0, 0.0, 0.0)), entries[1]),
        )
    segment = entry("segment-a", (1.0, 0.0), target_type=TargetType.VIDEO_SEGMENT)
    with pytest.raises(ValueError, match="target"):
        backend.add(
            index_generation_id=generation.generation_id,
            entries=(segment, entries[1]),
        )


def test_validation_detects_corruption_and_stale_reports(tmp_path: Path) -> None:
    entries = (entry("unit-a", (1.0, 0.0)),)
    backend, generation, _ = built_backend(tmp_path, entries)
    index_path = tmp_path / generation.generation_id / "index.faiss"
    index_path.write_bytes(index_path.read_bytes() + b"corrupt")

    report = backend.validate(generation)

    assert not report.valid
    assert "checksum_mismatch" in report.issue_codes
    with pytest.raises(ValueError, match="invalid or stale"):
        backend.activate(generation, report)


def test_unknown_native_ordinal_and_native_errors_are_sanitized(tmp_path: Path) -> None:
    entries = (entry("unit-a", (1.0, 0.0)),)
    fake = FakeFaiss()
    backend, generation, pin = built_backend(tmp_path, entries, fake=fake)
    fake.search_override = (
        np.asarray([[1.0]], dtype=np.float32),
        np.asarray([[999]], dtype=np.int64),
    )
    backend.close()  # The fake applies overrides when an index is opened.
    with pytest.raises(FaissOperationError, match="retrieval failed") as unknown:
        backend.retrieve(
            query_vector=(1.0, 0.0),
            embedding_space=generation.embedding_space,
            filters=SearchFilters(),
            generation=pin,
            top_k=1,
        )
    assert "999" not in str(unknown.value)

    fake.search_override = None
    fake.fail_search = True
    backend.close()
    with pytest.raises(FaissOperationError, match="retrieval failed") as native:
        backend.retrieve(
            query_vector=(1.0, 0.0),
            embedding_space=generation.embedding_space,
            filters=SearchFilters(),
            generation=pin,
            top_k=1,
        )
    assert "native token" not in str(native.value)


def test_payload_uses_only_documented_immutable_files(tmp_path: Path) -> None:
    entries = (entry("unit-a", (1.0, 0.0)),)
    _, generation, _ = built_backend(tmp_path, entries)
    generation_path = tmp_path / generation.generation_id

    assert {path.name for path in generation_path.iterdir()} == {
        "build-manifest.json",
        "index.faiss",
        "manifest.json",
        "ordinals.jsonl",
    }
    payload = json.loads((generation_path / "manifest.json").read_text(encoding="utf-8"))
    assert payload["hnsw"] == {"ef_construction": 200, "m": 32}
    assert payload["metric"] == "inner_product"
