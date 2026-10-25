"""RFC-0019: decode reuse never weakens verified graph reads."""

from __future__ import annotations

import os
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from capicove.contracts.graph_build import StageCoverage
from capicove.contracts.graph_build.snapshots import TABLE_ROLES
from capicove.processing import LocalProcessingWorkspace
from tests.graph_build_fixtures import (
    SCOPE,
    Gate,
    _catalog,
    _manifest,
    _rows,
)


pytestmark = pytest.mark.integration
pytest.importorskip("pyarrow")
IDENTITY = "a" * 64


@pytest.fixture(autouse=True)
def _fresh_memos() -> Iterator[None]:
    from capicove.graph_build import _decode_reuse

    memos = (
        _decode_reuse.GRAPH_MANIFESTS,
        _decode_reuse.GRAPH_TABLES,
        _decode_reuse.STAGE_ARTIFACTS,
    )
    for memo in memos:
        memo.clear()
    yield
    for memo in memos:
        memo.clear()


@pytest.fixture
def decodes(monkeypatch: pytest.MonkeyPatch) -> list[object]:
    """Count real Parquet decodes performed by the graph codec."""
    import pyarrow.parquet  # type: ignore[import-untyped]

    calls: list[object] = []
    original = pyarrow.parquet.ParquetFile

    def counting(*args: Any, **kwargs: Any) -> Any:
        calls.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(pyarrow.parquet, "ParquetFile", counting)
    return calls


def _codec(identity: str = IDENTITY) -> Any:
    from capicove.graph_build._adapters.parquet import ParquetTableCodec

    return ParquetTableCodec(publication_identity=identity)


def _part(tmp_path: Path, role: str = "documents") -> Path:
    path = tmp_path / f"{role}-part-00000.parquet"
    _codec().write(path, role, _rows()[role])
    return path


def _rewrite_same_size_keep_mtime(path: Path) -> None:
    observed = path.stat()
    payload = bytearray(path.read_bytes())
    payload[len(payload) // 2] ^= 0xFF
    path.write_bytes(bytes(payload))
    os.utime(path, ns=(observed.st_atime_ns, observed.st_mtime_ns))
    assert path.stat().st_size == observed.st_size
    assert path.stat().st_mtime_ns == observed.st_mtime_ns


def test_identical_part_bytes_decode_once(tmp_path: Path, decodes: list[object]) -> None:
    path = _part(tmp_path)
    first = _codec().read(path, "documents")
    second = _codec().read(path, "documents")
    assert first == second == _rows()["documents"]
    assert len(decodes) == 1


def test_same_size_rewrite_with_restored_mtime_is_rejected(tmp_path: Path) -> None:
    path = _part(tmp_path)
    _codec().read(path, "documents")
    _rewrite_same_size_keep_mtime(path)
    with pytest.raises((ValueError, TypeError)):
        _codec().read(path, "documents")


def test_same_bytes_under_another_role_or_identity_still_fail(tmp_path: Path) -> None:
    path = _part(tmp_path)
    _codec().read(path, "documents")
    with pytest.raises(ValueError):
        _codec("b" * 64).read(path, "documents")
    with pytest.raises(ValueError):
        _codec().read(path, "text_units")


def test_failed_decode_is_never_cached(tmp_path: Path, decodes: list[object]) -> None:
    path = _part(tmp_path)
    good = path.read_bytes()
    _codec().read(path, "documents")
    path.write_bytes(b"PAR1 not a parquet payload PAR1")
    with pytest.raises(ValueError):
        _codec().read(path, "documents")
    path.write_bytes(good)
    assert _codec().read(path, "documents") == _rows()["documents"]
    path.write_bytes(b"PAR1 not a parquet payload PAR1")
    with pytest.raises(ValueError):
        _codec().read(path, "documents")
    assert len(decodes) == 3


def test_unsafe_path_is_rejected_even_when_bytes_are_known(tmp_path: Path) -> None:
    path = _part(tmp_path)
    _codec().read(path, "documents")
    link = tmp_path / "link.parquet"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        _codec().read(link, "documents")


def test_memo_evicts_least_recently_used_entries_at_its_byte_bound() -> None:
    from capicove.graph_build._decode_reuse import VerifiedDecodeMemo

    memo = VerifiedDecodeMemo(max_bytes=10)
    memo.put("first", ("one",), 6)
    memo.put("second", ("two",), 3)
    assert memo.get("first") == ("one",)
    memo.put("third", ("three",), 4)
    assert memo.get("second") is None
    assert memo.get("first") == ("one",)
    assert memo.get("third") == ("three",)
    memo.put("oversized", ("big",), 11)
    assert memo.get("oversized") is None
    assert memo.get("first") == ("one",)


def test_concurrent_readers_receive_equal_rows(tmp_path: Path) -> None:
    path = _part(tmp_path)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: _codec().read(path, "documents"), range(32)))
    assert all(result == _rows()["documents"] for result in results)


def test_opt_out_disables_reuse(
    tmp_path: Path, decodes: list[object], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CAPICOVE_GRAPH_DECODE_REUSE", "0")
    path = _part(tmp_path)
    assert _codec().read(path, "documents") == _codec().read(path, "documents")
    assert len(decodes) == 2


def _warm_catalog(tmp_path: Path, gate: Gate) -> tuple[Any, Any, Any]:
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, gate)
    manifest = _manifest(workspace, rows=_rows())
    ref = catalog.register(manifest)
    catalog.get(ref.snapshot_id, SCOPE)
    return catalog, manifest, ref


def test_warm_catalog_reads_decode_each_table_part_at_most_once(
    tmp_path: Path, decodes: list[object]
) -> None:
    catalog, _, ref = _warm_catalog(tmp_path, Gate())
    after_first = len(decodes)
    catalog.get(ref.snapshot_id, SCOPE)
    catalog.get(ref.snapshot_id, SCOPE)
    assert 0 < after_first <= len(TABLE_ROLES)
    assert len(decodes) == after_first


def test_warm_catalog_read_rejects_same_size_rewrite(tmp_path: Path) -> None:
    from capicove.graph_build.snapshots import GraphValidationError

    catalog, manifest, ref = _warm_catalog(tmp_path, Gate())
    target = next(table.dataset_ref for table in manifest.tables if table.role == "entities")
    _rewrite_same_size_keep_mtime(Path(unquote(urlsplit(target.uri).path)) / "part-00000.parquet")
    with pytest.raises(GraphValidationError):
        catalog.get(ref.snapshot_id, SCOPE)


def test_warm_catalog_read_rechecks_revoked_evidence(tmp_path: Path) -> None:
    gate = Gate()
    catalog, _, ref = _warm_catalog(tmp_path, gate)
    gate.denied.add(_rows()["documents"][0].evidence_ref)
    with pytest.raises(PermissionError):
        catalog.get(ref.snapshot_id, SCOPE)


def _stage_chain(tmp_path: Path, gate: Gate) -> tuple[Any, tuple[str, ...]]:
    from capicove.graph_build._stage_artifacts import BuildStageStore, StageOutput

    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    store = BuildStageStore(tmp_path / "artifacts" / "stages.sqlite", gate, workspace)
    output = StageOutput(
        "identity_embeddings",
        "succeeded",
        StageCoverage("descriptors", 0, 0, 0, 0, 0, 0),
        (),
        (),
    )
    refs: list[str] = []
    for index, evidence in enumerate(("evidence:deep", "evidence:middle", "evidence:top")):
        refs.append(
            store.commit(
                SCOPE,
                output,
                semantics=f'{{"step":{index}}}'.encode(),
                dependencies=tuple(refs[-1:]),
                input_refs=(),
                evidence_refs=(evidence,),
            )
        )
    return store, tuple(refs)


def test_recursive_stage_read_uses_one_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store, refs = _stage_chain(tmp_path, Gate())
    connections: list[object] = []
    original = sqlite3.connect

    def counting(*args: Any, **kwargs: Any) -> Any:
        connections.append(args[0])
        return original(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", counting)
    value = store.read(refs[-1], SCOPE)
    assert value["dependencies"] == [refs[-2]]
    assert len(connections) == 1


def test_warm_stage_read_skips_repeated_canonical_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build import _stage_artifacts
    from capicove.graph_build.codec import canonical_bytes as original

    store, refs = _stage_chain(tmp_path, Gate())
    first = store.read(refs[-1], SCOPE)
    calls: list[object] = []

    def counting(value: Any) -> bytes:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(_stage_artifacts, "canonical_bytes", counting)
    assert store.read(refs[-1], SCOPE) == first
    assert calls == []


def test_warm_stage_read_detects_dependency_tampering(tmp_path: Path) -> None:
    store, refs = _stage_chain(tmp_path, Gate())
    store.read(refs[-1], SCOPE)
    with sqlite3.connect(tmp_path / "artifacts" / "stages.sqlite") as connection:
        payload = bytes(
            connection.execute(
                "SELECT payload FROM build_stage_artifacts WHERE artifact_ref = ?", (refs[0],)
            ).fetchone()[0]
        )
        tampered = payload.replace(b"evidence:deep", b"evidence:peed")
        assert tampered != payload
        connection.execute(
            "UPDATE build_stage_artifacts SET payload = ? WHERE artifact_ref = ?",
            (tampered, refs[0]),
        )
    with pytest.raises(ValueError, match="stage_artifact_integrity_failure"):
        store.read(refs[-1], SCOPE)


def test_warm_stage_read_rechecks_evidence_at_every_depth(tmp_path: Path) -> None:
    gate = Gate()
    store, refs = _stage_chain(tmp_path, gate)
    store.read(refs[-1], SCOPE)
    gate.denied.add("evidence:deep")
    with pytest.raises(PermissionError):
        store.read(refs[-1], SCOPE)


def _forbid_artifact_validation(catalog: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("eligibility checks must not re-verify artifacts")

    monkeypatch.setattr(catalog, "_validate", forbidden)


def test_require_eligible_reauthorizes_without_artifact_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog, manifest, ref = _warm_catalog(tmp_path, Gate())
    _forbid_artifact_validation(catalog, monkeypatch)
    assert catalog.require_eligible(ref.snapshot_id, SCOPE) == manifest


def test_require_eligible_rejects_revoked_evidence_and_policy_drift(tmp_path: Path) -> None:
    class Drifting(Gate):
        drift = False

        def require_access(self, evidence_refs: tuple[str, ...], scope: Any) -> None:
            super().require_access(evidence_refs, scope)
            if self.drift:
                self.revision += 1

    gate = Drifting()
    catalog, _, ref = _warm_catalog(tmp_path, gate)
    gate.denied.add(_rows()["documents"][0].evidence_ref)
    with pytest.raises(PermissionError):
        catalog.require_eligible(ref.snapshot_id, SCOPE)
    gate.denied.clear()
    gate.drift = True
    with pytest.raises(PermissionError):
        catalog.require_eligible(ref.snapshot_id, SCOPE)


def test_require_eligible_rejects_tombstones_corrupt_state_and_foreign_scope(
    tmp_path: Path,
) -> None:
    from capicove.contracts.graph_build import GraphScope
    from capicove.graph_build.snapshots import GraphValidationError

    catalog, _, ref = _warm_catalog(tmp_path, Gate())
    with pytest.raises(GraphValidationError):
        catalog.require_eligible(ref.snapshot_id, GraphScope("workspace:test", "scope:other"))
    with sqlite3.connect(tmp_path / "catalog" / "graph.sqlite") as connection:
        connection.execute("UPDATE graph_snapshots SET state_sha256 = ?", ("0" * 64,))
    with pytest.raises(GraphValidationError):
        catalog.require_eligible(ref.snapshot_id, SCOPE)

    fresh, _, fresh_ref = _warm_catalog(tmp_path / "fresh", Gate())
    fresh.mark_ineligible(_rows()["documents"][0].evidence_ref, SCOPE, "source_deleted")
    with pytest.raises(GraphValidationError):
        fresh.require_eligible(fresh_ref.snapshot_id, SCOPE)


def test_require_eligible_is_not_an_artifact_integrity_check(tmp_path: Path) -> None:
    """Content integrity stays with get/active/read and the binding check."""
    from capicove.graph_build.snapshots import GraphValidationError

    catalog, manifest, ref = _warm_catalog(tmp_path, Gate())
    target = next(table.dataset_ref for table in manifest.tables if table.role == "entities")
    _rewrite_same_size_keep_mtime(Path(unquote(urlsplit(target.uri).path)) / "part-00000.parquet")
    assert catalog.require_eligible(ref.snapshot_id, SCOPE) == manifest
    with pytest.raises(GraphValidationError):
        catalog.get(ref.snapshot_id, SCOPE)


def test_warm_manifest_decode_skips_canonical_reencoding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build import snapshots
    from capicove.graph_build.codec import manifest_bytes as original

    catalog, manifest, ref = _warm_catalog(tmp_path, Gate())
    calls: list[object] = []

    def counting(value: Any) -> bytes:
        calls.append(value)
        return original(value)

    monkeypatch.setattr(snapshots, "manifest_bytes", counting)
    assert catalog.get(ref.snapshot_id, SCOPE) == manifest
    assert catalog.require_eligible(ref.snapshot_id, SCOPE) == manifest
    assert calls == []


def test_warm_manifest_decode_rejects_corrupt_stored_manifest(tmp_path: Path) -> None:
    from capicove.graph_build.snapshots import GraphValidationError

    catalog, _, ref = _warm_catalog(tmp_path, Gate())
    with sqlite3.connect(tmp_path / "catalog" / "graph.sqlite") as connection:
        stored = bytes(connection.execute("SELECT manifest FROM graph_snapshots").fetchone()[0])
        connection.execute(
            "UPDATE graph_snapshots SET manifest = ?", (stored.replace(b"complete", b"partial "),)
        )
    with pytest.raises(GraphValidationError):
        catalog.get(ref.snapshot_id, SCOPE)
    with pytest.raises(GraphValidationError):
        catalog.require_eligible(ref.snapshot_id, SCOPE)


def test_text_reader_reauthorization_does_not_reverify_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.integration.graph_build.test_retrieval_binding import setup

    _, gate, catalog, _, ref, reader, _ = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    _forbid_artifact_validation(catalog, monkeypatch)
    reader.require_access(content)
    gate.denied.add("evidence:1")
    with pytest.raises(PermissionError):
        reader.require_access(content)


def test_artifact_mutation_during_projection_prevents_any_index_effect(tmp_path: Path) -> None:
    from tests.integration.graph_build.test_retrieval_binding import build, build_setup

    values = build_setup(tmp_path)
    _, _, graph_catalog, ref, _, generations, _, index, _, _, _ = values
    manifest = graph_catalog.get(ref.snapshot_id, SCOPE)
    target = next(table.dataset_ref for table in manifest.tables if table.role == "entities")
    _rewrite_same_size_keep_mtime(Path(unquote(urlsplit(target.uri).path)) / "part-00000.parquet")
    with pytest.raises(ValueError):
        build(values)
    assert index.events == []
    assert (
        generations.active(workspace_ref=SCOPE.workspace_ref, retriever_id=index.provider_id)
        is None
    )
