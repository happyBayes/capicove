"""Real optional PyArrow round trips and public workspace crash recovery."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlsplit

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.table_registry import required_table_roles
from capicove.processing import LocalProcessingWorkspace, ProcessingRecoveryError, RunState
from tests.unit.graph_build.test_dataset_schemas import representative_rows


pytestmark = pytest.mark.integration
SCOPE = GraphScope("workspace:私有", "scope:学习")


@pytest.fixture
def arrow() -> Any:
    return pytest.importorskip("pyarrow", reason="requires optional capicove[graph-data]")


def _path(ref: Any) -> Path:
    return Path(unquote(urlsplit(ref.uri).path))


@pytest.mark.parametrize("role", sorted(required_table_roles()))
@pytest.mark.parametrize("empty", [False, True])
def test_real_parquet_has_typed_columns_and_round_trips_every_role(
    tmp_path: Path, arrow: Any, role: str, empty: bool
) -> None:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    from capicove.graph_build._adapters.parquet import ParquetTableCodec
    from capicove.graph_build.datasets import schema_for_role

    codec = ParquetTableCodec()
    rows = () if empty else (representative_rows()[role],)
    path = tmp_path / "part-00000.parquet"
    codec.write(path, role, rows)
    assert codec.read(path, role) == rows
    payload = path.read_bytes()
    assert payload[:4] == payload[-4:] == b"PAR1"
    table = pq.read_table(path)
    assert table.num_rows == len(rows)
    assert table.column_names == [field.name for field in schema_for_role(role).fields]
    if rows:
        assert pq.ParquetFile(path).metadata.row_group(0).column(0).compression == "ZSTD"
    if role == "entities":
        assert arrow.types.is_struct(table.schema.field("scope").type)
        assert arrow.types.is_list(table.schema.field("aliases").type)
        assert not table.schema.field("aliases").type.value_field.nullable


@pytest.mark.parametrize(
    "damage",
    ["role", "schema", "metadata", "count", "corrupt", "symlink", "directory", "traversal"],
)
def test_parquet_reader_rejects_drift_and_unsafe_paths(
    tmp_path: Path, arrow: Any, damage: str
) -> None:
    import pyarrow.parquet as pq

    from capicove.graph_build._adapters.parquet import ParquetTableCodec

    codec = ParquetTableCodec()
    path = tmp_path / "part-00000.parquet"
    codec.write(path, "entities", (representative_rows()["entities"],))
    role = "entities"
    if damage == "role":
        role = "resolved_entities"
    elif damage in {"schema", "metadata", "count"}:
        table = pq.read_table(path)
        if damage == "schema":
            table = table.drop(["subject_ref"])
        else:
            metadata = dict(table.schema.metadata)
            metadata[b"capicove.row_count" if damage == "count" else b"capicove.serializer"] = (
                b"999"
            )
            table = table.replace_schema_metadata(metadata)
        pq.write_table(table, path)
    elif damage == "corrupt":
        path.write_bytes(b"PAR1-truncated-PAR1")
    elif damage == "symlink":
        link = tmp_path / "link.parquet"
        link.symlink_to(path)
        path = link
    elif damage == "directory":
        path = tmp_path
    else:
        path = tmp_path / "child" / ".." / path.name
    with pytest.raises((ValueError, OSError)):
        codec.read(path, role)


def _publish(
    workspace: LocalProcessingWorkspace, rows: tuple[object, ...] | None = None, **kwargs: Any
) -> Any:
    from capicove.graph_build.datasets import publish_table

    return publish_table(
        workspace,
        scope=SCOPE,
        role="entities",
        rows=(representative_rows()["entities"],) if rows is None else rows,
        parent_refs=kwargs.pop("parent_refs", ()),
        processing_fingerprint=kwargs.pop("processing_fingerprint", "b" * 64),
        **kwargs,
    )


def test_publication_reuses_exact_recorded_run_even_after_latest_changes(
    tmp_path: Path, arrow: Any
) -> None:
    workspace = LocalProcessingWorkspace(tmp_path)
    first = _publish(workspace)
    second = _publish(workspace, processing_fingerprint="c" * 64, parent_refs=(first,))
    assert second.version == 2
    assert second.parent_snapshot_ids == (first.snapshot_id,)
    assert workspace.latest(first.dataset_id) == second
    assert _publish(workspace) == first
    assert "私有" not in first.uri and "学习" not in first.dataset_id
    assert workspace.validate_snapshot(first.snapshot_id, validator=lambda _: True) == first
    assert not (_path(second) / ".capicove-pending.json").exists()


def test_row_order_is_canonical_duplicates_rejected_and_scope_bound(
    tmp_path: Path, arrow: Any
) -> None:
    from capicove.graph_build.datasets import publish_table

    workspace = LocalProcessingWorkspace(tmp_path)
    row: Any = representative_rows()["entities"]
    other = replace(row, entity_id="e:2")
    ref = _publish(workspace, (row, other))
    assert _publish(workspace, (other, row)) == ref
    with pytest.raises(ValueError, match="duplicate"):
        _publish(workspace, (row, row))
    other_scope = GraphScope("other", "scope:学习")
    with pytest.raises(ValueError, match="scope"):
        publish_table(
            workspace,
            scope=other_scope,
            role="entities",
            rows=(row,),
            parent_refs=(),
            processing_fingerprint="b" * 64,
        )
    scoped = publish_table(
        workspace,
        scope=other_scope,
        role="entities",
        rows=(replace(row, scope=other_scope),),
        parent_refs=(),
        processing_fingerprint="b" * 64,
    )
    assert scoped.dataset_id != ref.dataset_id


def test_invalid_parents_and_fingerprints_fail_before_publication(
    tmp_path: Path, arrow: Any
) -> None:
    workspace = LocalProcessingWorkspace(tmp_path)
    first = _publish(workspace)
    second = _publish(workspace, processing_fingerprint="c" * 64)
    for parents in (
        (first, first),
        tuple(sorted((first, second), key=lambda ref: ref.snapshot_id, reverse=True)),
        (replace(first, version=99),),
    ):
        with pytest.raises(ValueError):
            _publish(workspace, parent_refs=parents)
    with pytest.raises(ValueError):
        _publish(workspace, processing_fingerprint="not-a-hash")


def test_reuse_rejects_mutated_committed_payload(tmp_path: Path, arrow: Any) -> None:
    workspace = LocalProcessingWorkspace(tmp_path)
    first = _publish(workspace)
    (_path(first) / "part-00000.parquet").write_bytes(b"PAR1-corrupt-PAR1")
    with pytest.raises(ProcessingRecoveryError):
        _publish(workspace)


@pytest.mark.parametrize(
    "point", ["before_publish", "after_publish", "before_commit", "before_finalize"]
)
def test_public_workspace_recovers_injected_process_crash(
    tmp_path: Path, arrow: Any, point: str
) -> None:
    class Crash(BaseException):
        pass

    class Interrupted(LocalProcessingWorkspace):
        run_id: str = ""
        snapshot: Any = None

        def create(self, run: Any, /) -> None:
            self.run_id = run.run_id
            super().create(run)

        def publish(self, **kwargs: Any) -> Any:
            if point == "before_publish":
                raise Crash()
            self.snapshot = super().publish(**kwargs)
            if point == "after_publish":
                raise Crash()
            return self.snapshot

        def commit_success(self, *args: Any, **kwargs: Any) -> Any:
            if point == "before_commit":
                raise Crash()
            return super().commit_success(*args, **kwargs)

        def finalize(self, *args: Any, **kwargs: Any) -> bool:
            if point == "before_finalize":
                raise Crash()
            return super().finalize(*args, **kwargs)

    workspace = Interrupted(tmp_path)
    with pytest.raises(Crash):
        _publish(workspace)
    recovered = LocalProcessingWorkspace(tmp_path)
    run = recovered.get_run(workspace.run_id)
    assert run is not None
    assert run.state is (RunState.SUCCEEDED if point == "before_finalize" else RunState.FAILED)
    if workspace.snapshot is not None:
        assert _path(workspace.snapshot).exists() == (point == "before_finalize")
    assert list((tmp_path / ".staging").iterdir()) == []
    retry = _publish(recovered)
    assert retry.version == 1
    if point == "before_finalize":
        assert retry == workspace.snapshot


def test_concurrent_publishers_commit_monotonic_versions_and_reuse(
    tmp_path: Path, arrow: Any
) -> None:
    LocalProcessingWorkspace(tmp_path)

    def publish(index: int) -> Any:
        return _publish(LocalProcessingWorkspace(tmp_path), processing_fingerprint=f"{index:064x}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        refs = tuple(executor.map(publish, [1, 2, 1, 3, 2, 4]))
    assert refs[0] == refs[2]
    assert refs[1] == refs[4]
    assert sorted({ref.version for ref in refs}) == [1, 2, 3, 4]


@pytest.mark.parametrize("point", ["create", "write", "publish", "commit", "committed", "finalize"])
def test_ordinary_failures_leave_durable_failure_or_preserve_commit(
    tmp_path: Path, arrow: Any, monkeypatch: pytest.MonkeyPatch, point: str
) -> None:
    from capicove.graph_build._adapters.parquet import ParquetTableCodec

    class FailingWorkspace(LocalProcessingWorkspace):
        run_id: str = ""
        published: Any = None
        edges: Any = None

        def create(self, run: Any, /) -> None:
            self.run_id = run.run_id
            super().create(run)
            if point == "create":
                raise RuntimeError("injected")

        def publish(self, **kwargs: Any) -> Any:
            if point == "publish":
                raise RuntimeError("injected")
            self.published = super().publish(**kwargs)
            return self.published

        def commit_success(self, run: str, snapshot: Any, edges: Any, /, **kwargs: Any) -> Any:
            self.edges = edges
            if point == "commit":
                raise RuntimeError("injected")
            result = super().commit_success(run, snapshot, edges, **kwargs)
            if point == "committed":
                raise RuntimeError("injected")
            return result

        def finalize(self, *args: Any, **kwargs: Any) -> bool:
            if point == "finalize":
                raise RuntimeError("injected")
            return super().finalize(*args, **kwargs)

    workspace = FailingWorkspace(tmp_path)
    if point == "write":

        def failed_write(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("injected")

        monkeypatch.setattr(ParquetTableCodec, "write", failed_write)
    with pytest.raises(RuntimeError, match="injected"):
        _publish(workspace)
    run = workspace.get_run(workspace.run_id)
    assert run is not None
    committed = point in {"committed", "finalize"}
    assert run.state is (RunState.SUCCEEDED if committed else RunState.FAILED)
    assert list((tmp_path / ".staging").iterdir()) == []
    if workspace.published is not None:
        assert _path(workspace.published).exists() == committed
    LocalProcessingWorkspace(tmp_path)


@pytest.mark.parametrize("after", [False, True])
def test_actual_directory_rename_failure_recovers_via_public_workspace(
    tmp_path: Path, arrow: Any, monkeypatch: pytest.MonkeyPatch, after: bool
) -> None:
    import os

    original = os.replace

    def interrupted(source: Any, destination: Any) -> None:
        if Path(source).is_dir() and Path(destination).name == "v1":
            if after:
                original(source, destination)
            raise OSError("injected rename failure")
        original(source, destination)

    workspace = LocalProcessingWorkspace(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(os, "replace", interrupted)
        with pytest.raises(RuntimeError):
            _publish(workspace)
    recovered = LocalProcessingWorkspace(tmp_path)
    assert list((tmp_path / ".staging").iterdir()) == []
    ref = _publish(recovered)
    assert ref.version == 1


def test_parent_lineage_uses_real_materialization_run(tmp_path: Path, arrow: Any) -> None:
    from capicove.contracts.data import LineageEdge

    class ObservedWorkspace(LocalProcessingWorkspace):
        edges: tuple[LineageEdge, ...] = ()

        def commit_success(
            self, run: str, ref: Any, lineage: tuple[LineageEdge, ...], /, **kwargs: Any
        ) -> Any:
            self.edges = lineage
            return super().commit_success(run, ref, lineage, **kwargs)

    workspace = ObservedWorkspace(tmp_path)
    first = _publish(workspace)
    second = _publish(workspace, processing_fingerprint="c" * 64)
    parents = tuple(sorted((first, second), key=lambda ref: ref.snapshot_id))
    child = _publish(workspace, parent_refs=parents)
    assert tuple(edge.input_snapshot_id for edge in workspace.edges) == child.parent_snapshot_ids
    assert all(
        edge.output_snapshot_id == child.snapshot_id
        and edge.operator_ids == ("capicove.graph.materialize",)
        for edge in workspace.edges
    )
    run = workspace.get_run(workspace.edges[0].run_id)
    assert run is not None and run.state is RunState.SUCCEEDED
    assert run.backend_id == "capicove.graph.materialization"
    assert run.output_snapshot_id == child.snapshot_id


def test_input_iterable_is_consumed_before_workspace_lock(tmp_path: Path, arrow: Any) -> None:
    from collections.abc import Iterator
    from contextlib import contextmanager

    from capicove.graph_build.datasets import publish_table

    class ObservedWorkspace(LocalProcessingWorkspace):
        locked = False

        @contextmanager
        def execution_lock(self) -> Iterator[None]:
            with super().execution_lock():
                self.locked = True
                try:
                    yield
                finally:
                    self.locked = False

    workspace = ObservedWorkspace(tmp_path)

    def rows() -> Iterator[object]:
        assert not workspace.locked
        yield representative_rows()["entities"]

    result = publish_table(
        workspace,
        scope=SCOPE,
        role="entities",
        rows=rows(),
        parent_refs=(),
        processing_fingerprint="b" * 64,
    )
    assert workspace.get(result.snapshot_id) == result


@pytest.mark.parametrize("point", ["staged", "created", "running"])
def test_startup_recovers_crashes_before_materialization_writes(
    tmp_path: Path, arrow: Any, point: str
) -> None:
    class Crash(BaseException):
        pass

    class Interrupted(LocalProcessingWorkspace):
        run_id = ""

        def stage(self, run_id: str, /) -> Path:
            self.run_id = run_id
            result = super().stage(run_id)
            if point == "staged":
                raise Crash()
            return result

        def create(self, run: Any, /) -> None:
            self.run_id = run.run_id
            super().create(run)
            if point == "created":
                raise Crash()

        def transition(self, *args: Any, **kwargs: Any) -> Any:
            result = super().transition(*args, **kwargs)
            if point == "running" and result.state is RunState.RUNNING:
                raise Crash()
            return result

    workspace = Interrupted(tmp_path)
    with pytest.raises(Crash):
        _publish(workspace)

    recovered = LocalProcessingWorkspace(tmp_path)
    run = recovered.get_run(workspace.run_id)
    if point == "staged":
        assert run is None
    else:
        assert run is not None and run.state is RunState.FAILED
        assert run.detail == "RECOVERED_INCOMPLETE_PUBLICATION"
    assert list((tmp_path / ".staging").iterdir()) == []
    snapshot = _publish(recovered)
    assert snapshot.version == 1
    assert _publish(recovered) == snapshot


@pytest.mark.parametrize("point", ["stage", "create"])
def test_ordinary_failure_before_run_creation_leaves_no_run_or_staging(
    tmp_path: Path, arrow: Any, point: str
) -> None:
    class FailingWorkspace(LocalProcessingWorkspace):
        run_id = ""

        def stage(self, run_id: str, /) -> Path:
            self.run_id = run_id
            if point == "stage":
                raise RuntimeError("injected stage failure")
            return super().stage(run_id)

        def create(self, run: Any, /) -> None:
            self.run_id = run.run_id
            if point == "create":
                raise RuntimeError("injected create failure")
            super().create(run)

    workspace = FailingWorkspace(tmp_path)
    with pytest.raises(RuntimeError, match="injected"):
        _publish(workspace)
    assert workspace.get_run(workspace.run_id) is None
    assert list((tmp_path / ".staging").iterdir()) == []
