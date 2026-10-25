"""Real Parquet identity, authorization, and transactional catalog persistence."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import (
    GraphScope,
    GraphSnapshotManifest,
    StageCoverage,
    StageResult,
)
from capicove.graph_build.datasets import publish_table
from capicove.processing import LocalProcessingWorkspace
from tests.graph_build_fixtures import (
    SCOPE,
    Gate,
    Inputs,
    Stages,
    _catalog,
    _manifest,
    _module,
    _rows,
)


pytestmark = pytest.mark.integration


PublishedGraph = tuple[LocalProcessingWorkspace, GraphSnapshotManifest]


@pytest.fixture(scope="module")
def published_graph(tmp_path_factory: pytest.TempPathFactory) -> PublishedGraph:
    """Publish the 17 immutable tables once per worker; catalogs remain per test."""
    workspace = LocalProcessingWorkspace(tmp_path_factory.mktemp("snapshot-tables"))
    return workspace, _manifest(workspace, rows=_rows())


def test_partial_graph_never_replaces_active_by_default() -> None:
    allowed = _module().activation_allowed
    assert allowed("complete", True, True)
    assert not allowed("partial", True, True)
    assert not allowed("complete", False, True)
    assert not allowed("complete", True, False)


def test_stage_attestation_accepts_exact_input_snapshot(
    tmp_path: Path, published_graph: PublishedGraph
) -> None:
    workspace, manifest = published_graph

    ref = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate()).register(manifest)

    assert ref.snapshot_id.startswith("graph:")


def test_registration_validates_rows_after_real_parquet_reads(tmp_path: Path) -> None:
    """Keep the catalog-to-rule seam real; exhaustive corruptions are unit cases."""
    rows = _rows()
    rows["text_units"] = (replace(rows["text_units"][0], source_sha256="f" * 64),)
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    manifest = _manifest(workspace, rows=rows)
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())

    with pytest.raises(_module().GraphValidationError, match="text unit document mismatch"):
        catalog.register(manifest)
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.parametrize("damage", ["foreign", "same_snapshot_id_other_field"])
def test_stage_attestation_rejects_nonmatching_input_snapshot(tmp_path: Path, damage: str) -> None:
    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    manifest = _manifest(workspace)
    foreign = publish_table(
        workspace,
        scope=SCOPE,
        role="documents",
        rows=(),
        parent_refs=(),
        processing_fingerprint="b" * 64,
    )

    class WrongInputStages(Stages):
        def validate(self, *args: Any) -> Any:
            result = super().validate(*args)
            replacement = (
                foreign
                if damage == "foreign"
                else replace(
                    result.input_snapshot_ref, version=result.input_snapshot_ref.version + 1
                )
            )
            return replace(result, input_snapshot_ref=replacement)

    catalog = module.GraphSnapshotCatalog(
        tmp_path / "catalog" / "graph.sqlite",
        workspace,
        Gate(),
        Inputs(workspace),
        WrongInputStages(),
    )
    with pytest.raises(module.GraphValidationError, match="stage validation identity mismatch"):
        catalog.register(manifest)


def test_validated_stage_artifacts_rejects_non_typed_input_snapshot() -> None:
    module = _module()
    with pytest.raises(TypeError, match="input_snapshot_ref must be DatasetSnapshotRef"):
        module.ValidatedStageArtifacts(
            StageResult(
                "representations",
                "succeeded",
                ("artifact:test",),
                (),
                StageCoverage("documents", 0, 0, 0, 0, 0, 0),
            ),
            SCOPE,
            "not-a-snapshot-ref",
            "a" * 64,
            (),
            "c" * 64,
            (),
        )


def test_catalog_rejects_tampered_stage_input_snapshot(
    tmp_path: Path, published_graph: PublishedGraph
) -> None:
    module = _module()
    workspace, manifest = published_graph

    class TamperedStages(Stages):
        def validate(self, *args: Any) -> Any:
            result = super().validate(*args)
            object.__setattr__(result, "input_snapshot_ref", "tampered")
            return result

    catalog = module.GraphSnapshotCatalog(
        tmp_path / "catalog" / "graph.sqlite",
        workspace,
        Gate(),
        Inputs(workspace),
        TamperedStages(),
    )
    with pytest.raises(module.GraphValidationError, match="stage validation identity mismatch"):
        catalog.register(manifest)


@pytest.mark.slow
@pytest.mark.parametrize("damage", [None, "endpoint", "evidence", "competing"])
def test_pinned_alignment_endpoints_and_current_only_view(
    tmp_path: Path, damage: str | None
) -> None:
    from capicove.contracts.graph_build import EntityAlignment

    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())
    baseline = catalog.register(_manifest(workspace, rows=_rows()))
    rows = _rows()
    first = EntityAlignment("entity:A", "entity:A", "matched", (), ("mention:A",), "d" * 64)
    second = EntityAlignment("entity:B", "entity:B", "matched", (), ("mention:B",), "d" * 64)
    if damage == "endpoint":
        first = replace(first, previous_entity_id="missing")
    elif damage == "evidence":
        first = replace(first, evidence_refs=("foreign",))
    elif damage == "competing":
        second = replace(
            second, previous_entity_id="entity:A", evidence_refs=("mention:A", "mention:B")
        )
    rows["entity_alignments"] = (first, second)
    manifest = _manifest(workspace, rows=rows)
    manifest = replace(
        manifest,
        baseline_snapshot_id=baseline.snapshot_id,
        stages=tuple(
            replace(stage, issue_codes=(), coverage=StageCoverage("entities", 2, 2, 2, 0, 0, 0))
            if stage.stage_id == "alignment"
            else stage
            for stage in manifest.stages
        ),
    )
    if damage is None:
        ref = catalog.register(manifest)
        assert catalog.activate(ref.snapshot_id, SCOPE, 0) == 1
        assert catalog.get(ref.snapshot_id, SCOPE) == manifest
    else:
        with pytest.raises(module.GraphValidationError):
            catalog.register(manifest)


@pytest.mark.slow
@pytest.mark.parametrize("damage", ["missing", "wrong_stage", "duplicate", "foreign_ref"])
def test_stage_table_commitments_are_closed(
    tmp_path: Path, published_graph: PublishedGraph, damage: str
) -> None:
    module = _module()
    workspace, manifest = published_graph

    class InvalidStages(Stages):
        def validate(self, *args: Any) -> Any:
            result = super().validate(*args)
            if result.stage.stage_id != "resolution":
                return result
            refs = result.table_refs
            if damage == "missing":
                refs = ()
            elif damage == "duplicate":
                refs = refs + refs[:1]
            elif damage == "wrong_stage":
                refs = (next(table for table in manifest.tables if table.role == "documents"),)
            else:
                refs = (
                    replace(refs[0], dataset_ref=replace(refs[0].dataset_ref, version=999)),
                    *refs[1:],
                )
            return replace(result, table_refs=refs)

    catalog = module.GraphSnapshotCatalog(
        tmp_path / "catalog" / "graph.sqlite", workspace, Gate(), Inputs(workspace), InvalidStages()
    )
    with pytest.raises(module.GraphValidationError):
        catalog.register(manifest)


@pytest.mark.slow
def test_scope_lifecycle_denial_makes_no_changes(
    tmp_path: Path, published_graph: PublishedGraph
) -> None:
    workspace, manifest = published_graph
    gate = Gate()
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, gate)
    ref = catalog.register(manifest)
    catalog.activate(ref.snapshot_id, SCOPE, 0)
    gate.denied_scopes.add(SCOPE)
    with pytest.raises(PermissionError):
        catalog.mark_ineligible("evidence:1", SCOPE, "source_deleted")
    gate.denied_scopes.clear()
    assert catalog.active(SCOPE) == (ref, 1)


@pytest.mark.slow
def test_activation_rechecks_artifacts_inside_policy_guard(tmp_path: Path) -> None:
    from urllib.parse import unquote, urlsplit

    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    manifest = _manifest(workspace)
    target = next(table.dataset_ref for table in manifest.tables if table.role == "entities")

    class ChangingGate(Gate):
        change = False

        @contextmanager
        def guard(self, expected_revision: int) -> Iterator[None]:
            with super().guard(expected_revision):
                if self.change:
                    part = Path(unquote(urlsplit(target.uri).path)) / "part-00000.parquet"
                    part.write_bytes(b"corrupt after initial validation")
                    self.change = False
                yield

    gate = ChangingGate()
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, gate)
    ref = catalog.register(manifest)
    gate.change = True
    with pytest.raises(module.GraphValidationError):
        catalog.activate(ref.snapshot_id, SCOPE, 0)
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_filesystem_mutation_after_verified_read_is_rejected_on_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """SQLite durability does not make late external file writes atomic."""
    from urllib.parse import unquote, urlsplit

    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    path = tmp_path / "catalog" / "graph.sqlite"
    catalog = _catalog(path, workspace, Gate())
    manifest = _manifest(workspace)
    ref = catalog.register(manifest)
    target = next(table.dataset_ref for table in manifest.tables if table.role == "entities")
    part = Path(unquote(urlsplit(target.uri).path)) / "part-00000.parquet"
    original = sqlite3.connect

    class LateMutation(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = ()) -> Any:
            if sql.startswith("UPDATE graph_active"):
                part.write_bytes(b"external write after final verified read")
            return super().execute(sql, parameters)

    with monkeypatch.context() as patch:
        patch.setattr(
            sqlite3,
            "connect",
            lambda *args, **kwargs: original(*args, **kwargs, factory=LateMutation),
        )
        assert catalog.activate(ref.snapshot_id, SCOPE, 0) == 1
    with pytest.raises(module.GraphValidationError):
        catalog.active(SCOPE)
    with pytest.raises(module.GraphValidationError):
        catalog.get(ref.snapshot_id, SCOPE)
    with original(path) as connection:
        assert connection.execute("SELECT revision FROM graph_active").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM graph_history").fetchone()[0] == 1


def test_real_seventeen_role_manifest_codec_roundtrip(tmp_path: Path) -> None:
    from capicove.graph_build.codec import graph_from_json, graph_to_json, manifest_bytes

    manifest = _manifest(LocalProcessingWorkspace(tmp_path / "processing"))
    assert graph_from_json(graph_to_json(manifest)) == manifest
    assert manifest_bytes(manifest) == manifest_bytes(
        replace(manifest, tables=tuple(reversed(manifest.tables)))
    )


@pytest.mark.slow
def test_registration_is_inactive_idempotent_and_restart_persistent(
    tmp_path: Path, published_graph: PublishedGraph
) -> None:
    module = _module()
    workspace, manifest = published_graph
    gate = Gate()
    path = tmp_path / "catalog" / "graph.sqlite"
    catalog = _catalog(path, workspace, gate)
    assert catalog.active(SCOPE) == (None, 0)
    ref = catalog.register(manifest)
    assert catalog.register(manifest) == ref
    assert catalog.get(ref.snapshot_id, SCOPE) == manifest
    reopened = _catalog(path, workspace, gate)
    assert reopened.active(SCOPE) == (None, 0)
    assert reopened.activate(ref.snapshot_id, SCOPE, 0) == 1
    assert catalog.active(SCOPE) == (ref, 1)
    with pytest.raises(module.GraphActivationConflict):
        catalog.activate(ref.snapshot_id, SCOPE, 0)


@pytest.mark.slow
def test_partial_activation_and_foreign_scope_reads_fail(
    tmp_path: Path, published_graph: PublishedGraph
) -> None:
    module = _module()
    workspace, manifest = published_graph
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())
    ref = catalog.register(replace(manifest, completeness="partial"))
    with pytest.raises(module.GraphValidationError):
        catalog.activate(ref.snapshot_id, SCOPE, 0)
    other = replace(SCOPE, scope_ref="scope:other")
    with pytest.raises(module.GraphValidationError) as foreign:
        catalog.get(ref.snapshot_id, other)
    with pytest.raises(module.GraphValidationError) as missing:
        catalog.get("graph:" + "f" * 64, other)
    assert str(foreign.value) == str(missing.value)


@pytest.mark.slow
def test_evidence_invalidation_fanout_and_scope_isolation(tmp_path: Path) -> None:
    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    gate = Gate()
    path = tmp_path / "catalog" / "graph.sqlite"
    catalog = _catalog(path, workspace, gate)
    manifest = _manifest(workspace, rows=_rows())
    first = catalog.register(manifest)
    second = catalog.register(replace(manifest, completeness="partial"))
    other_scope = replace(SCOPE, scope_ref="scope:other")
    other = catalog.register(_manifest(workspace, scope=other_scope, rows=_rows(other_scope)))
    assert catalog.activate(first.snapshot_id, SCOPE, 0) == 1
    assert catalog.activate(other.snapshot_id, other_scope, 0) == 1
    assert callable(getattr(catalog, "mark_ineligible", None)), "evidence invalidation is missing"
    gate.denied.add("evidence:1")
    gate.revision += 1
    catalog.mark_ineligible("evidence:1", SCOPE, "source_deleted")
    assert catalog.active(SCOPE) == (None, 2)
    gate.denied.clear()
    gate.revision += 1
    reopened = _catalog(path, workspace, gate)
    for ref in (first, second):
        with pytest.raises(module.GraphValidationError):
            reopened.get(ref.snapshot_id, SCOPE)
        with pytest.raises(module.GraphValidationError):
            reopened.activate(ref.snapshot_id, SCOPE, 2)
    with pytest.raises(module.GraphValidationError):
        reopened.register(manifest)
    assert reopened.active(other_scope) == (other, 1)


@pytest.mark.slow
def test_competing_connections_have_one_cas_winner(
    tmp_path: Path, published_graph: PublishedGraph
) -> None:
    module = _module()
    workspace, manifest = published_graph
    path = tmp_path / "catalog" / "graph.sqlite"
    catalogs = [_catalog(path, workspace, Gate()), _catalog(path, workspace, Gate())]
    ref = catalogs[0].register(manifest)

    def activate(catalog: Any) -> str:
        try:
            assert catalog.activate(ref.snapshot_id, SCOPE, 0) == 1
        except module.GraphActivationConflict:
            return "conflict"
        return "activated"

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(activate, catalogs)) == ["activated", "conflict"]
    assert catalogs[0].active(SCOPE) == (ref, 1)


@pytest.mark.slow
@pytest.mark.parametrize("operation", ["register", "activate", "get", "active"])
def test_current_source_authorization_is_required(
    tmp_path: Path, published_graph: PublishedGraph, operation: str
) -> None:
    workspace, manifest = published_graph
    gate = Gate()
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, gate)
    ref = catalog.register(manifest)
    catalog.activate(ref.snapshot_id, SCOPE, 0)
    gate.denied.add("evidence:1")
    gate.revision += 1
    with pytest.raises(PermissionError):
        if operation == "register":
            catalog.register(manifest)
        elif operation == "activate":
            catalog.activate(ref.snapshot_id, SCOPE, 1)
        elif operation == "get":
            catalog.get(ref.snapshot_id, SCOPE)
        else:
            catalog.active(SCOPE)


@pytest.mark.slow
@pytest.mark.parametrize("operation", ["register", "activate"])
def test_policy_revision_drift_aborts_publication(
    tmp_path: Path, published_graph: PublishedGraph, operation: str
) -> None:
    module = _module()
    workspace, manifest = published_graph
    path = tmp_path / "catalog" / "graph.sqlite"
    gate = Gate()

    class DriftingStages(Stages):
        drift = False

        def validate(self, *args: Any) -> Any:
            result = super().validate(*args)
            if self.drift:
                gate.revision += 1
                self.drift = False
            return result

    stages = DriftingStages()
    catalog = module.GraphSnapshotCatalog(path, workspace, gate, Inputs(workspace), stages)
    ref = catalog.register(manifest) if operation == "activate" else None
    stages.drift = True
    with pytest.raises(PermissionError):
        if ref is None:
            catalog.register(manifest)
        else:
            catalog.activate(ref.snapshot_id, SCOPE, 0)
    assert catalog.active(SCOPE) == (None, 0)
    with sqlite3.connect(path) as connection:
        assert connection.execute("SELECT count(*) FROM graph_snapshots").fetchone()[0] == (
            ref is not None
        )


@pytest.mark.slow
@pytest.mark.parametrize(
    "damage", ["manifest", "validation", "dependencies", "pointer", "history", "schema"]
)
def test_catalog_tampering_fails_closed(
    tmp_path: Path, published_graph: PublishedGraph, damage: str
) -> None:
    module = _module()
    workspace, manifest = published_graph
    path = tmp_path / "catalog" / "graph.sqlite"
    catalog = _catalog(path, workspace, Gate())
    ref = catalog.register(manifest)
    catalog.activate(ref.snapshot_id, SCOPE, 0)
    statements = {
        "manifest": "UPDATE graph_snapshots SET manifest=x'7b7d'",
        "validation": "UPDATE graph_snapshots SET validation_sha256='stale'",
        "dependencies": "DELETE FROM graph_evidence",
        "pointer": "UPDATE graph_active SET snapshot_id=NULL",
        "history": "DELETE FROM graph_history",
        "schema": "CREATE TABLE forbidden_graph_rows (payload TEXT)",
    }
    with sqlite3.connect(path) as connection:
        connection.execute(statements[damage])
    with pytest.raises((module.GraphValidationError, ValueError)):
        catalog.active(SCOPE)


@pytest.mark.slow
@pytest.mark.parametrize(
    "damage", ["missing", "corrupt", "foreign_scope", "wrong_role", "changed_ref"]
)
def test_dataset_integrity_and_identity_are_rechecked(tmp_path: Path, damage: str) -> None:
    from urllib.parse import unquote, urlsplit

    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())
    manifest = _manifest(workspace)
    ref = catalog.register(manifest)
    target = next(table for table in manifest.tables if table.role == "entities")
    if damage in {"missing", "corrupt"}:
        part = Path(unquote(urlsplit(target.dataset_ref.uri).path)) / "part-00000.parquet"
        if damage == "missing":
            part.unlink()
        else:
            part.write_bytes(b"PAR1-damaged-PAR1")
        with pytest.raises(module.GraphValidationError):
            catalog.activate(ref.snapshot_id, SCOPE, 0)
    else:
        replacement = target.dataset_ref
        if damage == "foreign_scope":
            replacement = publish_table(
                workspace,
                scope=replace(SCOPE, scope_ref="foreign"),
                role="entities",
                rows=(),
                parent_refs=(),
                processing_fingerprint="a" * 64,
            )
        elif damage == "wrong_role":
            replacement = next(
                table.dataset_ref for table in manifest.tables if table.role == "resolved_entities"
            )
        else:
            replacement = replace(replacement, version=999)
        manifest = replace(
            manifest,
            tables=tuple(
                replace(table, dataset_ref=replacement) if table.role == "entities" else table
                for table in manifest.tables
            ),
        )
        with pytest.raises(module.GraphValidationError):
            catalog.register(manifest)


@pytest.mark.parametrize("reason", ["", "source deleted", "private\ntext", "x" * 129, 12])
def test_lifecycle_reason_is_machine_readable(tmp_path: Path, reason: Any) -> None:
    catalog = _catalog(
        tmp_path / "catalog" / "graph.sqlite",
        LocalProcessingWorkspace(tmp_path / "processing"),
        Gate(),
    )
    with pytest.raises(ValueError):
        catalog.mark_ineligible("evidence:1", SCOPE, reason)
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.parametrize("kind,depth", [("GraphSnapshotManifest", 33), ("Entity", 17)])
def test_closed_codec_depth_bounds_remain_enforced(kind: str, depth: int) -> None:
    from capicove.graph_build.codec import canonical_bytes, graph_from_json

    nested: Any = "leaf"
    for _ in range(depth):
        nested = [nested]
    with pytest.raises(ValueError, match="nesting"):
        graph_from_json(json.dumps({"schema_version": 1, "kind": kind, "value": nested}))
    with pytest.raises(ValueError, match="nesting"):
        canonical_bytes(nested)


@pytest.mark.slow
@pytest.mark.parametrize(
    "phase,after",
    [
        ("register", False),
        ("register", True),
        ("update", False),
        ("update", True),
        ("history", False),
        ("history", True),
    ],
)
def test_sqlite_crash_rolls_back_whole_transaction(
    tmp_path: Path,
    published_graph: PublishedGraph,
    monkeypatch: pytest.MonkeyPatch,
    phase: str,
    after: bool,
) -> None:
    workspace, manifest = published_graph
    path = tmp_path / "catalog" / "graph.sqlite"
    catalog = _catalog(path, workspace, Gate())
    ref = catalog.register(manifest) if phase != "register" else None
    original = sqlite3.connect
    needle = {
        "register": "INSERT INTO graph_snapshots",
        "update": "UPDATE graph_active",
        "history": "INSERT INTO graph_history",
    }[phase]

    class Crash(BaseException):
        pass

    class FailingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = ()) -> Any:
            if sql.startswith(needle) and not after:
                raise Crash()
            result = super().execute(sql, parameters)
            if sql.startswith(needle) and after:
                raise Crash()
            return result

    with monkeypatch.context() as patch:
        patch.setattr(
            sqlite3,
            "connect",
            lambda *args, **kwargs: original(*args, **kwargs, factory=FailingConnection),
        )
        with pytest.raises(Crash):
            if ref is None:
                catalog.register(manifest)
            else:
                catalog.activate(ref.snapshot_id, SCOPE, 0)
    reopened = _catalog(path, workspace, Gate())
    assert reopened.active(SCOPE) == (None, 0)
    with original(path) as connection:
        assert connection.execute("SELECT count(*) FROM graph_snapshots").fetchone()[0] == (
            ref is not None
        )
        assert connection.execute("SELECT count(*) FROM graph_history").fetchone()[0] == 0


def test_profile_cannot_mix_tables_from_another_stage_publication(tmp_path: Path) -> None:
    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())
    manifest = _manifest(workspace)
    replacement = publish_table(
        workspace,
        scope=SCOPE,
        role="entities",
        rows=(),
        parent_refs=(),
        processing_fingerprint="f" * 64,
    )
    mixed = replace(
        manifest,
        tables=tuple(
            replace(table, dataset_ref=replacement) if table.role == "entities" else table
            for table in manifest.tables
        ),
    )
    with pytest.raises(module.GraphValidationError):
        catalog.register(mixed)


@pytest.mark.slow
def test_baseline_is_pinned_authorized_and_invalidation_propagates(tmp_path: Path) -> None:
    module = _module()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    gate = Gate()
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, gate)
    baseline = catalog.register(_manifest(workspace, rows=_rows()))
    manifest = _manifest(workspace)
    manifest = replace(
        manifest,
        baseline_snapshot_id=baseline.snapshot_id,
        stages=tuple(
            replace(stage, issue_codes=()) if stage.stage_id == "alignment" else stage
            for stage in manifest.stages
        ),
    )
    gate.denied.add("evidence:1")
    with pytest.raises(PermissionError):
        catalog.register(manifest)
    gate.denied.clear()
    current = catalog.register(manifest)
    catalog.activate(current.snapshot_id, SCOPE, 0)
    catalog.mark_ineligible("evidence:1", SCOPE, "baseline_source_deleted")
    assert catalog.active(SCOPE) == (None, 2)
    with pytest.raises(module.GraphValidationError):
        catalog.get(current.snapshot_id, SCOPE)


@pytest.mark.slow
@pytest.mark.parametrize(
    "damage",
    [
        "input_scope",
        "input_ref",
        "input_documents",
        "stage_identity",
        "stage_profile",
        "stage_digest",
        "stale",
    ],
)
def test_external_attestations_are_exact_and_revalidated(
    tmp_path: Path, published_graph: PublishedGraph, damage: str
) -> None:
    module = _module()
    workspace, manifest = published_graph
    inputs = Inputs(workspace)
    stages = Stages()

    class BadInputs(Inputs):
        def validate(self, ref: Any, scope: GraphScope) -> Any:
            result = inputs.validate(ref, scope)
            if damage == "input_scope":
                return replace(result, scope=replace(scope, scope_ref="foreign"))
            if damage == "input_ref":
                return replace(result, snapshot_ref=replace(ref, version=99))
            if damage == "input_documents":
                return replace(result, document_refs=("foreign",))
            return result

    class BadStages(Stages):
        stale = False

        def validate(self, *args: Any) -> Any:
            result = stages.validate(*args)
            if damage == "stage_identity":
                return replace(result, stage=replace(result.stage, artifact_refs=("foreign",)))
            if damage == "stage_profile":
                return replace(result, profile_fingerprint="f" * 64)
            if damage == "stage_digest":
                return replace(result, validation_sha256="invalid")
            return replace(result, validation_sha256="e" * 64) if self.stale else result

    bad_stages = BadStages()
    catalog = module.GraphSnapshotCatalog(
        tmp_path / "catalog" / "graph.sqlite", workspace, Gate(), BadInputs(workspace), bad_stages
    )
    if damage == "stale":
        ref = catalog.register(manifest)
        bad_stages.stale = True
        with pytest.raises(module.GraphValidationError):
            catalog.activate(ref.snapshot_id, SCOPE, 0)
    else:
        with pytest.raises(module.GraphValidationError):
            catalog.register(manifest)


@pytest.mark.slow
@pytest.mark.parametrize("after", [False, True])
def test_commit_acknowledgement_crash_is_all_or_nothing(
    tmp_path: Path, published_graph: PublishedGraph, monkeypatch: pytest.MonkeyPatch, after: bool
) -> None:
    workspace, manifest = published_graph
    path = tmp_path / "catalog" / "graph.sqlite"
    catalog = _catalog(path, workspace, Gate())
    ref = catalog.register(manifest)
    original = sqlite3.connect

    class Crash(BaseException):
        pass

    class Connection(sqlite3.Connection):
        updated = False

        def execute(self, sql: str, parameters: Any = ()) -> Any:
            self.updated |= sql.startswith("UPDATE graph_active")
            return super().execute(sql, parameters)

        def commit(self) -> None:
            if self.updated and not after:
                raise Crash()
            super().commit()
            if self.updated and after:
                raise Crash()

    with monkeypatch.context() as patch:
        patch.setattr(
            sqlite3,
            "connect",
            lambda *args, **kwargs: original(*args, **kwargs, factory=Connection),
        )
        with pytest.raises(Crash):
            catalog.activate(ref.snapshot_id, SCOPE, 0)
    assert _catalog(path, workspace, Gate()).active(SCOPE) == ((ref, 1) if after else (None, 0))


@pytest.mark.slow
def test_partial_extraction_without_resolution_remains_inspectable(tmp_path: Path) -> None:
    rows = _rows()
    rows = {
        role: values
        for role, values in rows.items()
        if role in {"documents", "text_units", "entity_mentions", "relationship_mentions"}
    }
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    manifest = _manifest(workspace, rows=rows, completeness="partial")
    manifest = replace(
        manifest,
        stages=tuple(
            stage
            for stage in manifest.stages
            if stage.stage_id in {"representations", "text_units", "extraction"}
        ),
    )
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())
    ref = catalog.register(manifest)
    assert catalog.get(ref.snapshot_id, SCOPE) == manifest
