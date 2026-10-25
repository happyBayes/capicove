"""Offline concrete graph-stage composition and inactive durable publication."""

from __future__ import annotations

import importlib
import importlib.util
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.datasets import publish_table
from capicove.graph_build.snapshots import GraphSnapshotCatalog
from capicove.processing import LocalProcessingWorkspace
from tests.graph_build_fixtures import Gate, Inputs
from tests.unit.graph_build.test_extraction import build_profile


pytestmark = pytest.mark.integration
SCOPE = GraphScope("workspace:test", "scope:test")


def service_module() -> Any:
    assert importlib.util.find_spec("capicove.graph_build.service") is not None, (
        "Task 22 concrete build service is missing"
    )
    return importlib.import_module("capicove.graph_build.service")


class Inventory(Inputs):
    def documents(self, ref: Any, scope: GraphScope) -> tuple[Any, ...]:
        from capicove.graph_build._adapters.parquet import ParquetTableCodec

        rows: tuple[Any, ...] = ()

        def read(path: Path) -> bool:
            nonlocal rows
            rows = ParquetTableCodec(publication_identity=ref.processing_fingerprint).read(
                path / "part-00000.parquet", "documents"
            )
            return True

        assert self.workspace.validate_snapshot(ref.snapshot_id, validator=read) == ref
        return rows


class Reader:
    def __init__(self, gate: Gate) -> None:
        self.gate = gate
        self.representations: dict[str, Any] = {}

    def read(self, document_ref: str, scope: GraphScope) -> Any:
        self.gate.require_scope(scope)
        return self.representations[document_ref]

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        self.gate.require_access(evidence_refs, scope)


class Tokenizer:
    fingerprint = "c" * 64

    def count(self, text: str) -> int:
        return len(text)


class SourceProvenance:
    fingerprint = "9" * 64

    def classify(self, documents: Any, scope: GraphScope) -> tuple[Any, ...]:
        assert all(document.scope == scope for document in documents)
        dto = service_module().DocumentSourceProvenance
        return tuple(dto(document.document_ref, "source", None) for document in documents)


def harness(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    module = service_module()
    gate = Gate()
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    inputs = Inventory(workspace)
    stages = module.BuildStageStore(tmp_path / "artifacts" / "stages.sqlite", gate, workspace)
    catalog = GraphSnapshotCatalog(
        tmp_path / "catalog" / "graph.sqlite", workspace, gate, inputs, stages
    )
    profile = build_profile()
    service = module.GraphBuildService(
        tmp_path / "runs" / "runs.sqlite",
        reader=Reader(gate),
        gate=gate,
        profiles=(profile,),
        inputs=inputs,
        tokenizer=Tokenizer(),
        inference_runner=None,
        embedder=None,
        identity_space=None,
        partitioner=None,
        workspace=workspace,
        catalog=catalog,
        stages=stages,
        policy=module.BuildPolicy(subject_ref="subject:explicit"),
        source_provenance=SourceProvenance(),
    )
    ref = publish_table(
        workspace,
        scope=SCOPE,
        role="documents",
        rows=(),
        parent_refs=(),
        processing_fingerprint="a" * 64,
    )
    request = module.GraphBuildRequest(ref, SCOPE, profile.profile_id, None, "reuse")
    return service, request, catalog, stages


@pytest.mark.slow
def test_empty_build_is_complete_durable_and_inactive(tmp_path: Path) -> None:
    service, request, catalog, stages = harness(tmp_path)
    result = service.build(request)
    assert result.status == "succeeded"
    assert result.completeness == "complete"
    assert result.graph_ref is not None
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    assert len(manifest.tables) == 17
    alignment = next(stage for stage in manifest.stages if stage.stage_id == "alignment")
    assert alignment.issue_codes == ("no_baseline",)
    assert catalog.active(SCOPE) == (None, 0)
    assert service.resume(result.run_id, SCOPE) == result
    for stage in manifest.stages:
        assert stages.validate(stage, SCOPE, manifest.profile_fingerprint).stage == stage


@pytest.mark.slow
def test_stage_artifact_tampering_cannot_be_reused(tmp_path: Path) -> None:
    import sqlite3

    service, request, catalog, _ = harness(tmp_path)
    result = service.build(request)
    with sqlite3.connect(tmp_path / "artifacts" / "stages.sqlite") as connection:
        connection.execute("UPDATE build_stage_artifacts SET payload = ?", (b"{}",))
    with pytest.raises(ValueError):
        service.resume(result.run_id, SCOPE)
    assert catalog.active(SCOPE) == (None, 0)


def authored(tmp_path: Path) -> tuple[Any, Any, Any, Any]:
    import json
    import time

    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.datasets import DocumentRow
    from capicove.graph_build.inference import InferenceRunner, RawGeneration
    from capicove.graph_build.ports import CommunityRuntimeIdentity
    from capicove.graph_build.representations import text_representation
    from capicove.retrieval import TextEmbedding
    from tests.unit.graph_build.test_extraction import response
    from tests.unit.retrieval.test_text_embedding import text_space

    service, request, catalog, _stages = harness(tmp_path)
    representation = text_representation("document:one", "Alice Python", SCOPE)
    service.reader.representations[representation.document_ref] = representation
    document = DocumentRow(
        representation.document_ref,
        SCOPE,
        representation.source_sha256,
        representation.representation_ref,
        representation.representation_ref,
        "text/plain",
        "available",
        (),
        StageCoverage("codepoints", 12, 12, 12, 0, 0, 0),
    )
    ref = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(document,),
        parent_refs=(),
        processing_fingerprint="b" * 64,
    )
    request = replace(request, input_snapshot_ref=ref)

    class Generator:
        provider_id = "local"
        model_id = "qwen"
        model_revision = "sha256:model"

        def __init__(self) -> None:
            self.calls: list[str] = []

        def generate(self, request: Any, *, control: Any) -> Any:
            self.calls.append(request.role)
            if request.role == "report":
                task = json.loads(request.input_segments[0][2])
                value = {
                    "version": 1,
                    "task_key": task["cache_key"],
                    "report_kind": task["report_kind"],
                    "community_id": task["community_id"],
                    "title": "Evidence overview",
                    "sections": {
                        name: []
                        for name in (
                            "summary",
                            "observed_patterns",
                            "strengths",
                            "difficulties",
                            "possible_causes",
                            "existing_interventions",
                            "observed_outcomes",
                            "suggested_next_steps",
                            "counter_evidence",
                            "limitations",
                        )
                    },
                }
                value["sections"]["limitations"] = [
                    {
                        "local_id": "limit:" + str(i),
                        "text": text,
                        "kind": "limitation",
                        "assertion_refs": [],
                        "finding_refs": [],
                        "child_statement_refs": [],
                    }
                    for i, text in enumerate(task["required_limitations"])
                ]
                value["sections"]["counter_evidence"] = [
                    {
                        "local_id": "counter:" + str(i),
                        "text": "A selected assertion disputes the reported relationship.",
                        "kind": "hypothesis",
                        "assertion_refs": [assertion_ref],
                        "finding_refs": [],
                        "child_statement_refs": [],
                    }
                    for i, assertion_ref in enumerate(task["bundle"]["counter_assertion_ids"])
                ]
            elif request.role == "diagnostic":
                value = {"findings": []}
            else:
                value = response()
            return RawGeneration(json.dumps(value).encode(), "stop", 20, 20)

    class WireTokenizer:
        tokenizer_id = "qwen-tokenizer"
        tokenizer_revision = "sha256:tokenizer"

        def count_request(self, request: Any) -> int:
            return 100

    class Embedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: tuple[str, ...], *, space: Any, role: str) -> Any:
            assert role == "document"
            self.calls += 1
            return tuple(
                TextEmbedding(space, tuple([1.0] + [0.0] * (space.dimension - 1))) for _ in texts
            )

    class Partitioner:
        algorithm_id = "louvain"
        algorithm_version = "test-v1"
        runtime_identity = CommunityRuntimeIdentity("scripted", "1", ())

        def partition(self, graph: Any, profile: Any) -> Any:
            return (frozenset(graph.node_ids),)

    def factory(refs: tuple[str, ...]) -> FilesystemProtectedArtifacts:
        return FilesystemProtectedArtifacts(tmp_path / "raw", service.gate, evidence_refs=refs)

    generator = Generator()
    receipts = SqliteReceiptStore(
        tmp_path / "receipts" / "receipts.sqlite",
        service.gate,
        factory((representation.representation_ref,)),
    )
    service.inference_runner = InferenceRunner(
        generator=generator,
        tokenizer=WireTokenizer(),
        gate=service.gate,
        artifact_factory=factory,
        receipts=receipts,
        clock=time.monotonic,
        wall_clock=time.time,
        sleeper=lambda _: None,
        cancelled=lambda: False,
    )
    service.identity_space = replace(text_space(), space_id="bge-m3-v1")
    service.embedder = Embedder()
    service.partitioner = Partitioner()
    return service, request, catalog, generator


@pytest.mark.slow
def test_authored_build_replays_without_provider_or_embedding_calls(tmp_path: Path) -> None:
    service, request, catalog, generator = authored(tmp_path)
    first = service.build(request)
    assert first.completeness == "complete"
    assert generator.calls.count("user") == 1
    # The two evidence-bearing leaves generate; the source-free overview is local.
    assert generator.calls.count("report") == 2
    assert service.embedder.calls == 1
    calls = tuple(generator.calls)
    second = service.build(request)
    assert second.run_id != first.run_id
    assert second.graph_ref == first.graph_ref
    assert tuple(generator.calls) == calls
    assert service.embedder.calls == 1
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
@pytest.mark.parametrize("missing", ["inference_runner", "embedder", "partitioner"])
def test_missing_mandatory_runtime_publishes_inspectable_partial(
    tmp_path: Path, missing: str
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    setattr(service, missing, None)
    result = service.build(request)
    assert result.status == "partial"
    assert result.completeness == "partial"
    assert result.graph_ref is not None
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    assert any(stage.status == "failed" for stage in manifest.stages)
    assert catalog.active(SCOPE) == (None, 0)


def test_cancel_before_scheduling_and_resume_remains_cancelled(tmp_path: Path) -> None:
    service, request, catalog, generator = authored(tmp_path)
    profile = service._profile(request)
    from capicove.graph_build.profiles import graph_build_profile_fingerprint

    run = service.runs.create(
        request,
        profile_fingerprint=graph_build_profile_fingerprint(profile),
        input_digest=service.inputs.validate(request.input_snapshot_ref, SCOPE).validation_sha256,
        deadline_at=service.wall_clock() + profile.inference.build_deadline_seconds,
        runtime_payload=service._runtime_payload(profile),
    )
    service.cancel(run.run_id, SCOPE)
    result = service.resume(run.run_id, SCOPE)
    assert result.status == "cancelled"
    assert result.graph_ref is None
    assert generator.calls == []
    assert service.resume(run.run_id, SCOPE) == result
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_pinned_baseline_is_persisted_and_cannot_resurrect_deleted_documents(
    tmp_path: Path,
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    first = service.build(request)
    empty = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(),
        parent_refs=(),
        processing_fingerprint="a" * 64,
    )
    second = service.build(
        replace(request, input_snapshot_ref=empty, baseline_snapshot_id=first.graph_ref.snapshot_id)
    )
    assert second.status == "succeeded"
    run = service.runs.get(second.run_id, SCOPE)
    assert run.baseline_digest == first.graph_ref.manifest_sha256
    manifest = catalog.get(second.graph_ref.snapshot_id, SCOPE)
    alignment = next(stage for stage in manifest.stages if stage.stage_id == "alignment")
    assert "no_baseline" not in alignment.issue_codes
    alignment_output = service.stages.output(
        service.stages.read(alignment.artifact_refs[0], SCOPE)["output_ref"], SCOPE
    )
    assert dict(alignment_output.rows)["resolved_entities"] == ()
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
@pytest.mark.parametrize(
    "field", ["community", "report_prompt_version", "diagnostic_prompt_version"]
)
def test_unrelated_profile_changes_preserve_extraction_replay(tmp_path: Path, field: str) -> None:
    service, request, _, generator = authored(tmp_path)
    service.build(request)
    profile = service.profiles[0]
    if field == "community":
        profile = replace(profile, community=replace(profile.community, max_size=51))
    else:
        profile = replace(profile, **{field: "changed-v2"})
    service.profiles = (profile,)
    result = service.build(request)
    assert result.status == "succeeded"
    assert generator.calls.count("user") == 1
    assert service.embedder.calls == 1


@pytest.mark.slow
@pytest.mark.parametrize("after", [False, True])
def test_restart_after_stage_commit_reuses_completed_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after: bool
) -> None:
    import sqlite3

    service, request, catalog, generator = authored(tmp_path)
    request = replace(request, mode="regenerate")
    original = service.runs.record_stage

    class Crash(BaseException):
        pass

    def crash(run_id: str, scope: GraphScope, stage: str, artifact_ref: str) -> None:
        if stage == "extraction" and not after:
            raise Crash()
        original(run_id, scope, stage, artifact_ref)
        if stage == "extraction" and after:
            raise Crash()

    with monkeypatch.context() as patch:
        patch.setattr(service.runs, "record_stage", crash)
        with pytest.raises(Crash):
            service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    assert catalog.active(SCOPE) == (None, 0)
    result = service.resume(run_id, SCOPE)
    assert result.status == "succeeded"
    assert generator.calls.count("user") == 1


@pytest.mark.slow
def test_failed_new_document_version_keeps_no_old_contribution(tmp_path: Path) -> None:
    from capicove.contracts.graph_build import StageCoverage

    service, request, catalog, generator = authored(tmp_path)
    first = service.build(request)
    document = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    failed = replace(
        document,
        source_sha256="f" * 64,
        representation_ref=None,
        availability="unavailable",
        coverage=StageCoverage("codepoints", 12, 0, 0, 0, 12, 0),
    )
    ref = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(failed,),
        parent_refs=(),
        processing_fingerprint="d" * 64,
    )
    result = service.build(replace(request, input_snapshot_ref=ref))
    assert result.status == "partial"
    assert result.graph_ref != first.graph_ref
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    extraction = next(stage for stage in manifest.stages if stage.stage_id == "extraction")
    output = service.stages.output(
        service.stages.read(extraction.artifact_refs[0], SCOPE)["output_ref"], SCOPE
    )
    assert dict(output.rows)["entity_mentions"] == ()
    assert generator.calls.count("user") == 1


@pytest.mark.slow
def test_seal_revalidates_exact_published_rows(tmp_path: Path) -> None:
    service, request, catalog, _ = authored(tmp_path)
    result = service.build(request)
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    stage = next(item for item in manifest.stages if item.stage_id == "resolution")
    original = service.stages.read(stage.artifact_refs[0], SCOPE)["output_ref"]
    output = service.stages.output(original, SCOPE)
    entities = dict(output.rows)["entities"]
    changed = publish_table(
        service.workspace,
        scope=SCOPE,
        role="entities",
        rows=(replace(entities[0], display_name="Altered"), *entities[1:]),
        parent_refs=(request.input_snapshot_ref,),
        processing_fingerprint=manifest.profile_fingerprint,
    )
    from capicove.contracts.graph_build import TableRef

    tables = tuple(
        TableRef(item.role, changed) if item.role == "entities" else item
        for item in manifest.tables
        if item.role in {"entities", "entity_memberships", "resolution_memberships"}
    )
    with pytest.raises(ValueError):
        forged = service.stages.seal(
            original,
            SCOPE,
            manifest.profile_fingerprint,
            tables,
            input_snapshot_ref=request.input_snapshot_ref,
            runtime_payload=service._runtime_payload(service.profiles[0]),
        )
        service.stages.validate(forged, SCOPE, manifest.profile_fingerprint)


@pytest.mark.slow
def test_operational_stages_have_verifiable_artifacts(tmp_path: Path) -> None:
    service, request, _, _ = harness(tmp_path)
    result = service.build(request)
    records = dict(service.runs.stages(result.run_id, SCOPE))
    assert "retrieval" in records
    for stage in ("publication", "retrieval"):
        value = service.stages.read(records[stage], SCOPE)
        assert value["stage_id"] == stage
        assert value["kind"] == "operation"
    assert service.stages.read(records["retrieval"], SCOPE)["status"] == "skipped"
    publication = service.stages.read(records["publication"], SCOPE)
    assert len(publication["dependencies"]) == 1
    seal = service.stages.read(publication["dependencies"][0], SCOPE)
    assert service.stages.output(seal["output_ref"], SCOPE).stage_id == "reports"


@pytest.mark.slow
@pytest.mark.parametrize("after", [False, True])
def test_regenerate_recovery_between_receipt_and_artifact_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, after: bool
) -> None:
    import sqlite3

    service, request, _, generator = authored(tmp_path)
    original = service.stages.commit

    class Crash(BaseException):
        pass

    def commit(scope: GraphScope, output: Any, **kwargs: Any) -> str:
        if output.stage_id == "extraction" and not after:
            raise Crash()
        ref: str = original(scope, output, **kwargs)
        if output.stage_id == "extraction" and after:
            raise Crash()
        return ref

    with monkeypatch.context() as patch:
        patch.setattr(service.stages, "commit", commit)
        with pytest.raises(Crash):
            service.build(replace(request, mode="regenerate"))
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    assert service.resume(run_id, SCOPE).status == "succeeded"
    assert generator.calls.count("user") == 1


@pytest.mark.slow
def test_cancellation_during_inference_is_local_to_one_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, generator = authored(tmp_path)
    create = service.runs.create
    generate = generator.generate
    run_ids = []

    def created(*args: Any, **kwargs: Any) -> Any:
        run = create(*args, **kwargs)
        run_ids.append(run.run_id)
        return run

    def cancel_inside(request: Any, *, control: Any) -> Any:
        service.cancel(run_ids[-1], SCOPE)
        assert control.cancelled()
        return generate(request, control=control)

    with monkeypatch.context() as patch:
        patch.setattr(service.runs, "create", created)
        patch.setattr(generator, "generate", cancel_inside)
        result = service.build(request)
    assert result.status == "cancelled"
    assert result.graph_ref is None
    assert catalog.active(SCOPE) == (None, 0)
    next_result = service.build(request)
    assert next_result.status == "succeeded"


@pytest.mark.slow
# Real storage covers the first, middle, and final publication boundaries.
# All 34 orchestration combinations live in test_publication_orchestration.py.
@pytest.mark.parametrize("role", ["communities", "entity_mentions", "text_units"])
@pytest.mark.parametrize("after", [False, True])
def test_restart_around_each_table_publication_does_not_duplicate_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, role: str, after: bool
) -> None:
    import sqlite3

    module = service_module()
    service, request, catalog, _ = harness(tmp_path)
    original = module.publish_table

    class Crash(BaseException):
        pass

    def publish(*args: Any, **kwargs: Any) -> Any:
        if kwargs["role"] == role and not after:
            raise Crash()
        value = original(*args, **kwargs)
        if kwargs["role"] == role and after:
            raise Crash()
        return value

    with monkeypatch.context() as patch:
        patch.setattr(module, "publish_table", publish)
        with pytest.raises(Crash):
            service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    assert catalog.active(SCOPE) == (None, 0)
    result = service.resume(run_id, SCOPE)
    assert result.status == "succeeded"
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    assert all(
        table.dataset_ref.version == (2 if table.role == "documents" else 1)
        for table in manifest.tables
    )


@pytest.mark.slow
def test_cancel_between_table_publications_stops_new_writes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    module = service_module()
    service, request, catalog, _ = harness(tmp_path)
    original = module.publish_table
    calls = []

    def publish(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["role"])
        value = original(*args, **kwargs)
        with sqlite3.connect(service.runs.path) as connection:
            run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
        service.cancel(run_id, SCOPE)
        return value

    with monkeypatch.context() as patch:
        patch.setattr(module, "publish_table", publish)
        result = service.build(request)
    assert result.status == "cancelled"
    assert len(calls) == 1
    assert len(result.coverage) == 10
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_partial_extraction_rejections_cannot_claim_complete_coverage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from capicove.graph_build.inference import RawGeneration
    from tests.unit.graph_build.test_extraction import response

    service, request, _, generator = authored(tmp_path)
    original = generator.generate

    def generate(request: Any, *, control: Any) -> Any:
        if request.role != "user":
            return original(request, control=control)
        value = response()
        value["entities"][1]["source_spans"][0]["quote"] = "Wrong!"
        return RawGeneration(json.dumps(value).encode(), "stop", 20, 20)

    monkeypatch.setattr(generator, "generate", generate)
    result = service.build(request)
    assert result.status == "partial"
    extraction = next(stage for stage in result.coverage if stage.stage_id == "extraction")
    assert extraction.status == "partial"
    assert extraction.coverage.rejected == 1


def test_policy_revision_drift_during_generation_prevents_next_effect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, generator = authored(tmp_path)
    original = generator.generate

    def generate(request: Any, *, control: Any) -> Any:
        result = original(request, control=control)
        service.gate.revision += 1
        return result

    monkeypatch.setattr(generator, "generate", generate)
    with pytest.raises(PermissionError):
        service.build(request)
    assert service.embedder.calls == 0
    import sqlite3

    with sqlite3.connect(tmp_path / "receipts" / "receipts.sqlite") as connection:
        assert (
            connection.execute(
                "SELECT count(*) FROM inference_units WHERE status='succeeded'"
            ).fetchone()[0]
            == 0
        )
    assert catalog.active(SCOPE) == (None, 0)


def test_resume_does_not_renew_the_durable_build_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    service, request, catalog, generator = authored(tmp_path)
    now = [1000.0]
    service.wall_clock = lambda: now[0]
    original = service.runs.record_stage

    class Crash(BaseException):
        pass

    def record(run_id: str, scope: GraphScope, stage: str, artifact_ref: str) -> None:
        original(run_id, scope, stage, artifact_ref)
        if stage == "text_units":
            raise Crash()

    with monkeypatch.context() as patch:
        patch.setattr(service.runs, "record_stage", record)
        with pytest.raises(Crash):
            service.build(request)
    now[0] += 3601.0
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    result = service.resume(run_id, SCOPE)
    assert result.status == "failed"
    assert result.issue_codes == ("build_deadline_exceeded",)
    assert generator.calls == []
    assert {stage.stage_id for stage in result.coverage} == {"representations", "text_units"}
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.parametrize(
    "changes",
    [
        {"run_id": " "},
        {"status": "done"},
        {"status": True},
        {"completeness": "unknown"},
        {"graph_ref": "latest"},
        {"coverage": []},
        {"issue_codes": ["failed"]},
        {"retrieval_binding_ref": " "},
        {"status": "succeeded"},
    ],
)
def test_public_result_rejects_open_or_inconsistent_fields(changes: dict[str, Any]) -> None:
    module = service_module()
    result = module.GraphBuildResult("build:test", "failed", None, "partial", (), ("failed",), None)
    with pytest.raises((TypeError, ValueError)):
        replace(result, **changes)


@pytest.mark.slow
def test_input_read_and_chunk_read_cannot_return_different_versions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build.representations import text_representation

    service, request, _, generator = authored(tmp_path)
    original = service.reader.read
    calls = 0

    def read(document_ref: str, scope: GraphScope) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            return original(document_ref, scope)
        return text_representation(document_ref, "Alice Python!", scope)

    monkeypatch.setattr(service.reader, "read", read)
    result = service.build(request)
    assert result.status == "partial"
    assert generator.calls == []


def test_transport_neutral_service_is_publicly_available() -> None:
    import capicove.graph_build as graph

    assert getattr(graph, "GraphBuildService", None) is service_module().GraphBuildService
    assert getattr(graph, "GraphBuildRequest", None) is service_module().GraphBuildRequest
    assert getattr(graph, "GraphBuildResult", None) is service_module().GraphBuildResult


@pytest.mark.slow
def test_identity_pool_limit_is_enforced_before_embedding(tmp_path: Path) -> None:
    service, request, _, _ = authored(tmp_path)
    profile = service.profiles[0]
    service.profiles = (
        replace(profile, resolution=replace(profile.resolution, identity_pool_limit=1)),
    )
    result = service.build(request)
    assert result.status == "partial"
    assert service.embedder.calls == 0


@pytest.mark.slow
def test_current_runtime_identity_is_rechecked_before_recovered_effects(tmp_path: Path) -> None:
    service, request, _, _ = authored(tmp_path)
    result = service.build(request)
    service.tokenizer.fingerprint = "d" * 64
    with pytest.raises(ValueError, match="runtime_identity_changed"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
def test_actual_generator_identity_is_pinned_for_resume(tmp_path: Path) -> None:
    service, request, _, generator = authored(tmp_path)
    result = service.build(request)
    generator.model_revision = "sha256:changed"
    with pytest.raises(ValueError, match="runtime_identity_changed"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
def test_explicit_subject_is_passed_to_concrete_community_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    first = service.build(request)
    manifest = catalog.get(first.graph_ref.snapshot_id, SCOPE)
    resolution = next(stage for stage in manifest.stages if stage.stage_id == "resolution")
    seal = service.stages.read(resolution.artifact_refs[0], SCOPE)
    entities = dict(service.stages.output(seal["output_ref"], SCOPE).rows)["entities"]
    subject = next(entity for entity in entities if entity.entity_type == "PERSON")
    service.policy = replace(service.policy, subject_ref=subject.entity_id)
    original = service_module().community_projection
    owners = []

    def project(assertions: Any, entities: Any, owner: Any, profile: Any) -> Any:
        owners.append(owner)
        return original(assertions, entities, owner, profile)

    monkeypatch.setattr(service_module(), "community_projection", project)
    service.build(request)
    assert owners == [subject.entity_id]


@pytest.mark.slow
@pytest.mark.parametrize("different_type", [False, True])
def test_baseline_alignment_uses_bounded_embedding_candidates_and_abstains_same_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, different_type: bool
) -> None:
    import json

    from capicove.graph_build.inference import RawGeneration
    from capicove.graph_build.representations import text_representation
    from tests.unit.graph_build.test_extraction import response

    service, request, _, generator = authored(tmp_path)
    first = service.build(request)
    if different_type:
        generate = generator.generate

        def changed_generation(request: Any, *, control: Any) -> Any:
            if request.role != "user":
                return generate(request, control=control)
            value = response()
            value["entities"] = [value["entities"][0]]
            value["entities"][0]["entity_type"] = "RESOURCE"
            value["relationships"] = []
            return RawGeneration(json.dumps(value).encode(), "stop", 20, 20)

        monkeypatch.setattr(generator, "generate", changed_generation)
    document = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    representation = text_representation(
        "document:two", "Alice Python!" if different_type else "Alice Python", SCOPE
    )
    service.reader.representations[representation.document_ref] = representation
    changed = replace(
        document,
        document_ref=representation.document_ref,
        source_sha256=representation.source_sha256,
        representation_ref=representation.representation_ref,
        evidence_ref=representation.representation_ref,
    )
    current = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(changed,),
        parent_refs=(),
        processing_fingerprint="e" * 64,
    )
    original = service_module().identity_candidates
    original_cross = service_module().cross_identity_candidates
    pool_sizes = []
    cross_pool_sizes = []

    def candidates(entities: Any, vectors: Any, profile: Any) -> Any:
        pool_sizes.append(len(entities))
        return original(entities, vectors, profile)

    def cross_candidates(
        current_entities: Any,
        current_vectors: Any,
        baseline_entities: Any,
        baseline_vectors: Any,
        profile: Any,
    ) -> Any:
        cross_pool_sizes.append((len(current_entities), len(baseline_entities)))
        return original_cross(
            current_entities,
            current_vectors,
            baseline_entities,
            baseline_vectors,
            profile,
        )

    monkeypatch.setattr(service_module(), "identity_candidates", candidates)
    monkeypatch.setattr(service_module(), "cross_identity_candidates", cross_candidates)
    second = service.build(
        replace(
            request, input_snapshot_ref=current, baseline_snapshot_id=first.graph_ref.snapshot_id
        )
    )
    assert second.status == "succeeded"
    assert pool_sizes == ([1] if different_type else [2])
    assert cross_pool_sizes == ([(1, 2)] if different_type else [(2, 2)])
    refs = dict(service.runs.stages(second.run_id, SCOPE))
    alignment = dict(service.stages.output(refs["alignment"], SCOPE).rows)
    if different_type:
        assert alignment["entity_alignments"] == ()
    else:
        assert all(item.decision == "ambiguous" for item in alignment["entity_alignments"])
    assert len(alignment["resolved_entities"]) == (1 if different_type else 2)


@pytest.mark.slow
def test_baseline_alignment_top_k_is_not_crowded_by_current_entities(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each current entity retains a baseline candidate when dense scores tie."""
    import json

    from capicove.graph_build.inference import RawGeneration
    from capicove.graph_build.representations import text_representation
    from capicove.retrieval import TextEmbedding
    from tests.unit.graph_build.test_extraction import response

    service, request, _, generator = authored(tmp_path)
    service.profiles = (
        replace(
            service.profiles[0],
            resolution=replace(service.profiles[0].resolution, top_k=1),
        ),
    )
    original = generator.generate
    current_names = False

    class SegregatedEmbedder:
        def __init__(self) -> None:
            self.calls = 0

        def embed(self, texts: tuple[str, ...], *, space: Any, role: str) -> Any:
            assert role == "document"
            self.calls += 1
            return tuple(
                TextEmbedding(
                    space,
                    tuple(
                        [1.0, 0.0] + [0.0] * (space.dimension - 2)
                        if "current-" in text
                        else [0.0, 1.0] + [0.0] * (space.dimension - 2)
                    ),
                )
                for text in texts
            )

    service.embedder = SegregatedEmbedder()

    def generation(model_request: Any, *, control: Any) -> Any:
        if model_request.role != "user":
            return original(model_request, control=control)
        value = response()
        value["relationships"] = []
        prefix = "current" if current_names else "baseline"
        for index, entity in enumerate(value["entities"]):
            entity["entity_type"] = "PERSON"
            entity["name"] = f"{prefix}-{index}"
        return RawGeneration(json.dumps(value).encode(), "stop", 20, 20)

    monkeypatch.setattr(generator, "generate", generation)
    first = service.build(request)
    assert first.graph_ref is not None

    current_names = True
    document = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    representation = text_representation("document:two", "Alice Python!", SCOPE)
    service.reader.representations[representation.document_ref] = representation
    current_document = replace(
        document,
        document_ref=representation.document_ref,
        source_sha256=representation.source_sha256,
        representation_ref=representation.representation_ref,
        evidence_ref=representation.representation_ref,
    )
    current_input = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(current_document,),
        parent_refs=(),
        processing_fingerprint="f" * 64,
    )
    result = service.build(
        replace(
            request,
            input_snapshot_ref=current_input,
            baseline_snapshot_id=first.graph_ref.snapshot_id,
        )
    )

    stages = dict(service.runs.stages(result.run_id, SCOPE))
    current_entities = dict(service.stages.output(stages["resolution"], SCOPE).rows)["entities"]
    alignments = dict(service.stages.output(stages["alignment"], SCOPE).rows)["entity_alignments"]
    assert {item.current_entity_id for item in alignments} == {
        item.entity_id for item in current_entities
    }


@pytest.mark.slow
@pytest.mark.parametrize(
    "changes",
    [
        {"version": True},
        {"unexpected": "field"},
        {"status": "complete"},
        {"graph_ref": "unregistered"},
        {"dependencies": []},
    ],
)
def test_operational_artifact_reads_fail_closed_even_when_rehashed(
    tmp_path: Path, changes: dict[str, Any]
) -> None:
    import hashlib
    import sqlite3

    from capicove.graph_build.codec import canonical_bytes

    service, request, _, _ = harness(tmp_path)
    result = service.build(request)
    artifact = dict(service.runs.stages(result.run_id, SCOPE))["publication"]
    value = service.stages.read(artifact, SCOPE)
    value.update(changes)
    payload = canonical_bytes(value)
    ref = "build-artifact:" + hashlib.sha256(payload).hexdigest()
    with sqlite3.connect(tmp_path / "artifacts" / "stages.sqlite") as connection:
        connection.execute("INSERT INTO build_stage_artifacts VALUES (?, ?)", (ref, payload))
    with pytest.raises((ValueError, TypeError)):
        service.stages.read(ref, SCOPE)


def test_deadline_during_publication_is_a_durable_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, _ = harness(tmp_path)
    now = [10.0]
    service.monotonic_clock = lambda: now[0]
    original = service_module().publish_table
    calls = []

    def publish(*args: Any, **kwargs: Any) -> Any:
        calls.append(kwargs["role"])
        table = original(*args, **kwargs)
        now[0] += 4000.0
        return table

    monkeypatch.setattr(service_module(), "publish_table", publish)
    result = service.build(request)
    assert result.status == "failed"
    assert result.issue_codes == ("build_deadline_exceeded",)
    assert service.resume(result.run_id, SCOPE) == result
    assert len(calls) == 1
    assert catalog.active(SCOPE) == (None, 0)


def test_terminal_failure_coverage_is_bound_to_committed_stage_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A validly rehashed terminal result cannot omit retained stage evidence."""
    import hashlib
    import json
    import sqlite3

    from capicove.graph_build.codec import canonical_bytes

    service, request, _, _ = harness(tmp_path)
    now = [10.0]
    service.monotonic_clock = lambda: now[0]
    original = service_module().publish_table

    def publish(*args: Any, **kwargs: Any) -> Any:
        table = original(*args, **kwargs)
        now[0] += 4000.0
        return table

    monkeypatch.setattr(service_module(), "publish_table", publish)
    result = service.build(request)
    assert result.status == "failed" and result.coverage
    with sqlite3.connect(service.runs.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM build_results WHERE run_id = ?", (result.run_id,)
            ).fetchone()[0]
        )
        payload["coverage"] = []
        forged = canonical_bytes(payload)
        connection.execute(
            "UPDATE build_results SET payload = ?, digest = ? WHERE run_id = ?",
            (forged, hashlib.sha256(forged).hexdigest(), result.run_id),
        )

    with pytest.raises(ValueError, match="build_terminal_commitment_failure"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
@pytest.mark.parametrize(
    ("result_changes", "run_status"),
    [
        ({"issue_codes": ["forged_issue"]}, None),
        ({"status": "partial"}, "partial"),
    ],
)
def test_terminal_graph_result_is_reconstructed_from_authoritative_artifacts(
    tmp_path: Path,
    result_changes: dict[str, Any],
    run_status: str | None,
) -> None:
    """Rehashing mutable terminal metadata cannot change the graph outcome."""
    import hashlib
    import json
    import sqlite3

    from capicove.graph_build.codec import canonical_bytes

    service, request, _, _ = harness(tmp_path)
    result = service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        if run_status is not None:
            with pytest.raises(sqlite3.IntegrityError, match="immutable build terminal"):
                connection.execute(
                    "UPDATE build_runs SET status = ? WHERE run_id = ?",
                    (run_status, result.run_id),
                )
            return
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM build_results WHERE run_id = ?", (result.run_id,)
            ).fetchone()[0]
        )
        payload.update(result_changes)
        forged = canonical_bytes(payload)
        connection.execute(
            "UPDATE build_results SET payload = ?, digest = ? WHERE run_id = ?",
            (forged, hashlib.sha256(forged).hexdigest(), result.run_id),
        )

    with pytest.raises(ValueError, match="build_terminal_commitment_failure"):
        service.resume(result.run_id, SCOPE)


def test_terminal_failure_reason_is_reconstructed_from_run_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rehashed failure issue cannot replace the only supported no-graph reason."""
    import hashlib
    import json
    import sqlite3

    from capicove.graph_build.codec import canonical_bytes

    service, request, _, _ = harness(tmp_path)
    now = [10.0]
    service.monotonic_clock = lambda: now[0]
    original = service_module().publish_table

    def publish(*args: Any, **kwargs: Any) -> Any:
        table = original(*args, **kwargs)
        now[0] += 4000.0
        return table

    monkeypatch.setattr(service_module(), "publish_table", publish)
    result = service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM build_results WHERE run_id = ?", (result.run_id,)
            ).fetchone()[0]
        )
        payload["issue_codes"] = ["forged_issue"]
        forged = canonical_bytes(payload)
        connection.execute(
            "UPDATE build_results SET payload = ?, digest = ? WHERE run_id = ?",
            (forged, hashlib.sha256(forged).hexdigest(), result.run_id),
        )

    with pytest.raises(ValueError, match="build_terminal_commitment_failure"):
        service.resume(result.run_id, SCOPE)


def test_terminal_failure_revalidates_stage_input_and_dependency_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A foreign but well-formed output cannot be substituted into a terminal prefix."""
    import hashlib
    import json
    import sqlite3

    from capicove.graph_build.codec import canonical_bytes
    from capicove.graph_build.workflow import _stage_digest

    service, request, _, _ = harness(tmp_path)
    now = [10.0]
    service.monotonic_clock = lambda: now[0]
    original = service_module().publish_table

    def publish(*args: Any, **kwargs: Any) -> Any:
        table = original(*args, **kwargs)
        now[0] += 4000.0
        return table

    monkeypatch.setattr(service_module(), "publish_table", publish)
    result = service.build(request)
    stage = result.coverage[0]
    old_ref = stage.artifact_refs[0]
    value = service.stages.read(old_ref, SCOPE)
    value["input_refs"] = ['{"foreign":true}']
    forged_payload = canonical_bytes(value)
    forged_ref = "build-artifact:" + hashlib.sha256(forged_payload).hexdigest()
    with sqlite3.connect(tmp_path / "artifacts" / "stages.sqlite") as connection:
        connection.execute(
            "INSERT INTO build_stage_artifacts VALUES (?, ?)", (forged_ref, forged_payload)
        )
    with sqlite3.connect(service.runs.path) as connection:
        connection.execute(
            "UPDATE build_stages SET artifact_ref = ?, record_digest = ? "
            "WHERE run_id = ? AND stage = ?",
            (
                forged_ref,
                _stage_digest(result.run_id, stage.stage_id, forged_ref),
                result.run_id,
                stage.stage_id,
            ),
        )
        payload = json.loads(
            connection.execute(
                "SELECT payload FROM build_results WHERE run_id = ?", (result.run_id,)
            ).fetchone()[0]
        )
        payload["coverage"][0] = service_module().graph_to_json(
            replace(stage, artifact_refs=(forged_ref,))
        )
        forged_result = canonical_bytes(payload)
        connection.execute(
            "UPDATE build_results SET payload = ?, digest = ? WHERE run_id = ?",
            (forged_result, hashlib.sha256(forged_result).hexdigest(), result.run_id),
        )

    with pytest.raises(ValueError, match="build_terminal_commitment_failure"):
        service.resume(result.run_id, SCOPE)


def test_unknown_stage_exception_is_not_published_as_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Internal integrity failures remain exceptions and never become graph results."""
    import sqlite3

    service, request, _, _ = harness(tmp_path)
    original = service._compute_stage

    def compute(stage: str, *args: Any, **kwargs: Any) -> Any:
        if stage == "text_units":
            raise ValueError("internal_integrity_failure")
        return original(stage, *args, **kwargs)

    monkeypatch.setattr(service, "_compute_stage", compute)
    with pytest.raises(ValueError, match="internal_integrity_failure"):
        service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        assert connection.execute("SELECT count(*) FROM build_results").fetchone()[0] == 0


def test_reader_internal_value_error_is_not_downgraded_to_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only the reader port's explicit unavailable outcome may become partial."""
    import sqlite3

    service, request, _, _ = authored(tmp_path)

    def read(*args: Any, **kwargs: Any) -> Any:
        raise ValueError("reader_internal_corruption")

    monkeypatch.setattr(service.reader, "read", read)
    with pytest.raises(ValueError, match="reader_internal_corruption"):
        service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        assert connection.execute("SELECT count(*) FROM build_results").fetchone()[0] == 0


@pytest.mark.slow
def test_explicit_reader_unavailable_outcome_remains_partial(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A declared source read failure remains an inspectable partial graph."""
    from capicove.graph_build.ports import RepresentationUnavailableError

    service, request, _, generator = authored(tmp_path)

    def read(*args: Any, **kwargs: Any) -> Any:
        raise RepresentationUnavailableError("source_not_locally_available")

    monkeypatch.setattr(service.reader, "read", read)
    result = service.build(request)
    assert result.status == "partial"
    assert generator.calls == []


@pytest.mark.slow
def test_terminal_commit_fences_final_revision_and_evidence_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Revocation in the final check-to-finish window cannot commit success."""
    import sqlite3

    service, request, _, _ = authored(tmp_path)
    evidence = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0].evidence_ref
    original = service.runs.finish

    def finish(*args: Any, **kwargs: Any) -> Any:
        service.gate.denied.add(evidence)
        service.gate.revision += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service.runs, "finish", finish)
    with pytest.raises(PermissionError):
        service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        assert connection.execute("SELECT count(*) FROM build_results").fetchone()[0] == 0


@pytest.mark.slow
def test_terminal_resume_rejects_forged_retrieval_operation_when_not_requested(
    tmp_path: Path,
) -> None:
    """A no-retrieval run accepts only the exact durable skipped operation."""
    import sqlite3

    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build.workflow import _stage_digest

    service, request, _, _ = harness(tmp_path)
    result = service.build(request)
    records = dict(service.runs.stages(result.run_id, SCOPE))
    forged = service.stages.commit_operation(
        SCOPE,
        stage="retrieval",
        status="failed",
        dependencies=(records["publication"],),
        evidence_refs=(),
        graph_ref=service_module().graph_to_json(result.graph_ref),
        binding_ref=None,
        issues=("retrieval_build_failed",),
        coverage=StageCoverage("generations", 1, 0, 0, 0, 1, 0),
    )
    with sqlite3.connect(service.runs.path) as connection:
        connection.execute(
            "UPDATE build_stages SET artifact_ref = ?, record_digest = ? "
            "WHERE run_id = ? AND stage = 'retrieval'",
            (forged, _stage_digest(result.run_id, "retrieval", forged), result.run_id),
        )

    with pytest.raises(ValueError, match="result_retrieval_operation_mismatch"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
def test_terminal_resume_requires_retrieval_to_depend_on_recorded_publication(
    tmp_path: Path,
) -> None:
    """A valid retrieval operation cannot point at an unrecorded publication twin."""
    import sqlite3

    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build.workflow import _stage_digest

    service, request, _, _ = harness(tmp_path)
    result = service.build(request)
    records = dict(service.runs.stages(result.run_id, SCOPE))
    publication = service.stages.read(records["publication"], SCOPE)
    twin = service.stages.commit_operation(
        SCOPE,
        stage="publication",
        status="succeeded",
        dependencies=tuple(publication["dependencies"]),
        evidence_refs=("evidence:unrelated",),
        graph_ref=service_module().graph_to_json(result.graph_ref),
        binding_ref=None,
        issues=(),
        coverage=StageCoverage("tables", 17, 17, 17, 0, 0, 0),
    )
    forged = service.stages.commit_operation(
        SCOPE,
        stage="retrieval",
        status="skipped",
        dependencies=(twin,),
        evidence_refs=(),
        graph_ref=service_module().graph_to_json(result.graph_ref),
        binding_ref=None,
        issues=("retrieval_not_requested",),
        coverage=StageCoverage("generations", 0, 0, 0, 0, 0, 0),
    )
    with sqlite3.connect(service.runs.path) as connection:
        connection.execute(
            "UPDATE build_stages SET artifact_ref = ?, record_digest = ? "
            "WHERE run_id = ? AND stage = 'retrieval'",
            (forged, _stage_digest(result.run_id, "retrieval", forged), result.run_id),
        )

    with pytest.raises(ValueError, match="result_retrieval_operation_mismatch"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
def test_adding_one_document_replays_unchanged_extraction(tmp_path: Path) -> None:
    from capicove.graph_build.representations import text_representation

    service, request, _, generator = authored(tmp_path)
    service.build(request)
    generator.calls.clear()
    original = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    representation = text_representation("document:two", "Alice Python!", SCOPE)
    service.reader.representations[representation.document_ref] = representation
    changed = replace(
        original,
        document_ref=representation.document_ref,
        source_sha256=representation.source_sha256,
        representation_ref=representation.representation_ref,
        evidence_ref=representation.representation_ref,
    )
    current = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(original, changed),
        parent_refs=(),
        processing_fingerprint="e" * 64,
    )
    result = service.build(replace(request, input_snapshot_ref=current))
    assert result.status == "succeeded"
    assert generator.calls.count("user") == 1


@pytest.mark.slow
def test_replay_never_calls_provider_and_regenerate_has_distinct_runs(tmp_path: Path) -> None:
    service, request, catalog, generator = authored(tmp_path)
    cold = service.build(replace(request, mode="replay"))
    assert cold.status == "partial"
    assert generator.calls == []
    first = service.build(replace(request, mode="regenerate"))
    count = len(generator.calls)
    second = service.build(replace(request, mode="regenerate"))
    assert first.status == second.status == "succeeded"
    assert first.run_id != second.run_id
    assert len(generator.calls) == 2 * count
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_failed_report_keeps_graph_inspectable_without_semantic_repair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from capicove.graph_build.inference import RawGeneration

    service, request, catalog, generator = authored(tmp_path)
    original = generator.generate

    def generate(request: Any, *, control: Any) -> Any:
        if request.role == "report":
            value = json.loads(original(request, control=control).payload)
            value["task_key"] = "unrelated-task"
            return RawGeneration(json.dumps(value).encode(), "stop", 1, 1)
        return original(request, control=control)

    monkeypatch.setattr(generator, "generate", generate)
    result = service.build(request)
    assert result.status == "partial"
    assert generator.calls.count("report") == 1
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    assert (
        next(stage for stage in manifest.stages if stage.stage_id == "reports").status == "failed"
    )
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    assert len(dict(service.stages.output(refs["alignment"], SCOPE).rows)["resolved_entities"]) == 2


@pytest.mark.slow
@pytest.mark.parametrize("boundary", ["register", "finish"])
@pytest.mark.parametrize("after", [False, True])
def test_restart_around_graph_registration_and_result_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, boundary: str, after: bool
) -> None:
    import sqlite3

    service, request, catalog, generator = authored(tmp_path)
    target = catalog if boundary == "register" else service.runs
    original = getattr(target, boundary)

    class Crash(BaseException):
        pass

    def effect(*args: Any, **kwargs: Any) -> Any:
        if not after:
            raise Crash()
        original(*args, **kwargs)
        raise Crash() from None

    with monkeypatch.context() as patch:
        patch.setattr(target, boundary, effect)
        with pytest.raises(Crash):
            service.build(request)
    count = len(generator.calls)
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    result = service.resume(run_id, SCOPE)
    assert result.status == "succeeded"
    assert len(generator.calls) == count
    assert catalog.get(result.graph_ref.snapshot_id, SCOPE).completeness == "complete"
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_concurrent_same_scope_builds_have_distinct_runs_and_no_activation(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    service, request, catalog, _ = harness(tmp_path)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(service.build, request) for _ in range(2)]
        results = [future.result() for future in futures]
    assert all(result.status == "succeeded" for result in results)
    assert results[0].run_id != results[1].run_id
    assert results[0].graph_ref == results[1].graph_ref
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_terminal_result_coverage_must_match_registered_manifest(tmp_path: Path) -> None:
    import hashlib
    import sqlite3

    service, request, _, _ = harness(tmp_path)
    result = service.build(request)
    payload = service_module()._result_bytes(replace(result, coverage=()))
    with sqlite3.connect(service.runs.path) as connection:
        connection.execute(
            "UPDATE build_results SET payload=?, digest=? WHERE run_id=?",
            (payload, hashlib.sha256(payload).hexdigest(), result.run_id),
        )
    with pytest.raises(ValueError, match="build_terminal_commitment_failure"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
def test_resume_reconstructs_new_service_and_runner_instances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    service, request, _, _ = authored(tmp_path)
    original = service.runs.record_stage

    class Crash(BaseException):
        pass

    def record(run_id: str, scope: GraphScope, stage: str, artifact_ref: str) -> None:
        original(run_id, scope, stage, artifact_ref)
        if stage == "reports":
            raise Crash()

    monkeypatch.setattr(service.runs, "record_stage", record)
    with pytest.raises(Crash):
        service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    restarted, _, catalog, generator = authored(tmp_path)
    result = restarted.resume(run_id, SCOPE)
    assert result.status == "succeeded"
    assert generator.calls == []
    assert restarted.embedder.calls == 0
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.slow
def test_explicit_layout_exclusions_are_not_counted_as_extracted_codepoints(tmp_path: Path) -> None:
    service, request, _, generator = authored(tmp_path)
    document = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    rep = service.reader.representations[document.document_ref]
    rep = replace(
        rep, blocks=tuple(replace(block, exclusion_reason="layout") for block in rep.blocks)
    )
    service.reader.representations[document.document_ref] = rep
    document = replace(
        document, representation_ref=rep.representation_ref, evidence_ref=rep.representation_ref
    )
    current = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(document,),
        parent_refs=(),
        processing_fingerprint="f" * 64,
    )
    result = service.build(replace(request, input_snapshot_ref=current))
    stage = next(stage for stage in result.coverage if stage.stage_id == "text_units")
    assert stage.coverage.accepted == 0
    assert stage.coverage.omitted == 12
    assert generator.calls == []


@pytest.mark.slow
@pytest.mark.parametrize("reason", ["unauthorized", "incomplete", "incompatible"])
def test_pinned_baseline_is_rejected_before_new_effects(tmp_path: Path, reason: str) -> None:
    service, request, _, generator = authored(tmp_path)
    if reason == "incomplete":
        service.partitioner = None
    first = service.build(request)
    empty = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(),
        parent_refs=(),
        processing_fingerprint="e" * 64,
    )
    if reason == "unauthorized":
        service.gate.denied.add(
            service.inputs.documents(request.input_snapshot_ref, SCOPE)[0].evidence_ref
        )
    elif reason == "incompatible":
        service.profiles = (replace(service.profiles[0], report_prompt_version="report-v2"),)
    count = len(generator.calls)
    with pytest.raises((PermissionError, ValueError)):
        service.build(
            replace(
                request, input_snapshot_ref=empty, baseline_snapshot_id=first.graph_ref.snapshot_id
            )
        )
    assert len(generator.calls) == count


@pytest.mark.slow
def test_terminal_resume_rejects_policy_revision_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, _ = harness(tmp_path)
    result = service.build(request)
    original = catalog.get

    def get(*args: Any, **kwargs: Any) -> Any:
        manifest = original(*args, **kwargs)
        service.gate.revision += 1
        return manifest

    monkeypatch.setattr(catalog, "get", get)
    with pytest.raises(PermissionError):
        service.resume(result.run_id, SCOPE)


def test_runtime_drift_during_chunking_cannot_commit_new_stage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, _, generator = authored(tmp_path)
    original = service.tokenizer.count

    def count(text: str) -> int:
        service.tokenizer.fingerprint = "f" * 64
        return int(original(text))

    monkeypatch.setattr(service.tokenizer, "count", count)
    with pytest.raises(ValueError, match="runtime_identity_changed"):
        service.build(request)
    assert generator.calls == []


@pytest.mark.slow
def test_explicit_support_policy_uses_exact_source_occurrence_keys(tmp_path: Path) -> None:
    from capicove.contracts.graph_build import SourceSpan
    from capicove.graph_build.aggregation import source_occurrence_ref

    service, request, _, _ = authored(tmp_path)
    document = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    span = SourceSpan(document.evidence_ref, document.representation_ref, 0, 12, None, None)
    service.policy = replace(
        service.policy, support_groups=((source_occurrence_ref(span), "independent:event:1"),)
    )
    result = service.build(request)
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    assertions = dict(service.stages.output(refs["aggregation"], SCOPE).rows)[
        "relationship_assertions"
    ]
    assert assertions[0].support_group_id == "independent:event:1"


def test_evidence_labels_are_not_accepted_as_source_support_keys() -> None:
    with pytest.raises(ValueError):
        service_module().BuildPolicy("subject:1", support_groups=(("evidence:1", "group:1"),))


@pytest.mark.slow
@pytest.mark.parametrize("change", ["community", "extraction", "input"])
def test_qualification_seal_checks_exact_input_and_stage_semantics(
    tmp_path: Path, change: str
) -> None:
    from capicove.graph_build.profiles import graph_build_profile_fingerprint

    service, request, catalog, _ = authored(tmp_path)
    result = service.build(request)
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    original = dict(service.runs.stages(result.run_id, SCOPE))["extraction"]
    tables = tuple(
        table
        for table in manifest.tables
        if table.role in {"entity_mentions", "relationship_mentions"}
    )
    profile = service.profiles[0]
    if change == "community":
        profile = replace(
            profile, community=replace(profile.community, max_size=profile.community.max_size + 1)
        )
    elif change == "extraction":
        profile = replace(profile, extraction_prompt_version="extract-v2")
    input_ref = (
        replace(request.input_snapshot_ref, version=request.input_snapshot_ref.version + 1)
        if change == "input"
        else request.input_snapshot_ref
    )

    def seal() -> Any:
        return service.stages.seal(
            original,
            SCOPE,
            graph_build_profile_fingerprint(profile),
            tables,
            input_snapshot_ref=input_ref,
            runtime_payload=service._runtime_payload(profile),
        )

    if change == "community":
        value = seal()
        assert (
            service.stages.validate(
                value, SCOPE, graph_build_profile_fingerprint(profile)
            ).input_snapshot_ref
            == request.input_snapshot_ref
        )
    else:
        with pytest.raises(ValueError):
            seal()


@pytest.mark.slow
def test_trusted_counter_resolver_binds_disputed_mentions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from capicove.graph_build.inference import RawGeneration
    from tests.unit.graph_build.test_extraction import response

    service, request, _, generator = authored(tmp_path)
    original = generator.generate

    def generate(request: Any, *, control: Any) -> Any:
        if request.role != "user":
            return original(request, control=control)
        value = response()
        value["relationships"].append(
            {
                **value["relationships"][0],
                "local_id": "dispute",
                "epistemic_status": "disputed",
                "polarity": "positive",
            }
        )
        return RawGeneration(json.dumps(value).encode(), "stop", 20, 20)

    class CounterResolver:
        fingerprint = "a" * 64

        def resolve(self, mentions: Any) -> Any:
            explicit = next(item for item in mentions if item.epistemic_status == "explicit")
            disputed = next(item for item in mentions if item.epistemic_status == "disputed")
            return {disputed.mention_id: (explicit.mention_id,)}

    service.counter_evidence = CounterResolver()
    monkeypatch.setattr(generator, "generate", generate)
    result = service.build(request)
    assert result.status == "succeeded", result.issue_codes
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    assertions = dict(service.stages.output(refs["aggregation"], SCOPE).rows)[
        "relationship_assertions"
    ]
    assert len(assertions) == 2
    assert next(
        item for item in assertions if item.epistemic_status == "disputed"
    ).counter_assertion_ids


@pytest.mark.slow
@pytest.mark.parametrize("selection", ["one", "ambiguous", "alien"])
def test_explicit_trusted_subject_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection: str
) -> None:
    service, request, _, _ = authored(tmp_path)
    owners = []
    original = service_module().community_projection

    class Resolver:
        fingerprint = "a" * 64

        def resolve(self, subject_ref: str, entities: Any, scope: GraphScope) -> Any:
            assert subject_ref == "subject:explicit" and scope == SCOPE
            people = tuple(item.entity_id for item in entities if item.entity_type == "PERSON")
            if selection == "alien":
                return ("alien:person",)
            if selection == "ambiguous":
                return (*people, *people)
            return people

    def project(assertions: Any, entities: Any, owner: Any, profile: Any) -> Any:
        owners.append(owner)
        return original(assertions, entities, owner, profile)

    service.subject_resolver = Resolver()
    monkeypatch.setattr(service_module(), "community_projection", project)
    result = service.build(request)
    if selection == "one":
        assert result.status == "succeeded"
        assert owners and owners[0] is not None
    else:
        assert result.status == "partial"
        assert owners == []


def identity_authors(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Any, str]:
    """Keep author attestations independent of ordinary endpoint citations."""
    import json

    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build.inference import RawGeneration
    from capicove.graph_build.representations import text_representation

    service, request, _, generator = authored(tmp_path)
    original_document = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    documents = [original_document]
    for document_ref, text in (
        ("document:two", "Alice Python!"),
        ("document:identity", "Identity attestation for two authored documents."),
    ):
        representation = text_representation(document_ref, text, SCOPE)
        service.reader.representations[document_ref] = representation
        documents.append(
            replace(
                original_document,
                document_ref=document_ref,
                source_sha256=representation.source_sha256,
                representation_ref=representation.representation_ref,
                evidence_ref=representation.representation_ref,
                coverage=StageCoverage("codepoints", len(text), len(text), len(text), 0, 0, 0),
            )
        )
    input_ref = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=tuple(documents),
        parent_refs=(),
        processing_fingerprint="f" * 64,
    )
    generate = generator.generate

    def generate_document(request: Any, *, control: Any) -> Any:
        if request.role == "user" and any(
            "Identity attestation" in segment[2] for segment in request.input_segments
        ):
            return RawGeneration(
                json.dumps({"entities": [], "relationships": []}).encode(), "stop", 1, 1
            )
        return generate(request, control=control)

    monkeypatch.setattr(generator, "generate", generate_document)
    return service, replace(request, input_snapshot_ref=input_ref), documents[-1].evidence_ref


class AuthorIdentityEvidence:
    fingerprint = "a" * 64

    def __init__(self, source: str, same_author: bool = True) -> None:
        self.source = source
        self.same_author = same_author

    def resolve(self, entities: Any, mentions: Any, documents: Any, scope: Any) -> Any:
        assert scope == SCOPE
        by_mention = {mention.mention_id: mention for mention in mentions}
        document_evidence = {document.evidence_ref for document in documents}
        people = []
        for entity in entities:
            mention = by_mention[entity.mention_ids[0]]
            assert entity.scope == scope
            assert all(span.evidence_ref in document_evidence for span in mention.source_spans)
            if entity.entity_type == "PERSON":
                people.append(entity.entity_id)
        assert len(people) == 2
        return service_module().IdentityEvidence(
            aliases=((*sorted(people), self.source),) if self.same_author else ()
        )


@pytest.mark.slow
@pytest.mark.parametrize("same_author", [None, False, True])
def test_explicit_author_identity_context_controls_conservative_merging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_author: bool | None
) -> None:
    service, request, evidence = identity_authors(tmp_path, monkeypatch)
    service.identity_evidence = (
        AuthorIdentityEvidence(evidence, same_author) if same_author is not None else None
    )
    result = service.build(request)
    assert result.status == "succeeded", result.issue_codes
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    entities = dict(service.stages.output(refs["resolution"], SCOPE).rows)["entities"]
    assert sum(entity.entity_type == "PERSON" for entity in entities) == (1 if same_author else 2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "malformation",
    ["alien", "self", "reversed", "duplicate", "unsorted", "foreign_source", "mutable"],
)
def test_identity_context_rejects_noncanonical_or_foreign_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformation: str
) -> None:
    service, request, evidence = identity_authors(tmp_path, monkeypatch)

    class InvalidEvidence(AuthorIdentityEvidence):
        def resolve(self, entities: Any, mentions: Any, documents: Any, scope: Any) -> Any:
            ((left, right, source),) = super().resolve(entities, mentions, documents, scope).aliases
            cases = {
                "alien": ((left, "alien:entity", source),),
                "self": ((left, left, source),),
                "reversed": ((right, left, source),),
                "duplicate": ((left, right, source),) * 2,
                "unsorted": tuple(
                    sorted(
                        ((left, right, document.evidence_ref) for document in documents),
                        reverse=True,
                    )
                ),
                "foreign_source": ((left, right, "evidence:foreign"),),
                "mutable": [(left, right, source)],
            }
            return service_module().IdentityEvidence(aliases=cases[malformation])

    service.identity_evidence = InvalidEvidence(evidence)
    with pytest.raises(ValueError, match="identity_evidence"):
        service.build(request)
    assert service.embedder.calls == 0


@pytest.mark.slow
def test_identity_context_fingerprint_is_pinned_on_resume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, evidence = identity_authors(tmp_path, monkeypatch)
    resolver = AuthorIdentityEvidence(evidence)
    service.identity_evidence = resolver
    result = service.build(request)
    resolver.fingerprint = "b" * 64
    with pytest.raises(ValueError, match="runtime_identity_changed"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
@pytest.mark.parametrize("drift", ["runtime", "revision", "authorization"])
def test_identity_context_checks_authority_and_runtime_after_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: str
) -> None:
    service, request, evidence = identity_authors(tmp_path, monkeypatch)

    class DriftingEvidence(AuthorIdentityEvidence):
        def resolve(self, entities: Any, mentions: Any, documents: Any, scope: Any) -> Any:
            aliases = super().resolve(entities, mentions, documents, scope)
            if drift == "runtime":
                self.fingerprint = "b" * 64
            elif drift == "revision":
                service.gate.revision += 1
            else:
                service.gate.denied.add(self.source)
            return aliases

    service.identity_evidence = DriftingEvidence(evidence)
    with pytest.raises((ValueError, PermissionError)):
        service.build(request)
    assert service.embedder.calls == 0


def test_identity_evidence_resolver_is_a_public_startup_port() -> None:
    import capicove.graph_build as graph

    assert (
        getattr(graph, "IdentityEvidenceResolver", None)
        is service_module().IdentityEvidenceResolver
    )
    assert getattr(graph, "IdentityEvidence", None) is service_module().IdentityEvidence


def test_public_identity_port_preserves_lazy_optional_runtime_imports() -> None:
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import capicove.graph_build as graph; import sys; assert 'capicove.graph_build.service' not in sys.modules; assert graph.IdentityEvidenceResolver.__name__ == 'IdentityEvidenceResolver'; assert graph.IdentityEvidence.__name__ == 'IdentityEvidence'; assert not {'torch', 'transformers', 'sentence_transformers', 'pyarrow', 'networkx', 'igraph'} & sys.modules.keys()",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("fingerprint", ["", "a" * 63, "G" * 64, True])
def test_identity_context_requires_sha256_runtime_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fingerprint: Any
) -> None:
    service, request, evidence = identity_authors(tmp_path, monkeypatch)
    resolver = AuthorIdentityEvidence(evidence)
    resolver.fingerprint = fingerprint
    service.identity_evidence = resolver
    with pytest.raises(ValueError, match="resolver_fingerprint"):
        service.build(request)
    assert service.embedder.calls == 0


@pytest.mark.slow
def test_endpoint_citation_cannot_masquerade_as_independent_identity_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, _ = identity_authors(tmp_path, monkeypatch)
    endpoint_evidence = next(
        document.evidence_ref
        for document in service.inputs.documents(request.input_snapshot_ref, SCOPE)
        if document.document_ref == "document:one"
    )
    service.identity_evidence = AuthorIdentityEvidence(endpoint_evidence)
    result = service.build(request)
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    entities = dict(service.stages.output(refs["resolution"], SCOPE).rows)["entities"]
    assert sum(entity.entity_type == "PERSON" for entity in entities) == 2


def subject_identity_authors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: str | None = None
) -> tuple[Any, Any]:
    import json
    from itertools import combinations

    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build.inference import RawGeneration
    from capicove.graph_build.representations import text_representation
    from tests.unit.graph_build.test_extraction import response, span

    service, request, _, generator = authored(tmp_path)
    original = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    documents = []
    owners = ("document:owner1", "document:owner2", "document:owner3")
    for ref in (*owners, "document:book"):
        text = "Bobby pattern" if ref == "document:book" else "Alice pattern"
        rep = text_representation(ref, text, SCOPE)
        service.reader.representations[ref] = rep
        documents.append(
            replace(
                original,
                document_ref=ref,
                source_sha256=rep.source_sha256,
                representation_ref=rep.representation_ref,
                evidence_ref=rep.representation_ref,
                coverage=StageCoverage("codepoints", 13, 13, 13, 0, 0, 0),
            )
        )
    input_ref = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=tuple(documents),
        parent_refs=(),
        processing_fingerprint="f" * 64,
    )
    generate = generator.generate

    def generate_mentions(request: Any, *, control: Any) -> Any:
        if request.role != "user":
            return generate(request, control=control)
        person = (
            "Bobby" if any("Bobby" in segment[2] for segment in request.input_segments) else "Alice"
        )
        value = response()
        value["entities"][0].update(name=person, source_spans=[span(0, 5, person)])
        value["entities"][1].update(
            local_id="error",
            name="pattern",
            entity_type="ERROR_PATTERN",
            source_spans=[span(6, 13, "pattern")],
        )
        value["relationships"] = []
        return RawGeneration(json.dumps(value).encode(), "stop", 20, 20)

    class BoundEvidence:
        fingerprint = "d" * 64

        def resolve(self, entities: Any, mentions: Any, documents: Any, scope: Any) -> Any:
            dto = getattr(service_module(), "IdentityEvidence", None)
            assert dto is not None, "Missing closed subject-binding IdentityEvidence"
            by_mention = {mention.mention_id: mention for mention in mentions}
            by_rep = {document.representation_ref: document for document in documents}
            by_document: dict[str, dict[str, str]] = {}
            for entity in entities:
                document = by_rep[
                    by_mention[entity.mention_ids[0]].source_spans[0].representation_ref
                ]
                by_document.setdefault(document.document_ref, {})[entity.entity_type] = (
                    entity.entity_id
                )
            aliases = []
            for kind in ("PERSON", "ERROR_PATTERN"):
                for left, right in combinations(owners, 2):
                    source = next(
                        document.evidence_ref
                        for document in documents
                        if document.document_ref in owners
                        and document.document_ref not in {left, right}
                    )
                    if malformed == "unmerged" and kind == "PERSON":
                        source = next(
                            document.evidence_ref
                            for document in documents
                            if document.document_ref == left
                        )
                    aliases.append(
                        (*sorted((by_document[left][kind], by_document[right][kind])), source)
                    )
            bindings = []
            for index, document_ref in enumerate(owners):
                source = next(
                    document.evidence_ref
                    for document in documents
                    if document.document_ref == owners[(index + 1) % 3]
                )
                bindings.append(
                    (
                        by_document[document_ref]["ERROR_PATTERN"],
                        by_document[document_ref]["PERSON"],
                        source,
                    )
                )
            if malformed == "ambiguous":
                bindings.append(
                    (bindings[0][0], by_document["document:book"]["PERSON"], bindings[0][2])
                )
            elif malformed == "foreign":
                bindings[0] = (bindings[0][0], "alien:person", bindings[0][2])
            elif malformed == "wrong_type":
                bindings[0] = (bindings[0][1], bindings[0][0], bindings[0][2])
            elif malformed == "duplicate":
                bindings.append(bindings[0])
            elif malformed == "endpoint_only":
                bindings[0] = (
                    *bindings[0][:2],
                    next(
                        document.evidence_ref
                        for document in documents
                        if document.document_ref == owners[0]
                    ),
                )
            elif malformed == "runtime":
                self.fingerprint = "e" * 64
            result = tuple(sorted(bindings))
            if malformed == "unsorted":
                result = tuple(reversed(result))
            return dto(aliases=tuple(sorted(aliases)), subject_bindings=result)

    service.identity_evidence = BoundEvidence()
    monkeypatch.setattr(generator, "generate", generate_mentions)
    return service, replace(request, input_snapshot_ref=input_ref)


@pytest.mark.slow
def test_subject_bindings_merge_diagnostics_to_one_resolved_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request = subject_identity_authors(tmp_path, monkeypatch)
    result = service.build(request)
    assert result.status == "succeeded", result.issue_codes
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    entities = dict(service.stages.output(refs["resolution"], SCOPE).rows)["entities"]
    owners = [
        entity
        for entity in entities
        if entity.entity_type == "PERSON" and len(entity.mention_ids) == 3
    ]
    errors = [
        entity
        for entity in entities
        if entity.entity_type == "ERROR_PATTERN" and len(entity.mention_ids) == 3
    ]
    assert len(owners) == len(errors) == 1
    assert errors[0].subject_ref == owners[0].entity_id
    assert any(
        key.namespace == "capicove:subject_ref"
        and key.value == owners[0].entity_id
        and not key.authoritative
        for key in errors[0].identity_keys
    )
    assert sum(entity.entity_type == "PERSON" for entity in entities) == 2
    raw_mentions = dict(service.stages.output(refs["extraction"], SCOPE).rows)["entity_mentions"]
    assert all(
        key.namespace != "capicove:subject_ref"
        for mention in raw_mentions
        for key in mention.identity_keys
    )
    assert service.resume(result.run_id, SCOPE) == result


@pytest.mark.slow
@pytest.mark.parametrize(
    "malformed",
    [
        "ambiguous",
        "unmerged",
        "foreign",
        "wrong_type",
        "duplicate",
        "unsorted",
        "endpoint_only",
        "runtime",
    ],
)
def test_subject_bindings_fail_closed_on_ambiguous_or_untrusted_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, malformed: str
) -> None:
    service, request = subject_identity_authors(tmp_path, monkeypatch, malformed)
    with pytest.raises(ValueError):
        service.build(request)


def test_identity_evidence_result_is_closed_and_frozen() -> None:
    from dataclasses import FrozenInstanceError

    evidence = service_module().IdentityEvidence()
    with pytest.raises(FrozenInstanceError):
        evidence.subject_bindings = ()
    with pytest.raises(TypeError):
        service_module().IdentityEvidence(unknown=())


def test_source_provenance_is_a_public_closed_startup_port() -> None:
    from dataclasses import FrozenInstanceError

    import capicove.graph_build as graph

    row = graph.DocumentSourceProvenance("document:one", "generated", "report:one")
    assert row.derivation_ref == "report:one"
    assert graph.SourceProvenanceResolver is service_module().SourceProvenanceResolver
    with pytest.raises(FrozenInstanceError):
        row.source_kind = "source"  # type: ignore[misc]
    with pytest.raises(ValueError):
        graph.DocumentSourceProvenance("document:one", "source", "report:one")
    with pytest.raises(ValueError):
        graph.DocumentSourceProvenance("document:one", "generated", None)


@pytest.mark.parametrize(
    "malformation",
    ["missing", "duplicate", "foreign", "unsorted", "mutable", "invalid_kind", "drift"],
)
def test_source_provenance_classification_fails_closed(tmp_path: Path, malformation: str) -> None:
    service, request, _, _ = authored(tmp_path)

    class InvalidProvenance(SourceProvenance):
        def classify(self, documents: Any, scope: GraphScope) -> Any:
            rows = super().classify(documents, scope)
            row = rows[0]
            cases = {
                "missing": (),
                "duplicate": (row, row),
                "foreign": (
                    service_module().DocumentSourceProvenance("document:foreign", "source", None),
                ),
                "unsorted": (
                    service_module().DocumentSourceProvenance("z", "source", None),
                    row,
                ),
                "mutable": list(rows),
                "invalid_kind": object(),
            }
            if malformation == "drift":
                self.calls = getattr(self, "calls", 0) + 1
                if self.calls > 1:
                    return (
                        service_module().DocumentSourceProvenance(
                            row.document_ref, "generated", "report:changed"
                        ),
                    )
                return rows
            if malformation == "invalid_kind":
                return cases[malformation]
            return cases[malformation]

    service.source_provenance = InvalidProvenance()
    with pytest.raises((TypeError, ValueError), match="source_provenance"):
        service.build(request)


@pytest.mark.slow
def test_source_provenance_fingerprint_is_pinned_on_resume(tmp_path: Path) -> None:
    service, request, _, _ = authored(tmp_path)
    result = service.build(request)
    service.source_provenance.fingerprint = "8" * 64
    with pytest.raises(ValueError, match="runtime_identity_changed"):
        service.resume(result.run_id, SCOPE)


def test_nonempty_build_requires_explicit_source_provenance(tmp_path: Path) -> None:
    service, request, _, _ = authored(tmp_path)
    service.source_provenance = None
    with pytest.raises(ValueError, match="source_provenance_resolver_required"):
        service.build(request)


@pytest.mark.parametrize("drift", ["revision", "authorization", "fingerprint"])
def test_source_provenance_rechecks_authority_and_runtime_after_classification(
    tmp_path: Path, drift: str
) -> None:
    service, request, _, _ = authored(tmp_path)
    evidence = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0].evidence_ref

    class DriftingProvenance(SourceProvenance):
        def classify(self, documents: Any, scope: GraphScope) -> tuple[Any, ...]:
            rows = super().classify(documents, scope)
            if drift == "revision":
                service.gate.revision += 1
            elif drift == "authorization":
                service.gate.denied.add(evidence)
            else:
                self.fingerprint = "8" * 64
            return rows

    service.source_provenance = DriftingProvenance()
    with pytest.raises(PermissionError):
        service.build(request)


@pytest.mark.slow
def test_exact_source_classification_is_pinned_even_when_fingerprint_is_unchanged(
    tmp_path: Path,
) -> None:
    service, request, _, _ = authored(tmp_path)

    class MutableClassification(SourceProvenance):
        kind = "source"

        def classify(self, documents: Any, scope: GraphScope) -> tuple[Any, ...]:
            return tuple(
                service_module().DocumentSourceProvenance(
                    document.document_ref,
                    self.kind,
                    "report:changed" if self.kind == "generated" else None,
                )
                for document in documents
            )

    resolver = MutableClassification()
    service.source_provenance = resolver
    result = service.build(request)
    resolver.kind = "generated"
    with pytest.raises(ValueError, match="source_provenance_identity_changed"):
        service.resume(result.run_id, SCOPE)


@pytest.mark.slow
def test_generated_report_text_reingestion_cannot_become_source_evidence(
    tmp_path: Path,
) -> None:
    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build.representations import text_representation

    service, request, _, _ = authored(tmp_path)
    source_result = service.build(request)
    source_refs = dict(service.runs.stages(source_result.run_id, SCOPE))
    reports = dict(service.stages.output(source_refs["reports"], SCOPE).rows)["community_reports"]
    assert reports
    report = reports[0]
    report_text = "Alice Python\nGenerated report: " + report.title
    representation = text_representation("document:generated-report", report_text, SCOPE)
    service.reader.representations[representation.document_ref] = representation
    original = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0]
    document = replace(
        original,
        document_ref=representation.document_ref,
        source_sha256=representation.source_sha256,
        representation_ref=representation.representation_ref,
        coverage=StageCoverage(
            "codepoints", len(report_text), len(report_text), len(report_text), 0, 0, 0
        ),
    )
    input_ref = publish_table(
        service.workspace,
        scope=SCOPE,
        role="documents",
        rows=(original, document),
        parent_refs=(request.input_snapshot_ref,),
        processing_fingerprint="7" * 64,
    )

    class MixedReportProvenance:
        fingerprint = "6" * 64

        def classify(self, documents: Any, scope: GraphScope) -> tuple[Any, ...]:
            assert scope == SCOPE
            return tuple(
                service_module().DocumentSourceProvenance(
                    document.document_ref,
                    "generated"
                    if document.document_ref == representation.document_ref
                    else "source",
                    report.report_id
                    if document.document_ref == representation.document_ref
                    else None,
                )
                for document in documents
            )

    service.source_provenance = MixedReportProvenance()
    result = service.build(replace(request, input_snapshot_ref=input_ref))
    assert result.status == "succeeded", result.issue_codes
    refs = dict(service.runs.stages(result.run_id, SCOPE))
    assertions = dict(service.stages.output(refs["aggregation"], SCOPE).rows)[
        "relationship_assertions"
    ]
    assert {row.derivation_ref for row in assertions} == {None, report.report_id}
    diagnostic_metadata = service.stages.output(refs["diagnostics"], SCOPE).metadata
    assert importlib.import_module("json").loads(diagnostic_metadata)["bundles"] == []
    assert service.stages.output(refs["communities"], SCOPE).status == "succeeded"
    assert service.stages.output(refs["reports"], SCOPE).status == "succeeded"
