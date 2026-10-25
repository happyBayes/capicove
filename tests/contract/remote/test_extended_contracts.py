"""Extended remote schemas preserve canonical evidence and Book task variants."""

import importlib
import json
from pathlib import Path
from typing import Any, cast


ROOT = Path(__file__).resolve().parents[3]


def manifest() -> dict[str, Any]:
    path = ROOT / "docs/development/remote-mcp-contracts-extended.json"
    assert path.is_file(), "extended remote contracts are missing"
    return cast(dict[str, Any], json.loads(path.read_text()))


def test_document_payload_accepts_real_canonical_codec() -> None:
    from capicove.contracts import (
        DocumentPage,
        DocumentPageStatus,
        DocumentParseResult,
        DocumentParserReceipt,
    )
    from capicove.documents.codec import document_parse_result_to_json

    document = DocumentParseResult(
        "a" * 64,
        "b" * 64,
        (DocumentPage(0, None, 600, 800, DocumentPageStatus.SUCCEEDED, ()),),
        DocumentParserReceipt("fixture", "1", "fixture", 1, "c" * 64, "d" * 64, "e" * 64),
    )
    schemas = manifest()["payload_schemas"]
    schema = schemas["capicove.document.parse-result.v1"]
    importlib.import_module("jsonschema").validate(document_parse_result_to_json(document), schema)


def test_book_status_retains_every_canonical_result_field() -> None:
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    schema = operations["capicove.book.status"]["result_schema"]
    assert set(schema["properties"]) == {
        "book_id",
        "completed_tasks",
        "pending_tasks",
        "publishable",
        "structure_review_tasks",
        "total_tasks",
    }
    assert schema["additionalProperties"] is False


def test_book_summary_and_structure_submissions_remain_distinct() -> None:
    schemas = manifest()["payload_schemas"]
    summary = schemas["capicove.book.summary-submission.v1"]
    structure = schemas["capicove.book.structure-review-submission.v1"]
    assert "claims" in summary["required"]
    assert "operation" in structure["required"]
    assert "selected_block_ids" in structure["required"]


def test_optional_book_provider_provenance_round_trips_real_codec() -> None:
    from capicove.books.codec import summary_submission_to_json
    from capicove.books.tasks import AtomicClaim, SummarySubmission

    value = SummarySubmission(
        "task",
        "a" * 64,
        "Abstract",
        ("Point",),
        (AtomicClaim("claim:one", "Claim", ("citation",)),),
    )
    importlib.import_module("jsonschema").validate(
        summary_submission_to_json(value),
        manifest()["payload_schemas"]["capicove.book.summary-submission.v1"],
    )


def test_summary_schema_rejects_invalid_canonical_evidence() -> None:
    from copy import deepcopy

    import pytest

    from capicove.books.codec import summary_submission_to_json
    from capicove.books.tasks import AtomicClaim, SummarySubmission

    codec = summary_submission_to_json(
        SummarySubmission(
            "task",
            "a" * 64,
            "Abstract",
            ("Point",),
            (AtomicClaim("claim:one", "Claim", ("citation",)),),
        )
    )
    validator = importlib.import_module("jsonschema")
    schema = manifest()["payload_schemas"]["capicove.book.summary-submission.v1"]
    invalid: list[dict[str, Any]] = [
        dict(idempotency_key="invalid"),
        dict(abstract=" "),
        dict(key_points=[]),
        dict(claims=[]),
    ]
    for change in invalid:
        with pytest.raises(validator.ValidationError):
            validator.validate({**codec, **change}, schema)
    for change in (
        {"claim_id": "invalid"},
        {"citation_ids": []},
        {"citation_ids": ["same", "same"]},
    ):
        value = cast(dict[str, Any], deepcopy(codec))
        value["claims"][0].update(change)
        with pytest.raises(validator.ValidationError):
            validator.validate(value, schema)


def test_chunk_submission_contract_preserves_canonical_empty_summaries() -> None:
    from capicove.books.chunk_result_submission import validate_submission

    value = dict(
        schema_version=1,
        book_id="book",
        run_id="run",
        task_id="task",
        idempotency_key="caller-key",
        dependency_result_ids=[],
        abstract="No supported claims",
        key_points=[],
        claims=[],
        provenance={"caller_run_id": "fixture"},
    )
    canonical = validate_submission(value)
    schema = manifest()["payload_schemas"]["capicove.book.chunk-result-submission.v1"]
    importlib.import_module("jsonschema").validate(canonical, schema)


def test_chunk_task_and_status_schemas_preserve_real_service_results(tmp_path: Path) -> None:
    from capicove.books import BookChunkResultService
    from tests.unit.tools.test_book_chunk_results import _prepared

    workspace, book_id, run_id = _prepared(tmp_path)
    service = BookChunkResultService(workspace)
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    validator = importlib.import_module("jsonschema")
    for name, result in (
        ("chunk-tasks", service.list_tasks(book_id, run_id)),
        ("chunk-status", service.status(book_id, run_id)),
    ):
        validator.validate(result, operations["capicove.book." + name]["result_schema"])


def test_chunk_task_contract_includes_bounded_reduction_groups() -> None:
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    types = operations["capicove.book.chunk-tasks"]["result_schema"]["properties"]["items"][
        "items"
    ]["properties"]["task_type"]["enum"]
    assert "reduce_group" in types


def test_chunk_read_contracts_preserve_source_spans_and_context_policy(tmp_path: Path) -> None:
    from capicove.books.chunk_service import BookChunkService
    from tests.unit.tools.test_book_chunk_results import _prepared

    workspace, book_id, run_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    chunks = service.read_chunks(book_id, run_id)
    context = service.read_context(book_id, run_id, chunks["items"][0]["chunk_id"])
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    validator = importlib.import_module("jsonschema")
    for name, result in (
        ("chunks", chunks),
        ("chunks", service.read_chunks(book_id, run_id, tasks=True)),
        ("chunk-context", context),
    ):
        validator.validate(
            json.loads(json.dumps(result)), operations["capicove.book." + name]["result_schema"]
        )


def test_extended_contract_fingerprints_match_reviewed_sources_and_schemas() -> None:
    import hashlib

    data = manifest()
    for path, expected in data["source_manifest"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected
    for operation in data["operations"]:
        for key in ("parameters_schema", "result_schema"):
            raw = json.dumps(operation[key], sort_keys=True, separators=(",", ":")).encode()
            assert hashlib.sha256(raw).hexdigest() == operation[key + "_sha256"]
    for name, schema in data["payload_schemas"].items():
        raw = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
        assert hashlib.sha256(raw).hexdigest() == data["payload_schema_sha256"][name]


def test_wiki_read_contracts_preserve_public_service_output(tmp_path: Path) -> None:
    from datetime import date

    from capicove.cli.commands.wiki import _inventory, _lint, _search
    from capicove.wiki import WikiService

    service = WikiService(tmp_path)
    inventory = service.initialize(created_on=date(2026, 10, 23))
    source = tmp_path / "source.md"
    source.write_text("# Fixture\n\nA source fact about astronomy.\n")
    service.ingest(source, ingested_on=date(2026, 10, 23))
    service.build(built_on=date(2026, 10, 23))
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    validator = importlib.import_module("jsonschema")
    for name, value in (
        ("initialize", _inventory(inventory)),
        ("scan", _inventory(service.scan())),
        ("lint", _lint(service.lint())),
        ("query", _search(service.query("astronomy"))),
    ):
        validator.validate(value, operations["capicove.wiki." + name]["result_schema"])


def test_graph_build_contract_preserves_partial_stage_coverage() -> None:
    from dataclasses import asdict

    from capicove.contracts.graph_build import StageCoverage, StageResult
    from capicove.graph_build.service import GraphBuildResult

    result = GraphBuildResult(
        "run",
        "partial",
        None,
        "partial",
        (
            StageResult(
                "extraction",
                "partial",
                (),
                ("provider_unavailable",),
                StageCoverage("documents", 3, 1, 1, 0, 1, 1),
            ),
        ),
        ("provider_unavailable",),
        None,
    )
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    schema = operations["capicove.graph.build"]["result_schema"]
    importlib.import_module("jsonschema").validate(json.loads(json.dumps(asdict(result))), schema)


def test_extended_source_partition_cannot_hide_unmapped_capabilities() -> None:
    data = manifest()
    inventory = json.loads((ROOT / "docs/development/remote-mcp-operations.json").read_text())
    expected = {
        (row["operation_id"], row["provider_profile"])
        for row in inventory["operations"]
        if row["capability_id"] in data["scope"]
    }
    mapped = {
        (row["candidate_operation_id"], row["candidate_provider_profile"])
        for row in data["source_mappings"]
    }
    unresolved = {
        (row["operation_id"], row["provider_profile"]) for row in data["unresolved_candidates"]
    }
    assert mapped.isdisjoint(unresolved)
    assert mapped | unresolved == expected
    assert len(mapped) == len(data["source_mappings"])
    operations = {row["operation_id"] for row in data["operations"]}
    for row in data["source_mappings"]:
        assert row["enclosing_operations"]
        assert set(row["enclosing_operations"]) <= operations
    assert data["runtime_qualification"] == "not_run"


def test_graph_success_retains_manifest_integrity_digest() -> None:
    from dataclasses import asdict

    from capicove.contracts.graph_build import GraphSnapshotRef
    from capicove.graph_build.service import GraphBuildResult

    result = GraphBuildResult(
        "run",
        "succeeded",
        GraphSnapshotRef("graph:fixture", "a" * 64),
        "complete",
        (),
        (),
        "binding",
    )
    operation = next(
        row for row in manifest()["operations"] if row["operation_id"] == "capicove.graph.build"
    )
    importlib.import_module("jsonschema").validate(
        json.loads(json.dumps(asdict(result))), operation["result_schema"]
    )


def test_operator_projections_preserve_every_canonical_result_field() -> None:
    from capicove.tools.operator_models import MODEL_SPECS
    from capicove.tools.operator_traces import TRACE_SPECS

    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    for name, spec in {**MODEL_SPECS, **TRACE_SPECS}.items():
        projected = operations[name]["result_schema"]
        canonical = dict(spec.output_schema)
        if "properties" in canonical:
            additions = {
                "capicove.trace.export": {"manifest_ref"},
                "capicove.trace.validate": {"report_ref"},
            }.get(name, set())
            assert set(projected["properties"]) == (
                set(cast(dict[str, Any], canonical["properties"])) | additions
            )
        else:
            assert len(projected["anyOf"]) == len(cast(list[Any], canonical["anyOf"]))


def test_chunk_create_preview_preserve_actual_service_reports(tmp_path: Path) -> None:
    from capicove.books.chunk_config import resolve_chunk_config
    from capicove.books.chunk_service import BookChunkService
    from tests.integration.books.test_chunk_service import _prepared

    workspace, book_id = _prepared(tmp_path)
    service = BookChunkService(workspace)
    first = service.create(book_id, resolve_chunk_config())
    preview = service.create(
        book_id,
        resolve_chunk_config(overrides={"target_core_tokens": 700}),
        dry_run=True,
        compare_run=first["run_id"],
    )
    operations = {op["operation_id"]: op for op in manifest()["operations"]}
    for name, payload in (("chunk", first), ("chunk-preview", preview)):
        importlib.import_module("jsonschema").validate(
            json.loads(json.dumps(payload)), operations["capicove.book." + name]["result_schema"]
        )


def test_chunk_publication_contracts_preserve_actual_dag_and_wiki_receipts(tmp_path: Path) -> None:
    from dataclasses import asdict
    from datetime import date

    from capicove.books import BookChunkResultService
    from capicove.books.chunk_config import resolve_chunk_config
    from capicove.books.chunk_service import BookChunkService
    from capicove.wiki import WikiService, migrate_to_v2
    from tests.integration.books.test_chunk_result_publication import _submit_claim
    from tests.integration.books.test_chunk_service import _prepared

    workspace, book_id = _prepared(tmp_path)
    chunks = BookChunkService(workspace)
    run_id = str(chunks.create(book_id, resolve_chunk_config())["run_id"])
    service = BookChunkResultService(workspace)
    tasks = service.list_tasks(book_id, run_id)["items"]
    run, _, _ = chunks._load(book_id, run_id)
    by_chunk = {chunk.chunk_id: chunk for chunk in run.chunks}
    accepted: dict[str, dict[str, Any]] = {}
    citations: dict[str, dict[str, Any]] = {}
    for number, task in enumerate(tasks):
        dependencies = task["dependency_task_ids"]
        citation = (
            asdict(by_chunk[task["chunk_ids"][0]].core_spans[0])
            if task["task_type"] == "chunk_summary"
            else citations[dependencies[0]]
        )
        accepted[task["task_id"]] = _submit_claim(
            service,
            book_id,
            run_id,
            task,
            citation,
            f"contract-{number}",
            [accepted[item]["result_id"] for item in dependencies],
        )
        citations[task["task_id"]] = citation
    wiki = WikiService(workspace.root)
    wiki.initialize(created_on=date(2026, 9, 1))
    migrate_to_v2(wiki.wiki_path, preview=False)
    operations = {op["operation_id"]: op for op in manifest()["operations"]}
    for name, payload in (
        ("chunk-publication-manifest", service.publication_manifest(book_id, run_id)),
        ("chunk-publish", service.publish(book_id, run_id, wiki, published_on=date(2026, 9, 2))),
    ):
        importlib.import_module("jsonschema").validate(
            payload, operations["capicove.book." + name]["result_schema"]
        )


def test_book_task_lists_preserve_actual_service_codec(tmp_path: Path) -> None:
    from capicove.books import BookService
    from capicove.books.codec import structure_review_task_to_json, summary_task_to_json
    from tests.integration.books.test_chunk_service import _prepared

    workspace, book_id = _prepared(tmp_path)
    service = BookService(workspace)
    schemas = manifest()["payload_schemas"]
    for name, payload in (
        ("summary-tasks", [summary_task_to_json(task) for task in service.tasks(book_id)]),
        (
            "structure-tasks",
            [structure_review_task_to_json(task) for task in service.structure_tasks(book_id)],
        ),
    ):
        importlib.import_module("jsonschema").validate(
            payload, schemas["capicove.book." + name + ".v1"]
        )


def test_wiki_migration_preserves_preview_apply_and_idempotent_receipts(tmp_path: Path) -> None:
    from dataclasses import asdict
    from datetime import date

    from capicove.wiki import WikiService, migrate_to_v2

    service = WikiService(tmp_path)
    service.initialize(created_on=date(2026, 1, 1))
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    preview = operations["capicove.wiki.migration-preview"]
    apply = operations["capicove.wiki.migration-apply"]
    for operation, preview_only in ((preview, True), (apply, False), (apply, False)):
        receipt = json.loads(
            json.dumps(asdict(migrate_to_v2(service.wiki_path, preview=preview_only)))
        )
        importlib.import_module("jsonschema").validate(receipt, operation["result_schema"])
    assert "expected_inventory_sha256" in apply["parameters_schema"]["required"]
    assert "not atomic" in apply["semantic_audit"]["field_projection"]["mutation_boundary"]


def test_graph_catalog_contracts_preserve_real_activation_and_revocation(tmp_path: Path) -> None:
    from dataclasses import asdict

    from capicove.processing import LocalProcessingWorkspace
    from tests.graph_build_fixtures import SCOPE, Gate, _catalog, _manifest

    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    snapshot = _manifest(workspace)
    catalog = _catalog(tmp_path / "catalog" / "graph.sqlite", workspace, Gate())
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    active_schema = operations["capicove.graph.active"]["result_schema"]
    ref, revision = catalog.active(SCOPE)
    importlib.import_module("jsonschema").validate(
        {"graph_ref": None if ref is None else asdict(ref), "revision": revision}, active_schema
    )
    registered = catalog.register(snapshot)
    revision = catalog.activate(registered.snapshot_id, SCOPE, revision)
    importlib.import_module("jsonschema").validate(
        {"revision": revision}, operations["capicove.graph.activate"]["result_schema"]
    )
    ref, revision = catalog.active(SCOPE)
    assert ref is not None
    importlib.import_module("jsonschema").validate(
        {"graph_ref": asdict(ref), "revision": revision}, active_schema
    )
    result = catalog.mark_ineligible("source:fixture", SCOPE, "source_deleted")
    importlib.import_module("jsonschema").validate(
        result, operations["capicove.graph.mark-ineligible"]["result_schema"]
    )


def test_graph_manifest_projection_preserves_real_codec_and_exact_uri_replacements(
    tmp_path: Path,
) -> None:
    from capicove.graph_build.codec import manifest_bytes
    from capicove.processing import LocalProcessingWorkspace
    from tests.graph_build_fixtures import _manifest

    snapshot = _manifest(LocalProcessingWorkspace(tmp_path / "processing"))
    payload = json.loads(manifest_bytes(snapshot))
    original = json.loads(manifest_bytes(snapshot))
    datasets = [
        payload["input_snapshot_ref"],
        *[table["dataset_ref"] for table in payload["tables"]],
    ]
    for dataset in datasets:
        del dataset["uri"]
        dataset["dataset_artifact"] = {
            "artifact_id": "dataset:" + dataset["manifest_sha256"],
            "sha256": dataset["manifest_sha256"],
            "version": dataset["version"],
        }
    schema = manifest()["payload_schemas"]["capicove.graph.manifest-projection.v1"]
    importlib.import_module("jsonschema").validate(payload, schema)
    assert set(payload) == set(original)
    assert payload["stages"] == original["stages"]
    assert len(payload["tables"]) == 17


def test_extended_validator_rejects_missing_partition_and_forged_contracts() -> None:
    import copy

    from tests.contract.remote.test_operation_coverage import checker

    module = checker()
    baseline = manifest()
    assert module.validate_extended_manifest(baseline, ROOT) == []
    mutations = []
    missing = copy.deepcopy(baseline)
    missing["unresolved_candidates"].pop()
    mutations.append(missing)
    wrong = copy.deepcopy(baseline)
    wrong["source_mappings"][0]["candidate_provider_profile"] = "invented-provider"
    mutations.append(wrong)
    forged = copy.deepcopy(baseline)
    forged["operations"][0]["acceptance_receipt"] = "claimed-approval"
    mutations.append(forged)
    empty = copy.deepcopy(baseline)
    empty["operations"][0]["parameters_schema"] = {}
    empty["operations"][0]["parameters_schema_sha256"] = module.digest({})
    mutations.append(empty)
    wrong_payload = copy.deepcopy(baseline)
    wrong_payload["operations"][-1]["payload_contracts"] = {"manifest": "MissingPayloadV99"}
    mutations.append(wrong_payload)
    for changed in mutations:
        assert module.validate_extended_manifest(changed, ROOT)


def test_main_coverage_gate_validates_extended_manifest(monkeypatch: Any) -> None:
    from tests.contract.remote.test_operation_coverage import checker, inventory

    module = checker()
    changed = manifest()
    changed["unresolved_candidates"].pop()
    original = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path.name == "remote-mcp-contracts-extended.json":
            return json.dumps(changed)
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    errors = module.validate(inventory(), ROOT, "inventoried", None)
    assert any("extended candidate/provider partition" in error for error in errors)


def test_document_retrieval_preserves_real_partial_page_and_coordinates() -> None:
    from capicove.contracts import (
        DocumentBlock,
        DocumentBlockType,
        DocumentPage,
        DocumentPageStatus,
        DocumentParseResult,
        DocumentParserReceipt,
        NormalizedBox,
    )
    from capicove.documents.codec import document_parse_result_to_json

    block = DocumentBlock(
        "block:1",
        0,
        DocumentBlockType.TEXT,
        "Exact text",
        "Exact text",
        NormalizedBox(0.1, 0.2, 0.8, 0.9),
        0,
        0.9,
    )
    parsed = DocumentParseResult(
        "a" * 64,
        "b" * 64,
        (
            DocumentPage(0, "i", 600, 800, DocumentPageStatus.SUCCEEDED, (block,)),
            DocumentPage(1, None, 600, 800, DocumentPageStatus.FAILED, (), "parse_failed"),
        ),
        DocumentParserReceipt("fixture", "1", "fixture", 1, "c" * 64, "d" * 64, "e" * 64),
    )
    payload = cast(dict[str, Any], document_parse_result_to_json(parsed))
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    validator = importlib.import_module("jsonschema")
    for page in payload["pages"]:
        error = page.pop("error")
        page["error_ref"] = (
            None
            if error is None
            else {"artifact_id": "diagnostic:page", "sha256": "f" * 64, "version": 1}
        )
        validator.validate(page, operations["capicove.document.page"]["result_schema"])
    validator.validate(
        payload["pages"][0]["blocks"][0], operations["capicove.document.block"]["result_schema"]
    )
    validator.validate(payload["receipt"], operations["capicove.document.receipt"]["result_schema"])
    validator.validate(
        {
            "page_index": 0,
            "block_id": "block:1",
            "width": 600,
            "height": 800,
            "box": payload["pages"][0]["blocks"][0]["box"],
        },
        operations["capicove.document.coordinates"]["result_schema"],
    )


def test_extended_profiles_cannot_claim_readiness_or_foreign_operation() -> None:
    import copy

    from tests.contract.remote.test_operation_coverage import checker

    module = checker()
    for field, value in (
        ("enabled_by_default", True),
        ("qualification_receipt", "fabricated"),
        ("final_coverage_obligation", False),
        ("operation_ids", ["capicove.wiki.initialize"]),
    ):
        changed = copy.deepcopy(manifest())
        changed["provider_profiles"][0][field] = value
        assert module.validate_extended_manifest(changed, ROOT)


def test_graph_run_contracts_preserve_native_pins_and_terminal_bytes(tmp_path: Path) -> None:
    from dataclasses import asdict

    from capicove.graph_build.workflow import request_bytes
    from tests.unit.graph_build.test_workflow import SCOPE, _result_payload, _run, _store

    store = _store(tmp_path)
    run = _run(store)
    payload = asdict(store.get(run.run_id, SCOPE))
    payload["request"] = json.loads(request_bytes(run.request))
    dataset = payload["request"]["input_snapshot_ref"]
    del dataset["uri"]
    dataset["dataset_artifact"] = {
        "artifact_id": "dataset:fixture",
        "sha256": dataset["manifest_sha256"],
        "version": dataset["version"],
    }
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    validator = importlib.import_module("jsonschema")
    validator.validate(payload, operations["capicove.graph.status"]["result_schema"])
    validator.validate(
        store.result(run.run_id, SCOPE), operations["capicove.graph.result"]["result_schema"]
    )
    store.finish(
        run.run_id,
        SCOPE,
        "succeeded",
        _result_payload(run.run_id),
        expected_revision=store.gate.revision,
    )
    result = store.result(run.run_id, SCOPE)
    assert result is not None
    validator.validate(json.loads(result), operations["capicove.graph.result"]["result_schema"])


def test_graph_plan_retains_exact_native_stage_dependency_graph() -> None:
    from capicove.graph_build.workflow import stage_dependencies

    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    schema = operations["capicove.graph.plan"]["result_schema"]["properties"]["workflow"]
    native = {key: list(value) for key, value in stage_dependencies().items()}
    validator = importlib.import_module("jsonschema").Draft202012Validator(schema)
    validator.validate(native)
    mutated = dict(native)
    mutated["communities"] = ["diagnostics"]
    assert list(validator.iter_errors(mutated))
    discovery = operations["capicove.graph.run-discovery"]
    assert discovery["semantic_audit"]["implementation_kind"] == "new_adapter_required"
    assert discovery["result_schema"]["properties"]["runs"]["maxItems"] == 100


def test_batch_contracts_preserve_real_plan_queue_status_and_cancel(tmp_path: Path) -> None:
    from capicove.runtime.mcp_host import compose_mcp_host
    from tests.unit.runtime.test_batch_mcp import PNG, BatchSpace, call, enqueue, selectors

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    images = tmp_path / "images"
    images.mkdir()
    (images / "a.png").write_bytes(PNG)
    space = BatchSpace(root, images, root / "host.json")
    space.write()
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    validator = importlib.import_module("jsonschema")
    with compose_mcp_host(space.load()) as composed:
        bindings = {item.spec.tool_id: item for item in composed}
        planned = call(bindings, "plan", selectors())["result"]
        validator.validate(planned, operations["capicove.batch.plan"]["result_schema"])
        queued = enqueue(bindings)["result"]
        validator.validate(queued, operations["capicove.batch.enqueue"]["result_schema"])
        job = queued["job"]
        status = call(bindings, "status", {"job_id": job["job_id"]})["result"]
        status["next_cursor"] = None
        status["snapshot_revision"] = job["revision"]
        for item in status["items"]:
            item["result_ref"] = item.pop("result")
        validator.validate(status, operations["capicove.batch.status"]["result_schema"])
        cancelled = call(
            bindings, "cancel", {"job_id": job["job_id"], "expected_revision": job["revision"]}
        )["result"]
        validator.validate(cancelled, operations["capicove.batch.cancel"]["result_schema"])
        assert cancelled["job"]["state"] == "cancelled"


def test_batch_result_artifacts_have_six_exact_payload_contracts() -> None:
    data = manifest()
    operation = next(o for o in data["operations"] if o["operation_id"] == "capicove.batch.status")
    contracts = operation["payload_contracts"]
    assert set(contracts) == {
        "image_aspect_ratio_filter",
        "image_shape_filter",
        "image_size_filter",
        "image_subplot_filter",
        "video_filter",
        "video_split",
    }
    for reference in contracts.values():
        assert reference in data["payload_schemas"]


def test_batch_input_modes_reject_unconfigured_video_criteria() -> None:
    operations = {o["operation_id"]: o for o in manifest()["operations"]}
    validator = importlib.import_module("jsonschema").Draft202012Validator(
        operations["capicove.batch.plan"]["parameters_schema"]
    )
    value: dict[str, Any] = {
        "input_root": "videos",
        "operation": "video_filter",
        "profile_ref": "batch-video-local",
        "parameters": {
            "criteria": [{"type": "metadata", "criterion_id": "m"}],
            "aggregation": "all",
        },
    }
    validator.validate(value)
    value["parameters"]["criteria"][0]["type"] = "nsfw"
    assert list(validator.iter_errors(value))


def test_extended_existing_tool_schema_identity_cannot_be_forged() -> None:
    from tests.contract.remote.test_operation_coverage import checker

    data = manifest()
    batch = next(o for o in data["operations"] if o["operation_id"] == "capicove.batch.plan")
    batch["local_input_schema_digest"] = "0" * 64
    assert checker().validate_extended_manifest(data, ROOT)


def test_trace_export_manifest_preserves_real_package_codec(tmp_path: Path) -> None:
    from capicove.export import encode_trace_export_manifest, inspect_trace_export
    from tests.integration.export.test_validate_package import _export

    package = _export(tmp_path / "source")
    value, report = inspect_trace_export(package)
    assert report.valid
    payload = encode_trace_export_manifest(value)
    schema = manifest()["payload_schemas"]["capicove.trace.export-manifest.v1"]
    importlib.import_module("jsonschema").validate(payload, schema)
    assert set(payload) == set(schema["required"])
    operations = {item["operation_id"]: item for item in manifest()["operations"]}
    export = operations["capicove.trace.export"]
    assert export["payload_contracts"]["manifest"] == "capicove.trace.export-manifest.v1"
    assert "manifest_ref" in export["result_schema"]["required"]


def test_trace_validation_diagnostics_preserve_native_issue_context() -> None:
    from capicove.export.validation import TraceValidationIssue, ValidationSeverity

    issue = TraceValidationIssue(ValidationSeverity.ERROR, "EXPORT_INVALID", "Sensitive detail")
    schema = manifest()["payload_schemas"]["capicove.trace.validation-report.v1"]
    fields = schema["properties"]["issues"]["items"]["properties"]
    assert set(fields) == {
        "severity",
        "code",
        "message_ref",
        "shard",
        "line",
        "episode_id",
        "record_id",
    }
    assert issue.code
    assert "message" not in fields
