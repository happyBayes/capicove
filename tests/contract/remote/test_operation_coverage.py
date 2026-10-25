"""Operation coverage must fail closed on inventory and evidence drift."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts/check_remote_mcp_coverage.py"
INVENTORY = ROOT / "docs/development/remote-mcp-operations.json"


def checker() -> ModuleType:
    """Load the standalone checker without modifying the application runtime."""
    assert SCRIPT.exists(), "operation coverage checker is not implemented"
    spec = importlib.util.spec_from_file_location("coverage_checker", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def inventory() -> dict[str, Any]:
    assert INVENTORY.exists(), "operation inventory is not implemented"
    return cast(dict[str, Any], json.loads(INVENTORY.read_text()))


def test_inventoried_inventory_covers_current_sources() -> None:
    assert checker().validate(inventory(), ROOT, "inventoried", None) == []


@pytest.mark.parametrize("mutation", ["remove", "provider", "source", "schema", "duplicate"])
def test_inventory_mutations_are_rejected(mutation: str) -> None:
    module, data = checker(), inventory()
    row = next(row for row in data["operations"] if row["canonical_tool_id"])
    if mutation == "remove":
        data["operations"].remove(row)
    elif mutation == "provider":
        row["provider_profile"] = "invented-provider"
    elif mutation == "source":
        row["source_entrypoints"] = ["src/capicove/missing.py:nope"]
    elif mutation == "schema":
        row["input_schema_digest"] = "0" * 64
    else:
        data["operations"].append(row.copy())
    assert module.validate(data, ROOT, "inventoried", None)


@pytest.mark.parametrize("value", [None, [], {"operations": []}, {"format_version": True}])
def test_invalid_inventory_is_a_diagnostic_not_a_crash(value: object) -> None:
    assert checker().validate(value, ROOT, "inventoried", None)


@pytest.mark.parametrize(
    "receipts",
    [None, {}, {"receipts": []}, {"receipts": [{"status": "passed", "real_provider": True}]}],
)
def test_no_fabricated_qualification_from_empty_or_self_asserted_receipts(receipts: object) -> None:
    assert checker().validate(inventory(), ROOT, "qualified", receipts)


def test_required_nonaggregate_surfaces_are_present() -> None:
    data = inventory()
    rows = data["operations"]
    assert {row["capability_id"] for row in rows} == {f"C{i:02}" for i in range(1, 27)}
    assert any(row["capability_id"] == "C25" and "delete" in row["operation_id"] for row in rows)
    assert any(
        row["capability_id"] == "C21" and "count_tokens" in row["operation_id"] for row in rows
    )
    assert any(
        row["capability_id"] == "C17" and "capicove.data.project" in row["operation_id"]
        for row in rows
    )
    plugins = {row["provider_profile"] for row in rows if row["operation_id"].startswith("plugin:")}
    assert len(plugins) == 5


def test_source_surface_addition_requires_inventory_entry(tmp_path: Path) -> None:
    module = checker()
    before = module.discover_public_surfaces(tmp_path)
    path = tmp_path / "src/capicove/cli/commands/new_capability.py"
    path.parent.mkdir(parents=True)
    path.write_text('def public_operation():\n    """New public entry."""\n    pass\n')
    after = module.discover_public_surfaces(tmp_path)
    assert before != after
    assert any("public_operation" in key for key in after)


def test_existing_audio_tool_has_owning_source_and_local_wire_name() -> None:
    rows = inventory()["operations"]
    row = next(row for row in rows if row["canonical_tool_id"] == "capicove.audio.clip")
    assert row["capability_id"] == "C05"
    assert row["source_entrypoints"][0].startswith("src/capicove/tools/audio_clip.py:")
    assert row["mcp_name"] == "capicove_audio_clip"


def test_whole_source_manifest_prevents_unnoticed_nonfunction_drift() -> None:
    module, data = checker(), inventory()
    assert data.get("source_manifest"), (
        "module-level provider/configuration changes require fingerprints"
    )
    data["source_manifest"]["src/capicove/model/serving.py"] = "0" * 64
    assert module.validate(data, ROOT, "inventoried", None)


def test_dynamic_knowledge_tools_and_nonbusiness_classifications_are_explicit() -> None:
    rows = inventory()["operations"]
    assert any(row["canonical_tool_id"] == "capicove.knowledge.ask" for row in rows)
    helpers = [row for row in rows if row["classification"] == "internal_helper"]
    assert helpers
    assert all(row["classification_rationale"] and row["source_entrypoints"] for row in helpers)


def test_provider_adapters_and_cli_subcommands_have_individual_rows() -> None:
    rows = inventory()["operations"]
    assert any("TesseractCliOcrProvider" in row["provider_profile"] for row in rows)
    assert any(
        row["operation_id"].startswith("cli:") and "delete" in row["operation_id"] for row in rows
    )


def test_all_existing_tool_rows_record_local_wire_mapping() -> None:
    assert all(row["mcp_name"] for row in inventory()["operations"] if row["canonical_tool_id"])


def test_planned_mode_rejects_structural_only_inventory() -> None:
    errors = checker().validate(inventory(), ROOT, "planned", None)
    assert any("planning" in error for error in errors)


def test_c18_tracks_admission_and_worker_implementation_tasks() -> None:
    rows = [row for row in inventory()["operations"] if row["capability_id"] == "C18"]
    assert rows
    assert all({"R3.02", "R3.04"} <= set(row["task_ids"]) for row in rows)


def reviewed_test_plan() -> dict[str, Any]:
    """Synthetic planning fixture; not acceptance evidence for repository rows."""
    return {
        "classification": "business_tool",
        "classification_rationale": "Synthetic test business operation",
        "review": {"owner": "fixture-owner", "reference": "fixture-review"},
        "contract": {
            "tool_id": "capicove.fixture",
            "version": 1,
            "mcp_name": "capicove_fixture_v1",
            "input_schema_digest": "a" * 64,
            "output_schema_digest": "b" * 64,
            "review_reference": "fixture-contract",
        },
        "permissions": ["fixture:read"],
        "resource_policy": {"rule": "fixture scope"},
        "approval_policy": {"rule": "read-only fixture"},
        "execution_mode": "sync",
        "deadline": 30,
        "idempotency_policy": {"rule": "read-only fixture"},
        "test_ids": ["fixture.contract", "fixture.isolation"],
        "operation_provider_bindings": [
            {
                "operation_id": "fixture.operation",
                "provider_profile": "fixture-provider",
                "provider_source": "fixture-provider-source",
                "profile": {"id": "fixture-profile", "pins": {"version": "1"}},
            }
        ],
    }


def test_reviewed_planning_metadata_is_separate_from_source_defaults() -> None:
    module = checker()
    item = {
        "operation_id": "fixture.operation",
        "provider_profile": "fixture-provider",
        "planning": reviewed_test_plan(),
    }
    known = {
        ("fixture.operation", "fixture-provider"): item,
        ("fixture-provider-source", "fixture-provider"): {
            "operation_id": "fixture-provider-source"
        },
    }
    assert module.planning_errors(item, known) == []


@pytest.mark.parametrize(
    "field",
    [
        "review",
        "contract",
        "permissions",
        "resource_policy",
        "approval_policy",
        "test_ids",
        "operation_provider_bindings",
    ],
)
def test_planning_rejects_missing_required_mapping(field: str) -> None:
    plan = reviewed_test_plan()
    plan[field] = None
    item = {
        "operation_id": "fixture.operation",
        "provider_profile": "fixture-provider",
        "planning": plan,
    }
    assert checker().planning_errors(item, {})


def test_planning_rejects_unbound_provider_candidate() -> None:
    plan = reviewed_test_plan()
    plan["operation_provider_bindings"][0]["operation_id"] = "unknown.operation"
    item = {
        "operation_id": "provider:fixture",
        "provider_profile": "fixture-provider",
        "planning": plan,
    }
    assert checker().planning_errors(item, {})


def test_source_manifest_includes_provider_profiles_and_public_snapshot_services() -> None:
    manifest = checker().source_manifest(ROOT)
    for relative in (
        "src/capicove/model/registry.py",
        "src/capicove/model/ports.py",
        "src/capicove/graph_build/snapshots.py",
        "src/capicove/model/image_models.toml",
        "src/capicove/model/bge_m3.lock.toml",
    ):
        assert relative in manifest
    rows = inventory()["operations"]
    assert any("graph_build/snapshots.py:" in row["operation_id"] for row in rows)


def test_c24_tracks_all_identity_transport_execution_packages() -> None:
    expected = {"R1.01", "R1.02", "R1.03", "R1.04", "R2.01", "R3.01", "R3.05"}
    rows = [row for row in inventory()["operations"] if row["capability_id"] == "C24"]
    assert rows
    assert all(expected <= set(row["task_ids"]) for row in rows)


@pytest.mark.parametrize("field", ["classification", "execution_mode", "deadline"])
def test_malformed_planning_metadata_produces_diagnostics(field: str) -> None:
    plan = reviewed_test_plan()
    plan[field] = []
    item = {
        "operation_id": "fixture.operation",
        "provider_profile": "fixture-provider",
        "planning": plan,
    }
    assert checker().planning_errors(item, {})


def test_reviewed_mapping_does_not_rewrite_discovered_source_defaults() -> None:
    module, data = checker(), inventory()
    data["operations"][0]["planning"] = reviewed_test_plan()
    assert module.validate(data, ROOT, "inventoried", None) == []


def test_planning_rejects_placeholder_classification_review() -> None:
    plan = reviewed_test_plan()
    plan["review"]["reference"] = "pending"
    item = {
        "operation_id": "fixture.operation",
        "provider_profile": "fixture-provider",
        "planning": plan,
    }
    assert checker().planning_errors(item, {})


def test_foreign_loaded_capicove_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    foreign = ModuleType("capicove")
    foreign.__file__ = str(tmp_path / "capicove/__init__.py")
    import sys

    monkeypatch.setitem(sys.modules, "capicove", foreign)
    with pytest.raises(ValueError, match="source root"):
        checker().ensure_source_root(ROOT)


def test_cli_pins_requested_checkout_before_shared_environment(tmp_path: Path) -> None:
    import os
    import subprocess
    import sys

    foreign = tmp_path / "capicove"
    foreign.mkdir()
    (foreign / "__init__.py").write_text('raise RuntimeError("foreign checkout imported")\n')
    result = subprocess.run(
        [sys.executable, str(SCRIPT), "--inventory", str(INVENTORY), "--mode", "inventoried"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "discovery_consistent"
    assert report["g0_complete"] is False


@pytest.mark.parametrize("case", ["valid", "mixed", "empty", "sibling"])
def test_namespace_packages_require_all_paths_inside_selected_package(
    case: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    namespace = ModuleType("capicove.fixture_namespace")
    valid = str(ROOT / "src/capicove/fixture_namespace")
    locations = {
        "valid": [valid],
        "mixed": [valid, str(tmp_path)],
        "empty": [],
        "sibling": [str(ROOT / "src/unrelated_namespace")],
    }[case]
    namespace.__path__ = locations
    monkeypatch.setitem(sys.modules, namespace.__name__, namespace)
    if case == "valid":
        checker().ensure_source_root(ROOT)
    else:
        with pytest.raises(ValueError, match="source root"):
            checker().ensure_source_root(ROOT)


@pytest.mark.parametrize("candidate", ["fixture.operation", "fixture-provider-source"])
def test_planning_cannot_cover_another_provider_variant(candidate: str) -> None:
    item = {
        "operation_id": candidate,
        "provider_profile": "OTHER-PROVIDER",
        "planning": reviewed_test_plan(),
    }
    known = {
        ("fixture.operation", "OTHER-PROVIDER"): {"operation_id": "fixture.operation"},
        ("fixture.operation", "fixture-provider"): {"operation_id": "fixture.operation"},
        (candidate, "OTHER-PROVIDER"): item,
        ("fixture-provider-source", "fixture-provider"): {
            "operation_id": "fixture-provider-source"
        },
    }
    assert checker().planning_errors(item, known)


@pytest.mark.parametrize("deadline", [float("nan"), float("inf"), float("-inf")])
def test_planning_deadline_must_be_finite(deadline: float) -> None:
    plan = reviewed_test_plan()
    plan["deadline"] = deadline
    item = {
        "operation_id": "fixture.operation",
        "provider_profile": "fixture-provider",
        "planning": plan,
    }
    known = {
        ("fixture.operation", "fixture-provider"): item,
        ("fixture-provider-source", "fixture-provider"): {
            "operation_id": "fixture-provider-source"
        },
    }
    errors = checker().planning_errors(item, known)
    assert any("deadline" in error for error in errors)


def test_binding_provider_requires_matching_business_operation_variant() -> None:
    item = {
        "operation_id": "fixture-provider-source",
        "provider_profile": "fixture-provider",
        "planning": reviewed_test_plan(),
    }
    known = {
        ("fixture.operation", "OTHER-PROVIDER"): {"operation_id": "fixture.operation"},
        ("fixture-provider-source", "fixture-provider"): item,
    }
    assert checker().planning_errors(item, known)


CORE_MANIFEST = ROOT / "docs/development/remote-mcp-contracts-core.json"


def core_manifest() -> dict[str, Any]:
    assert CORE_MANIFEST.is_file(), "source-grounded core contract manifest is missing"
    return cast(dict[str, Any], json.loads(CORE_MANIFEST.read_text()))


def test_core_contract_manifest_has_verified_source_mappings() -> None:
    manifest = core_manifest()
    assert checker().validate_core_manifest(manifest, ROOT) == []
    assert manifest["approval_state"] == "draft_for_review"
    assert {operation["capability_id"] for operation in manifest["operations"]} == {
        "C01",
        "C02",
        "C03",
        "C04",
        "C05",
        "C06",
        "C07",
        "C08",
        "C09",
        "C10",
        "C25",
        "C26",
    }


@pytest.mark.parametrize("mutation", ["operation", "source", "profile", "schema", "helper_target"])
def test_core_mapping_cannot_drop_operations_or_forge_sources(mutation: str) -> None:
    manifest = core_manifest()
    operation = next(
        item for item in manifest["operations"] if item["local_tool_id"] == "capicove.image.ocr"
    )
    if mutation == "operation":
        manifest["operations"].remove(operation)
    elif mutation == "source":
        operation["source_entrypoints"] = ["src/capicove/nonexistent.py:fake"]
    elif mutation == "profile":
        operation["provider_bindings"][0]["profile_id"] = "unregistered-profile"
    elif mutation == "schema":
        operation["input_schema"]["additionalProperties"] = True
    else:
        manifest["source_mappings"][0]["enclosing_operations"] = ["missing-operation"]
    assert checker().validate_core_manifest(manifest, ROOT)


def test_core_artifact_contract_rejects_local_paths_and_foreign_authority_fields() -> None:
    Draft202012Validator = importlib.import_module("jsonschema").Draft202012Validator

    manifest = core_manifest()
    operation = next(
        item for item in manifest["operations"] if item["local_tool_id"] == "capicove.audio.import"
    )
    schema = operation["input_schema"]
    validator = Draft202012Validator(schema)
    valid = {
        "parameters": {"source": {"artifact_id": "artifact-1", "sha256": "a" * 64, "version": 1}},
        "profile_id": operation["provider_bindings"][0]["profile_id"],
        "idempotency_key": "request-1",
    }
    assert list(validator.iter_errors(valid)) == []
    valid["parameters"]["source"] = {"type": "path", "path": "/etc/passwd"}
    assert list(validator.iter_errors(valid))
    valid["tenant_id"] = "foreign"
    assert list(validator.iter_errors(valid))


def test_core_profiles_do_not_invent_provisioning_or_qualification() -> None:
    manifest = core_manifest()
    assert all(
        profile["provisioning_state"] == "not_verified" and profile["qualification_receipt"] is None
        for profile in manifest["profiles"]
    )
    assert all(
        operation["planned_test_suites"] and operation["test_ids"]
        for operation in manifest["operations"]
    )


def test_generic_search_does_not_claim_every_retriever_supports_it() -> None:
    rows = checker().expected_rows(ROOT)
    search = [row for row in rows if row["canonical_tool_id"] == "capicove.search"]
    assert {row["provider_profile"] for row in search} == {"pending-exact-provider-binding"}
    evidence = [row for row in rows if row["canonical_tool_id"] == "capicove.get_evidence"]
    assert {row["provider_profile"] for row in evidence} == {"builtin"}


@pytest.mark.parametrize("mutation", ["missing", "digest", "mapping"])
def test_inventory_requires_exact_core_manifest_linkage(mutation: str) -> None:
    data = inventory()
    if mutation == "missing":
        data.pop("core_contract_manifest", None)
    elif mutation == "digest":
        data["core_contract_manifest"] = {
            "path": "docs/development/remote-mcp-contracts-core.json",
            "sha256": "0" * 64,
        }
    else:
        row = next(row for row in data["operations"] if row.get("contract_mapping"))
        row["contract_mapping"]["enclosing_operations"] = ["forged"]
    assert any("core" in error for error in checker().validate(data, ROOT, "inventoried", None))


def test_core_video_profile_rejects_unsupported_modes() -> None:
    validator_type = importlib.import_module("jsonschema").Draft202012Validator
    manifest = core_manifest()
    split = next(
        op for op in manifest["operations"] if op["operation_id"] == "capicove.video.split"
    )
    payload = {
        "parameters": {
            "source": {"artifact_id": "a", "sha256": "a" * 64, "version": 1},
            "config": {"strategy": "duration", "split_duration": 2},
            "write_mode": "copy",
        },
        "profile_id": "video-pyav",
        "idempotency_key": "a",
    }
    validator = validator_type(split["input_schema"])
    assert list(validator.iter_errors(payload))
    payload["profile_id"] = "video-ffmpeg"
    assert list(validator.iter_errors(payload)) == []
    filtering = next(
        op for op in manifest["operations"] if op["operation_id"] == "capicove.video.filter"
    )
    filter_payload = {
        "parameters": {
            "sources": [
                {"item_id": "a", "source": {"artifact_id": "a", "sha256": "a" * 64, "version": 1}}
            ],
            "criteria": [{"type": "motion", "criterion_id": "c"}],
            "aggregation": "all",
        },
        "profile_id": "video-filter-metadata",
        "idempotency_key": "a",
    }
    validator = validator_type(filtering["input_schema"])
    assert list(validator.iter_errors(filter_payload))
    filter_payload["profile_id"] = "video-filter-motion"
    assert list(validator.iter_errors(filter_payload)) == []


@pytest.mark.parametrize(
    "mutation",
    [
        "erase_partition",
        "fake_candidate",
        "profile_claim",
        "tool_collision",
        "wire_collision",
        "empty_parameters",
    ],
)
def test_core_manifest_cannot_erase_obligations_or_forge_readiness(mutation: str) -> None:
    manifest = core_manifest()
    module = checker()
    if mutation == "erase_partition":
        manifest["source_mappings"] = []
        manifest["unresolved_candidates"] = []
    elif mutation == "fake_candidate":
        manifest["source_mappings"][0]["candidate_operation_id"] = "nonexistent"
    elif mutation == "profile_claim":
        manifest["profiles"][0].update(
            enabled_by_default=True,
            provisioning_state="qualified",
            qualification_receipt="fake",
            final_coverage_obligation=False,
        )
    elif mutation == "tool_collision":
        manifest["operations"][1]["proposed_tool_id"] = manifest["operations"][0][
            "proposed_tool_id"
        ]
    elif mutation == "wire_collision":
        manifest["operations"][1]["mcp_name"] = manifest["operations"][0]["mcp_name"]
    else:
        op = manifest["operations"][0]
        op["input_schema"]["properties"]["parameters"] = {}
        op["input_schema_digest"] = module.digest(op["input_schema"])
    assert module.validate_core_manifest(manifest, ROOT)


def test_core_service_queries_preserve_return_semantics() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}
    eviction = ops["capicove.generation.video.can_evict"]["output_schema"]["properties"]["result"]
    assert eviction["properties"]["can_evict"] == {"type": "boolean"}
    active = ops["capicove.generation.video.resolve_active"]
    assert "generation_id" not in active["input_schema"]["properties"]["parameters"].get(
        "required", []
    )
    assert "pin" in active["output_schema"]["properties"]["result"]["properties"]
    policy = ops["capicove.memory.memory_policy_for"]["output_schema"]["properties"]["result"]
    assert "policy" in policy["properties"]


@pytest.mark.parametrize(
    "candidate,target",
    [
        (
            "public:src/capicove/processing/audio/clipping.py:AudioClippingService.clip_file",
            "capicove.audio.clip",
        ),
        (
            "public:src/capicove/processing/image/ports.py:ImageOcrProvider.recognize",
            "capicove.image.ocr",
        ),
        (
            "public:src/capicove/processing/video/retrieval/ports.py:ASRProcessor.transcribe",
            "capicove.video.prepare",
        ),
        (
            "cli:src/capicove/cli/commands/audio_transcribe.py:transcribe",
            "capicove.audio.transcribe",
        ),
    ],
)
def test_core_source_aliases_keep_exact_business_owner(candidate: str, target: str) -> None:
    mappings = {row["candidate_operation_id"]: row for row in core_manifest()["source_mappings"]}
    assert candidate in mappings
    assert target in mappings[candidate]["enclosing_operations"]
    assert mappings[candidate]["source_evidence"]


def test_core_business_candidates_are_not_hidden_as_helpers() -> None:
    operations = {op["operation_id"]: op for op in core_manifest()["operations"]}
    for identity in (
        "capicove.retrieval.rerank",
        "capicove.retrieval.rank",
        "capicove.retrieval.embed_image",
        "capicove.retrieval.embed_texts",
        "capicove.memory.record_memory",
        "capicove.audio.capabilities",
    ):
        assert identity in operations
        assert operations[identity]["implementation_state"] == "adapter_required"
        assert operations[identity]["approval_state"] == "draft_for_review"


def test_core_exports_and_video_validation_have_explicit_obligations() -> None:
    operations = {op["operation_id"] for op in core_manifest()["operations"]}
    assert {
        "capicove.image.metric.binary_mask_iou",
        "capicove.image.metric.bounding_box_iou",
        "capicove.image.metric.coefficient_of_variation",
        "capicove.image.metric.information_gain",
        "capicove.generation.video.validate",
    } <= operations


def test_core_search_and_memory_preserve_reviewed_result_fields() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}
    search = ops["capicove.search.video"]
    assert {
        "targets",
        "detail_level",
        "time_range",
        "language",
        "artifact_refs",
        "media_types",
        "text_origins",
    } <= set(search["input_schema"]["properties"]["parameters"]["properties"])
    hit = search["output_schema"]["properties"]["result"]["properties"]["hits"]["items"]
    assert {"timestamp_ms", "preview_resource_ref", "lane_scores", "processing_fingerprint"} <= set(
        hit["properties"]
    )
    doctor = ops["capicove.memory.doctor"]["output_schema"]["properties"]["result"]
    assert {
        "status",
        "write_enabled",
        "backend",
        "journal_mode",
        "extractor",
        "dense",
        "worker",
    } <= set(doctor["properties"])
    work = ops["capicove.memory.work_extractions"]["output_schema"]["properties"]["result"]
    assert {"processed", "outcomes"} <= set(work["properties"])


def test_core_semantic_audit_never_treats_generic_receipt_as_preserved() -> None:
    for operation in core_manifest()["operations"]:
        audit = operation["semantic_audit"]
        assert audit["state"] in {"pending", "blocked", "source_fields_projected"}
        fields = set(operation["output_schema"]["properties"]["result"].get("properties", {}))
        if {"resource_id", "changed", "state", "warnings"} <= fields:
            assert audit["state"] == "blocked"
            assert audit["unresolved_fields"]


def test_core_semantic_blockers_are_part_of_planned_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module, data, manifest = checker(), inventory(), core_manifest()
    manifest["operations"][0]["semantic_audit"]["state"] = "pending"
    original_read = Path.read_text

    def read_text(path: Path, *args: Any, **kwargs: Any) -> str:
        if path == ROOT / "docs/development/remote-mcp-contracts-core.json":
            return json.dumps(manifest)
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read_text)
    assert any("semantic" in error for error in module.validate(data, ROOT, "planned", None))


def test_entity_resolution_check_is_a_write_obligation() -> None:
    operation = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.memory.check_entity_resolution"
    )
    assert operation["execution_mode"] == "durable"
    assert "memory:write" in operation["permissions"]


def test_core_ocr_parameters_and_difference_provenance_are_closed() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}
    parameters = ops["capicove.image.ocr"]["output_schema"]["properties"]["result"]["properties"][
        "results"
    ]["items"]["properties"]["parameters"]
    assert parameters["additionalProperties"] is False
    assert set(parameters["properties"]) == {"languages"}
    areas = ops["capicove.image.diff.caption"]["input_schema"]["properties"]["parameters"][
        "properties"
    ]["areas"]
    assert "artifact_id" in areas["properties"]


def test_core_generic_projection_cannot_claim_semantic_completeness() -> None:
    manifest = core_manifest()
    op = manifest["operations"][0]
    op["output_schema"]["properties"]["result"] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            field: {"type": "string"} for field in ("resource_id", "state", "changed", "warnings")
        },
    }
    op["output_schema_digest"] = checker().digest(op["output_schema"])
    op["semantic_audit"] = {
        "state": "source_fields_projected",
        "unresolved_fields": [],
        "approval_evidence": None,
    }
    assert checker().validate_core_manifest(manifest, ROOT)


def test_inventory_includes_public_reverse_image_lifecycle_services() -> None:
    rows = checker().expected_rows(ROOT)
    identities = {row["operation_id"] for row in rows}
    assert (
        "public:src/capicove/runtime/reverse_image.py:ReverseImageRuntime.activate_preview"
        in identities
    )
    assert (
        "public:src/capicove/runtime/reverse_image.py:ReverseImageRuntime.issue_preview_qualification"
        in identities
    )
    assert (
        "public:src/capicove/runtime/retrieval_build.py:PersonalImageGenerationAdministrator.build"
        in identities
    )


def test_reverse_image_activation_preserves_qualification_and_cas() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}
    activation = ops["capicove.generation.activate"]
    fields = activation["input_schema"]["properties"]["parameters"]["properties"]
    assert {"registration_ref", "qualification_id", "expected_previous_snapshot"} <= set(fields)
    for identity in (
        "capicove.generation.image.register_published",
        "capicove.generation.image.issue_preview_qualification",
        "capicove.generation.image.resolve_active_snapshot",
    ):
        assert identity in ops


def test_native_knowledge_preserves_evidence_claims_and_active_state() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}

    def result_fields(identity: str) -> set[str]:
        return set(ops[identity]["output_schema"]["properties"]["result"]["properties"])

    assert {"revision", "evidence", "graph_context", "retrieval"} <= result_fields(
        "capicove.knowledge.search"
    )
    assert {"claims", "status", "evidence"} <= result_fields("capicove.knowledge.ask")
    assert {
        "generation",
        "last_run",
        "graph_status",
        "semantic",
        "generation_enabled",
    } <= result_fields("capicove.knowledge.status")
    assert {"quote", "locator", "source_sha256", "original"} <= result_fields(
        "capicove.knowledge.evidence"
    )


def test_generation_inspect_and_cleanup_have_concrete_proposed_contracts() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}
    inspection = ops["capicove.generation.inspect"]
    assert "inspections" in inspection["output_schema"]["properties"]["result"]["properties"]
    cleanup = ops["capicove.generation.cleanup"]
    assert {"cleanup_plan", "expected_lease_revision"} <= set(
        cleanup["input_schema"]["properties"]["parameters"]["properties"]
    )
    assert {"deleted", "retained", "uncertain"} <= set(
        cleanup["output_schema"]["properties"]["result"]["properties"]
    )
    assert cleanup["implementation_state"] == "adapter_required"
    assert cleanup["approval_policy"]["required"] is True


@pytest.mark.parametrize("speech", [False, True])
def test_remote_transcription_preserves_real_canonical_serializer(speech: bool) -> None:
    from capicove.processing.audio.models import (
        AudioBackendLineage,
        AudioEvidence,
        AudioStreamSelection,
        AudioTranscript,
        AudioTranscriptionResult,
        AudioTranscriptionSegment,
        AudioWarningCode,
    )
    from capicove.tools.audio_transcribe import _serialize

    segments = (AudioTranscriptionSegment(0, 0.0, 1.0, "hello"),) if speech else ()
    transcript = AudioTranscript(
        "hello" if speech else "",
        "en",
        segments,
        () if speech else (AudioWarningCode.NO_SPEECH_DETECTED,),
        AudioBackendLineage("whisper.cpp", "fixture", "tiny", "a" * 64, "cpu", "f32"),
    )
    payload = _serialize(
        AudioTranscriptionResult(
            AudioEvidence(
                "sha256:" + "b" * 64, "b" * 64, 100, "audio/wav", "a.wav", Path("/owned/a.wav")
            ),
            transcript,
            AudioStreamSelection(0, "sole_stream_implicit", 1),
        )
    )
    operation = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.audio.transcribe"
    )
    validator = importlib.import_module("jsonschema").Draft202012Validator
    validator(operation["output_schema"]["properties"]["result"]).validate(payload)


def test_remote_audio_clip_preserves_every_canonical_result_field() -> None:
    from capicove.tools.audio_clip import AUDIO_CLIP_SPEC

    operation = next(
        op for op in core_manifest()["operations"] if op["operation_id"] == "capicove.audio.clip"
    )
    result = operation["output_schema"]["properties"]["result"]
    assert set(result["properties"]) == set(
        cast(dict[str, object], AUDIO_CLIP_SPEC.output_schema["properties"])
    )


@pytest.mark.parametrize("kind", ["aspect-ratio", "shape", "size"])
@pytest.mark.parametrize("empty", [False, True])
def test_remote_filter_accepts_real_empty_input_decision(kind: str, empty: bool) -> None:
    import io

    from PIL import Image

    from capicove.processing.image.filters.aspect_ratio import ImageAspectRatioFilter
    from capicove.processing.image.filters.shape import ImageShapeFilter
    from capicove.processing.image.filters.size import ImageSizeFilter
    from capicove.processing.image.models import ImageSource
    from capicove.tools.image import _serialize_result

    filters: dict[str, ImageAspectRatioFilter | ImageShapeFilter | ImageSizeFilter] = {
        "aspect-ratio": ImageAspectRatioFilter(),
        "shape": ImageShapeFilter(),
        "size": ImageSizeFilter(),
    }
    buffer = io.BytesIO()
    Image.new("RGB", (8, 4), "red").save(buffer, format="PNG")
    sources = () if empty else (ImageSource(data=buffer.getvalue()),)
    decision = filters[kind].evaluate(sources)
    payload, _ = _serialize_result(decision, decision.filter_name)
    operation = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == f"capicove.image.filter.{kind}"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        operation["output_schema"]["properties"]["result"]
    ).validate(payload)


@pytest.mark.parametrize("kind", ["manifest", "pin"])
def test_portable_video_payload_preserves_real_catalog_codec(kind: str) -> None:
    from capicove.retrieval._adapters.sqlite_video_generations import (
        _set_manifest_dict,
        _set_pin_dict,
    )
    from tests.integration.retrieval.test_sqlite_video_generation_catalog import _record

    record = _record("portable-fixture")
    payload = (
        _set_manifest_dict(record.manifest) if kind == "manifest" else _set_pin_dict(record.pin)
    )
    # Explicitly named capability fields, never recursive path-string scrubbing.
    payload["unit_catalog_ref"] = "owned:unit-catalog"
    if kind == "pin":
        for name in ("keyframe_visual", "segment_visual", "asr_text"):
            cast(dict[str, object], payload[name])["physical_index_ref"] = "owned:" + name
    path = ROOT / "docs/development/remote-mcp-core-payloads.json"
    document = json.loads(path.read_text())
    schema = {
        "$defs": document["$defs"],
        "$ref": "#/$defs/VideoGenerationSet" + ("Manifest" if kind == "manifest" else "Pin") + "V1",
    }
    validator = importlib.import_module("jsonschema").Draft202012Validator
    validator(schema).validate(payload)
    payload["unit_catalog_ref"] = "/tmp/local.sqlite"
    assert not validator(schema).is_valid(payload)


def test_core_payload_schema_linkage_rejects_missing_or_changed_digest() -> None:
    manifest = core_manifest()
    manifest["payload_schema_file"] = {
        "path": "docs/development/remote-mcp-core-payloads.json",
        "sha256": "0" * 64,
    }
    assert any(
        "payload schema" in error for error in checker().validate_core_manifest(manifest, ROOT)
    )


@pytest.mark.parametrize(
    "operation", ["doctor", "get_episode", "recall", "resolve_entity", "work_extractions"]
)
def test_memory_schema_accepts_actual_sqlite_service_payload(
    tmp_path: Path, operation: str
) -> None:
    from datetime import UTC, datetime

    from capicove.contracts.memory import MemoryQuery
    from capicove.tools.memory_tools import _json_value
    from tests.integration.agent.test_memory_lifecycle import make_agent

    agent = make_agent(tmp_path)
    receipt = agent.remember_episode(
        workspace="w",
        scope="s",
        content="Remember this fixture",
        event_time=datetime.now(UTC),
        idempotency_key="fixture",
    )
    result: object
    if operation == "doctor":
        result = agent.doctor()
    elif operation == "get_episode":
        result = agent.get_episode(workspace="w", scope="s", episode_id=receipt["episode_id"])
    elif operation == "recall":
        result = agent.recall(MemoryQuery("w", "s", "fixture"))
    elif operation == "resolve_entity":
        result = agent.resolve_entity(
            workspace="w", scope="s", name="Alice", entity_type="person", idempotency_key="entity"
        )
    else:
        with pytest.raises(RuntimeError, match="extraction_unavailable"):
            agent.work_extractions(workspace="w", scope="s")
        return
    payload = _json_value(result)
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.memory." + operation
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


@pytest.mark.parametrize("current", [False, True])
def test_memory_fact_history_preserves_actual_fact_lineage(tmp_path: Path, current: bool) -> None:
    from datetime import UTC, datetime

    from capicove.contracts.knowledge import TimeInterval, TypedLiteral
    from capicove.tools.memory_tools import _json_value
    from tests.integration.agent.test_memory_lifecycle import make_agent

    agent = make_agent(tmp_path)
    now = datetime.now(UTC)
    receipt = agent.assert_fact(
        workspace="w",
        scope="s",
        subject_name="Alice",
        subject_type="person",
        predicate="city",
        object=TypedLiteral("string", "Paris"),
        validity=TimeInterval(now, None),
        event_time=now,
        evidence_text="Alice lives in Paris",
        idempotency_key="fact",
    )
    facts = agent.get_fact_history(workspace="w", scope="s", fact_id=receipt["fact_id"])
    assert facts
    operation = "get_fact_history"
    if current:
        facts = agent.knowledge.current_facts(
            workspace="w", scope="s", subject=facts[0].subject_ref.entity_id, predicate="city"
        )
        operation = "current_facts"
    payload = {"facts": _json_value(facts), "next_cursor": None}
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.memory." + operation
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


@pytest.mark.parametrize("kind", ["phash", "lsh-search", "perceptual-deduplicate"])
def test_remote_hash_schema_preserves_real_algorithm_codec(kind: str) -> None:
    import io

    from PIL import Image

    from capicove.processing.image.deduplication.lsh import HashRecord, ImageLshSearch
    from capicove.processing.image.deduplication.perceptual import (
        ImagePerceptualDeduplicator,
        ImagePhash,
        ImageRecord,
    )
    from capicove.processing.image.models import ImageSource
    from capicove.tools.image import _serialize_result

    buffer = io.BytesIO()
    Image.new("RGB", (16, 16), "red").save(buffer, format="PNG")
    records = (ImageRecord("one", (ImageSource(data=buffer.getvalue()),)),)
    result: object
    if kind == "phash":
        result = ImagePhash().process(records)
    elif kind == "lsh-search":
        result = ImageLshSearch().search(
            (0,), (HashRecord("one", (0,)),), hash_version="phash-64-v1"
        )
    else:
        result = ImagePerceptualDeduplicator().group(records)
    payload, _ = _serialize_result(result, kind)
    contract = next(
        op for op in core_manifest()["operations"] if op["operation_id"] == "capicove.image." + kind
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


def test_durable_registration_schema_requires_authority_and_publication_binding() -> None:
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    schema = document["$defs"]["PublishedRegistrationRecordV1"]
    payload = {
        "registration_ref": "owned:registration",
        "authority_sha256": "a" * 64,
        "workspace_id": "w",
        "profile_ref": "personal-lite",
        "activation_set_id": "set",
        "manifest_sha256": "b" * 64,
        "published_root_ref": "owned:root",
        "corpus_ref": "corpus",
        "published_at": "2026-10-23T00:00:00Z",
        "expires_at": "2026-10-24T00:00:00Z",
        "revision": 1,
        "state": "registered",
    }
    validator = importlib.import_module("jsonschema").Draft202012Validator(schema)
    validator.validate(payload)
    for field in ("authority_sha256", "manifest_sha256", "published_root_ref"):
        invalid = dict(payload)
        invalid.pop(field)
        assert not validator.is_valid(invalid)
    payload["published_root_ref"] = "/tmp/publication"
    assert not validator.is_valid(payload)


def test_inspection_cursor_schema_binds_snapshot_authority_and_offset() -> None:
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    schema = document["$defs"]["InspectionCursorRecordV1"]
    fields = set(schema["required"])
    assert {
        "authority_sha256",
        "snapshot_sha256",
        "collection_ref",
        "collection_revision",
        "offset",
        "limit",
        "expires_at",
        "policy_epoch",
    } <= fields
    assert schema["additionalProperties"] is False


def test_metric_box_contract_preserves_fractional_source_geometry() -> None:
    from capicove.processing.image.metrics import bounding_box_iou
    from capicove.processing.image.models import BoundingBox

    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.image.metric.bounding_box_iou"
    )
    parameters = {
        "first": {"x": 0.5, "y": 0.5, "width": 1.5, "height": 1.5},
        "second": {"x": 0.5, "y": 0.5, "width": 1.5, "height": 1.5},
    }
    validator = importlib.import_module("jsonschema").Draft202012Validator
    validator(contract["input_schema"]["properties"]["parameters"]).validate(parameters)
    actual = bounding_box_iou(
        BoundingBox(**parameters["first"]), BoundingBox(**parameters["second"])
    )
    validator(contract["output_schema"]["properties"]["result"]).validate({"value": actual})
    assert actual == 1.0


@pytest.mark.parametrize("include_documents", [False, True])
def test_rerank_contract_preserves_canonical_render_modes(include_documents: bool) -> None:
    from capicove.retrieval.reranking import RerankResult, rank_scores, render_rerank_result

    result = RerankResult("model", "revision", rank_scores((1.0, 0.5), top_n=2), 13)
    payload = render_rerank_result(result, ("first", "second"), return_documents=include_documents)
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.retrieval.rerank"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


@pytest.mark.parametrize("kind", ["clip", "transcribe"])
def test_audio_capability_contract_preserves_actual_unavailable_profile(kind: str) -> None:
    from capicove.cli.commands.audio_clip import clip_capabilities
    from capicove.tools.audio_transcribe import audio_transcribe_capabilities

    if kind == "clip":
        payload = clip_capabilities("/nonexistent/capicove-ffmpeg", "/nonexistent/capicove-ffprobe")
    else:
        payload = audio_transcribe_capabilities(
            executable_available=False,
            vad_executable_available=False,
            decoder_available=False,
            probe_available=False,
            model_available=False,
            model_sha256=None,
            vad_model_available=False,
            vad_model_sha256=None,
            backend_version="unavailable",
        )
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.audio.capabilities"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


def test_native_knowledge_doctor_preserves_actual_readiness(tmp_path: Path) -> None:
    from capicove.runtime.knowledge_doctor import doctor
    from tests.unit.runtime.test_native_knowledge import configured

    config = configured(tmp_path)
    payload = doctor(config)
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.knowledge.doctor"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]["properties"]["readiness"]
    ).validate(payload)


def test_native_text_preparation_lineage_validates_portable_payload(tmp_path: Path) -> None:
    import hashlib

    from capicove.runtime.knowledge_prepare import prepare_source
    from capicove.runtime.media_handoff import StagedFile
    from tests.unit.runtime.test_native_knowledge import configured

    config = configured(tmp_path)
    path = config.source_roots["docs"] / "source.md"
    content = b"Immutable text evidence"
    path.write_bytes(content)
    _, lineage = prepare_source(
        config, StagedFile(path, path, hashlib.sha256(content).hexdigest(), len(content)), tmp_path
    )
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    schema = {"$defs": document["$defs"], "$ref": "#/$defs/NativeKnowledgeLineageV1"}
    importlib.import_module("jsonschema").Draft202012Validator(schema).validate(lineage)


def test_difference_area_contract_preserves_actual_pipeline_stages() -> None:
    import io

    from PIL import Image

    from capicove.processing.image.mappers.difference_area import (
        ImgDiffDifferenceAreaGeneratorMapper,
    )
    from capicove.processing.image.models import ImageSource
    from capicove.tools.image import _serialize_result

    sources = []
    for color in ("red", "blue"):
        buffer = io.BytesIO()
        Image.new("RGB", (16, 16), color).save(buffer, format="PNG")
        sources.append(ImageSource(data=buffer.getvalue()))
    result = ImgDiffDifferenceAreaGeneratorMapper().map(tuple(sources))
    payload, _ = _serialize_result(result, "imgdiff_difference_area_generator_mapper")
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.image.diff.areas"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


def test_empty_background_result_preserves_canonical_lineage() -> None:
    from capicove.tools.image import _serialize_result

    payload, _ = _serialize_result((), "image_remove_background_mapper")
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.image.remove-background"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


@pytest.mark.parametrize("kind", ["image", "video"])
def test_search_contract_accepts_actual_canonical_codec(kind: str) -> None:
    from capicove.contracts import FullHit, SearchResult, SearchStatus, TargetType
    from capicove.tools.image_search import _serialize_search_result
    from capicove.tools.search import encode_search_result

    if kind == "image":
        payload = _serialize_search_result(SearchResult(SearchStatus.SUCCEEDED, "query"))
    else:
        hit = FullHit(
            "keyframe",
            TargetType.KEYFRAME,
            0.5,
            "evidence",
            timestamp_ms=3,
            preview_resource_ref="resource:preview:keyframe",
        )
        payload = encode_search_result(SearchResult(SearchStatus.SUCCEEDED, "query", hits=(hit,)))
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.search." + kind
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


@pytest.mark.parametrize("operation", ["import", "forget"])
def test_source_ledger_contract_preserves_actual_receipt(tmp_path: Path, operation: str) -> None:
    from capicove.runtime.local_retrieval import forget_source, import_source
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
    from tests.unit.runtime.test_local_retrieval import Workspace

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "source.md").write_text("Immutable source evidence")
    workspace = Workspace(root, notes, root / "index", root / "host.json")
    config = load_mcp_host_config(workspace.write_config())
    assert isinstance(config, WorkspaceHostConfiguration)
    payload = import_source(config, alias="notes", relative_path="source.md")
    if operation == "forget":
        payload = forget_source(config, alias="notes", relative_path="source.md")
    assert payload["source_ref"] == "notes/source.md"
    payload["source_ref"] = "owned:source-one"
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.source." + operation
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


def test_local_evidence_sections_validate_portable_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from capicove.runtime.local_retrieval import import_source, open_local_retrieval
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config
    from tests.unit.runtime.test_local_retrieval import (
        Workspace,
        build,
        context_for,
        first_hit,
        search,
    )

    root = tmp_path / "workspace"
    root.mkdir(mode=0o700)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "source.md").write_text("Paris source evidence")
    workspace = Workspace(root, notes, root / "index", root / "host.json")
    config = load_mcp_host_config(workspace.write_config())
    assert isinstance(config, WorkspaceHostConfiguration)
    import_source(config, alias="notes", relative_path="source.md")
    assert build(capsys, workspace)[0] == 0
    code, found = search(capsys, workspace, "Paris")
    assert code == 0
    evidence_id = str(first_hit(found)["evidence_id"])
    result = open_local_retrieval(config).evidence_handler(
        context_for("workspace"), {"evidence_id": evidence_id, "sections": ["metadata", "lineage"]}
    )
    assert result.result is not None
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    for section in ("metadata", "lineage"):
        section_payload = result.result[section]
        assert isinstance(section_payload, dict)
        payload = dict(section_payload)
        if section == "metadata":
            payload["source_ref"] = "owned:source-one"
        schema = {
            "$defs": document["$defs"],
            "$ref": "#/$defs/TextEvidence" + section.title() + "V1",
        }
        importlib.import_module("jsonschema").Draft202012Validator(schema).validate(payload)


def test_duplicate_scan_and_manifest_preserve_real_file_results(tmp_path: Path) -> None:
    from dataclasses import asdict

    from capicove.processing.video.deduplication.file import deduplicate_video_files
    from capicove.processing.video.deduplication.manifest import write_deduplication_manifest

    for name in ("a.mp4", "b.mp4"):
        (tmp_path / name).write_bytes(b"identical immutable video fixture")
    result = deduplicate_video_files([tmp_path])
    assert len(result.duplicate_groups) == 1
    payload = asdict(result)
    handles = {record.path: "owned:" + record.path.name for record in result.files}
    payload["files"] = [{"path": handles[r.path], "size_bytes": r.size_bytes} for r in result.files]
    payload["unique_files"] = [handles[path] for path in result.unique_files]
    payload["duplicate_groups"] = [
        {
            **asdict(group),
            "keeper": handles[group.keeper],
            "eliminated": [handles[path] for path in group.eliminated],
        }
        for group in result.duplicate_groups
    ]
    payload["failures"] = []
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    validator = importlib.import_module("jsonschema").Draft202012Validator
    validator({"$defs": document["$defs"], "$ref": "#/$defs/DuplicateScanPayloadV1"}).validate(
        payload
    )
    manifest = tmp_path / "manifest.json"
    write_deduplication_manifest(manifest, result)
    data = json.loads(manifest.read_text())
    for item in data["items"]:
        for key in ("path", "keeper_path", "moved_to"):
            if item[key]:
                item[key] = handles[Path(item[key])]
    validator({"$defs": document["$defs"], "$ref": "#/$defs/DuplicateManifestPayloadV1"}).validate(
        data
    )


def test_video_split_contract_preserves_actual_codec_fields() -> None:
    from capicove.tools.video_split import _serialize_result
    from tests.unit.tools.test_video_split import make_processing_result

    payload = _serialize_result(make_processing_result())
    for item in cast(
        list[dict[str, Any]], [payload["original"], *cast(list[Any], payload["segments"])]
    ):
        item.pop("base64")
        item.pop("type")
        item["artifact"] = {
            "artifact_id": "video:immutable",
            "sha256": item["sha256"],
            "version": 1,
        }
    contract = next(
        op for op in core_manifest()["operations"] if op["operation_id"] == "capicove.video.split"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


def test_video_filter_contract_preserves_actual_codec_fields() -> None:
    from capicove.tools.video_filter import _serialize_result
    from tests.unit.tools.test_video_filter import _result

    payload = _serialize_result(_result())
    contract = next(
        op for op in core_manifest()["operations"] if op["operation_id"] == "capicove.video.filter"
    )
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["output_schema"]["properties"]["result"]
    ).validate(payload)


def test_video_publication_allows_initial_alias_and_requires_revision() -> None:
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.generation.video.publish_prepared"
    )
    parameters = {
        "prepared_manifest": {"artifact_id": "prepared:one", "sha256": "a" * 64, "version": 1},
        "expected_active_generation": None,
        "expected_active_revision": 0,
    }
    importlib.import_module("jsonschema").Draft202012Validator(
        contract["input_schema"]["properties"]["parameters"]
    ).validate(parameters)


def test_video_preparation_payload_preserves_real_service_serialization(tmp_path: Path) -> None:
    from dataclasses import asdict

    from capicove.processing.video.retrieval.pipeline import canonical_json
    from tests.unit.processing.video.retrieval.test_pipeline import request, setup_service

    service, _, _, _ = setup_service(tmp_path)
    result = service.prepare(request())
    payload = json.loads(canonical_json(asdict(result)))
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    importlib.import_module("jsonschema").Draft202012Validator(
        {"$defs": document["$defs"], "$ref": "#/$defs/VideoPreparationPayloadV1"}
    ).validate(payload)


def test_image_generation_payloads_accept_actual_canonical_codecs(tmp_path: Path) -> None:
    from capicove.retrieval.image_generations import encode_generation_set_manifest
    from tests.unit.runtime.test_retrieval_build import FUTURE, _build

    administrator, _, built = _build(tmp_path, count=3)
    validated = administrator.validate(built, deadline=FUTURE)
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    validator = importlib.import_module("jsonschema").Draft202012Validator
    payloads = {
        "ImageGenerationManifestV1": json.loads(encode_generation_set_manifest(validated.manifest)),
        "TargetRepresentationManifestV1": json.loads(
            built.representation_result.sealed_manifest.canonical_bytes
        ),
    }
    for name, payload in payloads.items():
        validator({"$defs": document["$defs"], "$ref": "#/$defs/" + name}).validate(payload)


@pytest.mark.parametrize("kind", ["text", "visual"])
def test_generation_member_report_payload_preserves_contract_fields(kind: str) -> None:
    from dataclasses import asdict

    from capicove.retrieval.generations import TextIndexValidationReport
    from capicove.retrieval.video_models import VisualIndexValidationReport

    report = (
        TextIndexValidationReport("generation", "a" * 64, "/index", "b" * 64, 1, 1, ())
        if kind == "text"
        else VisualIndexValidationReport("generation", "a" * 64, "/index", "b" * 64, 1, 1, 2, ())
    )
    payload = asdict(report)
    payload["physical_index_ref"] = "owned:index"
    payload["issue_codes"] = list(report.issue_codes)
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    importlib.import_module("jsonschema").Draft202012Validator(
        {"$defs": document["$defs"], "$ref": "#/$defs/" + kind.title() + "ValidationReportV1"}
    ).validate(payload)


def test_video_publication_inventories_exact_numpy_composition() -> None:
    contract = next(
        op
        for op in core_manifest()["operations"]
        if op["operation_id"] == "capicove.generation.video.publish_prepared"
    )
    assert "video-local-exact" in {
        binding["profile_id"] for binding in contract["provider_bindings"]
    }


def test_image_build_and_phase_inputs_form_typed_chain() -> None:
    ops = {op["operation_id"]: op for op in core_manifest()["operations"]}
    artifact = {"artifact_id": "source:snapshot", "sha256": "a" * 64, "version": 1}
    validator = importlib.import_module("jsonschema").Draft202012Validator
    validator(
        ops["capicove.generation.image.build"]["input_schema"]["properties"]["parameters"]
    ).validate(
        {
            "source_snapshot": artifact,
            "build_plan": {
                "build_id": "build:one",
                "activation_set_id": "set:one",
                "corpus_ref": "corpus:one",
                "batch_size": 64,
            },
            "retain_representation_evidence": False,
        }
    )
    for phase in ("validate", "publish", "abort"):
        validator(
            ops["capicove.generation.image." + phase]["input_schema"]["properties"]["parameters"]
        ).validate({"staging_handle": "owned:stage", "expected_revision": 1})
    validator(
        ops["capicove.generation.image.register_published"]["input_schema"]["properties"][
            "parameters"
        ]
    ).validate({"generation_root_handle": "owned:published"})


@pytest.mark.parametrize(
    "mutation", ["nested_any", "missing_payload", "empty_projection", "wrong_owner"]
)
def test_core_semantic_mutations_fail_closed(mutation: str) -> None:
    manifest, module = core_manifest(), checker()
    op = next(
        item for item in manifest["operations"] if item["operation_id"] == "capicove.audio.clip"
    )
    if mutation == "nested_any":
        op["input_schema"]["properties"]["parameters"]["properties"]["source"] = {}
        op["input_schema_digest"] = module.digest(op["input_schema"])
    elif mutation == "missing_payload":
        op["payload_contracts"] = {"result": "MissingPayloadV99"}
    elif mutation == "empty_projection":
        op["output_schema"] = {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
            "required": [],
        }
        op["output_schema_digest"] = module.digest(op["output_schema"])
        op["semantic_audit"]["field_projection"] = {}
    else:
        next(
            mapping
            for mapping in manifest["source_mappings"]
            if mapping["candidate_operation_id"]
            == "provider:plugins/retriever-elasticsearch/src/capicove_retriever_elasticsearch/plugin.py:ElasticsearchRetrieverPlugin"
        )["enclosing_operations"] = ["capicove.audio.import"]
    assert module.validate_core_manifest(manifest, ROOT)


def test_video_filter_composite_profile_covers_every_criterion() -> None:
    manifest = core_manifest()
    profile = next(
        p for p in manifest["profiles"] if p["profile_id"] == "video-filter-composite-local"
    )
    assert set(profile["criterion_bindings"]) == {
        "metadata",
        "motion",
        "ocr_area",
        "aesthetics",
        "nsfw",
        "watermark",
        "tags",
        "frame_text_similarity",
    }
    op = next(o for o in manifest["operations"] if o["operation_id"] == "capicove.video.filter")
    assert profile["profile_id"] in op["input_schema"]["properties"]["profile_id"]["enum"]


def test_image_build_source_snapshot_preserves_actual_native_summary() -> None:
    from dataclasses import asdict

    from capicove.processing.video.retrieval.pipeline import canonical_json
    from capicove.runtime.retrieval_representations import ImageBuildSource
    from tests.unit.runtime.test_retrieval_build import _sources

    source = ImageBuildSource(_sources(3))
    payload = {
        "schema_version": 1,
        "summary": json.loads(canonical_json(asdict(source.summary))),
        "sources": [
            {
                "ordinal": item.ordinal,
                "unit": json.loads(canonical_json(asdict(item.unit))),
                "source": {
                    "artifact_id": "source:" + str(item.ordinal),
                    "sha256": item.unit.source_hash,
                    "version": 1,
                },
                "media_type": item.source.media_type,
            }
            for item in source
        ],
    }
    document = json.loads((ROOT / "docs/development/remote-mcp-core-payloads.json").read_text())
    importlib.import_module("jsonschema").Draft202012Validator(
        {"$defs": document["$defs"], "$ref": "#/$defs/ImageBuildSourceSnapshotV1"}
    ).validate(payload)


def test_image_validation_has_successful_retained_evidence_finalization() -> None:
    operation = next(
        o
        for o in core_manifest()["operations"]
        if o["operation_id"] == "capicove.generation.image.validate"
    )
    parameters = operation["input_schema"]["properties"]["parameters"]
    importlib.import_module("jsonschema").validate(
        {
            "staging_handle": "owned:stage",
            "expected_revision": 1,
            "finalize_representation_evidence": True,
        },
        parameters,
    )
    assert "without deleting" in operation["lifecycle_transition"]["evidence_lease"]


def test_source_manifest_ignores_local_environments_and_hidden_directories(
    tmp_path: Path,
) -> None:
    root = tmp_path / "checkout"
    (root / "src/capicove/mcp").mkdir(parents=True)
    (root / "src/capicove/mcp/server.py").write_text("SERVED = True\n")
    (root / "docs/development").mkdir(parents=True)
    (root / "docs/development/remote-mcp-capability-coverage.md").write_text("# fixture\n")
    plugin = root / "plugins/processor-fixture"
    plugin.mkdir(parents=True)
    (plugin / "pyproject.toml").write_text("[project]\nname = 'capicove-fixture'\n")
    baseline = checker().source_manifest(root)
    for hidden in (".venv/lib/site-packages/torch/fx.py", ".tox/tool.toml", ".git/hook.json"):
        path = plugin / hidden
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("UNTRACKED = True\n")
    (root / "src/.cache/state.json").parent.mkdir(parents=True)
    (root / "src/.cache/state.json").write_text("{}\n")
    assert checker().source_manifest(root) == baseline
    assert "src/capicove/mcp/server.py" in baseline
