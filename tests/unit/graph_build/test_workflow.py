"""Closed scheduling and durable request identity survive hostile input/restart."""

import hashlib
import importlib
import importlib.util
import sqlite3
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.codec import canonical_bytes
from tests.graph_build_fixtures import Gate


SCOPE = GraphScope("workspace:test", "scope:test")


def workflow() -> Any:
    assert importlib.util.find_spec("capicove.graph_build.workflow") is not None, (
        "Task 22 closed workflow is missing"
    )
    return importlib.import_module("capicove.graph_build.workflow")


def test_diagnostics_do_not_feed_community_clustering() -> None:
    graph = workflow().stage_dependencies()
    assert "diagnostics" not in graph["communities"]
    assert {"communities", "diagnostics"} <= set(graph["reports"])
    assert graph["retrieval"] == ("publication",)
    with pytest.raises(TypeError):
        graph["communities"] = ("diagnostics",)
    assert graph is not workflow().stage_dependencies()


@pytest.mark.parametrize("damage", ["cycle", "dynamic", "missing", "document"])
def test_only_the_accepted_dag_can_be_scheduled(damage: str) -> None:
    module = workflow()
    graph = dict(module.stage_dependencies())
    if damage == "cycle":
        graph["representations"] = ("reports",)
    elif damage == "dynamic":
        graph["download"] = ("publication",)
    elif damage == "missing":
        del graph["alignment"]
    else:
        graph["communities"] = ("aggregation", "diagnostics")
    with pytest.raises(ValueError):
        module.validate_dependencies(graph)


def request() -> Any:
    from tests.unit.contracts.test_graph_snapshots import dataset

    return workflow().GraphBuildRequest(dataset(), SCOPE, "profile:test", None, "reuse")


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "retry"},
        {"mode": True},
        {"profile_id": ""},
        {"baseline_snapshot_id": " "},
        {"scope": "scope:test"},
        {"input_snapshot_ref": "latest"},
    ],
)
def test_build_request_rejects_unpinned_or_unknown_values(changes: dict[str, Any]) -> None:
    original = request()
    with pytest.raises((TypeError, ValueError)):
        replace(original, **changes)
    with pytest.raises(FrozenInstanceError):
        original.mode = "regenerate"


def test_run_identity_is_durable_fresh_and_scope_safe(tmp_path: Path) -> None:
    module = workflow()
    gate = Gate()
    store = module.BuildRunStore(tmp_path / "runs.sqlite", gate)
    original = request()
    runtime = module.canonical_bytes({"version": 1, "runtime": "test"})
    first = store.create(
        original,
        profile_fingerprint="a" * 64,
        input_digest="b" * 64,
        deadline_at=2_000_000_000.0,
        runtime_payload=runtime,
    )
    second = store.create(
        original,
        profile_fingerprint="a" * 64,
        input_digest="b" * 64,
        deadline_at=2_000_000_000.0,
        runtime_payload=runtime,
    )
    assert first.run_id != second.run_id
    reopened = module.BuildRunStore(tmp_path / "runs.sqlite", gate)
    assert reopened.get(first.run_id, SCOPE) == first
    assert first.request == original
    with pytest.raises(PermissionError):
        reopened.get(first.run_id, replace(SCOPE, scope_ref="other"))
    reopened.cancel(first.run_id, SCOPE)
    reopened.cancel(first.run_id, SCOPE)
    assert reopened.get(first.run_id, SCOPE).cancelled
    assert not reopened.get(second.run_id, SCOPE).cancelled
    gate.denied_scopes.add(SCOPE)
    with pytest.raises(PermissionError):
        reopened.cancel(second.run_id, SCOPE)


def test_committed_run_request_cannot_be_reinterpreted(tmp_path: Path) -> None:
    module = workflow()
    store = module.BuildRunStore(tmp_path / "runs.sqlite", Gate())
    run = store.create(
        request(),
        profile_fingerprint="a" * 64,
        input_digest="b" * 64,
        deadline_at=2_000_000_000.0,
    )
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute("UPDATE build_runs SET request = ?", (b"{}",))
    with pytest.raises(ValueError):
        store.get(run.run_id, SCOPE)


def _store(tmp_path: Path, gate: Gate | None = None) -> Any:
    return workflow().BuildRunStore(tmp_path / "runs.sqlite", gate or Gate())


def _run(store: Any, *, baseline: bool = False) -> Any:
    build_request = request()
    if baseline:
        build_request = replace(build_request, baseline_snapshot_id="snapshot:baseline")
    return store.create(
        build_request,
        profile_fingerprint="a" * 64,
        input_digest="b" * 64,
        baseline_digest="c" * 64 if baseline else None,
        deadline_at=2_000_000_000.0,
        runtime_payload=workflow().canonical_bytes({"version": 1, "runtime": "test"}),
    )


def _result_payload(run_id: str, status: str = "succeeded") -> bytes:
    return canonical_bytes(
        {
            "version": 1,
            "run_id": run_id,
            "status": status,
            "graph_ref": None,
            "completeness": "complete" if status == "succeeded" else "partial",
            "coverage": [],
            "issue_codes": [],
            "retrieval_binding_ref": None,
        }
    )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("profile_digest", "A" * 64),
        ("input_digest", "short"),
        ("baseline_digest", "not-a-digest"),
        ("deadline_at", -1.0),
        ("status", "complete"),
        ("cancelled", 2),
        ("workspace", "workspace:other"),
    ],
)
def test_run_rows_are_closed_and_integrity_checked(
    tmp_path: Path, field: str, value: object
) -> None:
    store = _store(tmp_path)
    run = _run(store, baseline=True)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(f"UPDATE build_runs SET {field} = ?", (value,))
    expected_error = PermissionError if field == "workspace" else ValueError
    with pytest.raises(expected_error):
        store.get(run.run_id, SCOPE)


def test_baseline_pin_presence_is_exact(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create(
            request(),
            profile_fingerprint="a" * 64,
            input_digest="b" * 64,
            baseline_digest="c" * 64,
            deadline_at=2_000_000_000.0,
        )
    with pytest.raises(ValueError):
        store.create(
            replace(request(), baseline_snapshot_id="snapshot:baseline"),
            profile_fingerprint="a" * 64,
            input_digest="b" * 64,
            deadline_at=2_000_000_000.0,
        )


@pytest.mark.parametrize("deadline", [True, 0.0, -1.0, float("inf"), float("nan")])
def test_deadline_pin_must_be_a_finite_positive_timestamp(tmp_path: Path, deadline: object) -> None:
    store = _store(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        store.create(
            request(),
            profile_fingerprint="a" * 64,
            input_digest="b" * 64,
            deadline_at=deadline,
        )


def test_deadline_and_runtime_pins_survive_restart(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runtime = workflow().canonical_bytes(
        {"version": 1, "embedding_space": "space:identity", "tokenizer": "local:v1"}
    )
    run = store.create(
        request(),
        profile_fingerprint="a" * 64,
        input_digest="b" * 64,
        deadline_at=2_000_000_000.25,
        runtime_payload=runtime,
    )
    reopened = _store(tmp_path)
    assert reopened.get(run.run_id, SCOPE).deadline_at == 2_000_000_000.25
    assert reopened.runtime(run.run_id, SCOPE) == runtime
    reopened.pin_runtime(run.run_id, SCOPE, runtime)
    with pytest.raises(ValueError):
        reopened.pin_runtime(
            run.run_id,
            SCOPE,
            workflow().canonical_bytes({"version": 1, "tokenizer": "changed"}),
        )


def test_runtime_pin_is_required_before_effects_and_integrity_checked(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    run = store.create(
        request(),
        profile_fingerprint="a" * 64,
        input_digest="b" * 64,
        deadline_at=2_000_000_000.0,
    )
    assert store.runtime(run.run_id, SCOPE) is None
    with pytest.raises(ValueError):
        store.record_stage(run.run_id, SCOPE, "representations", "build-artifact:" + "1" * 64)
    with pytest.raises(ValueError):
        store.pin_runtime(run.run_id, SCOPE, b' {"version":1}')
    payload = workflow().canonical_bytes({"version": 1, "runtime": "test"})
    store.pin_runtime(run.run_id, SCOPE, payload)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute("UPDATE build_runtime SET digest = ?", ("0" * 64,))
    with pytest.raises(ValueError):
        store.runtime(run.run_id, SCOPE)


@pytest.mark.parametrize("field", ["profile_digest", "input_digest", "deadline_at"])
def test_valid_looking_pin_substitution_is_detected(tmp_path: Path, field: str) -> None:
    store = _store(tmp_path)
    run = _run(store)
    value: object = 2_000_000_001.0 if field == "deadline_at" else "d" * 64
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(f"UPDATE build_runs SET {field} = ?", (value,))
    with pytest.raises(ValueError):
        store.get(run.run_id, SCOPE)


def test_valid_looking_runtime_and_stage_substitution_is_detected(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _run(store)
    changed = workflow().canonical_bytes({"version": 1, "runtime": "changed"})
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(
            "UPDATE build_runtime SET payload = ?, digest = ?",
            (changed, hashlib.sha256(changed).hexdigest()),
        )
    with pytest.raises(ValueError):
        store.runtime(run.run_id, SCOPE)

    second = _run(store)
    store.record_stage(second.run_id, SCOPE, "representations", "build-artifact:" + "1" * 64)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(
            "UPDATE build_stages SET artifact_ref = ? WHERE run_id = ?",
            ("build-artifact:" + "2" * 64, second.run_id),
        )
    with pytest.raises(ValueError):
        store.stages(second.run_id, SCOPE)


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_run_reads_and_new_effects_revalidate_runtime_pin(tmp_path: Path, damage: str) -> None:
    store = _store(tmp_path)
    run = _run(store)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        if damage == "missing":
            connection.execute("DELETE FROM build_runtime")
        else:
            changed = workflow().canonical_bytes({"version": 1, "runtime": "changed"})
            connection.execute(
                "UPDATE build_runtime SET payload = ?, digest = ?",
                (changed, hashlib.sha256(changed).hexdigest()),
            )
    with pytest.raises(ValueError):
        store.get(run.run_id, SCOPE)
    with pytest.raises(ValueError):
        store.record_stage(run.run_id, SCOPE, "representations", "build-artifact:" + "1" * 64)


@pytest.mark.parametrize("damage", ["duplicate", "noncanonical", "utf8", "deep"])
def test_persisted_request_requires_bounded_canonical_json(tmp_path: Path, damage: str) -> None:
    store = _store(tmp_path)
    run = _run(store)
    original = workflow().request_bytes(request())
    if damage == "duplicate":
        payload = original[:-1] + b',"mode":"reuse"}'
    elif damage == "noncanonical":
        payload = b" " + original
    elif damage == "utf8":
        payload = b'{"version":1,"bad":"\xff"}'
    else:
        payload = b'{"version":1,"input_snapshot_ref":' + b"[" * 40 + b"0" + b"]" * 40 + b"}"
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(
            "UPDATE build_runs SET request = ?, request_digest = ?",
            (payload, hashlib.sha256(payload).hexdigest()),
        )
    with pytest.raises(ValueError):
        store.get(run.run_id, SCOPE)


def test_store_refuses_unsafe_paths_and_schema_tampering(tmp_path: Path) -> None:
    module = workflow()
    with pytest.raises(ValueError):
        module.BuildRunStore(Path("relative.sqlite"), Gate())
    target = tmp_path / "target.sqlite"
    target.touch(mode=0o600)
    link = tmp_path / "link.sqlite"
    link.symlink_to(target)
    with pytest.raises((ValueError, OSError, RuntimeError)):
        module.BuildRunStore(link, Gate())

    store = _store(tmp_path)
    _run(store)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute("CREATE TABLE injected (value TEXT)")
    with pytest.raises(ValueError):
        store.get("build:" + "0" * 32, SCOPE)


def test_stage_records_are_an_exact_immutable_prefix(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _run(store)
    with pytest.raises(ValueError):
        store.record_stage(run.run_id, SCOPE, "text_units", "build-artifact:" + "1" * 64)
    stages = tuple(workflow().stage_dependencies())
    for index, stage in enumerate(stages[:7]):
        store.record_stage(run.run_id, SCOPE, stage, "build-artifact:" + f"{index:064x}")
    with pytest.raises(ValueError):
        store.record_stage(run.run_id, SCOPE, "communities", "build-artifact:" + "8" * 64)
    with pytest.raises(ValueError):
        store.record_stage(run.run_id, SCOPE, "representations", "build-artifact:" + "f" * 64)


@pytest.mark.parametrize(
    ("stage", "artifact_ref"),
    [("foreign", "build-artifact:" + "1" * 64), ("representations", " ")],
)
def test_stages_reject_tampered_names_and_references(
    tmp_path: Path, stage: str, artifact_ref: str
) -> None:
    store = _store(tmp_path)
    run = _run(store)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(
            "INSERT INTO build_stages VALUES (?, ?, ?, ?)",
            (run.run_id, stage, artifact_ref, "0" * 64),
        )
    with pytest.raises(ValueError):
        store.stages(run.run_id, SCOPE)


def test_terminal_result_is_canonical_identity_bound_and_immutable(tmp_path: Path) -> None:
    store = _store(tmp_path)
    run = _run(store)
    payload = _result_payload(run.run_id)
    with pytest.raises(ValueError):
        store.finish(run.run_id, SCOPE, "failed", payload, expected_revision=0)
    store.finish(run.run_id, SCOPE, "succeeded", payload, expected_revision=0)
    store.finish(run.run_id, SCOPE, "succeeded", payload, expected_revision=0)
    store.cancel(run.run_id, SCOPE)
    assert store.get(run.run_id, SCOPE).status == "succeeded"
    assert not store.get(run.run_id, SCOPE).cancelled
    with pytest.raises(ValueError):
        store.finish(
            run.run_id,
            SCOPE,
            "partial",
            _result_payload(run.run_id, "partial"),
            expected_revision=0,
        )
    with pytest.raises(ValueError):
        store.record_stage(run.run_id, SCOPE, "representations", "build-artifact:" + "1" * 64)


@pytest.mark.parametrize("damage", ["digest", "run_id", "status", "json"])
def test_terminal_result_tampering_is_detected(tmp_path: Path, damage: str) -> None:
    store = _store(tmp_path)
    run = _run(store)
    store.finish(
        run.run_id,
        SCOPE,
        "succeeded",
        _result_payload(run.run_id),
        expected_revision=0,
    )
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        if damage == "digest":
            connection.execute("UPDATE build_results SET digest = ?", ("0" * 64,))
        elif damage == "run_id":
            payload = _result_payload("build:" + "f" * 32)
            connection.execute(
                "UPDATE build_results SET payload = ?, digest = ?",
                (payload, hashlib.sha256(payload).hexdigest()),
            )
        elif damage == "status":
            with pytest.raises(sqlite3.IntegrityError, match="immutable build terminal"):
                connection.execute("UPDATE build_runs SET status = 'partial'")
            return
        else:
            payload = b" " + _result_payload(run.run_id)
            connection.execute(
                "UPDATE build_results SET payload = ?, digest = ?",
                (payload, hashlib.sha256(payload).hexdigest()),
            )
    with pytest.raises(ValueError):
        store.result(run.run_id, SCOPE)


def test_joint_terminal_status_cancel_and_result_rehash_cannot_change_cause(
    tmp_path: Path,
) -> None:
    """The terminal transition has an independent commitment beyond mutable flags."""
    store = _store(tmp_path)
    run = _run(store)
    original = _result_payload(run.run_id, "failed")
    store.finish(run.run_id, SCOPE, "failed", original, expected_revision=0)
    forged = original.replace(b'"issue_codes":[]', b'"issue_codes":["forged"]')
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute(
            "UPDATE build_results SET payload = ?, digest = ?",
            (forged, hashlib.sha256(forged).hexdigest()),
        )
        with pytest.raises(sqlite3.IntegrityError, match="immutable build terminal"):
            connection.execute(
                "UPDATE build_runs SET status = 'cancelled', cancelled = 1, "
                "terminal_prior_status = 'running', terminal_prior_cancelled = 0, "
                "terminal_digest = ?",
                ("f" * 64,),
            )

    with pytest.raises(ValueError, match="build_terminal_commitment_failure"):
        store.result(run.run_id, SCOPE)


def test_missing_terminal_immutability_trigger_fails_closed_on_reopen(tmp_path: Path) -> None:
    """Removing the write-once guard makes the entire run store unsupported."""
    store = _store(tmp_path)
    _run(store)
    with sqlite3.connect(tmp_path / "runs.sqlite") as connection:
        connection.execute("DROP TRIGGER immutable_build_terminal")

    with pytest.raises(ValueError, match="schema is unsupported or corrupt"):
        _store(tmp_path)


def test_revision_change_rolls_back_mutations(tmp_path: Path) -> None:
    class ChangingGate(Gate):
        change_on_exit = False

        def guard(self, expected_revision: int) -> Any:
            from contextlib import contextmanager

            @contextmanager
            def changing() -> Any:
                if self.revision != expected_revision:
                    raise PermissionError("policy changed")
                yield
                if self.change_on_exit:
                    self.revision += 1
                if self.revision != expected_revision:
                    raise PermissionError("policy changed")

            return changing()

    gate = ChangingGate()
    store = _store(tmp_path, gate)
    run = _run(store)
    gate.change_on_exit = True
    with pytest.raises(PermissionError):
        store.record_stage(run.run_id, SCOPE, "representations", "build-artifact:" + "1" * 64)
    gate.change_on_exit = False
    assert store.stages(run.run_id, SCOPE) == ()
