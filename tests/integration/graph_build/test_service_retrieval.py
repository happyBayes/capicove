"""Optional retrieval composition for the fixed graph build service."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from capicove.graph_build._service_retrieval import (
    BuildRetrievalRuntime,
    build_retrieval,
)
from capicove.retrieval import TextIndexGenerationState
from tests.graph_build_fixtures import SCOPE
from tests.integration.graph_build.test_build_service import authored
from tests.integration.graph_build.test_retrieval_binding import (
    RecordingIndex,
    build,
    build_setup,
    runtime,
)
from tests.unit.graph_build.test_retrieval_projection import module as retrieval_module
from tests.unit.retrieval.test_text_embedding import Encoder


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def retrieval_runtime(
    tmp_path: Path, encoder: Encoder | None = None
) -> tuple[BuildRetrievalRuntime, tuple[Any, ...]]:
    values = build_setup(tmp_path)
    module, _, _, _, reader, generations, bindings, index, *_ = values
    configured = BuildRetrievalRuntime(
        reader=reader,
        embedder=runtime(module, encoder),
        lifecycle=index,
        writer=index,
        generations=generations,
        bindings=bindings,
        index_format_version="recording-v1",
        batch_size=2,
    )
    return configured, values


def test_missing_optional_runtime_returns_bounded_unavailable_result() -> None:
    result = build_retrieval("run:one", None, SCOPE, None)

    assert result.status == "unavailable"
    assert result.binding_id is None
    assert result.issue_codes == ("retrieval_runtime_unavailable",)


def test_runtime_fingerprint_is_stable_and_configuration_specific(tmp_path: Path) -> None:
    configured, _ = retrieval_runtime(tmp_path)

    assert configured.fingerprint == configured.fingerprint
    assert configured.fingerprint != replace(configured, batch_size=1).fingerprint


def test_build_registers_exact_inactive_binding(tmp_path: Path) -> None:
    configured, values = retrieval_runtime(tmp_path)
    _, _, graph_catalog, graph_ref, _, generations, _, index, *_ = values

    result = build_retrieval("run:one", graph_ref, SCOPE, configured)

    assert result.status == "ready"
    assert result.binding_id is not None
    assert result.issue_codes == ()
    assert index.events == ["create", "write", "write", "validate"]
    assert graph_catalog.active(SCOPE) == (None, 0)
    assert (
        generations.active(workspace_ref=SCOPE.workspace_ref, retriever_id=index.provider_id)
        is None
    )


def test_ready_attempt_resumes_binding_without_duplicate_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    encoder = Encoder()
    configured, values = retrieval_runtime(tmp_path, encoder)
    _, _, _, graph_ref, _, _, bindings, index, *_ = values
    original = type(bindings).register
    calls = 0

    def interrupt_once(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("crash before binding commit")
        return original(self, *args, **kwargs)

    monkeypatch.setattr(type(bindings), "register", interrupt_once)
    first = build_retrieval("run:resume", graph_ref, SCOPE, configured)
    events_after_crash = tuple(index.events)
    embedding_calls_after_crash = tuple(encoder.calls)
    second = build_retrieval("run:resume", graph_ref, SCOPE, configured)

    assert first.status == "failed"
    assert first.issue_codes == ("retrieval_build_failed",)
    assert bindings.attempt("run:resume", SCOPE).status == "ready"
    assert second.status == "ready"
    assert second.binding_id is not None
    assert tuple(index.events) == events_after_crash
    assert tuple(encoder.calls) == embedding_calls_after_crash


def test_catalog_resolves_only_ready_attempt_to_validated_generation(tmp_path: Path) -> None:
    _, values = retrieval_runtime(tmp_path)
    _, _, _, _, _, _, bindings, _, _, _, commitment = values
    validated = build(values, "run:validated")

    assert bindings.validated_attempt("run:missing", SCOPE) is None
    assert bindings.validated_attempt("run:validated", SCOPE) == validated
    assert bindings.validated_attempt("run:validated", SCOPE).projection == commitment


def test_failed_partial_attempt_is_terminal_under_same_run_id(tmp_path: Path) -> None:
    configured, values = retrieval_runtime(tmp_path)
    _, _, _, graph_ref, _, generations, bindings, index, *_ = values
    index.fail_write = True

    first = build_retrieval("run:partial", graph_ref, SCOPE, configured)
    events_after_failure = tuple(index.events)
    second = build_retrieval("run:partial", graph_ref, SCOPE, configured)

    assert first.status == second.status == "failed"
    assert first.issue_codes == second.issue_codes == ("retrieval_build_failed",)
    assert tuple(index.events) == events_after_failure
    attempt = bindings.attempt("run:partial", SCOPE)
    assert attempt.status == "failed"
    assert bindings.validated_attempt("run:partial", SCOPE) is None
    generation = generations.get(next(iter(index.entries)))
    assert generation.state is TextIndexGenerationState.CORRUPT


def test_started_attempt_is_incomplete_and_never_appended_again(tmp_path: Path) -> None:
    configured, values = retrieval_runtime(tmp_path)
    module, gate, _, graph_ref, _, _, bindings, index, _, _, commitment = values
    attempt = module.GraphIndexAttempt(
        "run:started", index.provider_id, configured.index_format_version
    )
    bindings._reserve(commitment, attempt, gate.revision)

    result = build_retrieval("run:started", graph_ref, SCOPE, configured)

    assert result.status == "incomplete"
    assert result.issue_codes == ("retrieval_attempt_incomplete",)
    assert bindings.validated_attempt("run:started", SCOPE) is None
    assert index.events == []


def test_successful_resume_is_idempotent_without_duplicate_append(tmp_path: Path) -> None:
    configured, values = retrieval_runtime(tmp_path)
    _, _, _, graph_ref, _, _, _, index, *_ = values

    first = build_retrieval("run:idempotent", graph_ref, SCOPE, configured)
    events = tuple(index.events)
    second = build_retrieval("run:idempotent", graph_ref, SCOPE, configured)

    assert second == first
    assert tuple(index.events) == events


def test_authorization_failure_is_bounded_but_never_creates_attempt(tmp_path: Path) -> None:
    configured, values = retrieval_runtime(tmp_path)
    _, gate, _, graph_ref, _, _, bindings, index, *_ = values
    gate.denied.add("evidence:1")
    gate.revision += 1

    result = build_retrieval("run:denied", graph_ref, SCOPE, configured)

    assert result.status == "unavailable"
    assert result.issue_codes == ("retrieval_unauthorized",)
    assert bindings.attempt("run:denied", SCOPE) is None
    assert index.events == []


def test_runtime_rejects_wrong_provider_and_catalog_pairing(tmp_path: Path) -> None:
    configured, _ = retrieval_runtime(tmp_path)
    foreign_catalog: Any = object()

    with pytest.raises(ValueError, match="provider"):
        replace(configured, writer=replace_provider(configured.writer))
    with pytest.raises(ValueError, match="catalog"):
        replace(configured, generations=foreign_catalog)


def replace_provider(index: Any) -> RecordingIndex:
    replacement = RecordingIndex()
    replacement.provider_id = "foreign-text"
    return replacement


def configure_service_retrieval(
    tmp_path: Path, service: Any, catalog: Any, *, encoder: Encoder | None = None
) -> tuple[BuildRetrievalRuntime, Any, RecordingIndex]:
    from capicove.graph_build.retrieval import GraphRetrievalCatalog, GraphTextReader
    from capicove.retrieval._adapters.sqlite_generations import (
        SqliteTextIndexGenerationCatalog,
    )

    reader = GraphTextReader(catalog, service.workspace, service.gate, service.reader)
    generations = SqliteTextIndexGenerationCatalog(
        tmp_path / "service-retrieval" / "generations.sqlite"
    )
    bindings = GraphRetrievalCatalog(
        tmp_path / "service-retrieval" / "bindings.sqlite",
        reader,
        service.gate,
        generations,
    )
    index = RecordingIndex()
    configured = BuildRetrievalRuntime(
        reader,
        runtime(retrieval_module(), encoder),
        index,
        index,
        generations,
        bindings,
        "recording-v1",
        2,
    )
    service.retrieval = configured
    return configured, generations, index


def test_service_success_records_inactive_retrieval_binding(tmp_path: Path) -> None:
    service, request, catalog, _ = authored(tmp_path)
    configured, generations, index = configure_service_retrieval(tmp_path, service, catalog)

    result = service.build(request)

    assert result.status == "succeeded"
    assert result.completeness == "complete"
    assert result.retrieval_binding_ref is not None
    assert configured.bindings.get(result.retrieval_binding_ref, SCOPE) is not None
    retrieval = service.stages.read(
        dict(service.runs.stages(result.run_id, SCOPE))["retrieval"], SCOPE
    )
    assert retrieval["status"] == "succeeded"
    assert retrieval["binding_ref"] == result.retrieval_binding_ref
    assert catalog.active(SCOPE) == (None, 0)
    assert (
        generations.active(workspace_ref=SCOPE.workspace_ref, retriever_id=index.provider_id)
        is None
    )


def test_service_resume_after_binding_before_operation_commit_does_not_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sqlite3

    service, request, catalog, _ = authored(tmp_path)
    encoder = Encoder()
    _, _, index = configure_service_retrieval(tmp_path, service, catalog, encoder=encoder)
    original = service.stages.commit_operation

    class Crash(BaseException):
        pass

    def interrupt(scope: Any, *, stage: str, **kwargs: Any) -> Any:
        if stage == "retrieval":
            raise Crash()
        return original(scope, stage=stage, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(service.stages, "commit_operation", interrupt)
        with pytest.raises(Crash):
            service.build(request)
    with sqlite3.connect(service.runs.path) as connection:
        run_id = connection.execute("SELECT run_id FROM build_runs").fetchone()[0]
    provider_events = tuple(index.events)
    embedding_calls = tuple(encoder.calls)

    result = service.resume(run_id, SCOPE)

    assert result.status == "succeeded"
    assert result.retrieval_binding_ref is not None
    assert tuple(index.events) == provider_events
    assert tuple(encoder.calls) == embedding_calls
    assert catalog.active(SCOPE) == (None, 0)


def test_service_optional_provider_failure_preserves_complete_graph(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    _, _, index = configure_service_retrieval(tmp_path, service, catalog)

    def unavailable(manifest: Any) -> str:
        raise OSError("provider unavailable")

    monkeypatch.setattr(index, "create", unavailable)
    result = service.build(request)

    assert result.status == "succeeded"
    assert result.completeness == "complete"
    assert result.retrieval_binding_ref is None
    assert "retrieval_build_failed" in result.issue_codes
    assert catalog.get(result.graph_ref.snapshot_id, SCOPE).completeness == "complete"
    assert catalog.active(SCOPE) == (None, 0)


def test_configured_service_rejects_forged_runtime_unavailable_terminal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A configured retrieval run can never truthfully be runtime-unavailable."""
    import json
    import sqlite3

    from capicove.contracts.graph_build import StageCoverage
    from capicove.graph_build.codec import canonical_bytes, graph_to_json
    from capicove.graph_build.workflow import _stage_digest

    service, request, catalog, _ = authored(tmp_path)
    configure_service_retrieval(tmp_path, service, catalog)
    result = service.build(request)
    records = dict(service.runs.stages(result.run_id, SCOPE))
    forged_operation = service.stages.commit_operation(
        SCOPE,
        stage="retrieval",
        status="failed",
        dependencies=(records["publication"],),
        evidence_refs=(),
        graph_ref=graph_to_json(result.graph_ref),
        binding_ref=None,
        issues=("retrieval_runtime_unavailable",),
        coverage=StageCoverage("generations", 1, 0, 0, 0, 1, 0),
    )
    payload = json.loads(service.runs.result(result.run_id, SCOPE))
    payload["retrieval_binding_ref"] = None
    payload["issue_codes"] = sorted({*payload["issue_codes"], "retrieval_runtime_unavailable"})
    forged_result = canonical_bytes(payload)
    with sqlite3.connect(service.runs.path) as connection:
        connection.execute(
            "UPDATE build_stages SET artifact_ref = ?, record_digest = ? "
            "WHERE run_id = ? AND stage = 'retrieval'",
            (
                forged_operation,
                _stage_digest(result.run_id, "retrieval", forged_operation),
                result.run_id,
            ),
        )
    monkeypatch.setattr(service.runs, "result", lambda *args: forged_result)

    with pytest.raises(ValueError, match="result_retrieval_operation_mismatch"):
        service.resume(result.run_id, SCOPE)
