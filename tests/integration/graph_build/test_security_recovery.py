"""Assembly-level adversarial recovery for the offline graph build service."""

from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.contracts.graph_build.snapshots import TABLE_ROLES
from capicove.graph_build._adapters.parquet import ParquetTableCodec
from capicove.graph_build.aggregation import source_occurrence_ref
from capicove.graph_build.datasets import codec_for_role
from tests.integration.graph_build.test_build_service import SCOPE, authored
from tests.integration.graph_build.test_service_retrieval import configure_service_retrieval


pytestmark = [pytest.mark.integration, pytest.mark.slow]


class SimulatedCrash(BaseException):
    """Escape bounded exception handlers at one durable crash boundary."""


@dataclass(frozen=True, slots=True)
class ExpectedProvenance:
    """Freeze independently derived source lineage before the crash."""

    documents: frozenset[tuple[Any, ...]]
    document_spans: frozenset[tuple[Any, ...]]
    evidence_refs: frozenset[str]
    entity_mentions: frozenset[tuple[Any, ...]]
    relationship_mentions: frozenset[tuple[Any, ...]]
    occurrence_refs: frozenset[str]


def _span_key(span: Any) -> tuple[Any, ...]:
    return (
        span.evidence_ref,
        span.representation_ref,
        span.start_char,
        span.end_char,
        span.block_id,
        span.page_index,
    )


def _document_key(document: Any) -> tuple[Any, ...]:
    return (
        document.document_ref,
        document.scope,
        document.source_sha256,
        document.evidence_ref,
        document.representation_ref,
        document.media_type,
        document.availability,
        document.coverage,
    )


def _expected_provenance(service: Any, request: Any) -> ExpectedProvenance:
    from capicove.contracts.graph_build import SourceSpan

    documents = tuple(service.inputs.documents(request.input_snapshot_ref, SCOPE))
    inventory = service.inputs.validate(request.input_snapshot_ref, SCOPE)
    assert frozenset(inventory.evidence_refs) == frozenset(
        document.evidence_ref for document in documents
    )
    assert len(documents) == 1
    document = documents[0]
    representation = service.reader.read(document.document_ref, SCOPE)
    assert document.evidence_ref == document.representation_ref
    assert document.representation_ref == representation.representation_ref
    assert document.source_sha256 == representation.source_sha256
    assert representation.text == "Alice Python"

    def span(start: int, end: int) -> SourceSpan:
        return SourceSpan(
            document.evidence_ref,
            representation.representation_ref,
            start,
            end,
        )

    person = span(0, 5)
    skill = span(6, 12)
    relationship = span(0, 12)
    return ExpectedProvenance(
        frozenset(map(_document_key, documents)),
        frozenset({_span_key(relationship)}),
        frozenset(inventory.evidence_refs),
        frozenset(
            {
                ("Alice", "PERSON", (_span_key(person),)),
                ("Python", "SKILL", (_span_key(skill),)),
            }
        ),
        frozenset(
            {
                (
                    "HAS_SKILL",
                    "explicit",
                    "negative",
                    (("time_text", "Monday"),),
                    (_span_key(relationship),),
                )
            }
        ),
        frozenset({source_occurrence_ref(relationship)}),
    )


def _only_run_id(path: Path) -> str:
    with sqlite3.connect(path) as connection:
        rows = connection.execute("SELECT run_id FROM build_runs ORDER BY run_id").fetchall()
    assert len(rows) == 1
    return str(rows[0][0])


def _read_table(service: Any, ref: Any, role: str) -> tuple[Any, ...]:
    rows: tuple[Any, ...] = ()

    def read(path: Path) -> bool:
        nonlocal rows
        rows = ParquetTableCodec(publication_identity=ref.processing_fingerprint).read(
            path / "part-00000.parquet", role
        )
        return True

    assert service.workspace.validate_snapshot(ref.snapshot_id, validator=read) == ref
    return rows


def _assert_release_invariants(
    service: Any,
    catalog: Any,
    result: Any,
    root: Path,
    expected: ExpectedProvenance,
) -> None:
    assert result.status == "succeeded"
    assert result.completeness == "complete"
    assert result.graph_ref is not None
    manifest = catalog.get(result.graph_ref.snapshot_id, SCOPE)
    assert (
        manifest.input_snapshot_ref
        == service.runs.get(result.run_id, SCOPE).request.input_snapshot_ref
    )
    assert {table.role for table in manifest.tables} == set(TABLE_ROLES)
    assert len(manifest.tables) == 17

    rows_by_role: dict[str, tuple[Any, ...]] = {}
    for table in manifest.tables:
        rows = _read_table(service, table.dataset_ref, table.role)
        rows_by_role[table.role] = rows
        keys = codec_for_role(table.role).key_fields
        identities = [tuple(getattr(row, key) for key in keys) for row in rows]
        assert len(identities) == len(set(identities)), table.role

    documents = rows_by_role["documents"]
    entity_mentions = rows_by_role["entity_mentions"]
    relationship_mentions = rows_by_role["relationship_mentions"]
    assertions = rows_by_role["relationship_assertions"]
    assert frozenset(map(_document_key, documents)) == expected.documents
    assert (
        frozenset(_span_key(span) for document in documents for span in document.provenance)
        == expected.document_spans
    )
    assert frozenset(row.evidence_ref for row in documents) == expected.evidence_refs
    assert frozenset(row.representation_ref for row in documents) == expected.evidence_refs
    assert (
        frozenset(
            (mention.name, mention.entity_type, tuple(map(_span_key, mention.source_spans)))
            for mention in entity_mentions
        )
        == expected.entity_mentions
    )
    assert (
        frozenset(
            (
                mention.predicate,
                mention.epistemic_status,
                mention.polarity,
                mention.qualifiers,
                tuple(map(_span_key, mention.source_spans)),
            )
            for mention in relationship_mentions
        )
        == expected.relationship_mentions
    )
    all_spans = tuple(
        span
        for mention in (*entity_mentions, *relationship_mentions)
        for span in mention.source_spans
    )
    assert frozenset(span.evidence_ref for span in all_spans) == expected.evidence_refs
    assert frozenset(span.representation_ref for span in all_spans) == expected.evidence_refs
    mention_occurrences = frozenset(
        source_occurrence_ref(span)
        for mention in relationship_mentions
        for span in mention.source_spans
    )
    assertion_occurrences = frozenset(
        evidence_ref for assertion in assertions for evidence_ref in assertion.evidence_refs
    )
    assert mention_occurrences == expected.occurrence_refs
    relationship_mention_refs = frozenset(mention.mention_id for mention in relationship_mentions)
    expected_assertion_evidence = expected.evidence_refs | relationship_mention_refs
    assert assertion_occurrences == expected_assertion_evidence
    assert all(
        frozenset(assertion.evidence_refs) <= expected_assertion_evidence
        and frozenset(assertion.evidence_refs) & expected.evidence_refs
        and frozenset(assertion.evidence_refs) & relationship_mention_refs
        for assertion in assertions
    )

    assert catalog.active(SCOPE) == (None, 0)
    forbidden = {"memory", "temporal-memory", "temporal_memory", "wiki"}
    assert not any(
        any(token in part.lower() for token in forbidden)
        for path in root.rglob("*")
        for part in path.relative_to(root).parts
    )


def _expire_claims(path: Path) -> None:
    with sqlite3.connect(path) as connection:
        connection.execute("UPDATE inference_units SET lease_until=0 WHERE status='claimed'")


@pytest.mark.parametrize(
    "checkpoint",
    (
        "after_response_before_receipt",
        "after_table_rename_before_commit",
        "after_table_commit_before_manifest",
        "after_manifest_before_activation",
        "after_index_write_before_binding",
    ),
)
def test_restart_resume_closes_each_durable_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, checkpoint: str
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    expected = _expected_provenance(service, request)
    retrieval = checkpoint == "after_index_write_before_binding"
    if retrieval:
        configured, _, _ = configure_service_retrieval(tmp_path, service, catalog)

    with monkeypatch.context() as patch:
        if checkpoint == "after_response_before_receipt":
            receipts = service.inference_runner._receipts

            def crash_complete(*args: Any, **kwargs: Any) -> None:
                raise SimulatedCrash()

            patch.setattr(receipts, "complete", crash_complete)
        elif checkpoint == "after_table_rename_before_commit":

            def crash_commit(*args: Any, **kwargs: Any) -> None:
                raise SimulatedCrash()

            patch.setattr(service.workspace, "commit_success", crash_commit)
        elif checkpoint == "after_table_commit_before_manifest":

            def crash_register(*args: Any, **kwargs: Any) -> None:
                raise SimulatedCrash()

            patch.setattr(catalog, "register", crash_register)
        elif checkpoint == "after_manifest_before_activation":
            register = catalog.register

            def crash_after_register(*args: Any, **kwargs: Any) -> None:
                register(*args, **kwargs)
                raise SimulatedCrash()

            patch.setattr(catalog, "register", crash_after_register)
        else:
            register = type(configured.bindings).register

            def crash_after_index(self: Any, *args: Any, **kwargs: Any) -> None:
                raise SimulatedCrash()

            patch.setattr(type(configured.bindings), "register", crash_after_index)

        with pytest.raises(SimulatedCrash):
            service.build(request)

    run_id = _only_run_id(service.runs.path)
    assert catalog.active(SCOPE) == (None, 0)
    if checkpoint == "after_response_before_receipt":
        _expire_claims(tmp_path / "receipts" / "receipts.sqlite")

    restarted, _, restarted_catalog, _ = authored(tmp_path)
    generations = None
    index = None
    if retrieval:
        _, generations, index = configure_service_retrieval(tmp_path, restarted, restarted_catalog)
    result = restarted.resume(run_id, SCOPE)

    _assert_release_invariants(restarted, restarted_catalog, result, tmp_path, expected)
    assert restarted.resume(run_id, SCOPE) == result
    if retrieval:
        assert generations is not None
        assert index is not None
        assert result.retrieval_binding_ref is not None
        assert index.events == []
        assert (
            generations.active(workspace_ref=SCOPE.workspace_ref, retriever_id=index.provider_id)
            is None
        )


def test_cached_receipt_is_reauthorized_at_artifact_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts

    service, request, catalog, generator = authored(tmp_path)
    first = service.build(request)
    assert first.status == "succeeded"
    calls = tuple(generator.calls)
    evidence_ref = service.inputs.documents(request.input_snapshot_ref, SCOPE)[0].evidence_ref
    original = FilesystemProtectedArtifacts.read

    def revoke_then_read(self: Any, scope: GraphScope, artifact_ref: str) -> bytes:
        service.gate.denied.add(evidence_ref)
        service.gate.revision += 1
        return original(self, scope, artifact_ref)

    monkeypatch.setattr(FilesystemProtectedArtifacts, "read", revoke_then_read)
    with pytest.raises(PermissionError):
        service.build(request)
    assert tuple(generator.calls) == calls
    assert catalog.active(SCOPE) == (None, 0)


def test_symlink_replacement_of_cached_raw_artifact_fails_closed(
    tmp_path: Path,
) -> None:
    from capicove.graph_build._adapters.artifacts import ArtifactStoreError

    service, request, catalog, _ = authored(tmp_path)
    first = service.build(request)
    assert first.status == "succeeded"
    artifact = next((tmp_path / "raw").glob("*.json"))
    outside = tmp_path / "attacker.json"
    outside.write_bytes(b"{}")
    artifact.unlink()
    artifact.symlink_to(outside)

    with pytest.raises(ArtifactStoreError):
        service.build(request)
    assert catalog.active(SCOPE) == (None, 0)


def test_generated_report_cannot_be_recycled_as_source_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build.inference import RawGeneration

    service, request, catalog, generator = authored(tmp_path)
    generate = generator.generate

    def generated_source(request: Any, *, control: Any) -> Any:
        result = generate(request, control=control)
        if request.role != "report":
            return result
        payload = json.loads(result.payload)
        payload["sections"]["summary"] = [
            {
                "local_id": "recycled",
                "text": "Generated prose presented as source evidence.",
                "kind": "fact",
                "assertion_refs": ["generated_report:foreign"],
                "finding_refs": [],
                "child_statement_refs": [],
            }
        ]
        return RawGeneration(json.dumps(payload).encode(), "stop", 20, 20)

    monkeypatch.setattr(generator, "generate", generated_source)
    result = service.build(request)

    assert result.status == "partial"
    report = next(stage for stage in result.coverage if stage.stage_id == "reports")
    assert report.status == "failed"
    assert "report_generation_failed" in report.issue_codes
    assert catalog.active(SCOPE) == (None, 0)


def test_two_concurrent_builds_share_content_without_duplicate_identity(
    tmp_path: Path,
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    expected = _expected_provenance(service, request)

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = tuple(pool.map(lambda _: service.build(request), range(2)))

    assert all(result.status == "succeeded" for result in results)
    assert results[0].run_id != results[1].run_id
    assert results[0].graph_ref == results[1].graph_ref
    for result in results:
        _assert_release_invariants(service, catalog, result, tmp_path, expected)


def test_budget_cancellation_is_terminal_and_restart_safe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, request, catalog, _ = authored(tmp_path)
    publish = service.workspace.publish
    cancelled = False

    def cancel_after_publish(*args: Any, **kwargs: Any) -> Any:
        nonlocal cancelled
        result = publish(*args, **kwargs)
        if not cancelled and kwargs["dataset_id"].startswith("dataset:graph-"):
            cancelled = True
            run_id = _only_run_id(service.runs.path)
            service.cancel(run_id, SCOPE)
        return result

    monkeypatch.setattr(service.workspace, "publish", cancel_after_publish)
    result = service.build(request)
    assert result.status == "cancelled"
    assert result.graph_ref is None
    assert catalog.active(SCOPE) == (None, 0)

    restarted, _, restarted_catalog, _ = authored(tmp_path)
    assert restarted.resume(result.run_id, SCOPE) == result
    assert restarted_catalog.active(SCOPE) == (None, 0)
