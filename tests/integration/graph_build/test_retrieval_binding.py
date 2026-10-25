"""Real graph/generation catalogs around explicitly selected recording adapters."""

from __future__ import annotations

import sqlite3
from dataclasses import fields, is_dataclass, replace
from pathlib import Path
from typing import Any

import pytest

from capicove.graph_build.representations import text_representation
from capicove.processing import LocalProcessingWorkspace
from capicove.retrieval import (
    TextEmbeddingService,
    TextIndexGenerationState,
    TextIndexValidationReport,
)
from capicove.retrieval._adapters.sqlite_generations import SqliteTextIndexGenerationCatalog
from tests.graph_build_fixtures import (
    SCOPE,
    Gate,
    _catalog,
    _manifest,
    _rows,
)
from tests.unit.graph_build.test_retrieval_projection import module
from tests.unit.retrieval.test_text_embedding import Encoder, text_space


pytestmark = [pytest.mark.integration, pytest.mark.slow]


def setup(
    tmp_path: Path, *, source_text: str = "A B", rename: dict[str, str] | None = None
) -> tuple[Any, ...]:
    m = module()
    assert hasattr(m, "GraphTextReader"), "Missing committed graph text loader"
    representation = text_representation("document:1", source_text, SCOPE)

    def update(value: Any) -> Any:
        if isinstance(value, str):
            return {
                "representation:1": representation.representation_ref,
                "d" * 64: representation.source_sha256,
                "A B": source_text,
                **(rename or {}),
            }.get(value, value)
        if isinstance(value, tuple):
            return tuple(update(v) for v in value)
        if is_dataclass(value) and not isinstance(value, type):
            changes = {f.name: update(getattr(value, f.name)) for f in fields(value)}
            if "start_char" in changes and changes.get("end_char") == 3:
                changes["end_char"] = len(source_text)
            return replace(value, **changes)
        return value

    rows = {key: update(value) for key, value in _rows().items()}
    if rename:
        from capicove.graph_build.aggregation import aggregate_assertions

        rows["relationship_aggregates"] = aggregate_assertions(rows["relationship_assertions"])
    workspace = LocalProcessingWorkspace(tmp_path / "processing")
    gate = Gate()
    catalog = _catalog(tmp_path / "graph" / "catalog.sqlite", workspace, gate)
    manifest = _manifest(workspace, rows=rows)
    ref = catalog.register(manifest)

    class Sources:
        def read(self, document_ref: str, scope: Any) -> Any:
            gate.require_access(("evidence:1",), scope)
            assert document_ref == representation.document_ref
            return representation

        def require_access(self, evidence_refs: tuple[str, ...], scope: Any) -> None:
            gate.require_access(evidence_refs, scope)

    reader = m.GraphTextReader(catalog, workspace, gate, Sources())
    return m, gate, catalog, manifest, ref, reader, rows


def runtime(m: Any, encoder: Any = None, *, limit: int = 512) -> Any:
    class Tokenizer:
        fingerprint = "a" * 64
        space = text_space()
        verified_max_tokens = limit

        def count_document(self, text: str) -> int:
            return len(text) + 2

    return m.GraphEmbeddingRuntime(
        Tokenizer(), TextEmbeddingService(encoder or Encoder(), space=text_space()), "b" * 64, 2
    )


def test_committed_source_entity_report_text_has_distinct_exact_lineage(tmp_path: Path) -> None:
    m, _gate, catalog, _manifest_value, ref, reader, rows = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    assert len(content.texts) == 4
    source = content.texts[0]
    assert source.source_class == "source"
    assert source.text_unit == rows["text_units"][0]
    assert source.text == "A B"
    assert source.evidence_ref == "evidence:1"
    for value in content.texts[1:]:
        assert value.source_class in {"entity", "report"}
        assert value.evidence_ref.startswith("evidence:graph-derived-")
        assert value.evidence_ref != source.evidence_ref
        assert value.source_refs == ("evidence:1",)
        assert value.lineage_refs
    entries = m.graph_text_entries(ref, content, SCOPE, runtime(m), reader=reader)
    assert tuple(entry.text for entry in entries) == tuple(value.text for value in content.texts)
    assert entries[0].unit.text_locator.character_range.start_char == 0
    assert all(entry.corpus_refs == (ref.snapshot_id,) for entry in entries)
    assert catalog.get(ref.snapshot_id, SCOPE).completeness == "complete"
    assert catalog.active(SCOPE) == (None, 0)


@pytest.mark.parametrize("damage", ["text", "source", "lineage", "profile", "graph", "scope"])
def test_uncommitted_projection_is_rejected(tmp_path: Path, damage: str) -> None:
    m, _gate, _catalog_value, _manifest_value, ref, reader, _source_rows = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    if damage in {"text", "source", "lineage"}:
        changes = (
            {"text": "forged"}
            if damage == "text"
            else {"source_refs" if damage == "source" else "lineage_refs": ("alien:ref",)}
        )
        content = replace(content, texts=(replace(content.texts[0], **changes), *content.texts[1:]))
    elif damage == "profile":
        content = replace(content, profile_sha256="f" * 64)
    elif damage == "graph":
        content = replace(content, graph_ref=replace(ref, manifest_sha256="f" * 64))
    else:
        content = replace(content, scope=replace(SCOPE, scope_ref="scope:alien"))
    with pytest.raises(ValueError):
        m.graph_text_entries(ref, content, SCOPE, runtime(m), reader=reader)


def test_revocation_during_embedding_prevents_projection_release(tmp_path: Path) -> None:
    m, gate, _catalog_value, _manifest_value, ref, reader, _source_rows = setup(tmp_path)
    content = reader.read(ref, SCOPE)

    class RevokingEncoder(Encoder):
        def encode(self, *args: Any, **kwargs: Any) -> Any:
            result = super().encode(*args, **kwargs)
            gate.denied.add("evidence:1")
            gate.revision += 1
            return result

    with pytest.raises(PermissionError):
        m.graph_text_entries(ref, content, SCOPE, runtime(m, RevokingEncoder()), reader=reader)


class RecordingIndex:
    """Port-conforming local fake; catalogs and graph validation remain real."""

    provider_id = "recording-text"

    def __init__(self) -> None:
        self.entries: dict[str, tuple[Any, ...]] = {}
        self.fail_write = False
        self.events: list[str] = []

    def mapping_sha256(self, embedding_spaces: Any) -> str:
        assert embedding_spaces == (text_space(),)
        return "c" * 64

    def create(self, manifest: Any) -> str:
        self.events.append("create")
        self.entries[manifest.generation_id] = ()
        return "physical:" + str(manifest.generation_id)

    def add(self, *, index_generation_id: str, entries: tuple[Any, ...]) -> None:
        self.events.append("write")
        if self.fail_write:
            self.entries[index_generation_id] += entries[:1]
            raise OSError("uncertain partial append")
        self.entries[index_generation_id] += entries

    def validate(self, manifest: Any) -> TextIndexValidationReport:
        self.events.append("validate")
        entries = self.entries[manifest.generation_id]
        issues = []
        if len(entries) != manifest.expected_unit_count:
            issues.append("document_count_mismatch")
        if module().text_index_content_sha256(entries) != manifest.content_sha256:
            issues.append("content_mismatch")
        return TextIndexValidationReport(
            manifest.generation_id,
            manifest.manifest_sha256,
            "physical:" + manifest.generation_id,
            "c" * 64,
            manifest.expected_unit_count,
            len(entries),
            tuple(issues),
        )

    def activate(self, manifest: Any, validation: Any) -> Any:
        raise AssertionError("retrieval must not activate implicitly")

    def resolve_active(self) -> Any:
        raise AssertionError("graph binding must not read a global active index")


def build_setup(tmp_path: Path) -> tuple[Any, ...]:
    m, gate, graph_catalog, _manifest_value, ref, reader, _source_rows = setup(tmp_path)
    assert hasattr(m, "GraphRetrievalCatalog"), "Missing durable retrieval binding catalog"
    generations = SqliteTextIndexGenerationCatalog(tmp_path / "generations" / "catalog.sqlite")
    bindings = m.GraphRetrievalCatalog(
        tmp_path / "bindings" / "catalog.sqlite", reader, gate, generations
    )
    index = RecordingIndex()
    content = reader.read(ref, SCOPE)
    rt = runtime(m)
    entries = m.graph_text_entries(ref, content, SCOPE, rt, reader=reader)
    commitment = m.projection_commitment(
        content, rt, entries, index.mapping_sha256((text_space(),))
    )
    return (
        m,
        gate,
        graph_catalog,
        ref,
        reader,
        generations,
        bindings,
        index,
        content,
        entries,
        commitment,
    )


@pytest.fixture(scope="module")
def control_graph(tmp_path_factory: pytest.TempPathFactory) -> tuple[Any, ...]:
    """Reuse immutable graph evidence across control-boundary cases."""
    return build_setup(tmp_path_factory.mktemp("retrieval-control"))


@pytest.mark.parametrize("stop", ["cancel", "deadline"])
@pytest.mark.parametrize("boundary", ["write", "encode"])
def test_control_stops_retrieval_after_first_batch_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    control_graph: tuple[Any, ...],
    stop: str,
    boundary: str,
) -> None:
    m, gate, _, graph_ref, reader, _, _, _, content, entries, commitment = control_graph
    calls = 0

    def check_control() -> None:
        if calls:
            if stop == "cancel":
                raise InterruptedError("build_cancelled")
            raise TimeoutError("build_deadline_exceeded")

    error = InterruptedError if stop == "cancel" else TimeoutError
    if boundary == "encode":
        encoder = Encoder()
        original = encoder.encode

        def encode_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            result = original(*args, **kwargs)
            calls += 1
            return result

        monkeypatch.setattr(encoder, "encode", encode_once)
        with pytest.raises(error):
            m.graph_text_entries(
                graph_ref,
                content,
                SCOPE,
                runtime(m, encoder),
                reader=reader,
                check_control=check_control,
            )
    else:
        generations = SqliteTextIndexGenerationCatalog(tmp_path / "generations.sqlite")
        bindings = m.GraphRetrievalCatalog(tmp_path / "bindings.sqlite", reader, gate, generations)
        index = RecordingIndex()
        original_add = index.add

        def write_once(*args: Any, **kwargs: Any) -> Any:
            nonlocal calls
            result = original_add(*args, **kwargs)
            calls += 1
            return result

        monkeypatch.setattr(index, "add", write_once)
        with pytest.raises(error):
            m.build_graph_generation(
                commitment,
                entries,
                m.GraphIndexAttempt(f"run:{stop}", index.provider_id, "recording-v1"),
                runtime=runtime(m),
                lifecycle=index,
                writer=index,
                generations=generations,
                bindings=bindings,
                batch_size=2,
                check_control=check_control,
            )
        assert index.events[-1] == "write"
        attempt = bindings.attempt(f"run:{stop}", SCOPE)
        assert attempt is not None and attempt.status == "failed"

    assert calls == 1


def build(values: tuple[Any, ...], attempt_ref: str = "attempt:first") -> Any:
    (
        m,
        _gate,
        _graph_catalog,
        _ref,
        _reader,
        generations,
        bindings,
        index,
        _content,
        entries,
        commitment,
    ) = values
    return m.build_graph_generation(
        commitment,
        entries,
        m.GraphIndexAttempt(attempt_ref, index.provider_id, "recording-v1"),
        lifecycle=index,
        writer=index,
        generations=generations,
        bindings=bindings,
        batch_size=2,
        runtime=runtime(m),
    )


def test_generation_and_binding_are_exact_inactive_restartable(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    (
        m,
        gate,
        graph_catalog,
        ref,
        reader,
        generations,
        bindings,
        index,
        _content,
        _entries,
        commitment,
    ) = values
    validated = build(values)
    assert validated.record.state is TextIndexGenerationState.READY
    assert index.events == ["create", "write", "write", "validate"]
    binding = bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    assert binding == m.bind_retrieval(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    assert ref.snapshot_id == binding.graph_snapshot_id
    assert validated.record.manifest.manifest_sha256 in binding.generation_refs[0]
    reopened = m.GraphRetrievalCatalog(
        tmp_path / "bindings" / "catalog.sqlite", reader, gate, generations
    )
    assert reopened.get(binding.binding_id, SCOPE) == binding
    assert reopened.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE) == binding
    assert graph_catalog.active(SCOPE) == (None, 0)
    assert (
        generations.active(workspace_ref=SCOPE.workspace_ref, retriever_id=index.provider_id)
        is None
    )


def test_uncertain_partial_append_never_reuses_attempt_and_retry_is_fresh(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    (
        _m,
        _gate,
        graph_catalog,
        ref,
        _reader,
        generations,
        bindings,
        index,
        _content,
        entries,
        _commitment,
    ) = values
    index.fail_write = True
    with pytest.raises(OSError):
        build(values)
    failed_id = next(iter(index.entries))
    assert generations.get(failed_id).state is TextIndexGenerationState.CORRUPT
    assert bindings.attempt("attempt:first", SCOPE).status == "failed"
    assert graph_catalog.get(ref.snapshot_id, SCOPE).completeness == "complete"
    with pytest.raises(ValueError, match="attempt"):
        build(values)
    assert len(index.entries[failed_id]) == 1
    index.fail_write = False
    succeeded = build(values, "attempt:retry")
    assert succeeded.record.manifest.generation_id != failed_id
    assert index.entries[succeeded.record.manifest.generation_id] == entries


@pytest.mark.parametrize(
    "field",
    [
        "corpus_refs",
        "workspace_ref",
        "content_sha256",
        "mapping_sha256",
        "searchable_unit_manifest_sha256",
        "embedding_spaces",
    ],
)
def test_wrong_generation_commitment_rejects(tmp_path: Path, field: str) -> None:
    values = build_setup(tmp_path)
    (
        m,
        _gate,
        _graph_catalog,
        ref,
        _reader,
        _generations,
        _bindings,
        _index,
        _content,
        _entries,
        commitment,
    ) = values
    validated = build(values)
    bad: Any = "f" * 64
    if field == "corpus_refs":
        bad = ("graph:alien",)
    elif field == "workspace_ref":
        bad = "workspace:alien"
    elif field == "embedding_spaces":
        bad = (replace(text_space(), model_revision="different-same-dimension"),)
    manifest = replace(validated.record.manifest, **{field: bad})
    report = replace(validated.record.validation, manifest_sha256=manifest.manifest_sha256)
    if field == "mapping_sha256":
        report = replace(report, mapping_sha256=bad)
    record = replace(validated.record, manifest=manifest, validation=report)
    with pytest.raises(ValueError):
        candidate = replace(validated, record=record)
        m.bind_retrieval(ref, (candidate,), commitment.unit_manifest_sha256, SCOPE)


def test_binding_reads_recheck_revocation_and_graph_ineligibility(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    (
        _m,
        gate,
        graph_catalog,
        ref,
        _reader,
        _generations,
        bindings,
        _index,
        _content,
        _entries,
        commitment,
    ) = values
    validated = build(values)
    binding = bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    gate.denied.add("evidence:1")
    gate.revision += 1
    with pytest.raises(PermissionError):
        bindings.get(binding.binding_id, SCOPE)
    gate.denied.clear()
    gate.revision += 1
    graph_catalog.mark_ineligible("evidence:1", SCOPE, "source_deleted")
    with pytest.raises(ValueError):
        bindings.get(binding.binding_id, SCOPE)


def test_binding_interruption_rolls_back_without_rewriting_ready_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = build_setup(tmp_path)
    m, _, _, ref, reader, generations, bindings, index, _, _, commitment = values
    validated = build(values)
    expected = m.bind_retrieval(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    original = sqlite3.connect

    class Interrupted(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> Any:
            cursor = super().execute(sql, parameters)
            if sql.startswith("INSERT INTO graph_retrieval_bindings"):
                raise sqlite3.OperationalError("interrupted after insert before commit")
            return cursor

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if "/bindings/" in str(database):
            kwargs["factory"] = Interrupted
        result: sqlite3.Connection = original(database, *args, **kwargs)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        with pytest.raises(sqlite3.OperationalError):
            bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    reopened = m.GraphRetrievalCatalog(
        tmp_path / "bindings" / "catalog.sqlite", reader, values[1], generations
    )
    assert reopened.get(expected.binding_id, SCOPE) is None
    assert reopened.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE) == expected
    assert index.events == ["create", "write", "write", "validate"]


def test_binding_commit_rechecks_policy_before_sqlite_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = build_setup(tmp_path)
    m, gate, _, ref, _, _, bindings, _, _, _, commitment = values
    validated = build(values)
    expected = m.bind_retrieval(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    original = sqlite3.connect

    class Revoking(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> Any:
            cursor = super().execute(sql, parameters)
            if sql.startswith("INSERT INTO graph_retrieval_bindings"):
                gate.denied.add("evidence:1")
                gate.revision += 1
            return cursor

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if "/bindings/" in str(database):
            kwargs["factory"] = Revoking
        result: sqlite3.Connection = original(database, *args, **kwargs)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        with pytest.raises(PermissionError):
            bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    gate.denied.clear()
    gate.revision += 1
    assert bindings.get(expected.binding_id, SCOPE) is None


@pytest.mark.parametrize(
    "state", [TextIndexGenerationState.BUILDING, TextIndexGenerationState.CORRUPT]
)
def test_unvalidated_generation_cannot_bind(tmp_path: Path, state: Any) -> None:
    values = build_setup(tmp_path)
    m, _, _, ref, _, _, _, _, _, _, commitment = values
    validated = build(values)
    report = (
        None
        if state is TextIndexGenerationState.BUILDING
        else replace(validated.record.validation, issue_codes=("content_mismatch",))
    )
    record = replace(validated.record, state=state, validation=report)
    with pytest.raises(ValueError):
        m.bind_retrieval(
            ref, (replace(validated, record=record),), commitment.unit_manifest_sha256, SCOPE
        )


def test_wrong_graph_duplicate_generation_and_wrong_manifest_reject(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    m, _, _, ref, _, _, _, _, _, _, commitment = values
    validated = build(values)
    for graph_ref, records, digest, scope in (
        (
            replace(ref, snapshot_id="graph:alien"),
            (validated,),
            commitment.unit_manifest_sha256,
            SCOPE,
        ),
        (ref, (validated, validated), commitment.unit_manifest_sha256, SCOPE),
        (ref, (validated,), "f" * 64, SCOPE),
        (ref, (validated,), commitment.unit_manifest_sha256, replace(SCOPE, scope_ref="alien")),
    ):
        with pytest.raises(ValueError):
            m.bind_retrieval(graph_ref, records, digest, scope)


def test_same_count_wrong_text_fails_physical_content_validation(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    _, _, _, _, _, generations, bindings, index, _, _, _ = values
    original = index.add

    def corrupt(*, index_generation_id: str, entries: tuple[Any, ...]) -> None:
        original(
            index_generation_id=index_generation_id,
            entries=tuple(replace(entry, text="changed") for entry in entries),
        )

    index.add = corrupt
    with pytest.raises(ValueError, match="validated"):
        build(values)
    generation = generations.get(next(iter(index.entries)))
    assert generation.state is TextIndexGenerationState.CORRUPT
    assert generation.validation.observed_unit_count == generation.manifest.expected_unit_count
    assert bindings.attempt("attempt:first", SCOPE).status == "failed"


def test_embedding_response_count_and_space_are_checked(tmp_path: Path) -> None:
    from capicove.retrieval import TextEmbedding

    m, _, _, _, ref, reader, _ = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    for output in (
        (TextEmbedding(text_space(), (1.0, 0.0)),),
        (TextEmbedding(replace(text_space(), model_revision="alien"), (1.0, 0.0)),) * 2,
    ):
        with pytest.raises(ValueError):
            m.graph_text_entries(ref, content, SCOPE, runtime(m, Encoder(output)), reader=reader)


def test_binding_corrupt_bytes_and_foreign_scope_are_rejected(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    _, _, _, ref, _, _, bindings, _, _, _, commitment = values
    validated = build(values)
    binding = bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    assert bindings.get(binding.binding_id, replace(SCOPE, scope_ref="alien")) is None
    with sqlite3.connect(tmp_path / "bindings" / "catalog.sqlite") as connection:
        connection.execute("UPDATE graph_retrieval_bindings SET payload=?", (b"{}",))
    with pytest.raises(ValueError, match="digest"):
        bindings.get(binding.binding_id, SCOPE)


def test_changed_source_representation_is_rejected_before_encoding(tmp_path: Path) -> None:
    m, _, _, _, ref, reader, _ = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    reader.sources.read = lambda document_ref, scope: text_representation(
        document_ref, "X B", scope
    )
    with pytest.raises(ValueError, match="representation"):
        m.graph_text_entries(ref, content, SCOPE, runtime(m), reader=reader)


def test_long_source_and_derived_text_split_for_exact_document_wire_limit(tmp_path: Path) -> None:
    m, _, _, _, ref, reader, rows = setup(tmp_path, source_text="A🙂中B" * 200)
    content = reader.read(ref, SCOPE)
    encoder = Encoder()
    rt = runtime(m, encoder, limit=512)
    entries = m.graph_text_entries(ref, content, SCOPE, rt, reader=reader)
    source = tuple(entry for entry in entries if entry.unit.evidence_ref == "evidence:1")
    assert tuple(entry.unit.text_locator.character_range.end_char for entry in source) == (510, 800)
    assert "".join(entry.text for entry in source) == "A🙂中B" * 200
    assert rows["text_units"][0].core_locator.character_range.end_char == 800
    assert all(len(text) + 2 <= 512 for texts, _, _ in encoder.calls for text in texts)
    assert all(len(texts) <= 2 and role == "document" for texts, _, role in encoder.calls)


def test_duplicate_text_identity_rejects(tmp_path: Path) -> None:
    _, _, _, _, ref, reader, _ = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    with pytest.raises(ValueError, match="duplicate"):
        replace(content, texts=(*content.texts, content.texts[0]))


def test_source_and_graph_owner_id_collision_cannot_short_circuit_lineage(tmp_path: Path) -> None:
    _, _, _, _, ref, reader, _ = setup(tmp_path, rename={"entity:A": "evidence:1"})
    with pytest.raises(ValueError, match="lineage"):
        reader.read(ref, SCOPE)


def test_unavailable_provider_retains_graph_without_ready_binding(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    _, _, graph_catalog, ref, _, _, bindings, index, _, _, _ = values

    def unavailable(manifest: Any) -> str:
        raise OSError("explicit provider unavailable")

    index.create = unavailable
    with pytest.raises(OSError):
        build(values)
    assert bindings.attempt("attempt:first", SCOPE).status == "failed"
    assert graph_catalog.get(ref.snapshot_id, SCOPE).completeness == "complete"
    assert graph_catalog.active(SCOPE) == (None, 0)


def test_abandoned_full_append_is_never_bound_even_if_provider_validates(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    m, _, _, ref, _, generations, bindings, index, _, _, commitment = values
    original = index.add

    def uncertain(*, index_generation_id: str, entries: tuple[Any, ...]) -> None:
        original(index_generation_id=index_generation_id, entries=entries)
        if len(index.entries[index_generation_id]) == commitment.unit_count:
            raise OSError("lost write acknowledgement")

    index.add = uncertain
    with pytest.raises(OSError):
        build(values)
    record = generations.get(next(iter(index.entries)))
    assert record.state is TextIndexGenerationState.READY
    candidate = m.ValidatedGraphGeneration(
        record, commitment, m.GraphIndexAttempt("attempt:first", index.provider_id, "recording-v1")
    )
    with pytest.raises(ValueError, match="attempt"):
        bindings.register(ref, (candidate,), commitment.unit_manifest_sha256, SCOPE)


def test_projection_entry_order_and_exact_ranges_invalidate_commitment(tmp_path: Path) -> None:
    from capicove.contracts import CharacterRange

    values = build_setup(tmp_path)
    m, _, _, _, _, _, _, _, content, entries, commitment = values
    variants = (
        tuple(reversed(entries)),
        (replace(entries[0], text="changed"), *entries[1:]),
        (
            replace(
                entries[0],
                unit=replace(
                    entries[0].unit,
                    text_locator=replace(
                        entries[0].unit.text_locator, character_range=CharacterRange(1, 4)
                    ),
                ),
            ),
            *entries[1:],
        ),
    )
    for variant in variants:
        with pytest.raises(ValueError):
            m.projection_commitment(content, runtime(m), variant, commitment.mapping_sha256)


def test_binding_reconstructs_validation_report_invariants(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    m, _, _, ref, _, _, _, _, _, _, commitment = values
    validated = build(values)
    report = replace(validated.record.validation)
    object.__setattr__(report, "observed_unit_count", 0)
    record = replace(validated.record, validation=report)
    with pytest.raises(ValueError):
        m.bind_retrieval(
            ref, (replace(validated, record=record),), commitment.unit_manifest_sha256, SCOPE
        )


def _policy_change(gate: Gate, *, restore: bool) -> None:
    """Simulate policy changes even when final effective access remains allowed."""
    if restore:
        gate.denied.add("evidence:1")
        gate.revision += 1
        gate.denied.clear()
    gate.revision += 1


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("batch", [1, 2])
def test_projection_fences_one_policy_revision_across_encoder_batches(
    tmp_path: Path, restore: bool, batch: int
) -> None:
    m, gate, _, _, ref, reader, _ = setup(tmp_path)
    content = reader.read(ref, SCOPE)

    class ChangingEncoder(Encoder):
        def encode(self, *args: Any, **kwargs: Any) -> Any:
            result = super().encode(*args, **kwargs)
            if len(self.calls) == batch:
                _policy_change(gate, restore=restore)
            return result

    encoder = ChangingEncoder()
    with pytest.raises(PermissionError, match="policy"):
        m.graph_text_entries(ref, content, SCOPE, runtime(m, encoder), reader=reader)
    assert len(encoder.calls) == batch
    gate.require_access(("evidence:1",), SCOPE)


@pytest.mark.parametrize("restore", [False, True])
@pytest.mark.parametrize("effect", ["create", "write_first", "write_last", "validate"])
def test_generation_attempt_fences_policy_revision_across_provider_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, restore: bool, effect: str
) -> None:
    values = build_setup(tmp_path)
    m, gate, graph_catalog, ref, _, generations, bindings, index, _, _, commitment = values
    start_revision = gate.revision
    method = "add" if effect.startswith("write") else effect
    original = getattr(index, method)
    calls = 0
    target_call = 2 if effect == "write_last" else 1

    def changing(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        result = original(*args, **kwargs)
        calls += 1
        if calls == target_call:
            _policy_change(gate, restore=restore)
        return result

    with monkeypatch.context() as patch:
        patch.setattr(index, method, changing)
        with pytest.raises(PermissionError, match="policy"):
            build(values)
    state = bindings.attempt("attempt:first", SCOPE)
    assert state.status == "failed"
    assert state.policy_revision == start_revision
    record = generations.get(next(iter(index.entries)))
    if effect in {"create", "write_first"}:
        assert record.state is TextIndexGenerationState.CORRUPT
    else:
        assert record.state is TextIndexGenerationState.READY
        candidate = m.ValidatedGraphGeneration(record, commitment, state.attempt)
        with pytest.raises(ValueError, match="attempt"):
            bindings.register(ref, (candidate,), commitment.unit_manifest_sha256, SCOPE)
    assert graph_catalog.get(ref.snapshot_id, SCOPE).completeness == "complete"
    with pytest.raises(ValueError, match="attempt"):
        build(values)
    retried = build(values, "attempt:after-policy-change")
    assert retried.record.manifest.generation_id != record.manifest.generation_id
    assert bindings.attempt("attempt:after-policy-change", SCOPE).policy_revision == gate.revision


@pytest.mark.parametrize("inside_update", [False, True])
def test_ready_attempt_commit_is_fenced_by_original_policy_revision(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, inside_update: bool
) -> None:
    values = build_setup(tmp_path)
    m, gate, _, ref, _, generations, bindings, index, _, _, commitment = values
    start_revision = gate.revision
    changed = False
    original_finish = bindings._finish
    original_connect = sqlite3.connect

    def finish(*args: Any, **kwargs: Any) -> Any:
        nonlocal changed
        if args[2] == "ready" and not changed:
            changed = True
            _policy_change(gate, restore=True)
        return original_finish(*args, **kwargs)

    class ChangingConnection(sqlite3.Connection):
        def execute(self, sql: str, parameters: Any = (), /) -> Any:
            nonlocal changed
            cursor = super().execute(sql, parameters)
            if sql.startswith("UPDATE graph_index_attempts") and not changed:
                changed = True
                _policy_change(gate, restore=True)
            return cursor

    def connect(database: Any, *args: Any, **kwargs: Any) -> sqlite3.Connection:
        if "/bindings/" in str(database):
            kwargs["factory"] = ChangingConnection
        result: sqlite3.Connection = original_connect(database, *args, **kwargs)
        return result

    with monkeypatch.context() as patch:
        if inside_update:
            patch.setattr(sqlite3, "connect", connect)
        else:
            patch.setattr(bindings, "_finish", finish)
        with pytest.raises(PermissionError, match="policy"):
            build(values)
    state = bindings.attempt("attempt:first", SCOPE)
    assert state.status == "failed"
    assert state.policy_revision == start_revision
    record = generations.get(next(iter(index.entries)))
    assert record.state is TextIndexGenerationState.READY
    candidate = m.ValidatedGraphGeneration(record, commitment, state.attempt)
    with pytest.raises(ValueError, match="attempt"):
        bindings.register(ref, (candidate,), commitment.unit_manifest_sha256, SCOPE)


def test_binding_catalog_requires_one_shared_policy_revision_authority(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    m, _, _, _, reader, generations, _, _, _, _, _ = values
    with pytest.raises(ValueError, match="policy"):
        m.GraphRetrievalCatalog(
            tmp_path / "other-bindings" / "catalog.sqlite", reader, Gate(), generations
        )


def test_stable_attempt_remains_bindable_after_full_current_reauthorization(tmp_path: Path) -> None:
    values = build_setup(tmp_path)
    _, gate, _, ref, _, _, bindings, _, _, _, commitment = values
    validated = build(values)
    assert bindings.attempt("attempt:first", SCOPE).policy_revision == 0
    _policy_change(gate, restore=True)
    binding = bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)
    assert bindings.get(binding.binding_id, SCOPE) == binding
    assert bindings.attempt("attempt:first", SCOPE).policy_revision == 0


def test_unqualified_legacy_attempt_cannot_bind(tmp_path: Path) -> None:
    import hashlib
    import json

    from capicove.graph_build.codec import canonical_bytes

    values = build_setup(tmp_path)
    _, _, _, ref, _, _, bindings, _, _, _, commitment = values
    validated = build(values)
    with sqlite3.connect(tmp_path / "bindings" / "catalog.sqlite") as connection:
        row = connection.execute("SELECT payload FROM graph_index_attempts").fetchone()
        payload = json.loads(row[0])
        payload.pop("policy_revision")
        payload["version"] = 1
        encoded = canonical_bytes(payload)
        connection.execute(
            "UPDATE graph_index_attempts SET payload=?, sha256=?",
            (encoded, hashlib.sha256(encoded).hexdigest()),
        )
    with pytest.raises(ValueError, match="attempt"):
        bindings.register(ref, (validated,), commitment.unit_manifest_sha256, SCOPE)


def test_projection_checks_original_revision_immediately_before_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    m, gate, _, _, ref, reader, _ = setup(tmp_path)
    content = reader.read(ref, SCOPE)
    original = m.TextIndexEntry
    created = 0

    def entry(*args: Any, **kwargs: Any) -> Any:
        nonlocal created
        result = original(*args, **kwargs)
        created += 1
        if created == 4:
            _policy_change(gate, restore=True)
        return result

    monkeypatch.setattr(m, "TextIndexEntry", entry)
    with pytest.raises(PermissionError, match="policy"):
        m.graph_text_entries(ref, content, SCOPE, runtime(m), reader=reader)


def test_mapping_policy_drift_consumes_attempt_before_any_index_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = build_setup(tmp_path)
    _, gate, _, _, _, _, bindings, index, _, _, _ = values
    original = index.mapping_sha256

    def mapping(spaces: Any) -> str:
        result: str = original(spaces)
        _policy_change(gate, restore=True)
        return result

    monkeypatch.setattr(index, "mapping_sha256", mapping)
    with pytest.raises(PermissionError, match="policy"):
        build(values)
    assert bindings.attempt("attempt:first", SCOPE).status == "failed"
    assert index.events == []
