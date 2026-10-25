"""Exact derived text and explicit, inactive retrieval generation bindings."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Protocol

from capicove.contracts import (
    CharacterRange,
    EmbeddingSpaceRef,
    ProcessingFingerprint,
    SearchableUnit,
    TargetType,
    TextChunkLocator,
    TextOrigin,
)
from capicove.contracts.graph_build import (
    GraphRetrievalBinding,
    GraphScope,
    GraphSnapshotRef,
    TableRef,
    TextUnit,
)
from capicove.graph_build.codec import (
    canonical_bytes,
    fingerprint,
    graph_from_json,
    graph_to_json,
    manifest_bytes,
)
from capicove.graph_build.ports import AuthorizedRepresentationReader, TextTokenizer
from capicove.graph_build.snapshots import GraphSnapshotCatalog, SnapshotAccessGate
from capicove.processing import LocalProcessingWorkspace
from capicove.retrieval import (
    TextEmbeddingService,
    TextIndexEntry,
    TextIndexGenerationCatalog,
    TextIndexGenerationLifecycle,
    TextIndexGenerationManifest,
    TextIndexGenerationRecord,
    TextIndexGenerationService,
    TextIndexGenerationState,
    TextIndexValidationReport,
    TextIndexWriter,
)


def embedding_ranges(
    text: str, tokenizer: TextTokenizer, max_tokens: int
) -> tuple[tuple[int, int], ...]:
    """Cover UTF-8 text exactly using at most 3N bounded tokenizer calls.

    Counts need not be additive or monotonic. Each code point is checked first;
    greedy growth stops at its first overflow, without claiming maximal packing.
    Input is bounded to 65,536 code points and candidates to 4,096 code points.
    The tokenizer must be deterministic and must not truncate or acquire assets.
    """
    if type(text) is not str:
        raise TypeError("text must be a string")
    text.encode("utf-8")
    if len(text) > 65_536:
        raise ValueError("text exceeds bounded projection input")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise ValueError("max_tokens must be a positive strict integer")

    def count(value: str) -> int:
        result = tokenizer.count(value)
        if type(result) is not int or result < 0 or (value.strip() and result == 0):
            raise ValueError("invalid tokenizer count")
        return result

    for character in text:
        if count(character) > max_tokens:
            raise ValueError("single codepoint exceeds embedding limit")
    ranges = []
    start = 0
    while start < len(text):
        end = start + 1
        while end < min(len(text), start + 4096):
            if count(text[start : end + 1]) > max_tokens:
                break
            end += 1
        if count(text[start:end]) > max_tokens:
            raise ValueError("tokenizer changed during projection")
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def _digest(value: str) -> None:
    if type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("expected lowercase SHA-256")


def _refs(value: tuple[str, ...]) -> None:
    if (
        type(value) is not tuple
        or not value
        or any(type(ref) is not str or not ref.strip() for ref in value)
        or tuple(sorted(set(value))) != value
    ):
        raise ValueError("expected sorted unique nonempty lineage refs")


def _space_json(space: EmbeddingSpaceRef) -> dict[str, Any]:
    value = asdict(space)
    value["query_modalities"] = sorted(item.value for item in space.query_modalities)
    value["target_modalities"] = sorted(item.value for item in space.target_modalities)
    value["normalization"] = space.normalization.value
    value["distance_metric"] = space.distance_metric.value
    return value


class DocumentEmbeddingTokenizer(Protocol):
    """Qualified tokenizer for the encoder's exact document-role wire input.

    Identity and verified limit are immutable for the runtime lifetime. Counts
    include the configured model's document instructions and special tokens.
    The host qualifies correspondence to the injected encoder before startup;
    the encoder must reject overflow and must never truncate or download.
    """

    @property
    def fingerprint(self) -> str: ...

    @property
    def space(self) -> EmbeddingSpaceRef: ...

    @property
    def verified_max_tokens(self) -> int: ...

    def count_document(self, text: str) -> int: ...


@dataclass(frozen=True, slots=True)
class GraphEmbeddingRuntime:
    """Explicit qualified model, tokenizer, role, processing and batch limit."""

    tokenizer: DocumentEmbeddingTokenizer
    service: TextEmbeddingService
    profile_sha256: str
    batch_size: int = 32

    def __post_init__(self) -> None:
        _digest(self.profile_sha256)
        _digest(self.tokenizer.fingerprint)
        if not isinstance(self.service, TextEmbeddingService):
            raise TypeError("service must be TextEmbeddingService")
        if type(self.tokenizer.space) is not EmbeddingSpaceRef:
            raise TypeError("tokenizer must declare the complete embedding space")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("verified model input limit must be positive")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 256:
            raise ValueError("batch_size must be between 1 and 256")

    @property
    def space(self) -> EmbeddingSpaceRef:
        return self.tokenizer.space

    @property
    def max_tokens(self) -> int:
        return self.tokenizer.verified_max_tokens

    @property
    def fingerprint(self) -> str:
        return fingerprint(
            {
                "version": 1,
                "space": _space_json(self.space),
                "tokenizer": self.tokenizer.fingerprint,
                "max_tokens": self.max_tokens,
                "role": "document",
                "profile": self.profile_sha256,
            }
        )

    def count(self, text: str) -> int:
        """Adapt exact document wire counts to the pure range splitter."""
        return self.tokenizer.count_document(text)


@dataclass(frozen=True, slots=True)
class GraphText:
    """One committed source core or explicitly derived entity/report text.

    Source TextUnits remain unchanged. Derived content has no TextUnit and can
    never be expanded as an independent source assertion. Identity binds the
    complete graph, owner, derivation format, profile, text and source closure.
    """

    graph_ref: GraphSnapshotRef
    scope: GraphScope
    profile_sha256: str
    source_class: str
    owner_ref: str
    text: str
    source_sha256: str
    source_refs: tuple[str, ...]
    lineage_refs: tuple[str, ...]
    text_unit: TextUnit | None = None

    def __post_init__(self) -> None:
        if type(self.graph_ref) is not GraphSnapshotRef or type(self.scope) is not GraphScope:
            raise TypeError("text graph and scope must be typed")
        _digest(self.profile_sha256)
        _digest(self.source_sha256)
        _refs(self.source_refs)
        _refs(self.lineage_refs)
        if self.source_class not in {"source", "entity", "report"}:
            raise ValueError("unknown graph text source class")
        if type(self.owner_ref) is not str or not self.owner_ref.strip():
            raise ValueError("missing text owner")
        if type(self.text) is not str or not self.text.strip() or len(self.text) > 65_536:
            raise ValueError("graph text must be nonblank and bounded")
        self.text.encode("utf-8")
        if self.source_class == "source":
            if type(self.text_unit) is not TextUnit or len(self.source_refs) != 1:
                raise ValueError("source text requires its exact TextUnit and evidence")
        elif self.text_unit is not None:
            raise ValueError("derived text cannot own source assertions")

    @property
    def identity_sha256(self) -> str:
        return fingerprint({"version": 1, "text": json.loads(json.dumps(asdict(self)))})

    @property
    def evidence_ref(self) -> str:
        if self.source_class == "source":
            return self.source_refs[0]
        return "evidence:graph-derived-" + self.identity_sha256

    @property
    def representation_ref(self) -> str:
        if self.text_unit is not None:
            return self.text_unit.representation_ref
        return "representation:graph-derived-" + self.identity_sha256

    @property
    def artifact_ref(self) -> str:
        if self.source_class == "source":
            return "artifact:source-" + self.source_sha256
        return "artifact:graph-derived-" + self.identity_sha256


@dataclass(frozen=True, slots=True)
class GraphTextContent:
    """Closed immutable inventory, not an authorization or validation token."""

    graph_ref: GraphSnapshotRef
    scope: GraphScope
    profile_sha256: str
    texts: tuple[GraphText, ...]

    def __post_init__(self) -> None:
        if type(self.graph_ref) is not GraphSnapshotRef or type(self.scope) is not GraphScope:
            raise TypeError("content graph and scope must be typed")
        _digest(self.profile_sha256)
        if type(self.texts) is not tuple or any(type(item) is not GraphText for item in self.texts):
            raise TypeError("texts must be immutable GraphText values")
        if len({(item.source_class, item.owner_ref) for item in self.texts}) != len(self.texts):
            raise ValueError("duplicate graph text identity")

    @property
    def digest(self) -> str:
        return fingerprint(
            {
                "version": 1,
                "graph": asdict(self.graph_ref),
                "scope": asdict(self.scope),
                "profile": self.profile_sha256,
                "texts": [item.identity_sha256 for item in self.texts],
            }
        )


class GraphTextReader:
    """Reconstruct exact graph projections from verified public snapshot reads."""

    def __init__(
        self,
        catalog: GraphSnapshotCatalog,
        workspace: LocalProcessingWorkspace,
        gate: SnapshotAccessGate,
        sources: AuthorizedRepresentationReader,
    ) -> None:
        self.catalog = catalog
        self.workspace = workspace
        self.gate = gate
        self.sources = sources

    def read(self, graph_ref: GraphSnapshotRef, scope: GraphScope) -> GraphTextContent:
        """Verify graph membership, source bytes and exact transitive lineage."""
        from capicove.graph_build._adapters.parquet import ParquetTableCodec

        self.gate.require_scope(scope)
        revision = self.gate.revision
        manifest = self.catalog.get(graph_ref.snapshot_id, scope)
        if hashlib.sha256(manifest_bytes(manifest)).hexdigest() != graph_ref.manifest_sha256:
            raise ValueError("graph manifest digest mismatch")
        tables: dict[str, tuple[Any, ...]] = {}
        for table in manifest.tables:
            if table.role not in {
                "documents",
                "text_units",
                "entities",
                "entity_mentions",
                "relationship_mentions",
                "relationship_assertions",
                "diagnostic_findings",
                "communities",
                "community_reports",
            }:
                continue

            def validate(path: Path, table: TableRef = table) -> bool:
                tables[table.role] = ParquetTableCodec(
                    publication_identity=table.dataset_ref.processing_fingerprint
                ).read(path / "part-00000.parquet", table.role)
                return True

            verified = self.workspace.validate_snapshot(
                table.dataset_ref.snapshot_id, validator=validate
            )
            if verified != table.dataset_ref:
                raise ValueError("graph table commitment mismatch")
        documents = {row.document_ref: row for row in tables["documents"]}
        representations = {}
        for document in documents.values():
            self.gate.require_access((document.evidence_ref,), scope)
            representation = self.sources.read(document.document_ref, scope)
            if (
                representation.scope != scope
                or representation.document_ref != document.document_ref
                or representation.source_sha256 != document.source_sha256
                or representation.representation_ref != document.representation_ref
            ):
                raise ValueError("source representation commitment mismatch")
            representations[document.document_ref] = representation
        texts = []
        for unit in tables["text_units"]:
            document = documents[unit.document_ref]
            representation = representations[unit.document_ref]
            for segment in unit.segments:
                exact = "".join(
                    representation.text[span.start_char : span.end_char]
                    for span in segment.source_spans
                )
                if exact != segment.text:
                    raise ValueError("source text differs from committed TextUnit")
            core = unit.core_locator.character_range
            texts.append(
                GraphText(
                    graph_ref,
                    scope,
                    manifest.profile_fingerprint,
                    "source",
                    unit.text_unit_id,
                    representation.text[core.start_char : core.end_char],
                    unit.source_sha256,
                    (document.evidence_ref,),
                    (unit.text_unit_id,),
                    unit,
                )
            )
        # Exact closure follows graph IDs to source evidence; no derived output
        # is entered as a source leaf. Missing IDs and cycles fail closed.
        dependencies: dict[str, tuple[str, ...]] = {}
        source_refs = {document.evidence_ref for document in documents.values()}
        for role in ("entity_mentions", "relationship_mentions"):
            for row in tables[role]:
                dependencies[row.mention_id] = tuple(
                    sorted({span.evidence_ref for span in row.source_spans})
                )
        for row in tables["entities"]:
            dependencies[row.entity_id] = row.mention_ids
        for row in tables["relationship_assertions"]:
            dependencies[row.assertion_id] = row.evidence_refs
        for row in tables["diagnostic_findings"]:
            dependencies[row.finding_id] = (*row.support_refs, *row.counter_refs)
        for row in tables["communities"]:
            dependencies[row.community_id] = row.member_ids
        for row in tables["community_reports"]:
            refs = [row.community_id] if row.community_id is not None else []
            for field in fields(row.sections):
                for statement in getattr(row.sections, field.name):
                    refs.extend(statement.assertion_refs)
                    refs.extend(statement.finding_refs)
                    refs.extend(child.report_id for child in statement.child_statement_refs)
            dependencies[row.report_id] = tuple(sorted(set(refs)))

        dependency_count = sum(
            len(tables[role])
            for role in (
                "entity_mentions",
                "relationship_mentions",
                "entities",
                "relationship_assertions",
                "diagnostic_findings",
                "communities",
                "community_reports",
            )
        )
        if len(dependencies) != dependency_count or source_refs.intersection(dependencies):
            raise ValueError("ambiguous graph text lineage identity")

        def expand(ref: str, visiting: frozenset[str] = frozenset()) -> set[str]:
            if ref in source_refs:
                return {ref}
            if ref in visiting or ref not in dependencies or len(visiting) >= 128:
                raise ValueError("missing, alien or cyclic graph text lineage")
            return set().union(*(expand(child, visiting | {ref}) for child in dependencies[ref]))

        for role, source_class, id_field in (
            ("entities", "entity", "entity_id"),
            ("community_reports", "report", "report_id"),
        ):
            for row in tables[role]:
                owner = getattr(row, id_field)
                if source_class == "entity":
                    text = "\n".join((row.display_name, row.entity_type, *row.aliases))
                else:
                    text = "\n".join(
                        (
                            row.title,
                            *(
                                field.name + ": " + statement.text
                                for field in fields(row.sections)
                                for statement in getattr(row.sections, field.name)
                            ),
                        )
                    )
                source_closure = tuple(sorted(expand(owner)))
                if not source_closure:
                    if source_class == "report":
                        continue
                    raise ValueError("derived entity has no source evidence")
                texts.append(
                    GraphText(
                        graph_ref,
                        scope,
                        manifest.profile_fingerprint,
                        source_class,
                        owner,
                        text,
                        hashlib.sha256(text.encode()).hexdigest(),
                        source_closure,
                        tuple(
                            sorted(
                                {
                                    owner,
                                    "graph-row:" + fingerprint(asdict(row)),
                                    *dependencies[owner],
                                }
                            )
                        ),
                    )
                )
        content = GraphTextContent(graph_ref, scope, manifest.profile_fingerprint, tuple(texts))
        with self.gate.guard(revision):
            self.require_access(content)
        return content

    def require_access(self, content: GraphTextContent) -> None:
        """Recheck graph eligibility and all original sources before reuse.

        Content was fully verified by ``read``; ``_check_projection`` verifies it
        again before reservation and binding, so per-batch checks reauthorize
        without re-decoding graph artifacts (RFC-0019 Phase 2).
        """
        self.catalog.require_eligible(content.graph_ref.snapshot_id, content.scope)
        refs = tuple(sorted({ref for text in content.texts for ref in text.source_refs}))
        self.gate.require_scope(content.scope)
        self.gate.require_access(refs, content.scope)
        self.sources.require_access(refs, content.scope)


def _require_policy_revision(gate: SnapshotAccessGate, policy_revision: int) -> None:
    """Reject drift even if effective access was revoked and then restored."""
    if type(policy_revision) is not int or policy_revision < 0:
        raise ValueError("invalid policy revision")
    current = gate.revision
    if type(current) is not int or current != policy_revision:
        raise PermissionError("policy changed during retrieval operation")


def _capture_policy_revision(gate: SnapshotAccessGate, scope: GraphScope) -> int:
    revision = gate.revision
    _require_policy_revision(gate, revision)
    gate.require_scope(scope)
    _require_policy_revision(gate, revision)
    return revision


def _authorize_at_revision(
    reader: GraphTextReader, content: GraphTextContent, policy_revision: int
) -> None:
    """Reauthorize exact graph/source dependencies without advancing the fence."""
    _require_policy_revision(reader.gate, policy_revision)
    reader.require_access(content)
    _require_policy_revision(reader.gate, policy_revision)


def _uncontrolled() -> None:
    """Keep standalone projection calls compatible without a build controller."""


def graph_text_entries(
    graph_ref: GraphSnapshotRef,
    content: GraphTextContent,
    scope: GraphScope,
    embedder: GraphEmbeddingRuntime,
    *,
    reader: GraphTextReader,
    check_control: Callable[[], None] = _uncontrolled,
) -> tuple[TextIndexEntry, ...]:
    """Expand verified text into exact source-mapped, document-role dense entries."""
    if (
        type(content) is not GraphTextContent
        or content.graph_ref != graph_ref
        or content.scope != scope
    ):
        raise ValueError("graph text scope or graph mismatch")
    check_control()
    policy_revision = _capture_policy_revision(reader.gate, scope)
    if reader.read(graph_ref, scope) != content:
        raise ValueError("graph text is not committed by the snapshot")
    _authorize_at_revision(reader, content, policy_revision)
    runtime_identity = embedder.fingerprint
    pending = _planned_units(content, embedder, check_control=check_control)
    _authorize_at_revision(reader, content, policy_revision)
    entries: list[TextIndexEntry] = []
    for offset in range(0, len(pending), embedder.batch_size):
        check_control()
        _authorize_at_revision(reader, content, policy_revision)
        batch = pending[offset : offset + embedder.batch_size]
        vectors = embedder.service.encode(
            tuple(text for _, text in batch), space=embedder.space, role="document"
        )
        check_control()
        _authorize_at_revision(reader, content, policy_revision)
        if embedder.fingerprint != runtime_identity:
            raise ValueError("embedding runtime changed during projection")
        entries.extend(
            TextIndexEntry(unit, text, (graph_ref.snapshot_id,), "text/plain", embeddings=(vector,))
            for (unit, text), vector in zip(batch, vectors, strict=True)
        )
    _authorize_at_revision(reader, content, policy_revision)
    return tuple(entries)


def _planned_units(
    content: GraphTextContent,
    embedder: GraphEmbeddingRuntime,
    *,
    check_control: Callable[[], None] = _uncontrolled,
) -> tuple[tuple[SearchableUnit, str], ...]:
    projection = fingerprint({"content": content.digest, "runtime": embedder.fingerprint})
    pending = []
    for item in content.texts:
        check_control()
        for start, end in embedding_ranges(item.text, embedder, embedder.max_tokens):
            text = item.text[start:end]
            if not text.strip():
                raise ValueError("model limit cannot represent an exact whitespace subunit")
            base = item.text_unit.core_locator.character_range.start_char if item.text_unit else 0
            unit = SearchableUnit(
                unit_id="graph-text-unit:"
                + fingerprint(
                    {"projection": projection, "text": item.identity_sha256, "range": [start, end]}
                ),
                unit_type=TargetType.TEXT_CHUNK,
                source_hash=item.source_sha256,
                artifact_ref=item.artifact_ref,
                evidence_ref=item.evidence_ref,
                representation_refs=(item.representation_ref,),
                processing_run_ref="graph-projection:" + projection,
                processing_version="graph-text-v1",
                processing_fingerprint=ProcessingFingerprint(projection),
                embedding_spaces=(embedder.space,),
                text_locator=TextChunkLocator(
                    TextOrigin.DOCUMENT, CharacterRange(base + start, base + end)
                ),
            )
            pending.append((unit, text))
    return tuple(pending)


def _entry_json(entry: TextIndexEntry, *, vectors: bool) -> dict[str, Any]:
    value = asdict(entry)
    value["unit"]["embedding_spaces"] = [
        _space_json(space) for space in entry.unit.embedding_spaces
    ]
    value["embeddings"] = (
        [
            {"space": _space_json(embedding.space), "vector": embedding.vector}
            for embedding in entry.embeddings
        ]
        if vectors
        else []
    )
    return json.loads(json.dumps(value))  # type: ignore[no-any-return]


def text_index_content_sha256(entries: tuple[TextIndexEntry, ...]) -> str:
    """Commit ordered complete entries including vectors for provider validation."""
    return fingerprint(
        {"version": 1, "entries": [_entry_json(entry, vectors=True) for entry in entries]}
    )


@dataclass(frozen=True, slots=True)
class ProjectionCommitment:
    """Closed exact graph, unit, content, mapping and qualified runtime digests."""

    graph_ref: GraphSnapshotRef
    scope: GraphScope
    graph_text_sha256: str
    runtime_sha256: str
    unit_manifest_sha256: str
    content_sha256: str
    mapping_sha256: str
    embedding_spaces: tuple[EmbeddingSpaceRef, ...]
    unit_count: int

    def __post_init__(self) -> None:
        if type(self.graph_ref) is not GraphSnapshotRef or type(self.scope) is not GraphScope:
            raise TypeError("projection graph/scope must be typed")
        for value in (
            self.graph_text_sha256,
            self.runtime_sha256,
            self.unit_manifest_sha256,
            self.content_sha256,
            self.mapping_sha256,
        ):
            _digest(value)
        if (
            type(self.embedding_spaces) is not tuple
            or len(self.embedding_spaces) != 1
            or type(self.embedding_spaces[0]) is not EmbeddingSpaceRef
        ):
            raise ValueError("projection requires one exact embedding space")
        if type(self.unit_count) is not int or self.unit_count < 1:
            raise ValueError("projection must contain units")

    @property
    def digest(self) -> str:
        return fingerprint(_commitment_json(self))


def _commitment_json(value: ProjectionCommitment) -> dict[str, Any]:
    payload = asdict(value)
    payload["embedding_spaces"] = [_space_json(space) for space in value.embedding_spaces]
    return payload


def projection_commitment(
    content: GraphTextContent,
    runtime: GraphEmbeddingRuntime,
    entries: tuple[TextIndexEntry, ...],
    mapping_sha256: str,
) -> ProjectionCommitment:
    """Verify exact planned units and retain complete ordered vector commitments."""
    import math

    from capicove.contracts import DistanceMetric, VectorNormalization

    if (
        type(entries) is not tuple
        or not entries
        or any(type(entry) is not TextIndexEntry for entry in entries)
    ):
        raise ValueError("entries must be a nonempty immutable text inventory")
    if tuple((entry.unit, entry.text) for entry in entries) != _planned_units(content, runtime):
        raise ValueError("entries differ from exact graph projection")
    for entry in entries:
        if (
            entry.corpus_refs != (content.graph_ref.snapshot_id,)
            or entry.media_type != "text/plain"
            or entry.language is not None
        ):
            raise ValueError("entry metadata differs from graph projection")
        if len(entry.embeddings) != 1 or entry.embeddings[0].space != runtime.space:
            raise ValueError("entry embedding space mismatch")
        vector = entry.embeddings[0].vector
        if len(vector) != runtime.space.dimension or any(
            not math.isfinite(value) for value in vector
        ):
            raise ValueError("invalid embedding vector")
        norm = math.hypot(*vector)
        if (
            runtime.space.normalization is VectorNormalization.L2
            and not math.isclose(norm, 1.0, rel_tol=1e-5, abs_tol=1e-6)
        ) or (runtime.space.distance_metric is DistanceMetric.COSINE and norm == 0):
            raise ValueError("invalid embedding normalization")
    return ProjectionCommitment(
        content.graph_ref,
        content.scope,
        content.digest,
        runtime.fingerprint,
        fingerprint(
            {
                "version": 1,
                "content": content.digest,
                "runtime": runtime.fingerprint,
                "entries": [_entry_json(entry, vectors=False) for entry in entries],
            }
        ),
        text_index_content_sha256(entries),
        mapping_sha256,
        (runtime.space,),
        len(entries),
    )


@dataclass(frozen=True, slots=True)
class GraphIndexAttempt:
    """Caller-owned fresh attempt identity and explicitly selected backend format."""

    attempt_ref: str
    retriever_id: str
    index_format_version: str

    def __post_init__(self) -> None:
        for value in (self.attempt_ref, self.retriever_id, self.index_format_version):
            if type(value) is not str or not value.strip() or len(value) > 256:
                raise ValueError("attempt identity must be nonblank bounded text")
            value.encode("utf-8")


def _generation_manifest(
    projection: ProjectionCommitment, attempt: GraphIndexAttempt
) -> TextIndexGenerationManifest:
    identity = fingerprint({"projection": projection.digest, "attempt": asdict(attempt)})
    return TextIndexGenerationManifest(
        "graph-index:" + identity,
        projection.scope.workspace_ref,
        (projection.graph_ref.snapshot_id,),
        projection.unit_manifest_sha256,
        projection.content_sha256,
        projection.mapping_sha256,
        projection.embedding_spaces,
        attempt.retriever_id,
        attempt.index_format_version,
        ProcessingFingerprint(projection.digest),
        projection.unit_count,
    )


@dataclass(frozen=True, slots=True)
class ValidatedGraphGeneration:
    """Exact public catalog evidence plus graph projection and retry provenance."""

    record: TextIndexGenerationRecord
    projection: ProjectionCommitment
    attempt: GraphIndexAttempt

    def __post_init__(self) -> None:
        if (
            type(self.record) is not TextIndexGenerationRecord
            or type(self.projection) is not ProjectionCommitment
            or type(self.attempt) is not GraphIndexAttempt
        ):
            raise TypeError("validated generation fields must be closed typed records")
        if self.record.manifest != _generation_manifest(self.projection, self.attempt):
            raise ValueError("generation does not match exact graph projection/attempt")
        if (
            self.record.state
            not in {
                TextIndexGenerationState.READY,
                TextIndexGenerationState.ACTIVE,
                TextIndexGenerationState.SUPERSEDED,
            }
            or self.record.validation is None
            or not self.record.validation.valid
        ):
            raise ValueError("generation is not validated for pinned use")
        TextIndexValidationReport(
            **{
                field.name: getattr(self.record.validation, field.name)
                for field in fields(self.record.validation)
            }
        )
        # Reconstruct to recheck physical, count and mapping validation evidence.
        TextIndexGenerationRecord(
            **{field.name: getattr(self.record, field.name) for field in fields(self.record)}
        )


def bind_retrieval(
    graph_ref: GraphSnapshotRef,
    validated_generations: tuple[ValidatedGraphGeneration, ...],
    unit_manifest_sha256: str,
    scope: GraphScope,
) -> GraphRetrievalBinding:
    """Bind exact validated generations without resolving or activating aliases."""
    _digest(unit_manifest_sha256)
    if type(validated_generations) is not tuple or not validated_generations:
        raise ValueError("binding requires validated generations")
    generations = []
    spaces: dict[str, EmbeddingSpaceRef] = {}
    commitments = []
    for value in validated_generations:
        if type(value) is not ValidatedGraphGeneration:
            raise TypeError("binding requires ValidatedGraphGeneration")
        value.__post_init__()
        projection = value.projection
        if (
            projection.graph_ref != graph_ref
            or projection.scope != scope
            or projection.unit_manifest_sha256 != unit_manifest_sha256
        ):
            raise ValueError("binding graph, scope or unit manifest mismatch")
        for space in projection.embedding_spaces:
            if space.space_id in spaces:
                raise ValueError("binding has duplicate embedding space IDs")
            spaces[space.space_id] = space
        manifest = value.record.manifest
        generations.append(manifest.generation_id + "#" + manifest.manifest_sha256)
        commitments.append(projection.digest)
    if len(set(generations)) != len(generations):
        raise ValueError("binding has duplicate generations")
    identity = fingerprint(
        {
            "version": 1,
            "graph": asdict(graph_ref),
            "scope": asdict(scope),
            "generations": sorted(generations),
            "projections": sorted(commitments),
            "unit_manifest": unit_manifest_sha256,
        }
    )
    return GraphRetrievalBinding(
        "graph-binding:" + identity,
        graph_ref.snapshot_id,
        tuple(generations),
        tuple(spaces.values()),
        unit_manifest_sha256,
    )


@dataclass(frozen=True, slots=True)
class GraphBindingAttempt:
    """Bounded durable outcome separate from graph completeness and activation."""

    attempt: GraphIndexAttempt
    projection: ProjectionCommitment
    status: str
    issue_code: str | None
    policy_revision: int

    def __post_init__(self) -> None:
        if (
            type(self.attempt) is not GraphIndexAttempt
            or type(self.projection) is not ProjectionCommitment
        ):
            raise TypeError("attempt metadata must contain closed typed records")
        if type(self.policy_revision) is not int or self.policy_revision < 0:
            raise ValueError("invalid attempt policy revision")
        if self.status not in {"started", "ready", "failed"} or self.issue_code != (
            "retrieval_build_failed" if self.status == "failed" else None
        ):
            raise ValueError("invalid attempt status")


class GraphRetrievalCatalog:
    """Scoped immutable binding metadata and exclusive append attempt ownership.

    No vectors or graph rows are duplicated here. External stores cannot share
    this transaction: every reuse rechecks graph eligibility and catalog pins.
    """

    def __init__(
        self,
        database_path: Path,
        reader: GraphTextReader,
        gate: SnapshotAccessGate,
        generations: TextIndexGenerationCatalog,
    ) -> None:
        from capicove.graph_build._adapters.sqlite import SqliteBindingStore

        if reader.gate is not gate:
            raise ValueError("binding and projection must share one policy revision authority")
        self._store = SqliteBindingStore(database_path)
        self.reader = reader
        self.gate = gate
        self.generations = generations

    def _check_projection(self, projection: ProjectionCommitment) -> int:
        self.gate.require_scope(projection.scope)
        revision = self.gate.revision
        if type(revision) is not int or revision < 0:
            raise ValueError("invalid policy revision")
        if (
            self.reader.read(projection.graph_ref, projection.scope).digest
            != projection.graph_text_sha256
        ):
            raise ValueError("projection differs from current verified graph text")
        if self.gate.revision != revision:
            raise PermissionError("policy changed during binding validation")
        return revision

    def _reserve(
        self,
        projection: ProjectionCommitment,
        attempt: GraphIndexAttempt,
        policy_revision: int,
    ) -> None:
        _require_policy_revision(self.gate, policy_revision)
        if self._check_projection(projection) != policy_revision:
            raise PermissionError("policy changed before attempt reservation")
        state = GraphBindingAttempt(attempt, projection, "started", None, policy_revision)
        payload = _attempt_bytes(state)
        with self.gate.guard(policy_revision), self._store.transaction() as connection:
            existing = connection.execute(
                "SELECT 1 FROM graph_index_attempts WHERE workspace_ref=? AND scope_ref=? AND attempt_ref=?",
                (projection.scope.workspace_ref, projection.scope.scope_ref, attempt.attempt_ref),
            ).fetchone()
            if existing is not None:
                raise ValueError("attempt already consumed; retry requires a fresh attempt")
            connection.execute(
                "INSERT INTO graph_index_attempts VALUES (?, ?, ?, ?, ?)",
                (
                    projection.scope.workspace_ref,
                    projection.scope.scope_ref,
                    attempt.attempt_ref,
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                ),
            )
            _require_policy_revision(self.gate, policy_revision)

    def _finish(
        self,
        projection: ProjectionCommitment,
        attempt: GraphIndexAttempt,
        status: str,
        *,
        policy_revision: int,
        content: GraphTextContent,
    ) -> None:
        scope = projection.scope
        if (
            content.graph_ref != projection.graph_ref
            or content.scope != scope
            or content.digest != projection.graph_text_sha256
        ):
            raise ValueError("attempt completion content mismatch")
        if status == "ready":
            _authorize_at_revision(self.reader, content, policy_revision)
            commit_revision = policy_revision
        else:
            # Failed metadata preserves the attempted revision but is recorded
            # under current scope authority. It cannot qualify a generation.
            commit_revision = _capture_policy_revision(self.gate, scope)
        state = GraphBindingAttempt(
            attempt,
            projection,
            status,
            "retrieval_build_failed" if status == "failed" else None,
            policy_revision,
        )
        payload = _attempt_bytes(state)
        source_refs = tuple(sorted({ref for item in content.texts for ref in item.source_refs}))
        with self.gate.guard(commit_revision), self._store.transaction() as connection:
            _require_policy_revision(self.gate, commit_revision)
            if status == "ready":
                self.gate.require_scope(scope)
                self.gate.require_access(source_refs, scope)
                _require_policy_revision(self.gate, policy_revision)
            row = connection.execute(
                "SELECT payload, sha256 FROM graph_index_attempts WHERE workspace_ref=? AND scope_ref=? AND attempt_ref=?",
                (scope.workspace_ref, scope.scope_ref, attempt.attempt_ref),
            ).fetchone()
            if row is None:
                raise ValueError("attempt has not been reserved")
            previous = _decode_attempt(row[0], row[1])
            if (
                previous.attempt != attempt
                or previous.projection != projection
                or previous.status != "started"
                or previous.policy_revision != policy_revision
            ):
                raise ValueError("attempt terminal state is immutable")
            connection.execute(
                "UPDATE graph_index_attempts SET payload=?, sha256=? WHERE workspace_ref=? AND scope_ref=? AND attempt_ref=?",
                (
                    payload,
                    hashlib.sha256(payload).hexdigest(),
                    scope.workspace_ref,
                    scope.scope_ref,
                    attempt.attempt_ref,
                ),
            )
            if status == "ready":
                self.gate.require_scope(scope)
                self.gate.require_access(source_refs, scope)
            _require_policy_revision(self.gate, commit_revision)

    def attempt(self, attempt_ref: str, scope: GraphScope) -> GraphBindingAttempt | None:
        """Read currently authorized attempt metadata, including failed retrieval."""
        self.gate.require_scope(scope)
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT payload, sha256 FROM graph_index_attempts WHERE workspace_ref=? AND scope_ref=? AND attempt_ref=?",
                (scope.workspace_ref, scope.scope_ref, attempt_ref),
            ).fetchone()
        if row is None:
            return None
        state = _decode_attempt(row[0], row[1])
        if state.attempt.attempt_ref != attempt_ref or state.projection.scope != scope:
            raise ValueError("attempt identity is corrupt")
        self._check_projection(state.projection)
        return state

    def validated_attempt(
        self, attempt_ref: str, scope: GraphScope
    ) -> ValidatedGraphGeneration | None:
        """Resolve a ready attempt to its exact currently authorized generation."""
        state = self.attempt(attempt_ref, scope)
        if state is None or state.status != "ready":
            return None
        manifest = _generation_manifest(state.projection, state.attempt)
        record = self.generations.get(manifest.generation_id)
        if record is None:
            raise ValueError("ready attempt generation is unavailable")
        validated = ValidatedGraphGeneration(record, state.projection, state.attempt)
        self._validate((validated,))
        return validated

    def _validate(self, values: tuple[ValidatedGraphGeneration, ...]) -> int:
        revision = self.gate.revision
        for value in values:
            if self._check_projection(value.projection) != revision:
                raise PermissionError("binding policy changed")
            current = self.generations.get(value.record.manifest.generation_id)
            if (
                current is None
                or current.manifest != value.record.manifest
                or current.physical_index_ref != value.record.physical_index_ref
                or current.validation != value.record.validation
            ):
                raise ValueError("generation validation evidence changed or unavailable")
            ValidatedGraphGeneration(current, value.projection, value.attempt)
            state = self.attempt(value.attempt.attempt_ref, value.projection.scope)
            if (
                state is None
                or state.status != "ready"
                or state.projection != value.projection
                or state.attempt != value.attempt
            ):
                raise ValueError("generation has no successful owned append attempt")
            if state.policy_revision > revision:
                raise PermissionError("attempt policy revision exceeds current policy")
        _require_policy_revision(self.gate, revision)
        return revision

    def register(
        self,
        graph_ref: GraphSnapshotRef,
        values: tuple[ValidatedGraphGeneration, ...],
        unit_manifest_sha256: str,
        scope: GraphScope,
    ) -> GraphRetrievalBinding:
        """Commit exact binding bytes only after graph/source/generation revalidation."""
        binding = bind_retrieval(graph_ref, values, unit_manifest_sha256, scope)
        revision = self._validate(values)
        payload = canonical_bytes(
            {
                "version": 1,
                "binding": graph_to_json(binding),
                "attempts": sorted(value.attempt.attempt_ref for value in values),
            }
        )
        with self.gate.guard(revision):
            self._validate(values)
            with self._store.transaction() as connection:
                row = connection.execute(
                    "SELECT payload, sha256 FROM graph_retrieval_bindings WHERE workspace_ref=? AND scope_ref=? AND binding_id=?",
                    (scope.workspace_ref, scope.scope_ref, binding.binding_id),
                ).fetchone()
                if row is not None:
                    if row[0] != payload or row[1] != hashlib.sha256(payload).hexdigest():
                        raise ValueError("immutable binding identity is corrupt")
                else:
                    connection.execute(
                        "INSERT INTO graph_retrieval_bindings VALUES (?, ?, ?, ?, ?)",
                        (
                            scope.workspace_ref,
                            scope.scope_ref,
                            binding.binding_id,
                            payload,
                            hashlib.sha256(payload).hexdigest(),
                        ),
                    )
                self.gate.require_scope(scope)
                if self.gate.revision != revision:
                    raise PermissionError("policy changed before binding commit")
        return binding

    def get(self, binding_id: str, scope: GraphScope) -> GraphRetrievalBinding | None:
        """Resolve exact pins under current eligibility, never a global active index."""
        self.gate.require_scope(scope)
        with self._store.transaction() as connection:
            row = connection.execute(
                "SELECT payload, sha256 FROM graph_retrieval_bindings WHERE workspace_ref=? AND scope_ref=? AND binding_id=?",
                (scope.workspace_ref, scope.scope_ref, binding_id),
            ).fetchone()
        if row is None:
            return None
        payload = _decode_payload(row[0], row[1])
        if set(payload) != {"version", "binding", "attempts"} or payload["version"] != 1:
            raise ValueError("unsupported binding payload")
        binding = graph_from_json(payload["binding"])
        if type(binding) is not GraphRetrievalBinding or binding.binding_id != binding_id:
            raise ValueError("binding identity is corrupt")
        values = []
        for ref in payload["attempts"]:
            state = self.attempt(ref, scope)
            if state is None or state.status != "ready":
                raise ValueError("binding attempt unavailable")
            manifest = _generation_manifest(state.projection, state.attempt)
            record = self.generations.get(manifest.generation_id)
            if record is None:
                raise ValueError("binding generation unavailable")
            values.append(ValidatedGraphGeneration(record, state.projection, state.attempt))
        if not values:
            raise ValueError("binding has no generations")
        exact = bind_retrieval(
            values[0].projection.graph_ref, tuple(values), binding.unit_manifest_sha256, scope
        )
        if exact != binding:
            raise ValueError("binding metadata does not match exact generations")
        revision = self._validate(tuple(values))
        with self.gate.guard(revision):
            self._validate(tuple(values))
        return binding


def _attempt_bytes(value: GraphBindingAttempt) -> bytes:
    return canonical_bytes(
        {
            "version": 2,
            "attempt": asdict(value.attempt),
            "projection": _commitment_json(value.projection),
            "status": value.status,
            "issue_code": value.issue_code,
            "policy_revision": value.policy_revision,
        }
    )


def _decode_payload(payload: bytes, digest: str) -> dict[str, Any]:
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("binding metadata digest mismatch")
    value = json.loads(payload)
    if type(value) is not dict or canonical_bytes(value) != payload:
        raise ValueError("noncanonical binding metadata")
    return value


def _decode_attempt(payload: bytes, digest: str) -> GraphBindingAttempt:
    from capicove.contracts import DistanceMetric, QueryModality, VectorNormalization

    value = _decode_payload(payload, digest)
    if (
        set(value)
        != {"version", "attempt", "projection", "status", "issue_code", "policy_revision"}
        or type(value["version"]) is not int
        or value["version"] != 2
        or value["status"] not in {"started", "ready", "failed"}
        or value["issue_code"]
        != ("retrieval_build_failed" if value["status"] == "failed" else None)
    ):
        raise ValueError("invalid attempt state")
    projection = value["projection"]
    spaces = []
    for item in projection["embedding_spaces"]:
        spaces.append(
            EmbeddingSpaceRef(
                **{
                    **item,
                    "normalization": VectorNormalization(item["normalization"]),
                    "distance_metric": DistanceMetric(item["distance_metric"]),
                    "query_modalities": frozenset(
                        QueryModality(v) for v in item["query_modalities"]
                    ),
                    "target_modalities": frozenset(
                        TargetType(v) for v in item["target_modalities"]
                    ),
                }
            )
        )
    commitment = ProjectionCommitment(
        **{
            **projection,
            "graph_ref": GraphSnapshotRef(**projection["graph_ref"]),
            "scope": GraphScope(**projection["scope"]),
            "embedding_spaces": tuple(spaces),
        }
    )
    state = GraphBindingAttempt(
        GraphIndexAttempt(**value["attempt"]),
        commitment,
        value["status"],
        value["issue_code"],
        value["policy_revision"],
    )
    if _attempt_bytes(state) != payload:
        raise ValueError("attempt metadata is noncanonical")
    return state


def build_graph_generation(
    projection: ProjectionCommitment,
    entries: tuple[TextIndexEntry, ...],
    attempt: GraphIndexAttempt,
    *,
    runtime: GraphEmbeddingRuntime,
    lifecycle: TextIndexGenerationLifecycle,
    writer: TextIndexWriter,
    generations: TextIndexGenerationCatalog,
    bindings: GraphRetrievalCatalog,
    batch_size: int = 32,
    check_control: Callable[[], None] = _uncontrolled,
) -> ValidatedGraphGeneration:
    """Create, append once, validate and retain one explicitly owned attempt.

    Lifecycle.validate must attest exact content, count, mapping and all spaces
    against the supplied manifest. An uncertain writer exception permanently
    consumes this attempt even if the provider subsequently reports valid data.
    A retry must use a caller-supplied fresh attempt. No aliases are consulted.
    """
    if type(batch_size) is not int or not 1 <= batch_size <= 256:
        raise ValueError("invalid index batch size")
    if (
        writer.provider_id != attempt.retriever_id
        or lifecycle.provider_id != attempt.retriever_id
        or generations is not bindings.generations
    ):
        raise ValueError("generation runtime provider/catalog mismatch")
    check_control()
    policy_revision = _capture_policy_revision(bindings.gate, projection.scope)
    content = bindings.reader.read(projection.graph_ref, projection.scope)
    _authorize_at_revision(bindings.reader, content, policy_revision)
    manifest = _generation_manifest(projection, attempt)
    bindings._reserve(projection, attempt, policy_revision)
    service = TextIndexGenerationService(lifecycle, generations)
    try:
        check_control()
        _authorize_at_revision(bindings.reader, content, policy_revision)
        mapping_sha256 = lifecycle.mapping_sha256(projection.embedding_spaces)
        check_control()
        _authorize_at_revision(bindings.reader, content, policy_revision)
        expected = projection_commitment(content, runtime, entries, mapping_sha256)
        _authorize_at_revision(bindings.reader, content, policy_revision)
        if expected != projection:
            raise ValueError("generation projection commitment mismatch")
        if generations.get(manifest.generation_id) is not None:
            raise ValueError("generation already exists; append cannot resume")
        _authorize_at_revision(bindings.reader, content, policy_revision)
        check_control()
        service.create(manifest)
        check_control()
        _authorize_at_revision(bindings.reader, content, policy_revision)
        for offset in range(0, len(entries), batch_size):
            check_control()
            _authorize_at_revision(bindings.reader, content, policy_revision)
            writer.add(
                index_generation_id=manifest.generation_id,
                entries=entries[offset : offset + batch_size],
            )
            check_control()
            _authorize_at_revision(bindings.reader, content, policy_revision)
        _authorize_at_revision(bindings.reader, content, policy_revision)
        check_control()
        record = service.validate(manifest.generation_id)
        check_control()
        _authorize_at_revision(bindings.reader, content, policy_revision)
        result = ValidatedGraphGeneration(record, projection, attempt)
        _authorize_at_revision(bindings.reader, content, policy_revision)
        if bindings._check_projection(projection) != policy_revision:
            raise PermissionError("policy changed before attempt readiness")
        _authorize_at_revision(bindings.reader, content, policy_revision)
        check_control()
        bindings._finish(
            projection,
            attempt,
            "ready",
            policy_revision=policy_revision,
            content=content,
        )
        return result
    except Exception as error:
        # A failed write can have committed any prefix. Closing its validation
        # never makes this abandoned attempt eligible for graph binding.
        try:
            check_control()
            service.validate(manifest.generation_id)
        except Exception:
            error.add_note("Abandoned generation could not be validated; attempt remains unusable.")
        try:
            bindings._finish(
                projection,
                attempt,
                "failed",
                policy_revision=policy_revision,
                content=content,
            )
        except Exception:
            error.add_note("Attempt cleanup failed; the reserved attempt cannot be resumed.")
        raise
