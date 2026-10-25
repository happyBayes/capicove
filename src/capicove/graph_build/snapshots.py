"""Verified immutable graph registration and scoped compare-and-set activation.

Graph rows remain in processing's Parquet store. Host input/stage validators
attest their external schemas; this module verifies all graph table relations.
The injected policy guard defines policy consistency. SQLite transactions do
not make an external policy service or the filesystem globally atomic.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from dataclasses import asdict, dataclass, fields
from itertools import pairwise
from pathlib import Path
from typing import Any, Protocol

from capicove.contracts.data import DataFormat, DatasetSnapshotRef
from capicove.contracts.graph_build import (
    EntityMention,
    GraphScope,
    GraphSnapshotManifest,
    GraphSnapshotRef,
    SourceSpan,
    StageResult,
    TableRef,
)
from capicove.graph_build._adapters.sqlite import SqliteSnapshotStore
from capicove.graph_build._decode_reuse import GRAPH_MANIFESTS
from capicove.graph_build.codec import canonical_bytes, fingerprint, graph_from_json, manifest_bytes
from capicove.graph_build.datasets import codec_for_role
from capicove.graph_build.ports import AccessGate
from capicove.processing import LocalProcessingWorkspace, ProcessingRecoveryError


class GraphActivationConflict(RuntimeError):
    """The caller's expected active revision no longer owns the pointer."""


class GraphValidationError(ValueError):
    """A graph is unavailable, ineligible or fails immutable validation."""


class SnapshotAccessGate(AccessGate, Protocol):
    """Current monotonic policy revision with an explicit publication guard.

    Implementations serialize policy changes with guard entry/exit, reject an
    obsolete expected revision, and preserve the revision through its body.
    require_scope includes graph lifecycle authority for this configured host.
    A remote implementation must supply equivalent fencing semantics explicitly.
    """

    @property
    def revision(self) -> int:
        """Return a nonnegative, never-reused revision of the current policy."""
        ...

    def guard(self, expected_revision: int) -> AbstractContextManager[None]:
        """Fence current authorization through a short SQLite commit."""
        ...


@dataclass(frozen=True, slots=True)
class InputSnapshotInventory:
    """Verified external input-set identity and exact declared source inventory."""

    snapshot_ref: DatasetSnapshotRef
    scope: GraphScope
    document_refs: tuple[str, ...]
    evidence_refs: tuple[str, ...]
    validation_sha256: str


class InputSnapshotValidator(Protocol):
    """Read the host-owned document-set schema under current authorization.

    Verify source availability and exact immutable input content. The returned
    digest commits to the validation version and complete verified inventory.
    Never infer document membership from the output graph's documents table.
    """

    def validate(self, ref: DatasetSnapshotRef, scope: GraphScope) -> InputSnapshotInventory: ...


@dataclass(frozen=True, slots=True)
class ValidatedStageArtifacts:
    """Attest exact stage/coverage, all artifact refs and profile provenance."""

    stage: StageResult
    scope: GraphScope
    input_snapshot_ref: DatasetSnapshotRef
    profile_fingerprint: str
    evidence_refs: tuple[str, ...]
    validation_sha256: str
    table_refs: tuple[TableRef, ...]

    def __post_init__(self) -> None:
        if type(self.input_snapshot_ref) is not DatasetSnapshotRef:
            raise TypeError("input_snapshot_ref must be DatasetSnapshotRef")


class StageArtifactValidator(Protocol):
    """Verify every referenced stage artifact and its closed coverage semantics.

    Validate availability, content, the exact input snapshot and profile/version,
    and the declared denominator/population commitment. The digest includes all
    artifact bytes and validator version. Evidence refs include every source
    dependency. Every returned artifact attestation must bind to the input snapshot
    that produced it; an artifact from another build is never interchangeable.
    """

    def validate(
        self, stage: StageResult, scope: GraphScope, profile_fingerprint: str
    ) -> ValidatedStageArtifacts: ...


_DENOMINATORS = {
    "representations": "documents",
    "text_units": "codepoints",
    "extraction": "text_units",
    "identity_embeddings": "descriptors",
    "resolution": "mentions",
    "alignment": "entities",
    "aggregation": "assertions",
    "diagnostics": "bundles",
    "communities": "entities",
    "reports": "reports",
}

_STAGE_ROLES = {
    "representations": frozenset({"documents"}),
    "text_units": frozenset({"text_units"}),
    "extraction": frozenset({"entity_mentions", "relationship_mentions"}),
    "identity_embeddings": frozenset(),
    "resolution": frozenset({"entities", "entity_memberships", "resolution_memberships"}),
    "alignment": frozenset({"entity_alignments", "resolved_entities"}),
    "aggregation": frozenset(
        {"relationship_assertions", "relationship_aggregates", "resolved_relationships"}
    ),
    "diagnostics": frozenset({"diagnostic_findings"}),
    "communities": frozenset({"communities", "community_memberships"}),
    "reports": frozenset({"community_reports"}),
}


def activation_allowed(completeness: str, validated: bool, authorized: bool) -> bool:
    """Partial, unvalidated and unauthorized graphs never replace the pointer."""
    return completeness == "complete" and validated is True and authorized is True


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise GraphValidationError(message)


def _refs(value: tuple[str, ...]) -> tuple[str, ...]:
    _require(
        type(value) is tuple
        and all(type(ref) is str and bool(ref.strip()) for ref in value)
        and len(set(value)) == len(value),
        "invalid dependency inventory",
    )
    return tuple(sorted(value))


def _manifest_payload(manifest: GraphSnapshotManifest) -> bytes:
    """Return canonical bytes, reusing those verified for this exact object."""
    entry = GRAPH_MANIFESTS.get(("bytes", id(manifest)))
    if entry is not None and entry[0] is manifest:
        return bytes(entry[1])
    return manifest_bytes(manifest)


def _digest(value: str) -> None:
    _require(
        type(value) is str and re.fullmatch(r"[0-9a-f]{64}", value) is not None,
        "invalid validation digest",
    )


class GraphSnapshotCatalog:
    """Explicit local storage plus host authorities; construction grants no access."""

    def __init__(
        self,
        database_path: Path,
        workspace: LocalProcessingWorkspace,
        gate: SnapshotAccessGate,
        inputs: InputSnapshotValidator,
        stages: StageArtifactValidator,
    ) -> None:
        if not isinstance(workspace, LocalProcessingWorkspace):
            raise TypeError("workspace must be LocalProcessingWorkspace")
        self._workspace = workspace
        self._gate = gate
        self._inputs = inputs
        self._stages = stages
        self._store = SqliteSnapshotStore(database_path)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            with self._store.transaction() as connection:
                yield connection
        except (sqlite3.Error, OSError) as error:
            raise GraphValidationError("graph catalog is unavailable or corrupt") from error

    def _revision(self, scope: GraphScope) -> int:
        if type(scope) is not GraphScope:
            raise TypeError("scope must be GraphScope")
        self._gate.require_scope(scope)
        revision = self._gate.revision
        _require(type(revision) is int and revision >= 0, "invalid authorization revision")
        return revision

    def _authorize(self, scope: GraphScope, evidence: tuple[str, ...], revision: int) -> None:
        self._gate.require_scope(scope)
        self._gate.require_access(evidence, scope)
        if self._gate.revision != revision:
            raise PermissionError("authorization policy changed")

    def _table(self, ref: DatasetSnapshotRef, role: str, scope: GraphScope) -> tuple[Any, ...]:
        from capicove.graph_build._adapters.parquet import ParquetTableCodec

        codec = codec_for_role(role)
        expected_dataset = "dataset:graph-" + fingerprint({"scope": asdict(scope), "role": role})
        _require(
            ref.dataset_id == expected_dataset
            and ref.schema == codec.schema
            and ref.data_format is DataFormat.PARQUET,
            "graph table scope, role or schema mismatch",
        )
        rows: tuple[Any, ...] | None = None

        def validate(path: Path) -> bool:
            nonlocal rows
            reader = ParquetTableCodec(publication_identity=ref.processing_fingerprint)
            rows = reader.read(path / "part-00000.parquet", role)
            return True

        try:
            verified = self._workspace.validate_snapshot(ref.snapshot_id, validator=validate)
            _require(verified == ref and rows is not None, "graph table is unavailable")
        except (ProcessingRecoveryError, ValueError, OSError) as error:
            raise GraphValidationError("graph table failed verified read") from error
        if rows is None:
            raise GraphValidationError("graph table is unavailable")
        _require(all(getattr(row, "scope", scope) == scope for row in rows), "row scope mismatch")
        return codec.snapshot(rows)

    def _validate(
        self, manifest: GraphSnapshotManifest, revision: int, visiting: frozenset[str] = frozenset()
    ) -> tuple[str, tuple[str, ...]]:
        scope = manifest.scope
        self._authorize(scope, (), revision)
        inventory = self._inputs.validate(manifest.input_snapshot_ref, scope)
        _require(
            type(inventory) is InputSnapshotInventory
            and inventory.snapshot_ref == manifest.input_snapshot_ref
            and inventory.scope == scope,
            "input inventory identity mismatch",
        )
        _digest(inventory.validation_sha256)
        _refs(inventory.document_refs)
        evidence = set(_refs(inventory.evidence_refs))
        self._authorize(scope, tuple(sorted(evidence)), revision)
        try:
            verified = self._workspace.validate_snapshot(
                manifest.input_snapshot_ref.snapshot_id, validator=lambda _: True
            )
        except ProcessingRecoveryError as error:
            raise GraphValidationError("input snapshot failed verified read") from error
        _require(verified == manifest.input_snapshot_ref, "input snapshot is unavailable")
        stages = []
        commitments: dict[str, TableRef] = {}
        manifest_tables = {table.role: table for table in manifest.tables}
        for stage in manifest.stages:
            _require(
                stage.coverage.denominator == _DENOMINATORS[stage.stage_id],
                "stage coverage denominator mismatch",
            )
            result = self._stages.validate(stage, scope, manifest.profile_fingerprint)
            _require(
                type(result) is ValidatedStageArtifacts
                and result.stage == stage
                and result.scope == scope
                and type(result.input_snapshot_ref) is DatasetSnapshotRef
                and result.input_snapshot_ref == manifest.input_snapshot_ref
                and result.profile_fingerprint == manifest.profile_fingerprint,
                "stage validation identity mismatch",
            )
            _digest(result.validation_sha256)
            _require(
                type(result.table_refs) is tuple
                and all(type(ref) is TableRef for ref in result.table_refs),
                "stage table commitments must be typed immutable refs",
            )
            _require(
                stage.status in {"succeeded", "partial"} or not result.table_refs,
                "unsuccessful stage cannot commit output tables",
            )
            for table in result.table_refs:
                _require(
                    table.role in _STAGE_ROLES[stage.stage_id]
                    and table.role not in commitments
                    and manifest_tables.get(table.role) == table,
                    "stage table commitment mismatch",
                )
                commitments[table.role] = table
            evidence.update(_refs(result.evidence_refs))
            stages.append(result.validation_sha256)
        self._authorize(scope, tuple(sorted(evidence)), revision)
        tables = {
            table.role: self._table(table.dataset_ref, table.role, scope)
            for table in manifest.tables
        }
        required = (
            set(manifest_tables) - {"operation_diagnostics"}
            if manifest.completeness == "complete"
            else {role for role, rows in tables.items() if rows and role != "operation_diagnostics"}
        )
        _require(required <= commitments.keys(), "stage table coverage is incomplete")
        self._authorize(scope, tuple(sorted(evidence)), revision)
        _require(
            {row.document_ref for row in tables["documents"]} == set(inventory.document_refs),
            "documents do not match declared input set",
        )
        _require(
            {row.evidence_ref for row in tables["documents"]} == set(inventory.evidence_refs),
            "document evidence does not match declared input set",
        )
        baseline_tables: dict[str, tuple[Any, ...]] = {}
        baseline_digest = None
        if manifest.baseline_snapshot_id is not None:
            baseline = manifest.baseline_snapshot_id
            _require(baseline not in visiting, "cyclic baseline dependency")
            previous, previous_validation, previous_evidence = self._load(
                baseline, scope, revision, visiting | {baseline}
            )
            _require(
                previous.completeness == "complete"
                and previous.schema_version == manifest.schema_version
                and previous.profile_fingerprint == manifest.profile_fingerprint,
                "baseline compatibility mismatch",
            )
            baseline_tables = {
                table.role: self._table(table.dataset_ref, table.role, scope)
                for table in previous.tables
                if table.role in {"entities", "entity_mentions", "documents"}
            }
            evidence.update(previous_evidence)
            baseline_digest = previous_validation
        _validate_tables(manifest, tables, baseline_tables)
        dependencies = tuple(sorted(evidence))
        self._authorize(scope, dependencies, revision)
        return fingerprint(
            {
                "validator_version": 1,
                "input": inventory.validation_sha256,
                "stages": stages,
                "baseline": baseline_digest,
                "manifest": hashlib.sha256(_manifest_payload(manifest)).hexdigest(),
                "evidence": dependencies,
            }
        ), dependencies

    @staticmethod
    def _row(connection: sqlite3.Connection, snapshot_id: str, scope: GraphScope) -> sqlite3.Row:
        row: sqlite3.Row | None = connection.execute(
            "SELECT * FROM graph_snapshots WHERE workspace_ref=? AND scope_ref=? AND snapshot_id=?",
            (scope.workspace_ref, scope.scope_ref, snapshot_id),
        ).fetchone()
        if row is None:
            raise GraphValidationError("graph snapshot unavailable")
        return row

    @staticmethod
    def _decode(
        connection: sqlite3.Connection, row: sqlite3.Row, scope: GraphScope
    ) -> tuple[GraphSnapshotManifest, tuple[str, ...]]:
        try:
            payload = row["manifest"]
            digest = hashlib.sha256(payload).hexdigest()
            _require(
                digest == row["manifest_sha256"] and row["snapshot_id"] == "graph:" + digest,
                "manifest identity is corrupt",
            )
            # The digest above was recomputed from the stored bytes, so a hit is
            # the canonical decode of exactly these bytes (RFC-0019).
            manifest = GRAPH_MANIFESTS.get(("digest", digest))
            if manifest is None:
                manifest = graph_from_json(
                    '{"schema_version":1,"kind":"GraphSnapshotManifest","value":'
                    + payload.decode()
                    + "}"
                )
                _require(
                    type(manifest) is GraphSnapshotManifest and manifest_bytes(manifest) == payload,
                    "manifest is corrupt",
                )
                GRAPH_MANIFESTS.put(("digest", digest), manifest, len(payload))
                GRAPH_MANIFESTS.put(
                    ("bytes", id(manifest)), (manifest, bytes(payload)), len(payload)
                )
            _require(
                type(manifest) is GraphSnapshotManifest and manifest.scope == scope,
                "manifest is corrupt",
            )
            if not isinstance(manifest, GraphSnapshotManifest):
                raise GraphValidationError("manifest is corrupt")
            evidence = _refs(tuple(json.loads(row["evidence_json"])))
            _require(
                canonical_bytes(evidence).decode() == row["evidence_json"],
                "dependency inventory is corrupt",
            )
            state = fingerprint(
                {
                    "manifest": digest,
                    "validation": row["validation_sha256"],
                    "evidence": evidence,
                    "revision": row["authorization_revision"],
                }
            )
            _require(state == row["state_sha256"], "validation state is corrupt")
            _digest(row["validation_sha256"])
            deps = connection.execute(
                "SELECT evidence_ref FROM graph_evidence WHERE workspace_ref=? AND scope_ref=? AND snapshot_id=? ORDER BY evidence_ref",
                (scope.workspace_ref, scope.scope_ref, row["snapshot_id"]),
            ).fetchall()
            _require(
                tuple(item[0] for item in deps) == evidence, "reverse dependencies are corrupt"
            )
            invalid = connection.execute(
                "SELECT 1 FROM graph_evidence e JOIN graph_ineligible i USING (workspace_ref, scope_ref, evidence_ref) WHERE e.workspace_ref=? AND e.scope_ref=? AND e.snapshot_id=? LIMIT 1",
                (scope.workspace_ref, scope.scope_ref, row["snapshot_id"]),
            ).fetchone()
            _require(invalid is None, "graph snapshot unavailable")
            return manifest, evidence
        except (TypeError, KeyError, AttributeError, json.JSONDecodeError) as error:
            raise GraphValidationError("stored graph metadata is corrupt") from error

    def _load(
        self,
        snapshot_id: str,
        scope: GraphScope,
        revision: int,
        visiting: frozenset[str] = frozenset(),
    ) -> tuple[GraphSnapshotManifest, str, tuple[str, ...]]:
        with self._transaction() as connection:
            row = self._row(connection, snapshot_id, scope)
            manifest, evidence = self._decode(connection, row, scope)
        self._authorize(scope, evidence, revision)
        _require(
            type(row["authorization_revision"]) is int
            and 0 <= row["authorization_revision"] <= revision,
            "stored authorization revision is invalid",
        )
        validation, current_evidence = self._validate(manifest, revision, visiting)
        _require(
            validation == row["validation_sha256"] and current_evidence == evidence,
            "stored validation is stale",
        )
        return manifest, validation, evidence

    def register(self, manifest: GraphSnapshotManifest) -> GraphSnapshotRef:
        """Validate and durably register one inactive immutable manifest."""
        if type(manifest) is not GraphSnapshotManifest:
            raise TypeError("manifest must be GraphSnapshotManifest")
        payload = manifest_bytes(manifest)
        # Reconstruct so object.__setattr__ cannot bypass canonical constructors.
        decoded = graph_from_json(
            '{"schema_version":1,"kind":"GraphSnapshotManifest","value":' + payload.decode() + "}"
        )
        _require(decoded == manifest, "manifest is noncanonical")
        revision = self._revision(manifest.scope)
        validation, evidence = self._validate(manifest, revision)
        digest = hashlib.sha256(payload).hexdigest()
        ref = GraphSnapshotRef("graph:" + digest, digest)
        scope = manifest.scope
        with self._gate.guard(revision), self._transaction() as connection:
            self._authorize(scope, evidence, revision)
            for dependency in evidence:
                _require(
                    connection.execute(
                        "SELECT 1 FROM graph_ineligible WHERE workspace_ref=? AND scope_ref=? AND evidence_ref=?",
                        (scope.workspace_ref, scope.scope_ref, dependency),
                    ).fetchone()
                    is None,
                    "graph snapshot unavailable",
                )
            old = connection.execute(
                "SELECT * FROM graph_snapshots WHERE workspace_ref=? AND scope_ref=? AND snapshot_id=?",
                (scope.workspace_ref, scope.scope_ref, ref.snapshot_id),
            ).fetchone()
            if old is not None:
                previous, deps = self._decode(connection, old, scope)
                _require(
                    previous == manifest
                    and deps == evidence
                    and old["validation_sha256"] == validation,
                    "immutable registration conflicts with stored state",
                )
                return ref
            state = fingerprint(
                {
                    "manifest": digest,
                    "validation": validation,
                    "evidence": evidence,
                    "revision": revision,
                }
            )
            connection.execute(
                "INSERT INTO graph_snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    scope.workspace_ref,
                    scope.scope_ref,
                    ref.snapshot_id,
                    payload,
                    digest,
                    validation,
                    canonical_bytes(evidence).decode(),
                    revision,
                    state,
                ),
            )
            connection.executemany(
                "INSERT INTO graph_evidence VALUES (?, ?, ?, ?)",
                [(scope.workspace_ref, scope.scope_ref, dep, ref.snapshot_id) for dep in evidence],
            )
        return ref

    def get(self, snapshot_id: str, scope: GraphScope) -> GraphSnapshotManifest:
        """Read after current policy, dependency, table and validation checks."""
        revision = self._revision(scope)
        manifest, _, evidence = self._load(snapshot_id, scope, revision)
        with self._gate.guard(revision), self._transaction() as connection:
            self._authorize(scope, evidence, revision)
            self._decode(connection, self._row(connection, snapshot_id, scope), scope)
        return manifest

    def require_eligible(self, snapshot_id: str, scope: GraphScope) -> GraphSnapshotManifest:
        """Reauthorize current eligibility without re-verifying artifacts.

        Checks scope, policy revision, stored manifest and state digests, the
        evidence inventory, reverse dependencies, ineligibility tombstones and
        current access, exactly as ``get`` does. Table, stage and input
        artifacts are not reread: content integrity remains the job of ``get``,
        ``active`` and verified reads (RFC-0019 Phase 2).
        """
        revision = self._revision(scope)
        with self._transaction() as connection:
            row = self._row(connection, snapshot_id, scope)
            manifest, evidence = self._decode(connection, row, scope)
        self._authorize(scope, evidence, revision)
        _require(
            type(row["authorization_revision"]) is int
            and 0 <= row["authorization_revision"] <= revision,
            "stored authorization revision is invalid",
        )
        with self._gate.guard(revision), self._transaction() as connection:
            self._authorize(scope, evidence, revision)
            self._decode(connection, self._row(connection, snapshot_id, scope), scope)
        return manifest

    @staticmethod
    def _pointer(connection: sqlite3.Connection, scope: GraphScope) -> tuple[str | None, int]:
        keys = (scope.workspace_ref, scope.scope_ref)
        connection.execute("INSERT OR IGNORE INTO graph_active VALUES (?, ?, NULL, 0)", keys)
        row = connection.execute(
            "SELECT snapshot_id, revision FROM graph_active WHERE workspace_ref=? AND scope_ref=?",
            keys,
        ).fetchone()
        history = connection.execute(
            "SELECT revision, snapshot_id FROM graph_history WHERE workspace_ref=? AND scope_ref=? ORDER BY revision",
            keys,
        ).fetchall()
        _require(
            type(row[1]) is int
            and row[1] >= 0
            and len(history) == row[1]
            and all(item[0] == index for index, item in enumerate(history, 1))
            and (row[0] is None if not history else row[0] == history[-1][1]),
            "activation pointer/history is corrupt",
        )
        return row[0], row[1]

    def activate(self, snapshot_id: str, scope: GraphScope, expected_revision: int) -> int:
        """Atomically compare-and-set a complete graph and append its history."""
        if type(expected_revision) is not int or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        revision = self._revision(scope)
        manifest, _, evidence = self._load(snapshot_id, scope, revision)
        _require(
            activation_allowed(manifest.completeness, True, True), "partial graph cannot activate"
        )
        with self._gate.guard(revision):
            self._authorize(scope, evidence, revision)
            # Repeat bounded local artifact reads at the guarded publication
            # boundary, before taking SQLite's write lock. This detects changes
            # since preflight, not filesystem changes after the verified read.
            for table in manifest.tables:
                self._table(table.dataset_ref, table.role, scope)
            with self._transaction() as connection:
                self._authorize(scope, evidence, revision)
                self._decode(connection, self._row(connection, snapshot_id, scope), scope)
                _, current = self._pointer(connection, scope)
                if current != expected_revision:
                    raise GraphActivationConflict("active graph revision changed")
                cursor = connection.execute(
                    "UPDATE graph_active SET snapshot_id=?, revision=revision+1 WHERE workspace_ref=? AND scope_ref=? AND revision=?",
                    (snapshot_id, scope.workspace_ref, scope.scope_ref, expected_revision),
                )
                if cursor.rowcount != 1:
                    raise GraphActivationConflict("active graph revision changed")
                connection.execute(
                    "INSERT INTO graph_history VALUES (?, ?, ?, ?, ?)",
                    (scope.workspace_ref, scope.scope_ref, current + 1, snapshot_id, "activated"),
                )
        return current + 1

    def active(self, scope: GraphScope) -> tuple[GraphSnapshotRef | None, int]:
        """Return only a currently readable active graph, or a vacant pointer."""
        revision = self._revision(scope)
        with self._transaction() as connection:
            snapshot_id, current = self._pointer(connection, scope)
        if snapshot_id is None:
            self._authorize(scope, (), revision)
            return None, current
        manifest, _, evidence = self._load(snapshot_id, scope, revision)
        with self._gate.guard(revision), self._transaction() as connection:
            self._authorize(scope, evidence, revision)
            self._decode(connection, self._row(connection, snapshot_id, scope), scope)
            if self._pointer(connection, scope) != (snapshot_id, current):
                raise GraphActivationConflict("active graph revision changed during read")
        return GraphSnapshotRef(
            snapshot_id, hashlib.sha256(_manifest_payload(manifest)).hexdigest()
        ), current

    def mark_ineligible(self, evidence_ref: str, scope: GraphScope, reason_code: str) -> None:
        """Revoke dependent graphs without requiring access to deleted evidence.

        The configured scope gate grants lifecycle authority. A durable source
        tombstone also prevents future registration from resurrecting that source.
        Repeated invalidation is idempotent and retains the original reason.
        """
        revision = self._revision(scope)
        _refs((evidence_ref,))
        _require(
            type(reason_code) is str
            and re.fullmatch(r"[A-Za-z][A-Za-z0-9_.:-]{0,127}", reason_code) is not None,
            "reason_code must be a bounded machine-readable identifier",
        )
        keys = (scope.workspace_ref, scope.scope_ref)
        with self._gate.guard(revision), self._transaction() as connection:
            self._gate.require_scope(scope)
            if self._gate.revision != revision:
                raise PermissionError("authorization policy changed")
            # Check normalized dependencies against their committed inventory
            # before relying on them for fanout; damaged indexes fail closed.
            rows = connection.execute(
                "SELECT * FROM graph_snapshots WHERE workspace_ref=? AND scope_ref=?", keys
            ).fetchall()
            affected: set[str] = set()
            for row in rows:
                dependencies = tuple(json.loads(row["evidence_json"]))
                actual = connection.execute(
                    "SELECT evidence_ref FROM graph_evidence WHERE workspace_ref=? AND scope_ref=? AND snapshot_id=? ORDER BY evidence_ref",
                    (*keys, row["snapshot_id"]),
                ).fetchall()
                _require(
                    tuple(item[0] for item in actual) == dependencies,
                    "reverse dependencies are corrupt",
                )
                if evidence_ref in dependencies:
                    affected.add(row["snapshot_id"])
            snapshot_id, current = self._pointer(connection, scope)
            connection.execute(
                "INSERT OR IGNORE INTO graph_ineligible VALUES (?, ?, ?, ?)",
                (*keys, evidence_ref, reason_code),
            )
            if snapshot_id in affected:
                connection.execute(
                    "UPDATE graph_active SET snapshot_id=NULL, revision=revision+1 WHERE workspace_ref=? AND scope_ref=? AND revision=?",
                    (*keys, current),
                )
                connection.execute(
                    "INSERT INTO graph_history VALUES (?, ?, ?, NULL, ?)",
                    (*keys, current + 1, reason_code),
                )


def _validate_tables(
    manifest: GraphSnapshotManifest,
    tables: dict[str, tuple[Any, ...]],
    baseline: dict[str, tuple[Any, ...]],
) -> None:
    """Validate graph relations after verified typed table reads."""
    from capicove.graph_build.aggregation import aggregate_assertions
    from capicove.graph_build.ontology import relation_allowed

    documents = {row.document_ref: row for row in tables["documents"]}
    units = {row.text_unit_id: row for row in tables["text_units"]}
    mentions = {row.mention_id: row for row in tables["entity_mentions"]}
    relations = {row.mention_id: row for row in tables["relationship_mentions"]}
    entities = {row.entity_id: row for row in tables["entities"]}
    assertions = {row.assertion_id: row for row in tables["relationship_assertions"]}
    evidence = {row.evidence_ref for row in documents.values()}
    successful = {stage.stage_id for stage in manifest.stages if stage.status == "succeeded"}
    _require(not (mentions.keys() & relations.keys()), "mention identity collision")
    for document in documents.values():
        _require(
            all(
                span.evidence_ref == document.evidence_ref
                and span.representation_ref == document.representation_ref
                for span in document.provenance
            ),
            "document provenance mismatch",
        )
        if manifest.completeness == "complete":
            _require(
                document.availability == "available"
                and document.coverage.accepted == document.coverage.declared,
                "complete graph contains unavailable document",
            )
    unit_ranges: dict[str, list[tuple[int, int, int]]] = {}
    for unit in units.values():
        document = documents.get(unit.document_ref)
        _require(
            document is not None
            and document.availability == "available"
            and unit.source_sha256 == document.source_sha256
            and unit.representation_ref == document.representation_ref,
            "text unit document mismatch",
        )
        if document is None:
            raise GraphValidationError("text unit document mismatch")
        spans = tuple(span for segment in unit.segments for span in segment.source_spans)
        _require(
            all(
                span.evidence_ref == document.evidence_ref
                and span.representation_ref == document.representation_ref
                for span in spans
            ),
            "text unit evidence mismatch",
        )
        _require(
            all(any(_contains(parent, span) for parent in document.provenance) for span in spans),
            "text unit lies outside document provenance",
        )
        bounds = unit.core_locator.character_range
        unit_ranges.setdefault(unit.document_ref, []).append(
            (bounds.start_char, bounds.end_char, unit.ordinal)
        )
    for ranges in unit_ranges.values():
        ordered = sorted(ranges)
        _require(
            len({ordinal for _, _, ordinal in ordered}) == len(ordered)
            and all(left[1] <= right[0] for left, right in pairwise(ordered)),
            "overlapping cores or duplicate ordinals",
        )
    for mention in (*mentions.values(), *relations.values()):
        unit = units.get(mention.text_unit_id)
        if unit is None:
            raise GraphValidationError("mention text unit is missing")
        spans = tuple(span for segment in unit.segments for span in segment.source_spans)
        _require(
            all(any(_contains(parent, span) for parent in spans) for span in mention.source_spans),
            "mention evidence is outside source unit",
        )
        if isinstance(mention, EntityMention):
            core_spans = tuple(
                span
                for segment in unit.segments
                if segment.role == "core"
                for span in segment.source_spans
            )
            has_core = any(
                _contains(parent, span) for span in mention.source_spans for parent in core_spans
            )
            _require(
                (mention.anchor_role == "core") == has_core,
                "entity anchor role does not match core evidence ownership",
            )
        if mention.attribution_ref is not None:
            speaker = mentions.get(mention.attribution_ref)
            _require(
                speaker is not None
                and speaker.entity_type == "PERSON"
                and speaker.text_unit_id == mention.text_unit_id,
                "mention attribution is invalid",
            )
    for relation in relations.values():
        _require(
            all(
                ref in mentions and mentions[ref].text_unit_id == relation.text_unit_id
                for ref in (relation.source_mention_id, relation.target_mention_id)
            ),
            "relationship endpoint is missing or foreign",
        )
        _require(
            relation_allowed(
                mentions[relation.source_mention_id].entity_type,
                relation.predicate,
                mentions[relation.target_mention_id].entity_type,
            ),
            "relationship endpoint types are incompatible",
        )
        unit = units[relation.text_unit_id]
        core = [
            span
            for segment in unit.segments
            if segment.role == "core"
            for span in segment.source_spans
        ]
        _require(
            any(any(_contains(parent, span) for parent in core) for span in relation.source_spans),
            "relationship lacks core evidence",
        )
    membership_pairs = {(row.entity_id, row.mention_id) for row in tables["entity_memberships"]}
    expected_pairs = {
        (row.entity_id, mention) for row in entities.values() for mention in row.mention_ids
    }
    _require(
        membership_pairs == expected_pairs
        and {mention for _, mention in membership_pairs} <= mentions.keys()
        and len({mention for _, mention in membership_pairs}) == len(membership_pairs),
        "entity membership is incomplete or inconsistent",
    )
    if "resolution" in successful:
        _require(
            {mention for _, mention in membership_pairs} == mentions.keys(),
            "successful resolution must cover all mentions",
        )
    membership = {mention: entity for entity, mention in membership_pairs}
    for entity in entities.values():
        _require(
            all(
                ref in mentions and mentions[ref].entity_type == entity.entity_type
                for ref in entity.mention_ids
            ),
            "entity mention membership is invalid",
        )
    claimed_relations: set[str] = set()
    for assertion in assertions.values():
        _require(
            assertion.source_entity_id in entities and assertion.target_entity_id in entities,
            "assertion endpoint is missing",
        )
        _require(
            set(assertion.evidence_refs) <= evidence | relations.keys(),
            "assertion evidence is foreign",
        )
        _require(
            bool(set(assertion.evidence_refs) & evidence),
            "assertion requires original source evidence",
        )
        cited = [relations[ref] for ref in assertion.evidence_refs if ref in relations]
        _require(
            bool(cited),
            "assertion requires relationship mention lineage",
        )
        cited_ids = {relation.mention_id for relation in cited}
        _require(
            not (claimed_relations & cited_ids),
            "relationship mention has multiple assertion owners",
        )
        claimed_relations.update(cited_ids)
        for relation in cited:
            _require(
                (
                    assertion.source_entity_id,
                    assertion.target_entity_id,
                    assertion.predicate,
                    assertion.epistemic_status,
                    assertion.polarity,
                    assertion.attribution_ref,
                    tuple(sorted(assertion.qualifiers)),
                )
                == (
                    membership.get(relation.source_mention_id),
                    membership.get(relation.target_mention_id),
                    relation.predicate,
                    relation.epistemic_status,
                    relation.polarity,
                    membership.get(relation.attribution_ref) if relation.attribution_ref else None,
                    tuple(sorted(relation.qualifiers)),
                ),
                "assertion does not match source relationship",
            )
        _require(
            set(assertion.evidence_refs)
            == {
                ref
                for relation in cited
                for ref in (
                    relation.mention_id,
                    *(span.evidence_ref for span in relation.source_spans),
                )
            },
            "assertion evidence lineage mismatch",
        )
        if assertion.attribution_ref is not None:
            _require(
                assertion.attribution_ref in entities
                and entities[assertion.attribution_ref].entity_type == "PERSON",
                "assertion attribution is invalid",
            )
    try:
        aggregates = aggregate_assertions(tuple(assertions.values()))
    except ValueError as error:
        raise GraphValidationError("assertion counter links are invalid") from error
    _require(
        all(row in aggregates for row in tables["relationship_aggregates"]),
        "assertion aggregation mismatch",
    )
    if "aggregation" in successful:
        _require(
            claimed_relations == relations.keys(),
            "successful aggregation must cover all retained relationship mentions",
        )
        _require(
            tables["relationship_aggregates"] == aggregates, "incomplete assertion aggregation"
        )
    _require(
        all(row in tables["entities"] for row in tables["resolved_entities"])
        and all(
            row in tables["relationship_assertions"] for row in tables["resolved_relationships"]
        ),
        "resolved view must preserve current graph only",
    )
    resolved_ids = {row.entity_id for row in tables["resolved_entities"]}
    for assertion in tables["resolved_relationships"]:
        _require(
            assertion.source_entity_id in resolved_ids
            and assertion.target_entity_id in resolved_ids
            and (assertion.attribution_ref is None or assertion.attribution_ref in resolved_ids),
            "resolved relationship endpoint or attribution is missing from resolved view",
        )
    _require(
        {
            (row.analysis_entity_id, row.current_entity_id)
            for row in tables["resolution_memberships"]
        }
        == {(row.entity_id, row.entity_id) for row in tables["resolved_entities"]},
        "resolution membership must retain current identity",
    )
    if "alignment" in successful:
        _require(
            tables["resolved_entities"] == tables["entities"]
            and tables["resolved_relationships"] == tables["relationship_assertions"],
            "successful alignment must preserve complete current view",
        )
    previous = {row.entity_id: row for row in baseline.get("entities", ())}
    previous_mentions = {row.mention_id for row in baseline.get("entity_mentions", ())}
    baseline_evidence = {row.evidence_ref for row in baseline.get("documents", ())}
    matched_current: set[str] = set()
    matched_previous: set[str] = set()
    for alignment in tables["entity_alignments"]:
        _require(
            alignment.current_entity_id in entities and alignment.previous_entity_id in previous,
            "alignment endpoint is missing",
        )
        _require(
            set(alignment.evidence_refs)
            <= mentions.keys()
            | previous_mentions
            | evidence
            | baseline_evidence
            | entities.keys()
            | previous.keys(),
            "alignment evidence is foreign",
        )
        if alignment.decision == "matched":
            _require(
                alignment.current_entity_id not in matched_current
                and alignment.previous_entity_id not in matched_previous,
                "alignment correspondence is not one-to-one",
            )
            matched_current.add(alignment.current_entity_id)
            matched_previous.add(alignment.previous_entity_id)
            _require(
                entities[alignment.current_entity_id].entity_type
                == previous[alignment.previous_entity_id].entity_type,
                "alignment entity types differ",
            )
    for ref in entities.keys() & previous.keys():
        _require(entities[ref] == previous[ref], "shared entity ID has changed content")
    findings = {row.finding_id: row for row in tables["diagnostic_findings"]}
    for finding in findings.values():
        _require(
            finding.subject_ref in entities
            and finding.target_ref in entities
            and set(finding.support_refs + finding.counter_refs) <= assertions.keys(),
            "finding endpoint or source is missing",
        )
        _require(
            all(
                assertions[ref].derivation_ref is None
                for ref in finding.support_refs + finding.counter_refs
            ),
            "finding cites generated assertion",
        )
    _validate_communities(
        tables, {row.entity_id: row for row in tables["resolved_entities"]}, assertions, findings
    )
    if "communities" in successful:
        root_members = {
            ref for row in tables["communities"] if row.parent_id is None for ref in row.member_ids
        }
        _require(
            root_members == resolved_ids, "successful communities must cover every resolved entity"
        )
    for diagnostic in tables["operation_diagnostics"]:
        _require(
            diagnostic.unit_ref is None or diagnostic.unit_ref in units,
            "operation diagnostic unit is missing",
        )
    exact_populations = {
        "representations": len(documents),
        "extraction": len(units),
        "resolution": len(mentions),
        "aggregation": len(assertions),
        "communities": len(entities),
    }
    for stage in manifest.stages:
        if stage.stage_id in exact_populations:
            _require(
                stage.coverage.declared == exact_populations[stage.stage_id],
                "stage coverage/table mismatch",
            )
        if stage.stage_id == "reports":
            _require(
                stage.coverage.accepted == len(tables["community_reports"]),
                "report coverage/table mismatch",
            )


def _contains(parent: SourceSpan, child: SourceSpan) -> bool:
    return (
        parent.evidence_ref == child.evidence_ref
        and parent.representation_ref == child.representation_ref
        and parent.block_id == child.block_id
        and parent.page_index == child.page_index
        and parent.start_char <= child.start_char < child.end_char <= parent.end_char
    )


def _validate_communities(
    tables: dict[str, tuple[Any, ...]],
    entities: dict[str, Any],
    assertions: dict[str, Any],
    findings: dict[str, Any],
) -> None:
    communities = {row.community_id: row for row in tables["communities"]}
    children: dict[str | None, list[Any]] = {}
    for community in communities.values():
        _require(set(community.member_ids) <= entities.keys(), "community member is missing")
        children.setdefault(community.parent_id, []).append(community)
        if community.parent_id is not None:
            parent = communities.get(community.parent_id)
            _require(
                parent is not None
                and community.depth == parent.depth + 1
                and set(community.member_ids) < set(parent.member_ids),
                "invalid community hierarchy",
            )
    for parent_id, siblings in children.items():
        members: set[str] = set()
        for child in siblings:
            _require(not (members & set(child.member_ids)), "communities overlap")
            members.update(child.member_ids)
        if parent_id is not None:
            _require(
                members == set(communities[parent_id].member_ids),
                "community partition is incomplete",
            )
    _require(
        {(row.community_id, row.entity_id) for row in tables["community_memberships"]}
        == {(row.community_id, ref) for row in communities.values() for ref in row.member_ids},
        "community membership mismatch",
    )
    reports = {row.report_id: row for row in tables["community_reports"]}
    statements = {
        (report.report_id, statement.statement_id): statement
        for report in reports.values()
        for field in fields(report.sections)
        for statement in getattr(report.sections, field.name)
    }
    community_reports: set[str | None] = set()
    for report in reports.values():
        _require(report.community_id not in community_reports, "duplicate community report")
        community_reports.add(report.community_id)
        _require(
            report.community_id is None or report.community_id in communities,
            "report community is missing",
        )
        for field in fields(report.sections):
            for statement in getattr(report.sections, field.name):
                _require(
                    not (
                        (
                            field.name == "suggested_next_steps"
                            and statement.kind != "recommendation"
                        )
                        or (
                            statement.kind == "recommendation"
                            and field.name != "suggested_next_steps"
                        )
                        or (
                            field.name == "possible_causes"
                            and statement.kind not in {"hypothesis", "limitation"}
                        )
                        or (field.name == "limitations" and statement.kind != "limitation")
                        or (
                            field.name in {"existing_interventions", "observed_outcomes"}
                            and statement.kind != "fact"
                        )
                    ),
                    "report section kind mismatch",
                )
                _require(
                    set(statement.assertion_refs) <= assertions.keys()
                    and set(statement.finding_refs) <= findings.keys(),
                    "report source citation is missing",
                )
                for child in statement.child_statement_refs:
                    target = reports.get(child.report_id)
                    _require(
                        (child.report_id, child.statement_id) in statements and target is not None,
                        "child report citation is missing",
                    )
                    if target is None:
                        raise GraphValidationError("child report citation is missing")
                    _require(
                        target.community_id is not None
                        and communities[target.community_id].parent_id == report.community_id,
                        "report must cite only direct child communities",
                    )
                if statement.kind == "fact":
                    _require(
                        not statement.finding_refs
                        and all(
                            assertions[ref].epistemic_status == "explicit"
                            and assertions[ref].derivation_ref is None
                            for ref in statement.assertion_refs
                        )
                        and all(
                            statements[(ref.report_id, ref.statement_id)].kind == "fact"
                            for ref in statement.child_statement_refs
                        ),
                        "fact cites inferred or generated content",
                    )
