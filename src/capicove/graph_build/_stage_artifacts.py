"""Private closed stage artifacts verified recursively by the publication host."""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from capicove.contracts.data import DatasetSnapshotRef
from capicove.contracts.graph_build import (
    GraphScope,
    GraphSnapshotRef,
    StageCoverage,
    StageResult,
    TableRef,
)
from capicove.graph_build._adapters.sqlite import SqliteSnapshotStore
from capicove.graph_build._decode_reuse import STAGE_ARTIFACTS
from capicove.graph_build.codec import (
    _dataset_payload,
    _dataset_record,
    _manifest_canonical_bytes,
    canonical_bytes,
    graph_from_json,
)
from capicove.graph_build.datasets import codec_for_role
from capicove.graph_build.snapshots import SnapshotAccessGate, ValidatedStageArtifacts
from capicove.processing import LocalProcessingWorkspace, ProcessingRecoveryError


STAGE_ROLES = {
    "representations": ("documents",),
    "text_units": ("text_units",),
    "extraction": ("entity_mentions", "relationship_mentions"),
    "identity_embeddings": (),
    "resolution": ("entities", "entity_memberships", "resolution_memberships"),
    "alignment": ("entity_alignments", "resolved_entities"),
    "aggregation": ("relationship_assertions", "relationship_aggregates", "resolved_relationships"),
    "diagnostics": ("diagnostic_findings",),
    "communities": ("communities", "community_memberships"),
    "reports": ("community_reports",),
}
DENOMINATORS = dict(
    zip(
        STAGE_ROLES,
        (
            "documents",
            "codepoints",
            "text_units",
            "descriptors",
            "mentions",
            "entities",
            "assertions",
            "bundles",
            "entities",
            "reports",
        ),
        strict=True,
    )
)
_SCHEMA = (
    "CREATE TABLE build_stage_artifacts (artifact_ref TEXT PRIMARY KEY, payload BLOB NOT NULL)",
)
_COMMON_FIELDS = frozenset({"version", "scope", "kind", "dependencies", "evidence_refs"})
_FIELDS = {
    "output": frozenset(
        {
            "stage_id",
            "status",
            "coverage",
            "issues",
            "rows",
            "metadata",
            "semantics",
            "input_refs",
        }
    ),
    "seal": frozenset(
        {
            "profile_fingerprint",
            "output_ref",
            "tables",
            "input_snapshot_ref",
            "runtime_payload",
            "runtime_digest",
        }
    ),
    "operation": frozenset(
        {"stage_id", "status", "graph_ref", "binding_ref", "issues", "coverage"}
    ),
}


def _artifact_ref(payload: bytes) -> str:
    return "build-artifact:" + hashlib.sha256(payload).hexdigest()


class _Storage(SqliteSnapshotStore):
    @staticmethod
    def _schema(connection: sqlite3.Connection, initialize: bool) -> None:
        objects = connection.execute("SELECT type, sql FROM sqlite_master").fetchall()
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        if initialize and not objects and version == 0:
            for statement in _SCHEMA:
                connection.execute(statement)
            connection.execute("PRAGMA user_version=1")
            return
        if version != 1 or sorted((row[0], row[1] or "") for row in objects) != sorted(
            [
                ("table", _SCHEMA[0]),
                ("index", ""),
            ]
        ):
            raise ValueError("unsupported_build_stage_schema")


@dataclass(frozen=True, slots=True)
class StageOutput:
    """Verified closed output plus exact dependencies; rows use Task19 codecs."""

    stage_id: str
    status: str
    coverage: StageCoverage
    issues: tuple[str, ...]
    rows: tuple[tuple[str, tuple[Any, ...]], ...]
    metadata: bytes = b"{}"

    def __post_init__(self) -> None:
        if (
            self.stage_id not in STAGE_ROLES
            or self.coverage.denominator != DENOMINATORS[self.stage_id]
        ):
            raise ValueError("invalid_stage_output")
        StageResult(
            self.stage_id, self.status, ("artifact:validation",), self.issues, self.coverage
        )
        if tuple(role for role, _ in self.rows) != STAGE_ROLES[self.stage_id]:
            raise ValueError("stage_output_roles_mismatch")
        for role, values in self.rows:
            codec_for_role(role).snapshot(values)
        if canonical_bytes(json.loads(self.metadata)) != self.metadata:
            raise ValueError("noncanonical_stage_metadata")


class BuildStageStore:
    """Retain scope-protected immutable bytes, including successful empty stages.

    Table commitments are sealed after publication. Seal artifacts reference the
    previously committed stage output and never replace it. Reads recursively
    verify content identities and reauthorize every persisted evidence reference.
    """

    def __init__(
        self, path: Path, gate: SnapshotAccessGate, workspace: LocalProcessingWorkspace
    ) -> None:
        self.storage = _Storage(path)
        self.gate = gate
        self.workspace = workspace

    def _put(self, scope: GraphScope, value: dict[str, Any]) -> str:
        revision = self.gate.revision
        self.gate.require_scope(scope)
        self.gate.require_access(tuple(value["evidence_refs"]), scope)
        payload = canonical_bytes({"version": 1, "scope": asdict(scope), **value})
        ref = "build-artifact:" + hashlib.sha256(payload).hexdigest()
        with self.gate.guard(revision), self.storage.transaction() as connection:
            if self.gate.revision != revision:
                raise PermissionError("stage_policy_revision_changed")
            existing = connection.execute(
                "SELECT payload FROM build_stage_artifacts WHERE artifact_ref = ?", (ref,)
            ).fetchone()
            if existing is not None and bytes(existing[0]) != payload:
                raise ValueError("build_artifact_conflict")
            connection.execute(
                "INSERT OR IGNORE INTO build_stage_artifacts VALUES (?, ?)", (ref, payload)
            )
            if self.gate.revision != revision:
                raise PermissionError("stage_policy_revision_changed")
        return ref

    def read(self, ref: str, scope: GraphScope, *, depth: int = 0) -> dict[str, Any]:
        """Read exact bytes with bounded transitive integrity and access checks.

        The dependency closure is fetched in one short transaction and then
        verified and reauthorized node by node without holding the store lock.
        """
        self.gate.require_scope(scope)
        if depth > 16:
            raise ValueError("cyclic_stage_artifacts")
        return self._verified(ref, scope, self._fetch_closure(ref, depth), depth)

    def _fetch_closure(self, ref: str, depth: int) -> dict[str, bytes | None]:
        payloads: dict[str, bytes | None] = {}
        pending = [(ref, depth)]
        with self.storage.transaction() as connection:
            while pending:
                current, level = pending.pop()
                if current in payloads or level > 16:
                    continue
                row = connection.execute(
                    "SELECT payload FROM build_stage_artifacts WHERE artifact_ref = ?", (current,)
                ).fetchone()
                payload = None if row is None else bytes(row[0])
                payloads[current] = payload
                if payload is None or _artifact_ref(payload) != current:
                    continue
                try:
                    dependencies = json.loads(payload).get("dependencies")
                except (AttributeError, TypeError, ValueError):
                    continue
                if type(dependencies) is list:
                    pending.extend((item, level + 1) for item in dependencies if type(item) is str)
        return payloads

    def _payload(self, ref: str, payloads: dict[str, bytes | None]) -> bytes | None:
        if ref not in payloads:
            with self.storage.transaction() as connection:
                row = connection.execute(
                    "SELECT payload FROM build_stage_artifacts WHERE artifact_ref = ?", (ref,)
                ).fetchone()
            payloads[ref] = None if row is None else bytes(row[0])
        return payloads[ref]

    def _verified(
        self, ref: str, scope: GraphScope, payloads: dict[str, bytes | None], depth: int
    ) -> dict[str, Any]:
        self.gate.require_scope(scope)
        if depth > 16:
            raise ValueError("cyclic_stage_artifacts")
        payload = self._payload(ref, payloads)
        if payload is None:
            raise ValueError("stage_artifact_unavailable")
        if _artifact_ref(payload) != ref:
            raise ValueError("stage_artifact_integrity_failure")
        value = json.loads(payload)
        # Content-addressed bytes make these closed-shape checks pure; scope,
        # authorization and dependency checks below always run.
        shape_verified = STAGE_ARTIFACTS.get(ref) is not None
        if not shape_verified and (
            type(value["version"]) is not int
            or value["version"] != 1
            or canonical_bytes(value) != payload
        ):
            raise ValueError("stage_artifact_identity_mismatch")
        if value["scope"] != asdict(scope):
            raise ValueError("stage_artifact_identity_mismatch")
        kind = value.get("kind")
        if not shape_verified:
            if kind not in _FIELDS or set(value) != _COMMON_FIELDS | _FIELDS[kind]:
                raise ValueError("invalid_closed_stage_artifact")
            for name in ("dependencies", "evidence_refs"):
                if (
                    type(value[name]) is not list
                    or any(type(item) is not str or not item.strip() for item in value[name])
                    or len(set(value[name])) != len(value[name])
                ):
                    raise ValueError("invalid_stage_artifact_references")
            STAGE_ARTIFACTS.put(ref, True, len(payload))
        self.gate.require_access(tuple(value["evidence_refs"]), scope)
        dependencies = tuple(
            self._verified(dependency, scope, payloads, depth + 1)
            for dependency in value["dependencies"]
        )
        if kind == "seal" and (
            value["dependencies"] != [value["output_ref"]] or dependencies[0]["kind"] != "output"
        ):
            raise ValueError("invalid_sealed_output_dependency")
        if kind == "operation":
            stage = value["stage_id"]
            if (
                stage not in {"publication", "retrieval"}
                or type(graph_from_json(value["graph_ref"])) is not GraphSnapshotRef
            ):
                raise ValueError("invalid_operational_stage")
            coverage = StageCoverage(**value["coverage"])
            if (
                value["status"] not in {"succeeded", "failed", "partial", "skipped"}
                or type(value["issues"]) is not list
                or any(
                    type(code) is not str or re.fullmatch(r"[a-z][a-z0-9_]{0,127}", code) is None
                    for code in value["issues"]
                )
                or len(set(value["issues"])) != len(value["issues"])
            ):
                raise ValueError("invalid_operational_stage_status")
            if stage == "publication":
                if (
                    value["status"] != "succeeded"
                    or value["binding_ref"] is not None
                    or value["issues"]
                    or coverage != StageCoverage("tables", 17, 17, 17, 0, 0, 0)
                    or len(dependencies) != 1
                    or any(item["kind"] != "seal" for item in dependencies)
                ):
                    raise ValueError("invalid_publication_artifact")
                stages = {
                    self._verified(item["output_ref"], scope, payloads, depth + 1)["stage_id"]
                    for item in dependencies
                }
                if stages != {"reports"}:
                    raise ValueError("incomplete_publication_dependencies")
            elif (
                len(dependencies) != 1
                or dependencies[0]["kind"] != "operation"
                or dependencies[0]["stage_id"] != "publication"
                or dependencies[0]["graph_ref"] != value["graph_ref"]
                or coverage.denominator != "generations"
            ):
                raise ValueError("invalid_retrieval_artifact")
            elif value["status"] == "succeeded" and (
                type(value["binding_ref"]) is not str
                or not value["binding_ref"].strip()
                or coverage != StageCoverage("generations", 1, 1, 1, 0, 0, 0)
            ):
                raise ValueError("invalid_retrieval_binding")
            elif value["status"] != "succeeded" and value["binding_ref"] is not None:
                raise ValueError("unexpected_retrieval_binding")
        return cast(dict[str, Any], value)

    def commit(
        self,
        scope: GraphScope,
        output: StageOutput,
        *,
        semantics: bytes,
        dependencies: tuple[str, ...],
        input_refs: tuple[str, ...],
        evidence_refs: tuple[str, ...],
    ) -> str:
        """Commit complete rows and actual direct/transitive effect dependencies."""
        for dependency in dependencies:
            self.read(dependency, scope)
        return self._put(
            scope,
            {
                "kind": "output",
                "stage_id": output.stage_id,
                "status": output.status,
                "coverage": asdict(output.coverage),
                "issues": output.issues,
                "rows": {
                    role: [codec_for_role(role).encode(row) for row in rows]
                    for role, rows in output.rows
                },
                "metadata": output.metadata.decode(),
                "semantics": semantics.decode(),
                "dependencies": dependencies,
                "input_refs": input_refs,
                "evidence_refs": tuple(sorted(set(evidence_refs))),
            },
        )

    def output(self, ref: str, scope: GraphScope) -> StageOutput:
        """Reconstruct exact typed output from committed bytes."""
        value = self.read(ref, scope)
        if value["kind"] != "output" or set(value) != {
            "version",
            "scope",
            "kind",
            "stage_id",
            "status",
            "coverage",
            "issues",
            "rows",
            "metadata",
            "semantics",
            "dependencies",
            "input_refs",
            "evidence_refs",
        }:
            raise ValueError("invalid_closed_stage_artifact")
        stage = value["stage_id"]
        if stage not in STAGE_ROLES or set(value["rows"]) != set(STAGE_ROLES[stage]):
            raise ValueError("invalid_stage_artifact_roles")
        return StageOutput(
            stage,
            value["status"],
            StageCoverage(**value["coverage"]),
            tuple(value["issues"]),
            tuple(
                (role, tuple(codec_for_role(role).decode(row) for row in value["rows"][role]))
                for role in STAGE_ROLES[stage]
            ),
            value["metadata"].encode(),
        )

    def find_output(
        self, scope: GraphScope, stage: str, semantics: bytes, dependencies: tuple[str, ...]
    ) -> str | None:
        """Find only exact committed successful effects; reread current dependencies."""
        self.gate.require_scope(scope)
        with self.storage.transaction() as connection:
            rows = connection.execute(
                "SELECT artifact_ref, payload FROM build_stage_artifacts ORDER BY artifact_ref"
            ).fetchall()
        for row in rows:
            value = json.loads(bytes(row[1]))
            if (
                value.get("scope") == asdict(scope)
                and value.get("kind") == "output"
                and value.get("stage_id") == stage
                and value.get("semantics") == semantics.decode()
                and value.get("dependencies") == list(dependencies)
                and value.get("status") == "succeeded"
            ):
                self.output(row[0], scope)
                return str(row[0])
        return None

    def seal(
        self,
        ref: str,
        scope: GraphScope,
        profile_fingerprint: str,
        tables: tuple[TableRef, ...],
        *,
        input_snapshot_ref: DatasetSnapshotRef,
        runtime_payload: bytes,
    ) -> StageResult:
        """Bind published table identities to one immutable output and full profile."""
        output = self.output(ref, scope)
        input_pin = _manifest_canonical_bytes(_dataset_payload(input_snapshot_ref)).decode()
        self._qualification(ref, scope, profile_fingerprint, input_pin, runtime_payload)
        if output.status not in {"succeeded", "partial"}:
            tables = ()
        sealed = self._put(
            scope,
            {
                "kind": "seal",
                "profile_fingerprint": profile_fingerprint,
                "input_snapshot_ref": input_pin,
                "runtime_payload": runtime_payload.decode(),
                "runtime_digest": hashlib.sha256(runtime_payload).hexdigest(),
                "output_ref": ref,
                "tables": tuple(
                    (
                        table.role,
                        _manifest_canonical_bytes(_dataset_payload(table.dataset_ref)).decode(),
                    )
                    for table in tables
                ),
                "dependencies": (ref,),
                "evidence_refs": self.read(ref, scope)["evidence_refs"],
            },
        )
        return StageResult(
            output.stage_id, output.status, (sealed,), output.issues, output.coverage
        )

    def validate(
        self, stage: StageResult, scope: GraphScope, profile_fingerprint: str
    ) -> ValidatedStageArtifacts:
        """Implement Task20's host validator by rereading all committed artifacts."""
        if len(stage.artifact_refs) != 1:
            raise ValueError("stage_requires_one_closed_seal")
        ref = stage.artifact_refs[0]
        seal = self.read(ref, scope)
        if (
            set(seal)
            != {
                "version",
                "scope",
                "kind",
                "profile_fingerprint",
                "input_snapshot_ref",
                "runtime_payload",
                "runtime_digest",
                "output_ref",
                "tables",
                "dependencies",
                "evidence_refs",
            }
            or seal["kind"] != "seal"
        ):
            raise ValueError("invalid_stage_seal")
        output = self.output(seal["output_ref"], scope)
        runtime_payload = seal["runtime_payload"].encode()
        if hashlib.sha256(runtime_payload).hexdigest() != seal["runtime_digest"]:
            raise ValueError("stage_runtime_integrity_failure")
        self._qualification(
            seal["output_ref"],
            scope,
            profile_fingerprint,
            seal["input_snapshot_ref"],
            runtime_payload,
        )
        if (
            seal["profile_fingerprint"] != profile_fingerprint
            or StageResult(output.stage_id, output.status, (ref,), output.issues, output.coverage)
            != stage
        ):
            raise ValueError("stage_seal_identity_mismatch")
        tables = tuple(
            TableRef(role, _dataset_record(json.loads(payload))) for role, payload in seal["tables"]
        )
        if any(type(table) is not TableRef for table in tables):
            raise ValueError("invalid_stage_table_commitment")
        expected = dict(output.rows)
        if output.status in {"succeeded", "partial"}:
            if {table.role for table in tables} != set(expected) or len(tables) != len(expected):
                raise ValueError("stage_table_inventory_mismatch")
        elif tables:
            raise ValueError("unsuccessful_stage_tables")
        for table in tables:
            from capicove.graph_build._adapters.parquet import ParquetTableCodec

            def verify(path: Path, table: TableRef = table) -> bool:
                rows = ParquetTableCodec(
                    publication_identity=table.dataset_ref.processing_fingerprint
                ).read(path / "part-00000.parquet", table.role)
                if rows != codec_for_role(table.role).snapshot(expected[table.role]):
                    raise ValueError("stage_table_rows_mismatch")
                return True

            try:
                verified = self.workspace.validate_snapshot(
                    table.dataset_ref.snapshot_id, validator=verify
                )
            except ProcessingRecoveryError as error:
                raise ValueError("stage_table_validation_failed") from error
            if verified != table.dataset_ref:
                raise ValueError("stage_table_reference_mismatch")
        return ValidatedStageArtifacts(
            stage,
            scope,
            _dataset_record(json.loads(seal["input_snapshot_ref"])),
            profile_fingerprint,
            tuple(seal["evidence_refs"]),
            ref.removeprefix("build-artifact:"),
            tables,
        )

    def _qualification(
        self,
        ref: str,
        scope: GraphScope,
        profile_fingerprint: str,
        input_pin: str,
        runtime_payload: bytes,
    ) -> None:
        artifact = self.read(ref, scope)
        runtime = json.loads(runtime_payload)
        if (
            canonical_bytes(runtime) != runtime_payload
            or set(runtime)
            != {
                "version",
                "profile",
                "stages",
                "generation_available",
                "inference_runtime",
                "embedding_available",
                "retrieval",
                "source_provenance_digest",
            }
            or type(runtime["version"]) is not int
            or runtime["version"] != 1
            or set(runtime["stages"]) != set(STAGE_ROLES)
            or (
                runtime["source_provenance_digest"] is not None
                and (
                    type(runtime["source_provenance_digest"]) is not str
                    or re.fullmatch(r"[0-9a-f]{64}", runtime["source_provenance_digest"]) is None
                )
            )
        ):
            raise ValueError("invalid_stage_runtime_pin")
        if (
            runtime["profile"] != profile_fingerprint
            or runtime["stages"][artifact["stage_id"]] != artifact["semantics"]
        ):
            raise ValueError("stage_qualification_semantics_mismatch")
        input_ref = _dataset_record(json.loads(input_pin))
        if _manifest_canonical_bytes(_dataset_payload(input_ref)).decode() != input_pin or artifact[
            "input_refs"
        ] != [input_pin]:
            raise ValueError("stage_qualification_input_mismatch")

    def commit_operation(
        self,
        scope: GraphScope,
        *,
        stage: str,
        status: str,
        dependencies: tuple[str, ...],
        evidence_refs: tuple[str, ...],
        graph_ref: str,
        binding_ref: str | None,
        issues: tuple[str, ...],
        coverage: StageCoverage,
    ) -> str:
        """Commit closed post-manifest publication/retrieval execution evidence."""
        if stage not in {"publication", "retrieval"} or status not in {
            "succeeded",
            "failed",
            "partial",
            "skipped",
        }:
            raise ValueError("invalid_operational_stage")
        for dependency in dependencies:
            self.read(dependency, scope)
        return self._put(
            scope,
            {
                "kind": "operation",
                "stage_id": stage,
                "status": status,
                "dependencies": dependencies,
                "evidence_refs": evidence_refs,
                "graph_ref": graph_ref,
                "binding_ref": binding_ref,
                "issues": issues,
                "coverage": asdict(coverage),
            },
        )
