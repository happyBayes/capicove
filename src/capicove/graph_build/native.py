"""Public composition of the existing graph builder for native knowledge hosts."""

from __future__ import annotations

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from capicove.contracts import EmbeddingSpaceRef
from capicove.contracts.data import DatasetSnapshotRef
from capicove.contracts.graph_build import GraphScope, StageCoverage
from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
from capicove.graph_build._adapters.louvain import NetworkXLouvainPartitioner
from capicove.graph_build._adapters.parquet import ParquetTableCodec
from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
from capicove.graph_build._stage_artifacts import BuildStageStore
from capicove.graph_build.codec import fingerprint
from capicove.graph_build.datasets import DocumentRow, codec_for_role, publish_table
from capicove.graph_build.inference import InferenceRunner
from capicove.graph_build.ports import BatchTextEmbeddingProvider
from capicove.graph_build.profiles import ChunkingProfile, GraphBuildProfile, InferencePolicy
from capicove.graph_build.remote_generation import (
    OpenAICompatibleGenerator,
    OpenAICompatibleProfile,
    OpenAICompatibleTokenizer,
)
from capicove.graph_build.representations import TextRepresentation
from capicove.graph_build.service import (
    BuildPolicy,
    DocumentSourceProvenance,
    GraphBuildRequest,
    GraphBuildService,
)
from capicove.graph_build.snapshots import (
    GraphSnapshotCatalog,
    InputSnapshotInventory,
    SnapshotAccessGate,
)
from capicove.graph_build.vlm_generation import RegisteredGenerationProfile
from capicove.processing import LocalProcessingWorkspace


class _Inventory:
    def __init__(self, workspace: LocalProcessingWorkspace, gate: SnapshotAccessGate) -> None:
        self.workspace, self.gate = workspace, gate

    def rows(self, ref: DatasetSnapshotRef, role: str) -> tuple[Any, ...]:
        rows: tuple[Any, ...] = ()

        def read(path: Path) -> bool:
            nonlocal rows
            rows = ParquetTableCodec(publication_identity=ref.processing_fingerprint).read(
                path / "part-00000.parquet", role
            )
            return True

        if self.workspace.validate_snapshot(ref.snapshot_id, validator=read) != ref:
            raise ValueError("graph_table_integrity_failed")
        return rows

    def documents(self, ref: DatasetSnapshotRef, scope: GraphScope) -> tuple[DocumentRow, ...]:
        self.gate.require_scope(scope)
        rows = self.rows(ref, "documents")
        if any(type(row) is not DocumentRow or row.scope != scope for row in rows):
            raise ValueError("graph_input_scope_mismatch")
        return rows

    def validate(self, ref: DatasetSnapshotRef, scope: GraphScope) -> InputSnapshotInventory:
        rows = self.documents(ref, scope)
        refs = tuple(sorted({row.evidence_ref for row in rows}))
        self.gate.require_access(refs, scope)
        return InputSnapshotInventory(
            ref,
            scope,
            tuple(row.document_ref for row in rows),
            refs,
            fingerprint([codec_for_role("documents").encode(row) for row in rows]),
        )


class _Reader:
    def __init__(
        self, representations: tuple[TextRepresentation, ...], gate: SnapshotAccessGate
    ) -> None:
        self.values = {row.document_ref: row for row in representations}
        self.gate = gate

    def read(self, document_ref: str, scope: GraphScope) -> TextRepresentation:
        result = self.values[document_ref]
        self.require_access((result.representation_ref,), scope)
        return result

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        self.gate.require_access(evidence_refs, scope)


class _Sources:
    fingerprint = fingerprint("native-knowledge-source-only-v1")

    def classify(
        self, documents: tuple[DocumentRow, ...], scope: GraphScope
    ) -> tuple[DocumentSourceProvenance, ...]:
        return tuple(
            DocumentSourceProvenance(row.document_ref, "source", None) for row in documents
        )


class _NoSubject:
    fingerprint = fingerprint("native-knowledge-no-personal-subject-v1")

    def resolve(
        self, subject_ref: str, entities: tuple[Any, ...], scope: GraphScope
    ) -> tuple[str, ...]:
        return ()


def build_native_graph(
    root: Path,
    *,
    scope: GraphScope,
    representations: tuple[TextRepresentation, ...],
    gate: SnapshotAccessGate,
    generation: OpenAICompatibleProfile | RegisteredGenerationProfile,
    embedder: BatchTextEmbeddingProvider,
    space: EmbeddingSpaceRef,
) -> dict[str, Any]:
    """Build, validate and activate a graph using the established typed workflow.

    Generic corpus sources are never treated as the owner's self reports. The
    existing closed ontology is retained; no subject attribution is inferred.
    """
    workspace = LocalProcessingWorkspace(root / "processing")
    inventory = _Inventory(workspace, gate)
    reader = _Reader(representations, gate)
    tokenizer = OpenAICompatibleTokenizer(generation)
    profile = GraphBuildProfile(
        profile_id="native-knowledge-v1",
        ontology_version="personal-growth-v1",
        chunking=ChunkingProfile(),
        inference=InferencePolicy(
            generation.provider_id,
            generation.model,
            generation.model_revision,
            tokenizer.tokenizer_id,
            tokenizer.tokenizer_revision,
            "user",
            generation.context_limit,
        ),
        generation_profile_id="deterministic-json-v1",
        identity_embedding_space=space.space_id,
        retrieval_embedding_space=space.space_id,
        extraction_prompt_version="extract-native-v1",
        extraction_schema_version="extraction-native-v1",
        report_schema_version="report-native-v1",
        report_prompt_version="report-native-v1",
    )
    stages = BuildStageStore(root / "stages" / "stages.sqlite", gate, workspace)
    catalog = GraphSnapshotCatalog(
        root / "catalog" / "catalog.sqlite", workspace, gate, inventory, stages
    )

    def artifacts(refs: tuple[str, ...]) -> FilesystemProtectedArtifacts:
        return FilesystemProtectedArtifacts(root / "artifacts", gate, evidence_refs=refs)

    receipts = SqliteReceiptStore(
        root / "receipts" / "receipts.sqlite",
        gate,
        artifacts(tuple(row.representation_ref for row in representations)),
    )
    runner = InferenceRunner(
        generator=OpenAICompatibleGenerator(generation),
        tokenizer=tokenizer,
        gate=gate,
        artifact_factory=artifacts,
        receipts=receipts,
        clock=time.monotonic,
        wall_clock=time.time,
        sleeper=time.sleep,
        cancelled=lambda: False,
    )
    service = GraphBuildService(
        root / "runs" / "runs.sqlite",
        reader=reader,
        gate=gate,
        profiles=(profile,),
        inputs=inventory,
        tokenizer=tokenizer,
        inference_runner=runner,
        embedder=embedder,
        identity_space=space,
        partitioner=NetworkXLouvainPartitioner(),
        workspace=workspace,
        catalog=catalog,
        stages=stages,
        policy=BuildPolicy("subject:unassigned"),
        subject_resolver=_NoSubject(),
        source_provenance=_Sources(),
    )
    rows = tuple(
        DocumentRow(
            row.document_ref,
            scope,
            row.source_sha256,
            row.representation_ref,
            row.representation_ref,
            "text/plain",
            "available",
            (),
            StageCoverage("codepoints", len(row.text), len(row.text), len(row.text), 0, 0, 0),
        )
        for row in representations
    )
    input_ref = publish_table(
        workspace,
        scope=scope,
        role="documents",
        rows=rows,
        parent_refs=(),
        processing_fingerprint=fingerprint([asdict(row) for row in rows]),
    )
    result = service.build(GraphBuildRequest(input_ref, scope, profile.profile_id, None, "reuse"))
    receipt = {
        "run_id": result.run_id,
        "status": result.status,
        "completeness": result.completeness,
        "issue_codes": list(result.issue_codes),
        "snapshot_id": result.graph_ref.snapshot_id if result.graph_ref else None,
    }
    context: list[dict[str, Any]] = []
    if result.status == "succeeded" and result.graph_ref:
        manifest = catalog.get(result.graph_ref.snapshot_id, scope)
        for table in manifest.tables:
            if table.role not in {"resolved_entities", "community_reports"}:
                continue
            for row in inventory.rows(table.dataset_ref, table.role):
                encoded = codec_for_role(table.role).encode(row)
                context.append({"kind": table.role, "text": json_text(encoded)})
        _, revision = catalog.active(scope)
        catalog.activate(result.graph_ref.snapshot_id, scope, expected_revision=revision)
    return {"receipt": receipt, "context": context}


def json_text(value: Any) -> str:
    """Bound derived context; original evidence remains the citation authority."""
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)[:6000]
