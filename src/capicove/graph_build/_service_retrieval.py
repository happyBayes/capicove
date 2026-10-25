"""Optional retrieval composition for the fixed document-graph build service."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from capicove.contracts.graph_build import GraphScope, GraphSnapshotRef
from capicove.graph_build.codec import fingerprint
from capicove.graph_build.retrieval import (
    GraphEmbeddingRuntime,
    GraphIndexAttempt,
    GraphRetrievalCatalog,
    GraphTextReader,
    build_graph_generation,
    graph_text_entries,
    projection_commitment,
)
from capicove.retrieval import (
    TextIndexGenerationCatalog,
    TextIndexGenerationLifecycle,
    TextIndexWriter,
)


@dataclass(frozen=True, slots=True)
class BuildRetrievalRuntime:
    """Explicit local dependencies for one optional graph-index projection."""

    reader: GraphTextReader
    embedder: GraphEmbeddingRuntime
    lifecycle: TextIndexGenerationLifecycle
    writer: TextIndexWriter
    generations: TextIndexGenerationCatalog
    bindings: GraphRetrievalCatalog
    index_format_version: str
    batch_size: int = 32

    def __post_init__(self) -> None:
        if self.bindings.reader is not self.reader:
            raise ValueError("retrieval reader pairing mismatch")
        if self.bindings.generations is not self.generations:
            raise ValueError("retrieval catalog pairing mismatch")
        if self.lifecycle.provider_id != self.writer.provider_id:
            raise ValueError("retrieval provider pairing mismatch")
        if (
            type(self.index_format_version) is not str
            or not self.index_format_version.strip()
            or len(self.index_format_version) > 256
        ):
            raise ValueError("invalid retrieval index format version")
        if type(self.batch_size) is not int or not 1 <= self.batch_size <= 256:
            raise ValueError("invalid retrieval batch size")

    @property
    def fingerprint(self) -> str:
        """Return immutable startup identity without consulting provider state."""
        return fingerprint(
            {
                "version": 1,
                "embedding": self.embedder.fingerprint,
                "retriever": self.lifecycle.provider_id,
                "index_format": self.index_format_version,
                "batch_size": self.batch_size,
            }
        )


@dataclass(frozen=True, slots=True)
class BuildRetrievalResult:
    """Bounded optional outcome, kept separate from graph completeness."""

    status: str
    binding_id: str | None
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        allowed = {
            "ready": (),
            "unavailable": ("retrieval_runtime_unavailable", "retrieval_unauthorized"),
            "failed": ("retrieval_build_failed",),
            "incomplete": ("retrieval_attempt_incomplete",),
        }
        if self.status not in allowed:
            raise ValueError("invalid retrieval build status")
        if self.status == "ready":
            if not self.binding_id or self.issue_codes:
                raise ValueError("ready retrieval requires only a binding")
        elif self.binding_id is not None or len(self.issue_codes) != 1:
            raise ValueError("non-ready retrieval requires one bounded issue")
        if self.issue_codes and self.issue_codes[0] not in allowed[self.status]:
            raise ValueError("retrieval issue does not match status")


def _uncontrolled() -> None:
    """Allow standalone retrieval composition without a build controller."""


def build_retrieval(
    run_id: str,
    graph_ref: GraphSnapshotRef | None,
    scope: GraphScope,
    runtime: BuildRetrievalRuntime | None,
    *,
    check_control: Callable[[], None] = _uncontrolled,
) -> BuildRetrievalResult:
    """Build or resume one exact optional generation without activation.

    Provider I/O and authorization failures are bounded optional outcomes. Typed
    identity, pairing, and durable catalog corruption remain raised integrity
    violations and must stop the caller from persisting a misleading result.
    """
    if runtime is None:
        return BuildRetrievalResult("unavailable", None, ("retrieval_runtime_unavailable",))
    if type(run_id) is not str or not run_id.strip() or len(run_id) > 256:
        raise ValueError("invalid retrieval run identity")
    if type(graph_ref) is not GraphSnapshotRef or type(scope) is not GraphScope:
        raise TypeError("retrieval build requires typed graph and scope")
    attempt = GraphIndexAttempt(run_id, runtime.lifecycle.provider_id, runtime.index_format_version)
    try:
        check_control()
        state = runtime.bindings.attempt(run_id, scope)
        if state is not None:
            if (
                state.attempt != attempt
                or state.projection.graph_ref != graph_ref
                or state.projection.scope != scope
                or state.projection.runtime_sha256 != runtime.embedder.fingerprint
            ):
                raise ValueError("retrieval attempt identity or projection mismatch")
            if state.status == "failed":
                return BuildRetrievalResult("failed", None, ("retrieval_build_failed",))
            if state.status == "started":
                return BuildRetrievalResult("incomplete", None, ("retrieval_attempt_incomplete",))
            validated = runtime.bindings.validated_attempt(run_id, scope)
            if validated is None:
                raise ValueError("ready retrieval attempt cannot be validated")
            projection = state.projection
            check_control()
            binding = runtime.bindings.register(
                graph_ref, (validated,), projection.unit_manifest_sha256, scope
            )
            return BuildRetrievalResult("ready", binding.binding_id, ())

        content = runtime.reader.read(graph_ref, scope)
        entries = graph_text_entries(
            graph_ref,
            content,
            scope,
            runtime.embedder,
            reader=runtime.reader,
            check_control=check_control,
        )
        check_control()
        mapping = runtime.lifecycle.mapping_sha256((runtime.embedder.space,))
        check_control()
        projection = projection_commitment(content, runtime.embedder, entries, mapping)
        validated = build_graph_generation(
            projection,
            entries,
            attempt,
            runtime=runtime.embedder,
            lifecycle=runtime.lifecycle,
            writer=runtime.writer,
            generations=runtime.generations,
            bindings=runtime.bindings,
            batch_size=runtime.batch_size,
            check_control=check_control,
        )
        check_control()
        binding = runtime.bindings.register(
            graph_ref, (validated,), projection.unit_manifest_sha256, scope
        )
        return BuildRetrievalResult("ready", binding.binding_id, ())
    except PermissionError:
        check_control()
        return BuildRetrievalResult("unavailable", None, ("retrieval_unauthorized",))
    except (OSError, TimeoutError):
        check_control()
        return BuildRetrievalResult("failed", None, ("retrieval_build_failed",))


__all__ = ["BuildRetrievalResult", "BuildRetrievalRuntime", "build_retrieval"]
