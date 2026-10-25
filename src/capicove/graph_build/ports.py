"""Capability-owned input ports; implementations enforce current authorization."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from capicove.contracts.graph_build import GraphScope
from capicove.contracts.retrieval import EmbeddingSpaceRef
from capicove.graph_build.receipts import InferenceReceipt
from capicove.graph_build.representations import TextRepresentation


if TYPE_CHECKING:
    from capicove.graph_build.communities import WeightedGraph
    from capicove.graph_build.inference import DispatchControl, RawGeneration
    from capicove.graph_build.profiles import PartitionProfile, StructuredRequest
    from capicove.retrieval import TextEmbedding


class RepresentationUnavailableError(RuntimeError):
    """Expected source-local read failure, distinct from reader integrity defects."""


@dataclass(frozen=True, slots=True)
class CommunityRuntimeIdentity:
    """Declared runtime and immutable execution settings, separate from adapter version.

    Implementations must return a deterministic identity for the execution they
    perform. Scripted providers declare their own runtime rather than omitting it.
    Configuration pairs are sorted unique text keys and text values.
    """

    runtime_id: str
    runtime_version: str
    configuration: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if type(self.configuration) is not tuple or any(
            type(pair) is not tuple or len(pair) != 2 for pair in self.configuration
        ):
            raise TypeError("runtime configuration must be immutable pairs")
        for value in (
            self.runtime_id,
            self.runtime_version,
            *(v for p in self.configuration for v in p),
        ):
            if type(value) is not str or not value.strip():
                raise ValueError("runtime identity must contain nonblank text")
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                raise ValueError("runtime identity must be valid UTF-8") from None
        keys = tuple(key for key, _ in self.configuration)
        if keys != tuple(sorted(set(keys))):
            raise ValueError("runtime configuration keys must be unique and sorted")


class CommunityPartitioner(Protocol):
    """Explicit strategy over immutable capability values, never provider graphs.

    Identity is stable for the adapter lifetime. Implementations are reentrant;
    execution metadata must not be communicated through mutable last-run state.
    """

    @property
    def algorithm_id(self) -> str: ...

    @property
    def algorithm_version(self) -> str: ...

    @property
    def runtime_identity(self) -> CommunityRuntimeIdentity: ...

    def partition(
        self, graph: WeightedGraph, profile: PartitionProfile
    ) -> tuple[frozenset[str], ...]: ...


class BatchTextEmbeddingProvider(Protocol):
    """Embed derived graph text through an injected public retrieval service.

    Roles are exactly query/document, with model instructions selected at startup.
    Preserve tuple count/order and the complete requested embedding-space identity.
    """

    def embed(
        self, texts: tuple[str, ...], *, space: EmbeddingSpaceRef, role: str
    ) -> tuple[TextEmbedding, ...]: ...


class StructuredGenerator(Protocol):
    """Invoke only the explicitly configured, pinned provider/model.

    Implementations must bound transport time by control.remaining_seconds(),
    observe control.cancelled() during in-flight work, and bound response bytes by
    the request output limits. Preserve TimeoutError and RateLimitError so the
    runner can distinguish temporary failures from permanent errors. Control is
    operational context, never serialized model input or canonical request data.
    Synchronous invocation cannot be forcibly interrupted by the runner.
    """

    @property
    def provider_id(self) -> str:
        """Return the explicitly configured provider identity."""
        ...

    @property
    def model_id(self) -> str:
        """Return the immutable model identity."""
        ...

    @property
    def model_revision(self) -> str:
        """Return the pinned model revision."""
        ...

    def generate(self, request: StructuredRequest, *, control: DispatchControl) -> RawGeneration:
        """Return bounded raw output without hiding permanent error types."""
        ...


class RequestTokenizer(Protocol):
    """Count the complete provider wire prompt with the pinned tokenizer.

    count_request includes every segment, schema, separator, special token and
    provider chat-template overhead. The configured generator must use this same
    serialization; a text-only estimate is not a valid implementation.
    """

    @property
    def tokenizer_id(self) -> str:
        """Return the configured tokenizer identity."""
        ...

    @property
    def tokenizer_revision(self) -> str:
        """Return the pinned tokenizer revision."""
        ...

    def count_request(self, request: StructuredRequest) -> int:
        """Count exact complete input tokens without truncation or acquisition."""
        ...


class AccessGate(Protocol):
    """Enforce current host policy; references are not authorization tokens."""

    def require_scope(self, scope: GraphScope) -> None:
        """Raise unless current policy authorizes the compound scope."""
        ...

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        """Raise unless every associated source remains available and authorized."""
        ...


class ProtectedArtifacts(Protocol):
    """Store bytes bound to explicit immutable evidence lineage.

    Configure a writer for each unit's exact evidence set; do not reuse that writer
    for a different source set. Readers must authorize persisted lineage, including
    after restart, rather than assuming the writer's current binding describes it.
    """

    @property
    def evidence_refs(self) -> tuple[str, ...]:
        """Return the immutable evidence set associated with future puts."""
        ...

    def put(self, scope: GraphScope, payload: bytes) -> str:
        """Authorize scope and bound evidence; durably store protected bytes."""
        ...

    def read(self, scope: GraphScope, artifact_ref: str) -> bytes:
        """Authorize scope and persisted evidence, and verify content integrity."""
        ...


class ReceiptStore(Protocol):
    """Fence acceptance with durable leases; provider execution may be repeated."""

    def claim(self, scope: GraphScope, key: str, attempt_id: str, lease_until: float) -> bool:
        """Acquire an unowned/expired unit with a fresh attempt and Unix deadline."""
        ...

    def complete(
        self,
        scope: GraphScope,
        key: str,
        attempt_id: str,
        receipt: InferenceReceipt,
        *,
        success_guard: Callable[[], None] | None = None,
    ) -> None:
        """Accept a verified result from the live owner; exact repeats are idempotent.

        Before new success publication, invoke success_guard after artifact and
        lease verification inside the acceptance transaction. A raised exception
        must roll back success. The guard must be short and perform no I/O/store
        mutation. Failed/cancelled results bypass it so they can be retained after
        the dispatch deadline under their still-valid durable lease.
        """
        ...

    def get(self, scope: GraphScope, key: str) -> InferenceReceipt | None:
        """Return an authorized, integrity-checked terminal receipt, if available."""
        ...


class AuthorizedRepresentationReader(Protocol):
    """Read only in-scope available sources, including on cache hits.

    Implementations must reject unavailable or unauthorized evidence rather than
    treating it as empty text. No implementation or implicit acquisition is
    supplied here. Returned document/scope identity must match the request.
    """

    def read(self, document_ref: str, scope: GraphScope) -> TextRepresentation:
        """Authorize source dependencies and return immutable exact text."""
        ...

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        """Raise on any denied reference; possession of a hash grants no access."""
        ...


class TextTokenizer(Protocol):
    """Deterministic counts bound to a pinned tokenizer identity and revision."""

    @property
    def fingerprint(self) -> str:
        """Return the immutable lowercase SHA-256 of tokenizer configuration."""
        ...

    def count(self, text: str) -> int:
        """Count tokens without truncation, acquisition, or remote calls."""
        ...


class ControlledRequestTokenizer(RequestTokenizer, Protocol):
    """Optional remote-counting port carrying the caller's live execution budget."""

    def count_request_controlled(
        self, request: StructuredRequest, *, control: DispatchControl
    ) -> int:
        """Count exact input within the remaining deadline; never retry implicitly."""
        ...
