"""A runnable local video build-to-query composition using provisioned providers."""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import BoundedSemaphore
from types import TracebackType
from typing import TYPE_CHECKING, Protocol

from capicove.contracts import EmbeddingSpaceRef, InvocationContext, ProcessingFingerprint
from capicove.processing.video.ports import PathAuthorizer
from capicove.processing.video.retrieval.local import (
    create_local_video_preparation_service,
    read_local_preparation_resource,
)
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationRequest
from capicove.processing.video.retrieval.ports import ASRProcessor
from capicove.retrieval.ports import TextRetrieverBackend
from capicove.retrieval.video_evidence import VideoEvidenceReader
from capicove.retrieval.video_generations import VideoTextMemberValidator, open_exact_video_runtime
from capicove.retrieval.video_profiles import (
    StaticVideoRetrievalProfileRegistry,
    VideoRetrievalProfile,
)
from capicove.retrieval.video_publication import publish_prepared_video
from capicove.retrieval.video_search import VideoQueryEncoder, VideoSearchService
from capicove.tools.evidence import EvidenceToolHandler
from capicove.tools.search import SearchToolHandler


if TYPE_CHECKING:
    from fastapi import FastAPI

    from capicove.mcp.search import McpCallback, McpRegistrationServer
    from capicove.processing.image import ImageOcr


class SharedVideoEncoder(VideoQueryEncoder, Protocol):
    """One provisioned model serves offline previews and online text/image queries."""

    def encode_preview_batch(
        self, previews: tuple[bytes, ...]
    ) -> tuple[tuple[float, ...], ...]: ...


@dataclass(frozen=True, slots=True)
class VideoBuildResult:
    """Return the published snapshot and preparation degradation codes."""

    generation_set_id: str
    warning_codes: tuple[str, ...]


class LocalVideoSearchRuntime:
    """Own preparation, publication, leases, query workers and all search transports.

    Providers are explicitly provisioned by startup code. Construction never
    downloads models. The caller owns the text backend and its connection pool.
    """

    def __init__(
        self,
        *,
        workspace_root: Path,
        workspace_ref: str,
        corpus_refs: tuple[str, ...],
        embedding_space: EmbeddingSpaceRef,
        encoder: SharedVideoEncoder,
        text_backend: TextRetrieverBackend,
        text_index_format: str,
        path_authorizer: PathAuthorizer | None = None,
        policy_ref: str | None = None,
        asr_processor: ASRProcessor | None = None,
        image_ocr: ImageOcr | None = None,
        ocr_provider_fingerprint: ProcessingFingerprint | None = None,
        authorize_evidence: Callable[[InvocationContext, str, str], bool] | None = None,
    ) -> None:
        if not workspace_root.is_absolute() or not workspace_ref.strip() or not corpus_refs:
            raise ValueError(
                "absolute workspace root and explicit workspace/corpus identities required"
            )
        self._root = workspace_root
        self._workspace = workspace_ref
        self._corpora = corpus_refs
        self._text = text_backend
        self._text_format = text_index_format
        self._closed = False
        self._preparation_warnings = tuple(
            code
            for code, missing in (
                ("asr_not_configured", asr_processor is None),
                ("ocr_not_configured", image_ocr is None),
            )
            if missing
        )
        self.preparation = create_local_video_preparation_service(
            workspace_root,
            embedding_space=embedding_space,
            encode_batch=encoder.encode_preview_batch,
            path_authorizer=path_authorizer,
            policy_ref=policy_ref,
            asr_processor=asr_processor,
            image_ocr=image_ocr,
            ocr_provider_fingerprint=ocr_provider_fingerprint,
        )
        self.components = open_exact_video_runtime(
            generation_database_path=workspace_root / "generations.sqlite3",
            index_root=workspace_root / "indexes",
            validate_text_members=VideoTextMemberValidator(
                asr=text_backend.generation_lifecycle, ocr=text_backend.generation_lifecycle
            ),
        )
        self._executor = ThreadPoolExecutor(max_workers=8, thread_name_prefix="video-query")
        self.search = VideoSearchService(
            workspace_ref=workspace_ref,
            generation_sets=self.components.generation_sets,
            profiles=StaticVideoRetrievalProfileRegistry(
                (VideoRetrievalProfile.video_local_exact(),)
            ),
            catalog_resolver=self.components.catalog_resolver,
            visual_retriever=self.components.visual_backend,
            query_encoder=encoder,
            executor=self._executor,
            concurrency=BoundedSemaphore(4),
            text_backends={"asr_lexical": text_backend, "ocr_lexical": text_backend},
        )
        self.search_handler = SearchToolHandler(self.search)
        self.evidence_handler = EvidenceToolHandler(
            _LocalVideoEvidenceReader(self, authorize_evidence)
        )

    def build(self, request: VideoRetrievalPreparationRequest) -> VideoBuildResult:
        """Replace the active single-video snapshot after complete physical validation."""
        if self._closed:
            raise RuntimeError("video runtime is closed")
        if request.profile_ref != "video-local-exact":
            raise ValueError("local runtime requires video-local-exact")
        prepared = self.preparation.prepare(request)
        pin = publish_prepared_video(
            prepared,
            runtime=self.components,
            workspace_root=self._root,
            workspace_ref=self._workspace,
            corpus_refs=self._corpora,
            text_backend=self._text,
            text_index_format=self._text_format,
            read_resource=lambda ref: read_local_preparation_resource(self._root, ref),
        )
        return VideoBuildResult(pin.set_id, (*self._preparation_warnings, *prepared.warning_codes))

    def cli(self, argv: Sequence[str]) -> int:
        """Run the existing CLI over this instance."""
        from capicove.cli.app import main

        if self._closed:
            raise RuntimeError("video runtime is closed")
        return main(
            argv, search_handler=self.search_handler, evidence_handler=self.evidence_handler
        )

    def register_mcp(self, server: McpRegistrationServer) -> McpCallback:
        """Register the existing canonical MCP tool over this instance."""
        from capicove.mcp.search import register_search_mcp

        return register_search_mcp(server, self.search_handler)

    def register_evidence_mcp(
        self, server: McpRegistrationServer, *, context_factory: Callable[[], InvocationContext]
    ) -> McpCallback:
        """Opt in to evidence using a host-owned authenticated context."""
        from capicove.mcp.evidence import register_evidence_mcp

        return register_evidence_mcp(server, self.evidence_handler, context_factory=context_factory)

    def http_app(
        self, *, evidence_context_factory: Callable[[], InvocationContext] | None = None
    ) -> FastAPI:
        """Create the local HTTP app; hosting and network exposure are explicit."""
        from capicove.api.search import create_search_app

        return create_search_app(
            self.search,
            evidence_handler=self.evidence_handler if evidence_context_factory else None,
            evidence_context_factory=evidence_context_factory,
        )

    def close(self) -> None:
        """Drain workers before releasing indexes, including timed-out query leases."""
        if not self._closed:
            self._closed = True
            self._executor.shutdown(wait=True)
            self.components.close()

    def __enter__(self) -> LocalVideoSearchRuntime:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()


class _LocalVideoEvidenceReader:
    """Resolve existing visual receipts against one leased active generation.

    This reconstructs bindings from hash-verified receipts already written by
    preparation, without a new persistent index or a mutable latest-query cache.
    Text/diagnostic expansion requires a separately composed reader.
    """

    def __init__(
        self,
        runtime: LocalVideoSearchRuntime,
        authorize: Callable[[InvocationContext, str, str], bool] | None,
    ) -> None:
        self._runtime = runtime
        self._authorize = authorize

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]:
        runtime = self._runtime
        if runtime._closed:
            raise LookupError("video runtime is closed")
        if datetime.now(UTC) >= context.deadline:
            raise TimeoutError("evidence deadline exceeded")
        if (
            self._authorize is None
            or self._authorize(context, evidence_id, evidence_id) is not True
        ):
            raise PermissionError("evidence resource access denied")
        if re.fullmatch(r"resource:sha256:[0-9a-f]{64}", evidence_id) is None:
            raise LookupError("visual evidence receipt required")

        def read_resource(reference: str) -> bytes:
            try:
                return read_local_preparation_resource(runtime._root, reference)
            except FileNotFoundError as error:
                raise LookupError("evidence resource unavailable") from error

        receipt_bytes = read_resource(evidence_id)
        if len(receipt_bytes) > 16_384:
            raise ValueError("visual evidence receipt exceeds capacity")
        receipt = json.loads(receipt_bytes)
        if (
            not isinstance(receipt, dict)
            or receipt.get("schema_version") != 1
            or receipt.get("kind") != "video-visual-unit"
        ):
            raise LookupError("visual evidence receipt required")
        unit_id = receipt.get("unit_id")
        if (
            not isinstance(unit_id, str)
            or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,511}", unit_id) is None
        ):
            raise ValueError("invalid visual evidence unit")
        if self._authorize(context, evidence_id, unit_id) is not True:
            raise PermissionError("evidence unit access denied")
        with runtime.components.generation_sets.lease_active(
            workspace_ref=runtime._workspace, profile_ref="video-local-exact"
        ) as pin:
            reader = VideoEvidenceReader(
                catalog=runtime.components.catalog_resolver(pin),
                evidence_units={evidence_id: (unit_id,)},
                read_resource=read_resource,
                authorize_resource=self._authorize,
            )
            return reader.read(context=context, evidence_id=evidence_id, sections=sections)
