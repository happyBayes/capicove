"""Explicit local SigLIP2/Elasticsearch bootstrap for building and querying video."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import uuid4

from capicove.contracts import (
    DistanceMetric,
    EmbeddingSpaceRef,
    InvocationContext,
    ProcessingFingerprint,
    QueryModality,
    TargetType,
    VectorNormalization,
)
from capicove.model import load_packaged_manifest
from capicove.processing.video.models import VideoSource
from capicove.processing.video.retrieval.models import VideoRetrievalPreparationRequest
from capicove.retrieval.video_http_encoding import LocalSiglip2QueryEncoder
from capicove.runtime.video_search import LocalVideoSearchRuntime
from capicove.tools.evidence import SECTIONS


class _SourcePolicy:
    """Authorize only the exact source explicitly selected by the local operator."""

    def __init__(self, source: Path | None) -> None:
        self._source = source

    def authorize_read(self, policy_ref: str, path: Path) -> None:
        if policy_ref != "operator-source" or self._source is None or path != self._source:
            raise PermissionError("source is outside the operator selection")

    def authorize_write(self, policy_ref: str, path: Path) -> None:
        raise PermissionError("source writes are forbidden")


def _authorize_operator(context: InvocationContext, evidence_id: str, resource_ref: str) -> bool:
    """Grant reads only to this explicit local CLI workspace invocation."""
    return (
        context.workspace_id == "local"
        and context.actor_id == "capicove-cli"
        and context.environment_id == "local"
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Build an optional source and query a persistent single-video workspace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", required=True, type=Path)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--es-endpoint", required=True)
    parser.add_argument("--es-ca", required=True)
    parser.add_argument("--build-video", type=Path)
    parser.add_argument("--ocr-provider", choices=("paddleocr", "none"), default="paddleocr")
    parser.add_argument("--ocr-model-root", type=Path)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--query")
    action.add_argument("--get-evidence", help="Read a visual evidence ID from this workspace.")
    parser.add_argument("--section", choices=SECTIONS, action="append", default=[])
    args = parser.parse_args(argv)
    if args.section and args.get_evidence is None:
        parser.error("--section requires --get-evidence")
    if args.get_evidence is not None and args.build_video is not None:
        parser.error("--build-video cannot be combined with --get-evidence")
    image_ocr = None
    ocr_fingerprint = None
    if args.build_video is not None and args.ocr_provider != "none":
        from capicove.processing.image import ImageOcr
        from capicove.processing.image.ocr_configuration import ImageOcrConfiguration

        provider = ImageOcrConfiguration(model_root=args.ocr_model_root).create_provider()
        assert provider is not None
        image_ocr = ImageOcr(provider)
        ocr_fingerprint = provider.fingerprint
    # Optional plugin loading happens only after this explicit bootstrap invocation.
    from capicove_retriever_elasticsearch.adapter import ElasticsearchTextRetriever
    from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration
    from elasticsearch import Elasticsearch

    spec = next(m for m in load_packaged_manifest().models if m.key == "siglip2-base-patch16-224")
    space = EmbeddingSpaceRef(
        "siglip2-video-local-v1",
        spec.repository,
        spec.revision,
        spec.preprocess_revision,
        768,
        VectorNormalization.L2,
        DistanceMetric.INNER_PRODUCT,
        frozenset({QueryModality.TEXT, QueryModality.IMAGE}),
        frozenset({TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}),
    )
    encoder = LocalSiglip2QueryEncoder(
        endpoint=args.model_endpoint, model_id=spec.repository, embedding_space=space
    )
    client = Elasticsearch(args.es_endpoint, ca_certs=args.es_ca)
    backend = ElasticsearchTextRetriever(
        ElasticsearchConfiguration(
            hosts=(args.es_endpoint,),
            ca_certs=args.es_ca,
            index_prefix="capicove-video-local",
            active_alias="capicove-video-local-active",
        ),
        client_factory=lambda config: client,
    )
    source = args.build_video.resolve(strict=True) if args.build_video else None
    try:
        with LocalVideoSearchRuntime(
            workspace_root=args.workspace.resolve(),
            workspace_ref="local-video",
            corpus_refs=("operator-video",),
            embedding_space=space,
            encoder=encoder,
            text_backend=backend,
            text_index_format="elasticsearch-text-v1",
            path_authorizer=_SourcePolicy(source),
            policy_ref="operator-source",
            authorize_evidence=_authorize_operator,
            image_ocr=image_ocr,
            ocr_provider_fingerprint=ocr_fingerprint,
        ) as runtime:
            if source is not None:
                identity = uuid4().hex
                fingerprint = ProcessingFingerprint.from_manifest({"bootstrap": "video-local-v1"})
                result = runtime.build(
                    VideoRetrievalPreparationRequest(
                        VideoSource(path=source),
                        f"artifact:{identity}",
                        f"evidence:{identity}",
                        f"representation:{identity}",
                        f"run:{identity}",
                        "video-local-exact",
                        fingerprint,
                        fingerprint,
                        fingerprint,
                        datetime.now(UTC) + timedelta(minutes=30),
                    )
                )
                # Keep stdout reserved for the canonical query response.
                import sys

                print(
                    json.dumps(
                        {
                            "generation_set_id": result.generation_set_id,
                            "warning_codes": result.warning_codes,
                        }
                    ),
                    file=sys.stderr,
                )
            if args.get_evidence is not None:
                sections = [value for section in args.section for value in ("--section", section)]
                return runtime.cli(["get-evidence", args.get_evidence, *sections])
            return runtime.cli(["search", args.query, "--target", "video_segment"])
    finally:
        client.close()


if __name__ == "__main__":
    raise SystemExit(main())
