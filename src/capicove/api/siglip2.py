"""Optional FastAPI boundary for local SigLIP2 retrieval inference."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel, Field

from capicove.api.model_http import run_worker
from capicove.model import (
    load_packaged_manifest,
    verify_model_installation,
)
from capicove.processing.image.preparation import (
    DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES,
    ImageCompressionError,
    ImageFormatError,
    ImageInputTooLargeError,
    ImagePixelLimitError,
    ImagePreparationError,
    prepare_image,
)
from capicove.retrieval import Siglip2RetrievalService
from capicove.retrieval._adapters.siglip2 import HuggingFaceSiglip2Provider


DEFAULT_MAX_UPLOAD_BYTES = 10 * 1024 * 1024
DEFAULT_MAX_IMAGE_PIXELS = 40_000_000
ALLOWED_IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG", "WEBP"})


class TextEmbeddingRequest(BaseModel):
    """Request one or more text embeddings."""

    texts: list[str] = Field(min_length=1, max_length=128)


def build_service(
    model_directory: str | Path,
    *,
    expected_model_manifest_sha256: str,
    device: str = "auto",
) -> Siglip2RetrievalService:
    """Compose a service from a trusted, explicitly downloaded installation."""
    manifest = load_packaged_manifest()
    model = next(item for item in manifest.models if item.key == "siglip2-base-patch16-224")
    installation = verify_model_installation(
        Path(model_directory),
        model,
        manifest_sha256=manifest.source_sha256,
        expected_model_manifest_sha256=expected_model_manifest_sha256,
    )
    provider = HuggingFaceSiglip2Provider(installation, device=device)
    provider.load()
    return Siglip2RetrievalService(provider)


def create_app(
    service: Siglip2RetrievalService,
    *,
    max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
    max_image_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
    compression_target_bytes: int = DEFAULT_IMAGE_COMPRESSION_TARGET_BYTES,
) -> FastAPI:
    """Create an HTTP application around one configured retrieval service.

    Uploads over ``compression_target_bytes`` are normalized and compressed in
    memory before inference. ``max_upload_bytes`` bounds the file payload read by
    this handler after multipart parsing; a deployment exposed beyond trusted local
    clients must also enforce a total request-body limit before multipart parsing.
    """
    if max_upload_bytes <= 0 or max_image_pixels <= 0 or compression_target_bytes <= 0:
        raise ValueError("image limits must be positive")
    app = FastAPI(
        title="Capicove SigLIP2 Retrieval API",
        version="1.0.0",
        description="Offline image-text ranking and aligned embedding inference.",
    )

    async def decoded_image(upload: UploadFile) -> Image.Image:
        payload = await upload.read(max_upload_bytes + 1)
        await upload.close()
        if not payload:
            raise HTTPException(status_code=422, detail="image must not be empty")
        try:
            prepared = await run_worker(
                prepare_image,
                payload,
                target_bytes=compression_target_bytes,
                max_input_bytes=max_upload_bytes,
                max_image_pixels=max_image_pixels,
                allowed_formats=ALLOWED_IMAGE_FORMATS,
            )
        except ImageInputTooLargeError as error:
            raise HTTPException(
                status_code=413,
                detail=f"image exceeds the {max_upload_bytes}-byte limit",
            ) from error
        except ImagePixelLimitError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ImageFormatError as error:
            raise HTTPException(status_code=415, detail="unsupported image format") from error
        except ImageCompressionError as error:
            raise HTTPException(status_code=413, detail=str(error)) from error
        except ImagePreparationError as error:
            raise HTTPException(status_code=422, detail="invalid image") from error
        return prepared.image.convert("RGB")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {
            "status": "ok",
            "model_id": service.model_id,
            "model_revision": service.model_revision,
            "device": service.device,
        }

    @app.post("/v1/similarity")
    async def similarity(
        image: Annotated[UploadFile, File()],
        texts: Annotated[str, Form()],
    ) -> dict[str, Any]:
        try:
            raw_texts = json.loads(texts)
        except json.JSONDecodeError as error:
            raise HTTPException(status_code=422, detail="texts must be a JSON array") from error
        if not isinstance(raw_texts, list) or any(not isinstance(item, str) for item in raw_texts):
            raise HTTPException(status_code=422, detail="texts must be a JSON array of strings")
        decoded = await decoded_image(image)
        try:
            result = await run_worker(service.rank, decoded, tuple(raw_texts))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return {
            "model_id": result.model_id,
            "model_revision": result.model_revision,
            "device": result.device,
            "matches": [
                {
                    "text": item.text,
                    "logit": item.logit,
                    "match_probability": item.match_probability,
                    "candidate_probability": item.candidate_probability,
                    "rank": item.rank,
                }
                for item in result.matches
            ],
        }

    @app.post("/v1/embeddings/image")
    async def image_embedding(
        image: Annotated[UploadFile, File()],
    ) -> dict[str, Any]:
        decoded = await decoded_image(image)
        return _embedding_payload(await run_worker(service.embed_image, decoded))

    @app.post("/v1/embeddings/text")
    def text_embedding(request: TextEmbeddingRequest) -> dict[str, Any]:
        try:
            result = service.embed_texts(tuple(request.texts))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _embedding_payload(result)

    return app


def _embedding_payload(result: Any) -> dict[str, Any]:
    return {
        "model_id": result.model_id,
        "model_revision": result.model_revision,
        "device": result.device,
        "dimension": result.dimension,
        "embeddings": [list(vector) for vector in result.embeddings],
    }
