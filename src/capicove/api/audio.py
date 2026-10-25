"""Bounded multipart transcription endpoint over the public audio provider."""

from __future__ import annotations

import hashlib
import re
import tempfile
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path
from threading import Lock

import anyio
from fastapi import FastAPI, Request
from starlette.datastructures import UploadFile
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse, PlainTextResponse, Response
from starlette.types import Message

from capicove.api.model_http import authenticate, error, run_worker
from capicove.model.audio import AudioTranscriptionProvider
from capicove.processing.audio.models import AudioFailureCode
from capicove.processing.audio.ports import AudioValidationError
from capicove.processing.audio.transcription import AudioTranscriptionError


def create_app(
    provider: AudioTranscriptionProvider,
    *,
    model: str,
    api_key: str | None = None,
    max_upload_bytes: int = 25 * 1024 * 1024,
    timeout_seconds: int = 300,
    temp_directory: Path | None = None,
) -> FastAPI:
    """Own one ASR provider and admit one upload through inference at a time."""
    if type(max_upload_bytes) is not int or not 1 <= max_upload_bytes <= 100 * 1024 * 1024:
        raise ValueError("invalid upload limit")
    if type(timeout_seconds) is not int or not 1 <= timeout_seconds <= 3600:
        raise ValueError("invalid transcription timeout")
    if not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", model):
        raise ValueError("invalid model name")
    if temp_directory is not None and (
        not temp_directory.is_absolute()
        or temp_directory.resolve() != temp_directory
        or not temp_directory.is_dir()
    ):
        raise ValueError("invalid temporary directory")
    admission = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            yield
        finally:
            await run_worker(provider.close)

    app = FastAPI(title="Capicove Audio Transcription", lifespan=lifespan)
    authenticate(app, api_key)

    @app.get("/health")
    async def health() -> dict[str, object]:
        return {"status": "ok", "models": [model]}

    @app.get("/v1/models")
    async def models() -> dict[str, object]:
        return {
            "object": "list",
            "data": [{"id": model, "object": "model", "created": 0, "owned_by": "capicove"}],
        }

    def execute(upload: UploadFile, language: str, response_format: str) -> Response:
        with tempfile.TemporaryDirectory(
            prefix="capicove-upload-", dir=temp_directory
        ) as directory:
            # Ignore the supplied filename, including extensions and traversal components.
            path = Path(directory) / "audio"
            digest = hashlib.sha256()
            size = 0
            with path.open("xb") as target:
                while chunk := upload.file.read(65536):
                    size += len(chunk)
                    if size > max_upload_bytes:
                        return error(413, "audio_too_large")
                    digest.update(chunk)
                    target.write(chunk)
            if not size:
                return error(400, "empty_audio")
            result = provider.transcribe(
                path,
                language=language,
                deadline=datetime.now(UTC) + timedelta(seconds=timeout_seconds),
            )
            if response_format == "text":
                return PlainTextResponse(result.text)
            payload: dict[str, object] = {"text": result.text}
            if response_format == "verbose_json":
                payload.update(
                    task="transcribe",
                    language=result.language,
                    duration=result.input_duration_seconds,
                    segments=[
                        {
                            "id": s.index,
                            "start": s.start_seconds,
                            "end": s.end_seconds,
                            "text": s.text,
                        }
                        for s in result.segments
                    ],
                    capicove={
                        "source_sha256": digest.hexdigest(),
                        "model_sha256": result.lineage.model_sha256,
                        "vad_model_sha256": result.lineage.vad_model_sha256,
                        "backend_version": result.lineage.backend_version,
                        "warnings": [w.value for w in result.warnings],
                    },
                )
            return JSONResponse(payload)

    @app.post(
        "/v1/audio/transcriptions",
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "multipart/form-data": {
                        "schema": {
                            "type": "object",
                            "required": ["file", "model"],
                            "additionalProperties": False,
                            "properties": {
                                "file": {"type": "string", "format": "binary"},
                                "model": {"type": "string"},
                                "language": {"type": "string", "default": "auto"},
                                "response_format": {
                                    "type": "string",
                                    "enum": ["json", "text", "verbose_json"],
                                    "default": "json",
                                },
                            },
                        }
                    }
                },
            }
        },
    )
    async def transcribe(request: Request) -> Response:
        if not admission.acquire(blocking=False):
            return error(429, "server_busy")
        size = 0

        async def receive() -> Message:
            nonlocal size
            message = await request.receive()
            size += len(message.get("body", b""))
            if size > max_upload_bytes + 65536:
                # Starlette closes partial spooled uploads for HTTPException.
                raise HTTPException(413, "request_too_large")
            return message

        bounded = Request(request.scope, receive)
        try:
            with anyio.fail_after(30):
                payload = await bounded.body()

            async def replay() -> Message:
                return {"type": "http.request", "body": payload, "more_body": False}

            with anyio.CancelScope(shield=True):
                form = await Request(request.scope, replay).form(
                    max_files=1, max_fields=3, max_part_size=1024
                )
            try:
                items = form.multi_items()
                if len(items) != len(form) or set(form) - {
                    "file",
                    "model",
                    "language",
                    "response_format",
                }:
                    return error(400, "invalid_fields")
                if form.get("model") != model:
                    return error(404, "model_not_found")
                upload = form.get("file")
                language, fmt = form.get("language", "auto"), form.get("response_format", "json")
                if (
                    not isinstance(upload, UploadFile)
                    or not isinstance(language, str)
                    or not re.fullmatch(r"auto|[a-z]{2,3}(?:-[A-Za-z]{2,8})?", language)
                    or fmt not in ("json", "text", "verbose_json")
                ):
                    return error(400, "invalid_request")
                if upload.size is not None and upload.size > max_upload_bytes:
                    return error(413, "audio_too_large")
                return await run_worker(execute, upload, language, str(fmt))
            finally:
                with anyio.CancelScope(shield=True):
                    await form.close()
        except (AudioValidationError, AudioTranscriptionError) as exc:
            status = 504 if exc.code == AudioFailureCode.TIMEOUT else 400
            if exc.code in {
                AudioFailureCode.MODEL_UNAVAILABLE,
                AudioFailureCode.MODEL_LOAD_FAILED,
                AudioFailureCode.DECODER_UNAVAILABLE,
                AudioFailureCode.BACKEND_UNAVAILABLE,
            }:
                status = 503
            return error(status, exc.code.value.lower())
        except HTTPException as exc:
            return error(exc.status_code, "invalid_upload")
        except TimeoutError:
            return error(408, "upload_timeout")
        except ValueError:
            return error(400, "invalid_audio")
        except Exception:
            return error(500, "transcription_failed")
        finally:
            admission.release()

    return app
