"""Evidence-preserving multimodal preparation using existing public capabilities."""

from __future__ import annotations

from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.runtime.knowledge_config import KnowledgeConfiguration
from capicove.runtime.media_handoff import StagedFile


_TEXT = {".txt", ".md", ".markdown"}
_IMAGE = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
_AUDIO = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
_VIDEO = {".mp4", ".mkv", ".mov", ".webm"}


def prepare_source(
    config: KnowledgeConfiguration,
    source: StagedFile,
    store: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Preserve exact derived text and original page or time coordinates."""
    suffix = source.original.suffix.lower()
    if suffix in _TEXT:
        try:
            text = source.staged.read_bytes().decode("utf-8")
        except UnicodeError:
            raise ValueError("source_must_be_utf8") from None
        return [{"text": text, "kind": "text"}], {"adapter": "utf8-v1"}
    if suffix == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(source.staged))
        if reader.is_encrypted or not 1 <= len(reader.pages) <= 200:
            raise ValueError("pdf_page_limit_or_encryption")
        if any(page.images.keys() for page in reader.pages):
            return prepare_scanned_pdf(config, source)
        return _document(config, source, suffix)
    if suffix in _IMAGE:
        return _document(config, source, suffix)
    if suffix in _VIDEO:
        return prepare_video(config, source)
    if suffix in _AUDIO:
        return _audio(config, source)
    raise ValueError("unsupported_knowledge_source_type")


def _document(
    config: KnowledgeConfiguration,
    source: StagedFile,
    suffix: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.models is None:
        raise ValueError("document_models_not_configured")
    from capicove.contracts import DocumentParseRequest, LocalPathSource
    from capicove.contracts.graph_build import GraphScope
    from capicove.graph_build.document_preparation import DocumentGraphConfiguration
    from capicove.model import ModelFactory, load_configuration
    from capicove.processing.image import ImageSource

    with ModelFactory(load_configuration(config.models)) as factory:
        preparation = DocumentGraphConfiguration(
            model_factory=factory, parser_binding="knowledge.parser", ocr_binding="knowledge.ocr"
        )
        scope = GraphScope(config.workspace_id, "knowledge-private")
        if suffix == ".pdf":
            parsed, representation = preparation.prepare_pdf(
                DocumentParseRequest(
                    LocalPathSource(str(source.staged)),
                    source.sha256,
                    "application/pdf",
                    "book-balanced",
                    2,
                ),
                document_ref="source:" + source.sha256,
                scope=scope,
            )
            segments = [
                {
                    "text": block.exact_text,
                    "kind": "pdf",
                    "page_index": page.page_index,
                    "block_id": block.block_id,
                }
                for page in parsed.pages
                for block in page.blocks
            ]
            return segments, {
                "adapter": "paddle-document",
                "receipt": asdict(parsed.receipt),
                "representation_ref": representation.representation_ref,
                "warnings": list(parsed.warnings),
            }
        representation = preparation.prepare_image(
            ImageSource(path=str(source.staged)),
            document_ref="source:" + source.sha256,
            scope=scope,
        )
        if representation.source_sha256 != source.sha256:
            raise ValueError("ocr_source_identity_mismatch")
        return [{"text": representation.text, "kind": "image", "page_index": 0}], {
            "adapter": "image-ocr",
            "representation_ref": representation.representation_ref,
            "processing": list(representation.issue_codes),
        }


def _audio(
    config: KnowledgeConfiguration, source: StagedFile
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if config.audio is None:
        raise ValueError("audio_runtime_not_configured")
    from capicove.processing.audio.local_runtime import transcribe_local
    from capicove.runtime.knowledge_native import native_runtime

    with native_runtime(config, models=True) as (executables, assets, raw):
        transcript = transcribe_local(
            source.staged,
            model=assets["model"],
            vad_model=assets["vad_model"],
            whisper_cli=executables["whisper_cli"],
            whisper_vad_cli=executables["whisper_vad_cli"],
            ffmpeg=executables["ffmpeg"],
            ffprobe=executables["ffprobe"],
            backend_version=raw["backend_version"],
            deadline=datetime.now(UTC) + timedelta(seconds=300),
        )
    return [
        {
            "text": segment.text,
            "kind": "audio",
            "start_seconds": segment.start_seconds,
            "end_seconds": segment.end_seconds,
        }
        for segment in transcript.segments
    ], {
        "adapter": "whisper.cpp",
        "receipt": asdict(transcript.lineage),
        "language": transcript.language,
        "warnings": list(transcript.warnings),
    }


def prepare_scanned_pdf(
    config: KnowledgeConfiguration,
    source: StagedFile,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """OCR bounded rendered pages without claiming exact PDF text-layer extraction."""
    import io

    import pypdfium2 as pdfium  # type: ignore[import-untyped]

    from capicove.contracts.graph_build import GraphScope
    from capicove.graph_build.document_preparation import DocumentGraphConfiguration
    from capicove.model import ModelFactory, load_configuration
    from capicove.processing.image import ImageSource

    if config.models is None:
        raise ValueError("document_models_not_configured")
    segments = []
    receipts = []
    with ModelFactory(load_configuration(config.models)) as factory:
        preparation = DocumentGraphConfiguration(model_factory=factory, ocr_binding="knowledge.ocr")
        with pdfium.PdfDocument(str(source.staged)) as document:
            if not 1 <= len(document) <= 200:
                raise ValueError("pdf_page_limit_exceeded")
            for index in range(len(document)):
                page = document[index]
                try:
                    width, height = page.get_size()
                    if width <= 0 or height <= 0 or width * height * 4 > 12_000_000:
                        raise ValueError("pdf_page_pixel_limit_exceeded")
                    bitmap = page.render(scale=2)
                    try:
                        rendered = bitmap.to_pil()
                        encoded = io.BytesIO()
                        rendered.save(encoded, "PNG")
                        representation = preparation.prepare_image(
                            ImageSource(data=encoded.getvalue()),
                            document_ref=f"source:{source.sha256}:page:{index}",
                            scope=GraphScope(config.workspace_id, "knowledge-private"),
                        )
                        segments.append(
                            {"text": representation.text, "kind": "pdf", "page_index": index}
                        )
                        receipts.append(
                            {
                                "page_index": index,
                                "render_sha256": representation.source_sha256,
                                "representation_ref": representation.representation_ref,
                                "processing": list(representation.issue_codes),
                            }
                        )
                    finally:
                        bitmap.close()
                finally:
                    page.close()
    return segments, {
        "adapter": "pdf-page-ocr-v1",
        "render_scale": 2,
        "pages": receipts,
        "warnings": ["ocr_text_requires_source_verification"],
    }


def prepare_video(
    config: KnowledgeConfiguration, source: StagedFile
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Prepare speech and bounded sampled-frame OCR with explicit coverage."""
    import json
    import math

    # Only verified local media executables are invoked, without a shell.
    import subprocess  # nosec B404
    import tempfile

    from capicove.runtime.knowledge_native import native_runtime

    if config.models is None:
        raise ValueError("video_ocr_not_configured")
    from capicove.runtime.media_handoff import is_video_container

    with source.staged.open("rb") as stream:
        if not is_video_container(stream.read(192)):
            raise ValueError("invalid_video_container")
    demuxer = "mov" if source.original.suffix.lower() in {".mp4", ".mov"} else "matroska"
    segments = []
    audio_lineage = None
    with native_runtime(config, models=False) as (executables, _, _raw):
        # Pinned executable, fixed arguments, and explicit local protocols.
        probe = subprocess.run(  # nosec B603
            [
                executables["ffprobe"],
                "-v",
                "error",
                "-protocol_whitelist",
                "file,pipe",
                "-f",
                demuxer,
                "-show_entries",
                "format=duration:stream=codec_type",
                "-of",
                "json",
                str(source.staged),
            ],
            capture_output=True,
            check=True,
            timeout=30,
        )
        if len(probe.stdout) > 65536:
            raise ValueError("invalid_video_inventory")
        inventory = json.loads(probe.stdout)
        duration = float(inventory["format"]["duration"])
        kinds = [stream["codec_type"] for stream in inventory["streams"]]
        if (
            not math.isfinite(duration)
            or not 0 < duration <= 3600
            or kinds.count("video") != 1
            or kinds.count("audio") > 1
        ):
            raise ValueError("video_duration_or_stream_selection_unsupported")
        with tempfile.TemporaryDirectory(prefix="frames-", dir=source.staged.parent) as directory:
            frame = Path(directory) / "frame.png"
            receipts = []
            for second in range(0, math.ceil(duration), 30):
                # Pinned ffmpeg, fixed arguments, and explicit local protocols.
                subprocess.run(  # nosec B603
                    [
                        executables["ffmpeg"],
                        "-v",
                        "error",
                        "-nostdin",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-f",
                        demuxer,
                        "-ss",
                        str(second),
                        "-i",
                        str(source.staged),
                        "-map",
                        "0:v:0",
                        "-frames:v",
                        "1",
                        "-vf",
                        "scale=1600:1600:force_original_aspect_ratio=decrease",
                        "-y",
                        str(frame),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=30,
                )
                import hashlib

                encoded = frame.read_bytes()
                staged = StagedFile(frame, frame, hashlib.sha256(encoded).hexdigest(), len(encoded))
                prepared, receipt = _document(config, staged, ".png")
                for segment in prepared:
                    segments.append(
                        {
                            "text": segment["text"],
                            "kind": "video_frame",
                            "start_seconds": float(second),
                            "end_seconds": min(float(second) + 0.001, duration),
                        }
                    )
                receipts.append({"at_seconds": second, "receipt": receipt})
            if "audio" in kinds:
                # The audio capability accepts only audio containers. Extract the
                # selected track through the verified video decoder first.
                audio_path = Path(directory) / "track.wav"
                # Pinned ffmpeg, fixed arguments, and explicit local protocols.
                subprocess.run(  # nosec B603
                    [
                        executables["ffmpeg"],
                        "-v",
                        "error",
                        "-nostdin",
                        "-protocol_whitelist",
                        "file,pipe",
                        "-f",
                        demuxer,
                        "-i",
                        str(source.staged),
                        "-map",
                        "0:a:0",
                        "-vn",
                        "-ar",
                        "16000",
                        "-ac",
                        "1",
                        "-c:a",
                        "pcm_s16le",
                        "-y",
                        str(audio_path),
                    ],
                    capture_output=True,
                    check=True,
                    timeout=60,
                )
                audio_staged = StagedFile(
                    audio_path, audio_path, "0" * 64, audio_path.stat().st_size
                )
                audio, audio_lineage = _audio(config, audio_staged)
                segments.extend(audio)
    return segments, {
        "adapter": "video-evidence-v1",
        "coverage": "sampled_frames_and_audio",
        "frame_interval_seconds": 30,
        "duration_seconds": duration,
        "frame_receipts": receipts,
        "audio": audio_lineage,
        "warnings": ["visual_coverage_is_sampled_ocr_not_scene_understanding"],
    }
