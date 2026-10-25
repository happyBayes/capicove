"""Lazy local PP-StructureV3 backend."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, cast

from capicove.contracts import (
    DocumentOutlineEntry,
    DocumentParseRequest,
    DocumentParseResult,
    DocumentParserReceipt,
    LocalPathSource,
)
from capicove.documents import DocumentBackendUnavailableError, DocumentRejectedError
from capicove_processor_paddleocr.models import (
    load_book_balanced_manifest,
    verify_installed_models,
)
from capicove_processor_paddleocr.pdf_metadata import extract_pdf_metadata
from capicove_processor_paddleocr.runtime import (
    enforce_offline_runtime_policy,
    import_verified_runtime_module,
    resolve_runtime_identity,
    verify_loaded_runtime_modules,
)
from capicove_processor_paddleocr.translation import translate_pages


def pipeline_configuration(model_root: Path, *, device: str) -> dict[str, object]:
    """Resolve the closed local-only PP-StructureV3 configuration."""
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a nonblank string")
    manifest = load_book_balanced_manifest()
    root = model_root.expanduser().resolve()
    directories = {model.model_parameter: root / model.key for model in manifest.models}
    missing = [path.name for path in directories.values() if not path.is_dir()]
    if missing:
        raise ValueError(f"missing model directories: {', '.join(missing)}")
    configuration: dict[str, object] = dict(manifest.pipeline_configuration)
    if configuration.get("device") != "$requested_device":
        raise ValueError("model manifest device policy is invalid")
    configuration["device"] = device
    configuration.update({name: str(path) for name, path in directories.items()})
    return configuration


class PaddleOcrDocumentParserBackend:
    """Parse an authorized local PDF through an explicitly configured pipeline."""

    parser_id = "paddleocr"

    def __init__(
        self, model_root: Path, *, device: str = "cpu", enable_mkldnn: bool | None = None
    ) -> None:
        self._model_root = model_root.expanduser().resolve()
        self._configuration = pipeline_configuration(self._model_root, device=device)
        if enable_mkldnn is not None:
            if type(enable_mkldnn) is not bool:
                raise ValueError("enable_mkldnn must be a boolean")
            self._configuration["enable_mkldnn"] = enable_mkldnn

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        """Parse without model acquisition or network fallback."""
        if not isinstance(request.source, LocalPathSource):
            raise ValueError("PaddleOCR parser requires an authorized local path")
        manifest = load_book_balanced_manifest()
        if (
            request.profile_id != manifest.profile_id
            or request.profile_version != manifest.profile_version
        ):
            raise ValueError("request profile does not match the configured PaddleOCR profile")
        try:
            enforce_offline_runtime_policy()
            runtime_identity = resolve_runtime_identity(manifest.inference_engine)
            verify_loaded_runtime_modules()
            verify_installed_models(manifest, self._model_root)
        except (ValueError, RuntimeError, ImportError, OSError) as error:
            raise DocumentBackendUnavailableError(
                "Paddle parser assets or runtime are unavailable"
            ) from error
        path = Path(request.source.path)
        if not path.is_file():
            raise ValueError("authorized PDF path is unavailable")
        with tempfile.TemporaryDirectory(prefix="capicove-source-") as temporary:
            snapshot = Path(temporary) / "source.pdf"
            _snapshot_authorized_source(path, snapshot, request.source_sha256)
            outline, page_labels = extract_pdf_metadata(snapshot)
            verify_loaded_runtime_modules()
            raw_pages = _predict_pages(snapshot, request, self._configuration)
            verify_loaded_runtime_modules()
        page_indices = frozenset(_page_index(raw) for raw in raw_pages)
        outline = _restrict_outline(outline, page_indices)
        configuration_payload = json.dumps(
            {
                "pipeline_configuration": self._configuration,
                "runtime_identity": dict(runtime_identity),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        request_payload = json.dumps(
            {
                "source_sha256": request.source_sha256,
                "media_type": request.media_type,
                "profile_id": request.profile_id,
                "profile_version": request.profile_version,
                "language_hints": list(request.language_hints),
                "page_range": None
                if request.page_range is None
                else {
                    "start": request.page_range.start_page_index,
                    "end": request.page_range.end_page_index,
                },
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        processing_payload = request_payload + configuration_payload
        receipt = DocumentParserReceipt(
            "capicove.processor.paddleocr",
            "0.1.0",
            manifest.profile_id,
            manifest.profile_version,
            manifest.source_sha256,
            hashlib.sha256(configuration_payload).hexdigest(),
            hashlib.sha256(processing_payload).hexdigest(),
        )
        return translate_pages(
            raw_pages,
            source_sha256=request.source_sha256,
            receipt=receipt,
            outline=outline,
            page_labels=page_labels,
        )


def _snapshot_authorized_source(source: Path, destination: Path, expected_sha256: str) -> None:
    """Copy and hash the exact bytes that downstream PDF processing will read."""
    digest = hashlib.sha256()
    with source.open("rb") as input_stream, destination.open("xb") as output_stream:
        while chunk := input_stream.read(1024 * 1024):
            digest.update(chunk)
            output_stream.write(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("source hash does not match the authorized PDF")


def _predict_pages(
    path: Path, request: DocumentParseRequest, configuration: dict[str, object]
) -> tuple[Mapping[str, object], ...]:
    """Bound native inference to one requested page and discard image tensors."""
    pypdf = import_verified_runtime_module("pypdf")
    reader = pypdf.PdfReader(path)
    if reader.is_encrypted:
        raise DocumentRejectedError("encrypted PDFs are not supported")
    start, end = 0, len(reader.pages)
    if request.page_range is not None:
        start = request.page_range.start_page_index
        end = request.page_range.end_page_index
    if not 0 <= start < end <= len(reader.pages):
        raise DocumentRejectedError("requested page range exceeds PDF page bounds")
    _validate_ocr_language_hints(request.language_hints)
    pipeline: Any | None = None
    pages: list[Mapping[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="capicove-pdf-") as temporary:
        single = Path(temporary) / "page.pdf"
        for index in range(start, end):
            page = reader.pages[index]
            if _has_raster_images(page):
                raise DocumentRejectedError(
                    "book-balanced does not accept PDF page index "
                    f"{index} containing raster images without independent source-image "
                    "script validation"
                )
            embedded_text = _embedded_text(page)
            if embedded_text is None or not embedded_text.strip():
                raise DocumentRejectedError(
                    "book-balanced does not accept PDF page index "
                    f"{index} without a reliable nonblank text layer"
                )
            if _dominant_unsupported_script(embedded_text) is not None:
                raise DocumentRejectedError(
                    "book-balanced does not support the dominant PDF text script on "
                    f"page index {index}"
                )
            if pipeline is None:
                paddleocr = import_verified_runtime_module("paddleocr")
                pipeline = paddleocr.PPStructureV3(**configuration)
            writer = pypdf.PdfWriter()
            writer.add_page(page)
            writer.write(single)
            output = iter(pipeline.predict(input=str(single)))
            try:
                raw = _result_json(next(output))
            except StopIteration as error:
                raise ValueError(f"parser returned no result for PDF page index {index}") from error
            if next(output, None) is not None:
                raise ValueError("parser returned multiple results for a single PDF page")
            value = raw.get("res", raw)
            if not isinstance(value, Mapping):
                raise ValueError("PP-StructureV3 page result must be an object")
            evidence_text = _ocr_evidence_text(value)
            if not evidence_text.strip():
                raise ValueError(
                    f"parser returned no usable OCR evidence for PDF page index {index}"
                )
            if _dominant_unsupported_script(evidence_text) is not None:
                raise DocumentRejectedError(
                    "book-balanced OCR output contains a dominant unsupported script on "
                    f"page index {index}"
                )
            pages.append({**value, "page_index": index})
            del output, writer
            logging.getLogger(__name__).info(
                "Parsed PDF page index %d (%d/%d)", index, index - start + 1, end - start
            )
    return tuple(pages)


def _has_raster_images(page: object) -> bool:
    """Fail closed when pypdf cannot prove a page has no raster images."""
    try:
        images = cast(Any, page).images
        return bool(len(images))
    except (AttributeError, KeyError, TypeError, ValueError) as error:
        raise ValueError("PDF page image inventory is unavailable") from error
    except Exception as error:
        if any(
            error_type.__module__.startswith("pypdf") and error_type.__name__ == "PyPdfError"
            for error_type in type(error).__mro__
        ):
            raise ValueError("PDF page image inventory is unavailable") from error
        raise


def _embedded_text(page: object) -> str | None:
    """Read an optional valid text layer without making it authoritative."""
    extract_text = getattr(page, "extract_text", None)
    if not callable(extract_text):
        return None
    try:
        value = extract_text()
    except (AttributeError, KeyError, TypeError, ValueError):
        return None
    except Exception as error:
        if any(
            error_type.__module__.startswith("pypdf") and error_type.__name__ == "PyPdfError"
            for error_type in type(error).__mro__
        ):
            return None
        raise
    if not isinstance(value, str) or "\ufffd" in value:
        return None
    if any(
        character not in "\t\n\r" and unicodedata.category(character) in {"Cc", "Cs", "Co", "Cn"}
        for character in value
    ):
        return None
    return value


def _dominant_unsupported_script(value: str | None) -> str | None:
    """Detect a dominant script that this parser profile must reject."""
    if value is None:
        return None
    letters = 0
    unsupported_letters = 0
    for character in value:
        category = unicodedata.category(character)
        if not category.startswith("L"):
            continue
        letters += 1
        name = unicodedata.name(character, "")
        if not (
            "LATIN" in name or name.startswith("CJK ") or "HIRAGANA" in name or "KATAKANA" in name
        ):
            unsupported_letters += 1
    if unsupported_letters * 2 <= letters:
        return None
    return value


def _validate_ocr_language_hints(hints: tuple[str, ...]) -> None:
    supported = {"en", "ja", "zh"}
    unsupported = sorted(
        hint
        for hint in hints
        if hint.casefold().replace("_", "-").split("-", 1)[0] not in supported
    )
    if unsupported:
        raise DocumentRejectedError(f"unsupported OCR language hint: {', '.join(unsupported)}")


def _ocr_evidence_text(value: Mapping[str, object]) -> str:
    """Collect text that PP-StructureV3 may promote into exact page evidence."""
    texts: list[str] = []
    blocks = value.get("parsing_res_list")
    if isinstance(blocks, Sequence) and not isinstance(blocks, (str, bytes)):
        for block in blocks:
            if isinstance(block, Mapping):
                content = block.get("block_content")
                if isinstance(content, str) and content:
                    texts.append(content)
    overall = value.get("overall_ocr_res")
    if isinstance(overall, Mapping):
        recognized = overall.get("rec_texts")
        if isinstance(recognized, Sequence) and not isinstance(recognized, (str, bytes)):
            texts.extend(text for text in recognized if isinstance(text, str) and text)
    return "\n".join(texts)


def _result_json(value: object) -> Mapping[str, object]:
    raw: Any = getattr(value, "json", value)
    if not isinstance(raw, Mapping):
        raise ValueError("PP-StructureV3 returned an unsupported result")
    return dict(raw)


def select_requested_pages(
    raw_pages: tuple[Mapping[str, object], ...],
    *,
    start: int,
    end: int,
) -> tuple[Mapping[str, object], ...]:
    """Select a half-open request range while retaining original page indices."""
    return tuple(raw for raw in raw_pages if start <= _page_index(raw) < end)


def _page_index(raw: Mapping[str, object]) -> int:
    value = raw.get("res", raw)
    if not isinstance(value, Mapping):
        raise ValueError("PP-StructureV3 page result must be an object")
    page_index = value.get("page_index")
    if isinstance(page_index, bool) or not isinstance(page_index, int):
        raise ValueError("PP-StructureV3 page_index must be an integer")
    return page_index


def _restrict_outline(
    entries: tuple[DocumentOutlineEntry, ...],
    page_indices: frozenset[int],
) -> tuple[DocumentOutlineEntry, ...]:
    restricted: list[DocumentOutlineEntry] = []
    for entry in entries:
        children = _restrict_outline(entry.children, page_indices)
        if entry.page_index not in page_indices:
            restricted.extend(children)
            continue
        restricted.append(
            DocumentOutlineEntry(
                entry.title,
                entry.level,
                entry.page_index,
                children,
            )
        )
    return tuple(restricted)


__all__ = [
    "PaddleOcrDocumentParserBackend",
    "pipeline_configuration",
    "select_requested_pages",
]
