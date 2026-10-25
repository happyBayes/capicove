"""Prepare authorized document evidence through public parser and image ports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from capicove.contracts import DocumentPageStatus, DocumentParseRequest, DocumentParseResult
from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.representations import BlockSpan, TextRepresentation
from capicove.processing.image import ImageOcr, ImageSource
from capicove.processing.image.ocr_configuration import ImageOcrConfiguration


if TYPE_CHECKING:
    from capicove.model import ModelFactory
    from capicove.processing.image.ocr_configuration import ConfiguredImageOcrProvider


@dataclass(frozen=True, slots=True)
class DocumentGraphConfiguration:
    """Choose Paddle by default without changing the text-only graph core."""

    parser_id: str = "paddleocr"
    model_root: Path | None = None
    device: str = "cpu"
    image_ocr_provider: str = "paddleocr"
    model_factory: ModelFactory | None = None
    parser_binding: str = "graph.parser"
    ocr_binding: str = "graph.ocr"

    def prepare_pdf(
        self, request: DocumentParseRequest, *, document_ref: str, scope: GraphScope
    ) -> tuple[DocumentParseResult, TextRepresentation]:
        """Parse caller-authorized PDF bytes and retain the complete result."""
        from capicove.plugins import load_document_parser_plugin

        if self.model_factory is not None:
            with self.model_factory.document_parser(self.parser_binding) as backend:
                result = backend.parse(request)
        else:
            if self.model_root is None:
                raise ValueError("document graph preparation requires explicit model_root")
            plugin = load_document_parser_plugin(self.parser_id)
            backend = plugin.create({"model_root": str(self.model_root), "device": self.device})
            result = backend.parse(request)
        if result.source_sha256 != request.source_sha256:
            raise ValueError("parser source identity mismatch")
        if (result.receipt.profile_id, result.receipt.profile_version) != (
            request.profile_id,
            request.profile_version,
        ):
            raise ValueError("parser profile identity mismatch")
        return result, parsed_document_representation(document_ref, result, scope)

    def prepare_image(
        self, source: ImageSource, *, document_ref: str, scope: GraphScope
    ) -> TextRepresentation:
        """Prepare exact OCR text; the caller retains the original image evidence."""
        if self.model_factory is not None:
            with self.model_factory.image_ocr(self.ocr_binding) as configured_provider:
                return self._prepare_image(
                    configured_provider, source, document_ref=document_ref, scope=scope
                )
        provider = ImageOcrConfiguration(
            self.image_ocr_provider, self.model_root, self.device
        ).create_provider()
        if provider is None:
            raise ValueError("document graph image OCR is disabled")
        return self._prepare_image(provider, source, document_ref=document_ref, scope=scope)

    def _prepare_image(
        self,
        provider: ConfiguredImageOcrProvider,
        source: ImageSource,
        *,
        document_ref: str,
        scope: GraphScope,
    ) -> TextRepresentation:
        """Keep OCR and fingerprint access inside the provider's lease."""
        result = ImageOcr(provider).recognize((source,))[0]
        fingerprint = provider.fingerprint
        return TextRepresentation(
            document_ref,
            scope,
            result.source_hash,
            result.full_text,
            (BlockSpan("ocr:0", 0, len(result.full_text), 0, "paragraph", ()),),
            ("image-ocr:" + fingerprint.sha256,),
        )


def parsed_document_representation(
    document_ref: str, result: DocumentParseResult, scope: GraphScope
) -> TextRepresentation:
    """Project exact block text, pages and parser lineage without normalization.

    Failed pages are rejected. The retained DocumentParseResult remains the
    authority for boxes and OCR provenance; graph character spans address this
    derived text representation, rather than claiming PDF character offsets.
    """
    text = ""
    spans: list[BlockSpan] = []
    for page in result.pages:
        if page.status is not DocumentPageStatus.SUCCEEDED:
            raise ValueError("document graph preparation requires successful pages")
        for block in page.blocks:
            if spans:
                start = len(text)
                text += "\n\n"
                spans.append(
                    BlockSpan(
                        f"separator:{len(spans)}",
                        start,
                        len(text),
                        page.page_index,
                        "separator",
                        (),
                    )
                )
            start = len(text)
            text += block.exact_text
            spans.append(
                BlockSpan(
                    "document:" + block.block_id, start, len(text), page.page_index, "paragraph", ()
                )
            )
    return TextRepresentation(
        document_ref,
        scope,
        result.source_sha256,
        text,
        tuple(spans),
        (
            "document-parse:" + result.representation_sha256,
            "parser-processing:" + result.receipt.processing_sha256,
        ),
    )
