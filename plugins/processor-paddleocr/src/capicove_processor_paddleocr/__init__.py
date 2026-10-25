"""PP-StructureV3 document parser plugin for Capicove."""

from capicove_processor_paddleocr.image_ocr import PaddleImageOcrProvider
from capicove_processor_paddleocr.plugin import paddleocr_document_parser_plugin


__all__ = ["PaddleImageOcrProvider", "paddleocr_document_parser_plugin"]
