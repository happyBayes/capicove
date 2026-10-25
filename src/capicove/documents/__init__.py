"""Document parsing ports exposed by Capicove core."""

from capicove.documents.codec import (
    document_parse_result_from_json,
    document_parse_result_to_json,
)
from capicove.documents.ports import DocumentParserBackend
from capicove.documents.service import (
    DocumentBackendUnavailableError,
    DocumentInputError,
    DocumentLimitError,
    DocumentLimits,
    DocumentParserBinding,
    DocumentParsingService,
    DocumentProcessingError,
    DocumentRejectedError,
    DocumentResultError,
)


__all__ = [
    "DocumentBackendUnavailableError",
    "DocumentInputError",
    "DocumentLimitError",
    "DocumentLimits",
    "DocumentParserBackend",
    "DocumentParserBinding",
    "DocumentParsingService",
    "DocumentProcessingError",
    "DocumentRejectedError",
    "DocumentResultError",
    "document_parse_result_from_json",
    "document_parse_result_to_json",
]
