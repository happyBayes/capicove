"""Ports consumed by document parsing orchestration."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from capicove.contracts import DocumentParseRequest, DocumentParseResult


@runtime_checkable
class DocumentParserBackend(Protocol):
    """Parse authorized immutable document sources."""

    @property
    def parser_id(self) -> str:
        """Return the stable parser implementation identifier."""

    def parse(self, request: DocumentParseRequest, /) -> DocumentParseResult:
        """Parse without acquiring models or source data.

        Backends raise DocumentRejectedError for explicitly unsupported inputs
        and DocumentBackendUnavailableError for assets/runtime failures. Unknown
        exceptions are processing failures, never implicit input rejection.
        """


__all__ = ["DocumentParserBackend"]
