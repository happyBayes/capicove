from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime, timedelta

from capicove.books import BookStatus
from capicove.contracts import InvocationContext, PrivacyClass, ToolStatus
from capicove.tools.book import BOOK_TOOL_SPECS, BookToolHandler


class _Service:
    def status(self, book_id: str) -> BookStatus:
        return BookStatus(book_id, 2, 1, 1, 0, False)


def _context() -> InvocationContext:
    spec = BOOK_TOOL_SPECS["status"]
    return InvocationContext(
        "request",
        "trace",
        "workspace",
        "agent",
        "local",
        datetime.now(UTC) + timedelta(minutes=1),
        spec.required_permissions,
        PrivacyClass.CONFIDENTIAL,
    )


def test_book_specs_are_confidential_offline_and_cover_full_journey() -> None:
    assert set(BOOK_TOOL_SPECS) == {"prepare", "next-task", "submit", "status", "publish"}
    assert {spec.tool_id for spec in BOOK_TOOL_SPECS.values()} == {
        "capicove.book.prepare",
        "capicove.book.next-task",
        "capicove.book.submit",
        "capicove.book.status",
        "capicove.book.publish",
    }
    assert all(spec.network_access is False for spec in BOOK_TOOL_SPECS.values())
    assert all(spec.privacy_class is PrivacyClass.CONFIDENTIAL for spec in BOOK_TOOL_SPECS.values())
    prepare = BOOK_TOOL_SPECS["prepare"]
    assert prepare.contract_version == 2
    properties = prepare.input_schema["properties"]
    assert isinstance(properties, Mapping)
    assert properties["profile_version"] == {
        "type": "integer",
        "minimum": 1,
        "default": 2,
    }
    assert properties["language_hints"] == {
        "type": "array",
        "items": {"type": "string", "minLength": 1, "maxLength": 64},
        "maxItems": 16,
        "uniqueItems": True,
        "default": [],
    }


def test_status_tool_returns_canonical_progress_envelope() -> None:
    handler = BookToolHandler("status", service_factory=lambda workspace: _Service())

    result = handler(
        _context(),
        {"workspace": "/authorized/books", "book_id": "book:" + "1" * 24},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert isinstance(result.result, Mapping)
    assert result.result["completed_tasks"] == 1
    assert result.result["publishable"] is False
