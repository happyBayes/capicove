"""Closed caller-owned submissions for Book chunk-result tasks."""

from __future__ import annotations

import json
from typing import Any

from capicove.books.chunk_models import SourceSpan


_REQUIRED = frozenset(
    {
        "schema_version",
        "book_id",
        "run_id",
        "task_id",
        "idempotency_key",
        "dependency_result_ids",
        "abstract",
        "key_points",
        "claims",
    }
)
_CITATION_FIELDS = frozenset({"block_id", "page_index", "start_char", "end_char", "text_sha256"})


def canonical_json(value: object) -> bytes:
    """Encode one closed result object deterministically."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()


def _bounded_text(value: object, name: str, maximum: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"invalid {name}")
    try:
        size = len(value.encode("utf-8"))
    except UnicodeError as error:
        raise ValueError(f"invalid {name}") from error
    if size > maximum:
        raise ValueError(f"invalid {name}")
    return value


def validate_submission(value: object) -> dict[str, Any]:
    """Validate the closed v1 shape before any result can be persisted."""
    if (
        not isinstance(value, dict)
        or not _REQUIRED <= value.keys()
        or not set(value) <= (_REQUIRED | {"provenance"})
    ):
        raise ValueError("invalid_submission")
    if type(value["schema_version"]) is not int or value["schema_version"] != 1:
        raise ValueError("invalid_submission")
    for name in ("book_id", "run_id", "task_id"):
        _bounded_text(value[name], name, 128)
    _bounded_text(value["idempotency_key"], "idempotency_key", 128)
    _bounded_text(value["abstract"], "abstract", 16384)
    dependencies = value["dependency_result_ids"]
    if (
        not isinstance(dependencies, list)
        or len(dependencies) > 32
        or any(not isinstance(item, str) or not item for item in dependencies)
    ):
        raise ValueError("invalid_submission")
    points = value["key_points"]
    if not isinstance(points, list) or len(points) > 32:
        raise ValueError("invalid_submission")
    for point in points:
        _bounded_text(point, "key_point", 2048)
    claims = value["claims"]
    if not isinstance(claims, list) or len(claims) > 64:
        raise ValueError("invalid_submission")
    for claim in claims:
        if not isinstance(claim, dict) or set(claim) != {"text", "citations"}:
            raise ValueError("invalid_submission")
        _bounded_text(claim["text"], "claim_text", 2048)
        citations = claim["citations"]
        if not isinstance(citations, list) or not 1 <= len(citations) <= 16:
            raise ValueError("invalid_submission")
        for citation in citations:
            if not isinstance(citation, dict) or set(citation) != _CITATION_FIELDS:
                raise ValueError("invalid_submission")
            SourceSpan(**citation)
    if "provenance" in value:
        provenance = value["provenance"]
        if not isinstance(provenance, dict) or not set(provenance) <= {
            "provider_id",
            "model_revision",
            "caller_run_id",
        }:
            raise ValueError("invalid_submission")
        for field in provenance:
            _bounded_text(provenance[field], field, 256)
    try:
        payload = canonical_json(value)
    except (TypeError, UnicodeError, ValueError) as error:
        raise ValueError("invalid_submission") from error
    if len(payload) > 1024 * 1024:
        raise ValueError("invalid_submission")
    return dict(value)
