"""Versioned exact Markdown representations, without Unicode composition."""

import hashlib
import json
from dataclasses import dataclass


MAX_SOURCE_BYTES = 16 * 1024 * 1024
MAX_TEXT_CHARS = 2_000_000


@dataclass(frozen=True, slots=True)
class MarkdownRepresentation:
    """Immutable derived text bound to the exact original source."""

    source_sha256: str
    representation_sha256: str
    text: str
    encoded: bytes


def markdown_representation(raw: bytes) -> MarkdownRepresentation:
    """Decode strict UTF-8, strip one BOM and normalize only line endings."""
    if len(raw) > MAX_SOURCE_BYTES:
        raise ValueError("capacity_exceeded: source bytes")
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("evidence_unavailable: invalid UTF-8") from error
    text = text.removeprefix("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    if len(text) > MAX_TEXT_CHARS:
        raise ValueError("capacity_exceeded: representation text")
    digest = hashlib.sha256(raw).hexdigest()
    encoded = json.dumps(
        {"source_sha256": digest, "transform": "wiki-markdown-v1", "text": text},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return MarkdownRepresentation(digest, hashlib.sha256(encoded).hexdigest(), text, encoded)
