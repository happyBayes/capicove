"""Closed version-1 serialization for independent chunk runs."""

from __future__ import annotations

import json
from dataclasses import asdict, fields
from typing import Any

from capicove.books.chunk_config import persisted_chunk_config
from capicove.books.chunk_models import BookChunk, ChunkRun, SourceSpan


def encode_run(run: ChunkRun) -> bytes:
    return (
        json.dumps(
            {"schema_version": 1, "run": asdict(run)},
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def _closed(value: object, expected: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError("invalid closed chunk record")
    return dict(value)


def decode_run(raw: bytes) -> ChunkRun:
    """Reject incompatible versions and unknown fields before reconstructing DTOs."""
    try:
        envelope = _closed(json.loads(raw), {"schema_version", "run"})
        if type(envelope["schema_version"]) is not int or envelope["schema_version"] != 1:
            raise ValueError("unsupported chunk run version")
        value = _closed(envelope["run"], {field.name for field in fields(ChunkRun)})
        persisted_chunk_config(json.loads(value["config_json"]))
        chunks: list[BookChunk] = []
        for raw_chunk in value["chunks"]:
            chunk = _closed(raw_chunk, {field.name for field in fields(BookChunk)})
            for key in ("core_spans", "before_spans", "after_spans"):
                if not isinstance(chunk[key], list):
                    raise ValueError("chunk spans must be arrays")
                chunk[key] = tuple(
                    SourceSpan(**_closed(span, {field.name for field in fields(SourceSpan)}))
                    for span in chunk[key]
                )
            for key in ("heading_path", "content_types", "warnings"):
                if not isinstance(chunk[key], list):
                    raise ValueError("chunk labels must be arrays")
                chunk[key] = tuple(chunk[key])
            chunks.append(BookChunk(**chunk))
        value["chunks"] = tuple(chunks)
        value["exclusions"] = tuple(tuple(item) for item in value["exclusions"])
        return ChunkRun(**value)
    except (KeyError, TypeError, AttributeError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid chunk run encoding") from error
