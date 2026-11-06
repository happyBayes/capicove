"""Shared helpers for stack adapters: label matching, corpora and fingerprints."""

from __future__ import annotations

import dataclasses
import enum
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime, timedelta
from pathlib import Path, PurePosixPath
from typing import Any

from capicove.evaluation.harness import EvalDataset, EvalQuery, EvaluationError, sha256_json


_DOCUMENT_PATH = re.compile(r"^[a-z0-9][a-z0-9_./-]{0,200}\.(md|markdown)$")


def jsonable(value: Any) -> Any:
    """Project dataclasses, enums, tuples and paths onto canonical JSON values."""
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: jsonable(getattr(value, item.name)) for item in dataclasses.fields(value)
        }
    if isinstance(value, enum.Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(item) for item in value]
    if isinstance(value, (frozenset, set)):
        return sorted(jsonable(item) for item in value)
    if isinstance(value, Path):
        return str(value)
    return value


def fingerprint(value: Any) -> str:
    """Return the canonical SHA-256 of a dataclass or JSON-like policy value."""
    return sha256_json(jsonable(value))


def profile_identity(profile_id: str, version: int, policy: Any) -> dict[str, Any]:
    """Describe a query-time policy by identifier, version and fingerprint."""
    return {"id": profile_id, "version": version, "fingerprint": fingerprint(policy)}


def text_labels(query: EvalQuery, document: str, text: str | None) -> tuple[str, ...]:
    """Return labels whose document matches and whose required quote is present."""
    satisfied = []
    for label in query.payload.get("labels", ()):
        if label["document"] != document:
            continue
        if label["contains"] is None or (text is not None and label["contains"] in text):
            satisfied.append(label["label"])
    return tuple(satisfied)


def union_labels(groups: Iterable[Iterable[str]]) -> tuple[str, ...]:
    """Merge label groups into one sorted tuple."""
    return tuple(sorted({label for group in groups for label in group}))


def corpus_documents(dataset: EvalDataset) -> tuple[dict[str, str], ...]:
    """Validate authored Markdown documents as ``id``/``path``/``text`` rows."""
    rows = dataset.corpus.get("documents")
    if not isinstance(rows, list) or not rows:
        raise EvaluationError("dataset corpus must contain documents")
    seen: set[str] = set()
    paths: set[str] = set()
    documents = []
    for row in rows:
        if not isinstance(row, dict) or set(row) - {"graph"} != {"id", "path", "text"}:
            raise EvaluationError("corpus documents must contain id, path and text")
        if not isinstance(row["path"], str) or not _DOCUMENT_PATH.fullmatch(row["path"]):
            raise EvaluationError("corpus document paths must be relative Markdown names")
        if ".." in PurePosixPath(row["path"]).parts:
            raise EvaluationError("corpus document paths must stay inside the corpus")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise EvaluationError("corpus documents must contain text")
        if row["id"] in seen or row["path"] in paths:
            raise EvaluationError("corpus document IDs and paths must be unique")
        seen.add(row["id"])
        paths.add(row["path"])
        documents.append({"id": row["id"], "path": row["path"], "text": row["text"]})
    return tuple(documents)


def write_documents(root: Path, documents: tuple[dict[str, str], ...]) -> None:
    """Write authored documents beneath a private corpus root."""
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    for document in documents:
        path = root / document["path"]
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        path.write_text(document["text"], encoding="utf-8")


def deadline(seconds: float = 300.0) -> datetime:
    """Return an absolute UTC deadline for runtime calls."""
    return datetime.now(UTC) + timedelta(seconds=seconds)


def host_config(workspace: Path, capabilities: Mapping[str, Any], tools: list[str]) -> Any:
    """Compose an in-memory offline version-2 workspace host configuration."""
    import json

    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config

    body = {
        "schema_version": 2,
        "server_name": "capicove-eval",
        "workspace_id": "evaluation",
        "actor_id": "evaluator",
        "environment_id": "local",
        "workspace_root": str(workspace),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": tools,
        "trace_store": None,
        "capabilities": dict(capabilities),
        "network": {"enabled": False, "source_hosts": []},
    }
    config = parse_mcp_host_config(json.dumps(body).encode())
    if not isinstance(config, WorkspaceHostConfiguration):
        raise EvaluationError("evaluation host configuration is invalid")
    return config


def invocation_context() -> Any:
    """Return a read-only evaluation invocation context."""
    from capicove.contracts import InvocationContext

    return InvocationContext(
        request_id="eval",
        trace_id="eval",
        workspace_id="evaluation",
        actor_id="evaluator",
        environment_id="local",
        deadline=deadline(),
        permissions=frozenset({"evidence:read", "evidence:search"}),
    )


__all__ = [
    "corpus_documents",
    "deadline",
    "fingerprint",
    "host_config",
    "invocation_context",
    "jsonable",
    "profile_identity",
    "text_labels",
    "union_labels",
    "write_documents",
]


def render_image(spec: Mapping[str, Any]) -> bytes:
    """Render an authored synthetic PNG deterministically from a closed spec.

    ``color`` names a palette color; ``variant`` seeds a block pattern so images
    of one color have distinct bytes; ``size`` is the square side in pixels.
    """
    import hashlib
    import io

    from PIL import Image

    from capicove.evaluation.fakes import COLOR_WORDS

    if set(spec) != {"color", "variant", "size"} or spec["color"] not in COLOR_WORDS:
        raise EvaluationError("image specs must name a palette color, variant and size")
    if type(spec["variant"]) is not int or type(spec["size"]) is not int:
        raise EvaluationError("image variant and size must be integers")
    if not 16 <= spec["size"] <= 512:
        raise EvaluationError("image size must be from 16 to 512 pixels")
    base = COLOR_WORDS[spec["color"]]
    seed = hashlib.sha256(f"{spec['color']}:{spec['variant']}".encode()).digest()
    image = Image.new("RGB", (spec["size"], spec["size"]), base)
    block = max(1, spec["size"] // 8)
    for index in range(64):
        shade = (seed[index % 32] - 128) // 4
        x, y = (index % 8) * block, (index // 8) * block
        color = tuple(max(0, min(255, channel + shade)) for channel in base)
        image.paste(color, (x, y, x + block, y + block))
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=6)
    return buffer.getvalue()


def corpus_images(dataset: EvalDataset) -> tuple[dict[str, Any], ...]:
    """Validate authored image rows as ``id``/``path``/``spec``."""
    rows = dataset.corpus.get("images", [])
    if not isinstance(rows, list):
        raise EvaluationError("corpus images must be a list")
    images = []
    seen: set[str] = set()
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"id", "path", "spec"}:
            raise EvaluationError("corpus images must contain id, path and spec")
        path = row["path"]
        if (
            not isinstance(path, str)
            or not re.fullmatch(r"^[a-z0-9][a-z0-9_./-]{0,200}\.png$", path)
            or ".." in PurePosixPath(path).parts
        ):
            raise EvaluationError("corpus image paths must be relative PNG names")
        if row["id"] in seen:
            raise EvaluationError("corpus image IDs must be unique")
        seen.add(row["id"])
        images.append(dict(row))
    return tuple(images)
