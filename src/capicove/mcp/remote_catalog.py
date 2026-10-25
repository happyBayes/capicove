"""SDK-independent bounded catalog pages with authority-bound signed cursors."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import time
from collections.abc import Mapping
from dataclasses import astuple, dataclass
from typing import Any

from capicove.contracts.remote import AuthorityKey
from capicove.mcp.bindings import McpToolBinding, output_schema, validate_bindings


MAX_PAGE_BYTES = 512 * 1024
# Reserve framing, metadata, a <=2048-character cursor and the host's bounded
# request ID. Entry bytes alone are not a complete response-size bound.
PAGE_ENVELOPE_RESERVE = 8192
MAX_CATALOG_BYTES = 16 * 1024 * 1024
# Wire name of the always-offered remote readiness Tool (`capicove.remote.readiness`).
READINESS_MCP_NAME = "capicove_remote_readiness"
# Wire names of the remote control-plane Tools (`capicove.tools.remote_control`).
REMOTE_CONTROL_MCP_NAMES = {
    "capicove.artifact.begin-upload": "capicove_artifact_begin_upload",
    "capicove.artifact.write-chunk": "capicove_artifact_write_chunk",
    "capicove.artifact.upload-status": "capicove_artifact_upload_status",
    "capicove.artifact.abort-upload": "capicove_artifact_abort_upload",
    "capicove.artifact.commit-upload": "capicove_artifact_commit_upload",
    "capicove.artifact.stat": "capicove_artifact_stat",
    "capicove.artifact.read-chunk": "capicove_artifact_read_chunk",
    "capicove.artifact.delete": "capicove_artifact_delete",
    "capicove.operation.status": "capicove_operation_status",
    "capicove.operation.result": "capicove_operation_result",
    "capicove.operation.cancel": "capicove_operation_cancel",
}


def _plain(value: object) -> Any:
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        _plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()


def catalog_entry(binding: McpToolBinding) -> dict[str, Any]:
    """Project one canonical Tool/version without loading the optional MCP SDK."""
    schemas = {"input": binding.spec.input_schema, "output": output_schema(binding.spec)}
    return {
        "name": binding.name,
        "description": binding.spec.description,
        "inputSchema": _plain(schemas["input"]),
        "outputSchema": _plain(schemas["output"]),
        "annotations": binding.annotations(),
        "_meta": {
            "capicove": {
                "tool_id": binding.spec.tool_id,
                "contract_version": binding.spec.contract_version,
                "schema_sha256": hashlib.sha256(_canonical(schemas)).hexdigest(),
            }
        },
    }


@dataclass(frozen=True, slots=True)
class CatalogPage:
    """Pin a catalog slice and optional continuation to its complete digest."""

    bindings: tuple[McpToolBinding, ...]
    next_cursor: str | None
    catalog_digest: str


class CatalogPager:
    """Issue short-lived cursors that cannot cross sessions or policy ceilings."""

    def __init__(self, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or not 32 <= len(signing_key) <= 128:
            raise ValueError("An explicit strong ephemeral catalog signing key is required")
        self._key = signing_key

    def _encode(self, payload: dict[str, object]) -> str:
        raw = _canonical(payload)
        signature = hmac.digest(self._key, raw, "sha256")
        return base64.urlsafe_b64encode(raw + signature).rstrip(b"=").decode("ascii")

    def _decode(self, cursor: str) -> dict[str, Any]:
        try:
            if not isinstance(cursor, str) or not 1 <= len(cursor) <= 2048:
                raise ValueError("Invalid catalog cursor")
            raw = base64.b64decode(cursor + "=" * (-len(cursor) % 4), altchars=b"-_", validate=True)
            payload, signature = raw[:-32], raw[-32:]
            if len(signature) != 32 or not hmac.compare_digest(
                signature, hmac.digest(self._key, payload, "sha256")
            ):
                raise ValueError("Invalid catalog cursor")
            value = json.loads(payload)
            if not isinstance(value, dict) or set(value) != {"v", "binding", "offset", "expires"}:
                raise ValueError("Invalid catalog cursor")
            if (
                type(value["v"]) is not int
                or value["v"] != 1
                or type(value["offset"]) is not int
                or value["offset"] <= 0
                or type(value["expires"]) is not int
            ):
                raise ValueError("Invalid catalog cursor")
            return value
        except (ValueError, TypeError, UnicodeError):
            raise ValueError("Invalid catalog cursor") from None

    def page(
        self,
        bindings: tuple[McpToolBinding, ...],
        *,
        authority: AuthorityKey,
        session_id: str,
        permissions: frozenset[str],
        cursor: str | None = None,
    ) -> CatalogPage:
        """Return at most fifty Tools and enforce a separate metadata byte budget."""
        if (
            not isinstance(authority, AuthorityKey)
            or not isinstance(session_id, str)
            or not 1 <= len(session_id) <= 256
            or not isinstance(permissions, frozenset)
            or any(not isinstance(item, str) for item in permissions)
        ):
            raise ValueError("Complete catalog authority, session and ceiling required")
        if not isinstance(bindings, tuple) or len(bindings) > 4096:
            raise ValueError("Catalog exceeds its entry limit")
        if bindings:
            validate_bindings(bindings)
        ordered = tuple(sorted(bindings, key=lambda item: item.name))
        digest = hashlib.sha256()
        sizes: list[int] = []
        total = 0
        for item in ordered:
            raw = _canonical(catalog_entry(item))
            if len(raw) > MAX_PAGE_BYTES - PAGE_ENVELOPE_RESERVE:
                raise ValueError("One catalog entry exceeds the page byte limit")
            total += len(raw)
            if total > MAX_CATALOG_BYTES:
                raise ValueError("Catalog exceeds its metadata byte limit")
            digest.update(len(raw).to_bytes(8, "big"))
            digest.update(raw)
            sizes.append(len(raw))
        catalog_digest = digest.hexdigest()
        binding = hashlib.sha256(
            _canonical(
                {
                    "authority": astuple(authority),
                    "session": session_id,
                    "permissions": sorted(permissions),
                    "catalog": catalog_digest,
                }
            )
        ).hexdigest()
        now = time.time()
        if not math.isfinite(now):
            raise ValueError("Catalog clock unavailable")
        offset = 0
        if cursor is not None:
            value = self._decode(cursor)
            if (
                value["binding"] != binding
                or value["expires"] <= now
                or value["expires"] > now + 301
                or value["offset"] >= len(ordered)
            ):
                raise ValueError("Catalog cursor is stale or foreign")
            offset = value["offset"]
        end, used = offset, PAGE_ENVELOPE_RESERVE
        for size in sizes[offset : offset + 50]:
            if used + size > MAX_PAGE_BYTES:
                break
            end += 1
            used += size
        next_cursor = (
            self._encode({"v": 1, "binding": binding, "offset": end, "expires": int(now) + 300})
            if end < len(ordered)
            else None
        )
        return CatalogPage(ordered[offset:end], next_cursor, catalog_digest)
