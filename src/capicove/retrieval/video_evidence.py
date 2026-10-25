"""Authorized bounded expansion of published video unit evidence.

Text expansion uses canonical character/time locators and verified representations,
not provider-specific receipts. Query-score persistence remains unavailable.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from types import MappingProxyType

from capicove.contracts import InvocationContext, SearchableUnit
from capicove.contracts.retrieval import TextOrigin
from capicove.retrieval.video_ports import VideoUnitCatalogReader


_SUPPORTED = frozenset({"metadata", "preview", "lineage", "transcript", "ocr"})
_KNOWN = _SUPPORTED | {"transcript", "ocr", "scores", "provider_diagnostics"}
_OPAQUE = re.compile(r"[A-Za-z0-9][A-Za-z0-9:._-]{0,511}")
_CONTENT = re.compile(r"resource:(?:sha256|preview):([0-9a-f]{64})")
_MAX_UNITS = 32
_MAX_RESPONSE_BYTES = 4096
_MAX_RESOURCE_BYTES = 16 * 1024 * 1024


def video_evidence_bindings(units: tuple[SearchableUnit, ...]) -> Mapping[str, tuple[str, ...]]:
    """Derive immutable lookup bindings from actual published canonical units.

    The mapping is a composition value, not a new persistent storage format.
    Shared historical evidence may map to several units, bounded at read time.
    """
    grouped: dict[str, list[str]] = {}
    seen = set()
    for unit in units:
        if unit.unit_id in seen:
            raise ValueError("duplicate evidence unit ID")
        seen.add(unit.unit_id)
        grouped.setdefault(unit.evidence_ref, []).append(unit.unit_id)
    return MappingProxyType({ref: tuple(sorted(ids)) for ref, ids in grouped.items()})


class VideoEvidenceReader:
    """Read only authorized evidence and never expose source paths or media bytes.

    authorize_resource receives (InvocationContext, requested evidence ID, exact
    resource reference) and must return True. It is called for the receipt, each
    unit ID before catalog access, and each preview before content access on every
    invocation. Omitting the callback denies all access.
    """

    def __init__(
        self,
        *,
        catalog: VideoUnitCatalogReader,
        evidence_units: Mapping[str, tuple[str, ...]],
        read_resource: Callable[[str], bytes],
        authorize_resource: Callable[[InvocationContext, str, str], bool] | None = None,
    ) -> None:
        self._catalog = catalog
        self._bindings = MappingProxyType(
            {key: tuple(values) for key, values in evidence_units.items()}
        )
        self._read_resource = read_resource
        self._authorize_resource = authorize_resource

    @staticmethod
    def _deadline(context: InvocationContext) -> None:
        if datetime.now(UTC) >= context.deadline:
            raise TimeoutError("evidence deadline exceeded")

    def _authorize(self, context: InvocationContext, evidence_id: str, reference: str) -> None:
        self._deadline(context)
        if (
            "evidence:read" not in context.permissions
            or self._authorize_resource is None
            or self._authorize_resource(context, evidence_id, reference) is not True
        ):
            raise PermissionError("evidence resource access denied")
        self._deadline(context)

    def _verified_resource(self, context: InvocationContext, reference: str) -> bytes:
        match = _CONTENT.fullmatch(reference)
        if match is None:
            raise LookupError("content-addressed evidence resource unavailable")
        try:
            payload = self._read_resource(reference)
        except (FileNotFoundError, KeyError) as error:
            raise LookupError("evidence resource unavailable") from error
        self._deadline(context)
        if not isinstance(payload, bytes):
            raise ValueError("invalid evidence resource bytes")
        if len(payload) > _MAX_RESOURCE_BYTES:
            raise LookupError("evidence resource capacity exceeded")
        if hashlib.sha256(payload).hexdigest() != match[1]:
            raise ValueError("evidence resource hash mismatch")
        return payload

    def read(
        self, *, context: InvocationContext, evidence_id: str, sections: tuple[str, ...]
    ) -> Mapping[str, object]:
        """Project requested supported sections after fresh resource authorization."""
        self._deadline(context)
        if not isinstance(evidence_id, str) or _OPAQUE.fullmatch(evidence_id) is None:
            raise ValueError("invalid evidence identity")
        if (
            not sections
            or len(sections) != len(set(sections))
            or any(section not in _KNOWN for section in sections)
        ):
            raise ValueError("invalid evidence sections")
        self._authorize(context, evidence_id, evidence_id)
        if set(sections) - _SUPPORTED:
            raise LookupError("unsupported evidence section")
        ids = self._bindings.get(evidence_id)
        if not ids:
            raise LookupError("evidence unit unavailable")
        if len(ids) > _MAX_UNITS:
            raise LookupError("evidence unit capacity exceeded")
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate evidence binding")
        units = []
        for unit_id in ids:
            if _OPAQUE.fullmatch(unit_id) is None:
                raise ValueError("invalid opaque unit identity")
            self._authorize(context, evidence_id, unit_id)
            unit = self._catalog.get(unit_id)
            if unit is None:
                raise LookupError("evidence unit unavailable")
            if unit.unit_id != unit_id or unit.evidence_ref != evidence_id:
                raise ValueError("evidence binding mismatch")
            units.append(unit)
        # A deleted or corrupted receipt invalidates all sections, including metadata.
        self._verified_resource(context, evidence_id)
        result: dict[str, object] = {}
        for section, origin in (("transcript", TextOrigin.ASR), ("ocr", TextOrigin.OCR)):
            if section not in sections:
                continue
            excerpts = []
            for unit in units:
                locator = unit.text_locator
                if locator is None or locator.origin != origin:
                    continue
                if locator.time_range is None or len(unit.representation_refs) != 1:
                    raise LookupError("unambiguous temporal text unavailable")
                reference = unit.representation_refs[0]
                self._authorize(context, evidence_id, reference)
                text = self._verified_resource(context, reference).decode("utf-8")
                bounds = locator.character_range
                if bounds.end_char > len(text):
                    raise ValueError("text locator exceeds representation")
                excerpts.append(
                    {
                        "unit_id": unit.unit_id,
                        "text": text[bounds.start_char : bounds.end_char],
                        "representation_ref": reference,
                        "time_range": {
                            "start_ms": locator.time_range.start_ms,
                            "end_ms": locator.time_range.end_ms,
                        },
                    }
                )
            if not excerpts:
                raise LookupError("requested temporal text unavailable")
            result[section] = excerpts
        if "metadata" in sections:
            metadata = []
            for unit in units:
                item: dict[str, object] = {
                    "unit_id": unit.unit_id,
                    "target_type": unit.unit_type.value,
                    "source_sha256": unit.source_hash,
                }
                if unit.timestamp_ms is not None:
                    item["timestamp_ms"] = unit.timestamp_ms
                if unit.time_range is not None:
                    item["time_range"] = {
                        "start_ms": unit.time_range.start_ms,
                        "end_ms": unit.time_range.end_ms,
                    }
                if unit.scene_range is not None:
                    item["scene_range"] = {
                        "start_ms": unit.scene_range.start_ms,
                        "end_ms": unit.scene_range.end_ms,
                    }
                metadata.append(item)
            result["metadata"] = metadata
        if "lineage" in sections:
            # These hashes come from published canonical values, not provider claims.
            result["lineage"] = [
                {
                    "unit_id": u.unit_id,
                    "source_sha256": u.source_hash,
                    "processing_fingerprint": u.processing_fingerprint.sha256,
                }
                for u in units
            ]
        if "preview" in sections:
            previews = {
                ref
                for unit in units
                for ref in unit.representation_refs
                if ref.startswith("resource:preview:")
            }
            if len(previews) != 1:
                raise LookupError("unique evidence preview unavailable")
            reference = next(iter(previews))
            if _CONTENT.fullmatch(reference) is None:
                raise ValueError("invalid opaque preview reference")
            self._authorize(context, evidence_id, reference)
            payload = self._verified_resource(context, reference)
            if payload.startswith(b"\x89PNG\r\n\x1a\n"):
                media_type = "image/png"
            elif payload.startswith(b"\xff\xd8\xff"):
                media_type = "image/jpeg"
            elif payload.startswith(b"RIFF") and payload[8:12] == b"WEBP":
                media_type = "image/webp"
            else:
                raise LookupError("supported preview media unavailable")
            result["preview"] = {"resource_ref": reference, "media_type": media_type}
        if (
            len(
                json.dumps(
                    {"evidence_id": evidence_id, **result}, ensure_ascii=False, allow_nan=False
                ).encode()
            )
            > _MAX_RESPONSE_BYTES
        ):
            raise LookupError("evidence response capacity exceeded")
        self._deadline(context)
        return result


__all__ = ["VideoEvidenceReader", "video_evidence_bindings"]
