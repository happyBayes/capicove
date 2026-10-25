"""Closed video query decoding and projection shared by search transports."""

from __future__ import annotations

import base64
from collections.abc import Mapping
from dataclasses import asdict

from capicove.contracts import (
    ArtifactSource,
    CompactHit,
    FullHit,
    ImageQuery,
    InlineBytesSource,
    LocalPathSource,
    MountSource,
    TargetType,
    TextQuery,
)


MAX_INLINE_BYTES = 8 * 1024 * 1024


def _string(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("expected non-blank string")
    return value


def decode_video_query(value: Mapping[str, object]) -> TextQuery | ImageQuery:
    """Reject unknown fields before constructing a canonical query."""
    kind = value.get("type")
    if kind == "text" and not set(value) - {"type", "text", "language"}:
        language = value.get("language")
        return TextQuery(
            _string(value.get("text")), None if language is None else _string(language)
        )
    if kind != "image" or set(value) != {"type", "source"}:
        raise ValueError("unsupported_query_target")
    source = value["source"]
    if not isinstance(source, Mapping):
        raise ValueError("invalid source")
    fields = set(source)
    kind = source.get("type")
    if kind == "artifact" and fields == {"type", "artifact_ref"}:
        return ImageQuery(ArtifactSource(_string(source["artifact_ref"])))
    if kind == "local_path" and fields == {"type", "path"}:
        return ImageQuery(LocalPathSource(_string(source["path"])))
    if kind == "mount" and fields == {"type", "mount_ref", "relative_path"}:
        return ImageQuery(
            MountSource(_string(source["mount_ref"]), _string(source["relative_path"]))
        )
    if kind == "inline_bytes" and fields == {"type", "data_base64", "media_type"}:
        encoded = _string(source["data_base64"])
        if len(encoded) > 4 * ((MAX_INLINE_BYTES + 2) // 3):
            raise ValueError("inline image exceeds byte limit")
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_INLINE_BYTES:
            raise ValueError("inline image exceeds byte limit")
        return ImageQuery(InlineBytesSource(data, _string(source["media_type"])))
    raise ValueError("unsupported source fields")


def encode_video_hit(hit: CompactHit) -> dict[str, object]:
    """Require previews on v3 projections without breaking legacy constructors."""
    if hit.target_type not in {TargetType.KEYFRAME, TargetType.VIDEO_SEGMENT}:
        raise ValueError("unsupported target")
    preview = hit.preview_resource_ref
    if preview is None or not preview.startswith("resource:preview:"):
        raise ValueError("preview_resource_ref must be an opaque preview reference")
    result: dict[str, object] = {
        "id": hit.id,
        "target_type": hit.target_type.value,
        "brief_score": hit.brief_score,
        "evidence_id": hit.evidence_id,
        "preview_resource_ref": preview,
    }
    if hit.target_type is TargetType.KEYFRAME:
        result["timestamp_ms"] = hit.timestamp_ms
    else:
        if hit.time_range is None:
            raise ValueError("segment range missing")
        result["time_range"] = asdict(hit.time_range)
    if isinstance(hit, FullHit):
        result.update(
            {
                "artifact_ref": hit.artifact_ref,
                "searchable_unit_ref": hit.searchable_unit_ref,
                "representation_refs": list(hit.representation_refs),
                "match_kinds": [kind.value for kind in hit.match_kinds],
                "lane_scores": [asdict(score) for score in hit.lane_scores],
                "embedding_spaces": [asdict(space) for space in hit.embedding_spaces],
                "processing_fingerprint": None
                if hit.processing_fingerprint is None
                else asdict(hit.processing_fingerprint),
                "index_generation_ids": list(hit.index_generation_ids),
                "provider_ids": list(hit.provider_ids),
                "lineage_refs": list(hit.lineage_refs),
                "warning_codes": list(hit.warning_codes),
            }
        )
    return result
