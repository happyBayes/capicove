"""Canonical Tool contracts and handlers for Capicove image processing."""

from __future__ import annotations

import base64
import binascii
import dataclasses
import os
import re
import stat
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path, PurePath
from types import MappingProxyType
from typing import TypeVar

from capicove.contracts import (
    InvocationContext,
    PrivacyClass,
    SideEffect,
    ToolError,
    ToolErrorCode,
    ToolResources,
    ToolResult,
    ToolSpec,
    ToolStatus,
)
from capicove.processing.image.deduplication.lsh import (
    HashRecord,
    LshSearchResult,
)
from capicove.processing.image.deduplication.perceptual import (
    ImagePhashResult,
    ImageRecord,
    PerceptualDeduplicationResult,
)
from capicove.processing.image.invocation import (
    ImageDeadlineExceeded,
    image_invocation_deadline,
)
from capicove.processing.image.mappers.remove_background import (
    BackgroundRemovalOptions,
    ImageRemoveBackgroundMapper,
)
from capicove.processing.image.models import (
    BoundingBox,
    DerivedImage,
    DifferenceAreaResult,
    DifferenceCaptionResult,
    FilterDecision,
    ImageOcrResult,
    ImageSource,
)
from capicove.processing.image.ocr import (
    DEFAULT_OCR_LANGUAGES,
    ImageOcr,
    OcrProcessingError,
    OcrProviderUnavailableError,
    validate_ocr_languages,
)
from capicove.tools.registry import ToolRegistry


_MAX_INLINE_IMAGE_BYTES = 32 * 1024 * 1024
_MAX_TOTAL_IMAGE_BYTES = 128 * 1024 * 1024
_MAX_SOURCES = 256
_MAX_RECORDS = 1024
_MAX_LSH_CANDIDATES = 100_000
_MAX_TEXT_LENGTH = 8192
_MAX_DIFFERENCE_REGIONS = 64
_MAX_BASE64_LENGTH = ((_MAX_INLINE_IMAGE_BYTES + 2) // 3) * 4
_WindowsPathT = TypeVar("_WindowsPathT", bound=PurePath)

_SOURCE_SCHEMA: dict[str, object] = {
    "oneOf": [
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["path"],
            "properties": {"path": {"type": "string"}},
        },
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["base64"],
            "properties": {
                "base64": {
                    "type": "string",
                    "contentEncoding": "base64",
                    "x-capicove-max-decoded-bytes": _MAX_INLINE_IMAGE_BYTES,
                },
                "media_type": {"type": "string"},
            },
        },
    ]
}

_SOURCES_SCHEMA: dict[str, object] = {
    "type": "array",
    "maxItems": _MAX_SOURCES,
    "items": _SOURCE_SCHEMA,
}
_PAIR_SOURCES_SCHEMA: dict[str, object] = {
    "type": "array",
    "minItems": 2,
    "maxItems": 2,
    "items": _SOURCE_SCHEMA,
}

_OCR_INPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["sources"],
    "properties": {
        "sources": _SOURCES_SCHEMA,
        "languages": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "uniqueItems": True,
            "items": {"type": "string", "pattern": "^[A-Za-z0-9_]+$"},
        },
    },
}


def _source_input(*, pair: bool = False, text: bool = False) -> dict[str, object]:
    properties: dict[str, object] = {
        "sources": _PAIR_SOURCES_SCHEMA if pair else _SOURCES_SCHEMA,
    }
    required = ["sources"]
    if text:
        properties["text"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": _MAX_TEXT_LENGTH,
        }
        required.append("text")
    return {
        "type": "object",
        "additionalProperties": False,
        "required": sorted(required),
        "properties": properties,
    }


_BACKGROUND_INPUT = _source_input()
_BACKGROUND_INPUT["properties"]["options"] = {  # type: ignore[index]
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "alpha_matting": {"type": "boolean"},
        "foreground_threshold": {"type": "integer", "minimum": 0, "maximum": 255},
        "background_threshold": {"type": "integer", "minimum": 0, "maximum": 255},
        "erode_size": {"type": "integer", "minimum": 0},
        "background_color": {
            "type": ["array", "null"],
            "minItems": 4,
            "maxItems": 4,
            "items": {"type": "integer", "minimum": 0, "maximum": 255},
        },
    },
}

_BOX_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["height", "width", "x", "y"],
    "properties": {
        "x": {"type": "number", "minimum": 0},
        "y": {"type": "number", "minimum": 0},
        "width": {"type": "number", "exclusiveMinimum": 0},
        "height": {"type": "number", "exclusiveMinimum": 0},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
    },
}

_PARAMETER_VALUE_SCHEMA: dict[str, object] = {
    "type": ["string", "number", "integer", "boolean", "null"],
}
_PARAMETERS_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": _PARAMETER_VALUE_SCHEMA,
}
_STAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["reason", "retained", "score", "stage"],
    "properties": {
        "stage": {"type": "string"},
        "score": {"type": "number", "minimum": 0},
        "retained": {"type": "boolean"},
        "reason": {"type": "string"},
    },
}
_LINEAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["operation_version", "parameters", "source_hashes"],
    "properties": {
        "operation_version": {"type": "integer", "minimum": 1},
        "parameters": _PARAMETERS_SCHEMA,
        "source_hashes": {"type": "array", "items": {"type": "string"}},
        "provider_id": {"type": "string"},
        "provider_version": {"type": "string"},
        "model_id": {"type": "string"},
        "model_revision": {"type": "string"},
        "model_manifest_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "preprocess_revision": {"type": "string"},
        "device": {"type": "string"},
        "stages": {"type": "array", "items": _STAGE_SCHEMA},
        "artifacts": {"type": "array", "items": {"type": "string"}},
    },
}

_DIFFERENCE_CAPTION_INPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["areas", "sources"],
    "properties": {
        "sources": _PAIR_SOURCES_SCHEMA,
        "areas": {
            "type": "object",
            "additionalProperties": False,
            "required": ["lineage", "regions"],
            "properties": {
                "regions": {
                    "type": "array",
                    "maxItems": _MAX_DIFFERENCE_REGIONS,
                    "items": _BOX_SCHEMA,
                },
                "lineage": _LINEAGE_SCHEMA,
            },
        },
    },
}

_DEDUP_INPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["records"],
    "properties": {
        "records": {
            "type": "array",
            "maxItems": _MAX_RECORDS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["images", "record_id"],
                "properties": {
                    "record_id": {"type": "string", "minLength": 1},
                    "images": _SOURCES_SCHEMA,
                    "text": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": _MAX_TEXT_LENGTH,
                    },
                },
            },
        }
    },
}

_PHASH_INPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["records"],
    "properties": {
        "records": {
            **_DEDUP_INPUT["properties"]["records"],  # type: ignore[index]
            "minItems": 1,
        },
        "max_hamming_distance": {"type": "integer", "minimum": 0, "maximum": 64},
        "enable_lsh": {"type": "boolean"},
    },
}

_PHASH_HEX_SCHEMA: dict[str, object] = {
    "type": "string",
    "pattern": "^[0-9A-Fa-f]{16}$",
}
_TEXT_HASH_SCHEMA: dict[str, object] = {
    "type": "string",
    "pattern": "^[0-9a-f]{64}$",
}
_HASH_COLLECTION_SCHEMA: dict[str, object] = {
    "type": "array",
    "minItems": 1,
    "maxItems": _MAX_SOURCES,
    "items": _PHASH_HEX_SCHEMA,
}
_LSH_QUERY_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hashes"],
    "properties": {
        "hashes": _HASH_COLLECTION_SCHEMA,
        "text_hash": _TEXT_HASH_SCHEMA,
    },
}
_LSH_CANDIDATE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hashes", "record_id"],
    "properties": {
        "record_id": {"type": "string", "minLength": 1},
        "hashes": _HASH_COLLECTION_SCHEMA,
        "text_hash": _TEXT_HASH_SCHEMA,
    },
}
_LSH_SEARCH_INPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["candidates", "hash_version", "query"],
    "properties": {
        "query": _LSH_QUERY_SCHEMA,
        "candidates": {
            "type": "array",
            "maxItems": _MAX_LSH_CANDIDATES,
            "items": _LSH_CANDIDATE_SCHEMA,
        },
        "hash_version": {"type": "string", "minLength": 1},
        "max_hamming_distance": {"type": "integer", "minimum": 0, "maximum": 64},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 1_000},
    },
}

_METRIC_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["name", "provider_id", "values"],
    "properties": {
        "name": {"type": "string"},
        "values": {"type": "array", "items": {"type": "number"}},
        "provider_id": {"type": "string"},
        "provider_version": {"type": ["string", "null"]},
        "model_id": {"type": ["string", "null"]},
        "model_revision": {"type": ["string", "null"]},
        "model_manifest_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "preprocess_revision": {"type": ["string", "null"]},
        "device": {"type": ["string", "null"]},
    },
}
_FILTER_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["applicable", "keep", "lineage", "metrics"],
    "properties": {
        "applicable": {"type": "boolean"},
        "keep": {"type": "boolean"},
        "metrics": {"type": "array", "items": _METRIC_SCHEMA},
        "lineage": _LINEAGE_SCHEMA,
    },
}
_ARTIFACT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "byte_size",
        "height",
        "media_type",
        "operation",
        "operation_version",
        "parameters",
        "path",
        "provider_id",
        "sha256",
        "source_hashes",
        "width",
    ],
    "properties": {
        "path": {"type": "string"},
        "sha256": {"type": "string"},
        "media_type": {"type": "string"},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "byte_size": {"type": "integer", "minimum": 1},
        "source_hashes": {"type": "array", "items": {"type": "string"}},
        "operation": {"type": "string"},
        "provider_id": {"type": "string"},
        "provider_version": {"type": ["string", "null"]},
        "operation_version": {"type": "integer", "minimum": 1},
        "parameters": {"type": "array"},
        "model_id": {"type": ["string", "null"]},
        "model_revision": {"type": ["string", "null"]},
        "model_manifest_sha256": {
            "type": ["string", "null"],
            "pattern": "^[0-9a-f]{64}$",
        },
        "preprocess_revision": {"type": ["string", "null"]},
        "device": {"type": ["string", "null"]},
    },
}
_ARTIFACT_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["artifacts", "lineage"],
    "properties": {
        "artifacts": {"type": "array", "items": _ARTIFACT_SCHEMA},
        "lineage": _LINEAGE_SCHEMA,
    },
}
_AREA_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineage", "regions"],
    "properties": {
        "regions": {"type": "array", "items": _BOX_SCHEMA},
        "lineage": _LINEAGE_SCHEMA,
    },
}
_CAPTION_REGION_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "box",
        "caption_difference_score",
        "difference_caption",
        "left_caption",
        "left_match_score",
        "right_caption",
        "right_match_score",
    ],
    "properties": {
        "box": _BOX_SCHEMA,
        "left_caption": {"type": "string"},
        "right_caption": {"type": "string"},
        "difference_caption": {"type": "string"},
        "left_match_score": {"type": "number", "minimum": -1, "maximum": 1},
        "right_match_score": {"type": "number", "minimum": -1, "maximum": 1},
        "caption_difference_score": {"type": "number", "minimum": 0, "maximum": 1},
    },
}
_CAPTION_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineage", "regions"],
    "properties": {
        "regions": {"type": "array", "items": _CAPTION_REGION_SCHEMA},
        "lineage": _LINEAGE_SCHEMA,
    },
}
_OCR_WORD_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "block",
        "confidence",
        "height",
        "line",
        "page",
        "paragraph",
        "text",
        "width",
        "word",
        "x",
        "y",
    ],
    "properties": {
        "text": {"type": "string", "minLength": 1},
        "x": {"type": "integer", "minimum": 0},
        "y": {"type": "integer", "minimum": 0},
        "width": {"type": "integer", "minimum": 1},
        "height": {"type": "integer", "minimum": 1},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "page": {"type": "integer", "minimum": 1},
        "block": {"type": "integer", "minimum": 1},
        "paragraph": {"type": "integer", "minimum": 1},
        "line": {"type": "integer", "minimum": 1},
        "word": {"type": "integer", "minimum": 1},
    },
}
_OCR_RESULT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "full_text",
        "languages",
        "operation_version",
        "parameters",
        "provider_id",
        "provider_version",
        "source_hash",
        "words",
    ],
    "properties": {
        "source_hash": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        "full_text": {"type": "string"},
        "words": {"type": "array", "maxItems": 100000, "items": _OCR_WORD_SCHEMA},
        "provider_id": {"type": "string"},
        "provider_version": {"type": ["string", "null"]},
        "languages": {
            "type": "array",
            "minItems": 1,
            "maxItems": 8,
            "items": {"type": "string"},
        },
        "operation_version": {"type": "integer", "minimum": 1},
        "parameters": _PARAMETERS_SCHEMA,
    },
}
_OCR_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineage", "results"],
    "properties": {
        "results": {"type": "array", "items": _OCR_RESULT_SCHEMA},
        "lineage": _LINEAGE_SCHEMA,
    },
}
_DUPLICATE_GROUP_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["distances_from_kept", "duplicate_record_ids", "kept_record_id"],
    "properties": {
        "kept_record_id": {"type": "string"},
        "duplicate_record_ids": {"type": "array", "items": {"type": "string"}},
        "distances_from_kept": {"type": "array", "items": {"type": "array"}},
    },
}
_DEDUP_LINEAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "hash_version",
        "max_hamming_distance",
        "provider_id",
        "provider_version",
        "record_source_hashes",
    ],
    "properties": {
        "provider_id": {"type": "string"},
        "provider_version": {"type": ["string", "null"]},
        "hash_version": {"type": "string"},
        "max_hamming_distance": {"type": "integer", "minimum": 0, "maximum": 64},
        "record_source_hashes": {"type": "array", "items": {"type": "array"}},
    },
}
_DEDUP_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["duplicate_groups", "lineage", "retained_record_ids"],
    "properties": {
        "duplicate_groups": {"type": "array", "items": _DUPLICATE_GROUP_SCHEMA},
        "retained_record_ids": {"type": "array", "items": {"type": "string"}},
        "lineage": _DEDUP_LINEAGE_SCHEMA,
    },
}

_PHASH_RECORD_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["hashes", "record_id", "source_hashes", "text_hash"],
    "properties": {
        "record_id": {"type": "string"},
        "hashes": {"type": "array", "items": _PHASH_HEX_SCHEMA},
        "source_hashes": {
            "type": "array",
            "items": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
        },
        "text_hash": {"anyOf": [_TEXT_HASH_SCHEMA, {"type": "null"}]},
    },
}
_PHASH_LINEAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "bit_width",
        "candidate_count",
        "candidate_strategy",
        "comparison_count",
        "hash_version",
        "max_hamming_distance",
        "provider_id",
        "provider_version",
    ],
    "properties": {
        "provider_id": {"type": "string"},
        "provider_version": {"type": ["string", "null"]},
        "hash_version": {"type": "string"},
        "bit_width": {"const": 64},
        "max_hamming_distance": {"type": "integer", "minimum": 0, "maximum": 64},
        "candidate_strategy": {"enum": ["exhaustive", "lsh"]},
        "candidate_count": {"type": "integer", "minimum": 0},
        "comparison_count": {"type": "integer", "minimum": 0},
    },
}
_PHASH_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "duplicate_groups",
        "lineage",
        "records",
        "retained_record_ids",
    ],
    "properties": {
        "records": {"type": "array", "items": _PHASH_RECORD_SCHEMA},
        "duplicate_groups": {"type": "array", "items": _DUPLICATE_GROUP_SCHEMA},
        "retained_record_ids": {"type": "array", "items": {"type": "string"}},
        "lineage": _PHASH_LINEAGE_SCHEMA,
    },
}
_LSH_MATCH_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["distance", "hashes", "record_id"],
    "properties": {
        "record_id": {"type": "string"},
        "hashes": _HASH_COLLECTION_SCHEMA,
        "distance": {"type": "integer", "minimum": 0},
    },
}
_LSH_LINEAGE_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "bit_width",
        "candidate_count",
        "candidate_strategy",
        "comparison_count",
        "hash_version",
        "max_hamming_distance",
    ],
    "properties": {
        "hash_version": {"type": "string"},
        "bit_width": {"type": "integer", "minimum": 64},
        "max_hamming_distance": {"type": "integer", "minimum": 0, "maximum": 64},
        "candidate_strategy": {"enum": ["exhaustive", "lsh"]},
        "candidate_count": {"type": "integer", "minimum": 0},
        "comparison_count": {"type": "integer", "minimum": 0},
    },
}
_LSH_SEARCH_OUTPUT: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["lineage", "matches"],
    "properties": {
        "matches": {"type": "array", "maxItems": 1_000, "items": _LSH_MATCH_SCHEMA},
        "lineage": _LSH_LINEAGE_SCHEMA,
    },
}


def _spec(
    tool_id: str,
    description: str,
    input_schema: Mapping[str, object],
    output_schema: Mapping[str, object],
    *,
    write: bool = False,
    model: bool = False,
    filesystem_read: bool = True,
    subprocesses: tuple[str, ...] = (),
) -> ToolSpec:
    permissions = {"image:read"}
    if write:
        permissions.update(("artifact:write", "image:write"))
    if model:
        permissions.add("model:execute")
    return ToolSpec(
        tool_id=tool_id,
        contract_version=1,
        description=description,
        input_schema=input_schema,
        output_schema=output_schema,
        modalities=("image", "text"),
        required_permissions=frozenset(permissions),
        privacy_class=PrivacyClass.CONFIDENTIAL,
        side_effect=SideEffect.WRITE if write else SideEffect.READ,
        network_access=False,
        deterministic=not model,
        idempotent=True,
        timeout_seconds=300.0,
        redacted_fields=(
            "sources.*.base64",
            "records.*.images.*.base64",
            "records.*.text",
            "text",
        ),
        resources=ToolResources(
            filesystem_read=filesystem_read,
            filesystem_write=write,
            subprocesses=subprocesses,
        ),
    )


IMAGE_TOOL_SPECS_BY_NAME: Mapping[str, ToolSpec] = MappingProxyType(
    {
        "image_remove_background_mapper": _spec(
            "capicove.image.remove-background",
            "Remove image backgrounds into immutable PNG artifacts.",
            _BACKGROUND_INPUT,
            _ARTIFACT_OUTPUT,
            write=True,
            model=True,
        ),
        "imgdiff_difference_area_generator_mapper": _spec(
            "capicove.image.diff.areas",
            "Generate deterministic changed regions for one aligned image pair.",
            _source_input(pair=True),
            _AREA_OUTPUT,
        ),
        "imgdiff_difference_caption_generator_mapper": _spec(
            "capicove.image.diff.caption",
            "Generate structured captions for validated changed image-pair regions.",
            _DIFFERENCE_CAPTION_INPUT,
            _CAPTION_OUTPUT,
            model=True,
        ),
        "image_aesthetics_filter": _spec(
            "capicove.image.filter.aesthetics",
            "Filter images by local-model aesthetics score.",
            _source_input(),
            _FILTER_OUTPUT,
            model=True,
        ),
        "image_aspect_ratio_filter": _spec(
            "capicove.image.filter.aspect-ratio",
            "Filter images by width-to-height ratio.",
            _source_input(),
            _FILTER_OUTPUT,
        ),
        "image_nsfw_filter": _spec(
            "capicove.image.filter.nsfw",
            "Filter images by local-model NSFW probability.",
            _source_input(),
            _FILTER_OUTPUT,
            model=True,
        ),
        "image_shape_filter": _spec(
            "capicove.image.filter.shape",
            "Filter images by decoded width and height.",
            _source_input(),
            _FILTER_OUTPUT,
        ),
        "image_size_filter": _spec(
            "capicove.image.filter.size",
            "Filter images by encoded source byte size.",
            _source_input(),
            _FILTER_OUTPUT,
        ),
        "image_subplot_filter": _spec(
            "capicove.image.filter.subplot",
            "Filter grid-like subplot images using deterministic OpenCV analysis.",
            _source_input(),
            _FILTER_OUTPUT,
        ),
        "image_text_matching_filter": _spec(
            "capicove.image.filter.text-matching",
            "Filter images by local-model image-text matching probability.",
            _source_input(text=True),
            _FILTER_OUTPUT,
            model=True,
        ),
        "image_text_similarity_filter": _spec(
            "capicove.image.filter.text-similarity",
            "Filter images by local-model image-text cosine similarity.",
            _source_input(text=True),
            _FILTER_OUTPUT,
            model=True,
        ),
        "image_watermark_filter": _spec(
            "capicove.image.filter.watermark",
            "Filter images by local-model watermark probability.",
            _source_input(),
            _FILTER_OUTPUT,
            model=True,
        ),
        "image_pair_similarity_filter": _spec(
            "capicove.image.filter.pair-similarity",
            "Filter exactly two images by visual cosine similarity.",
            _source_input(pair=True),
            _FILTER_OUTPUT,
            model=True,
        ),
        "image_phash": _spec(
            "capicove.image.phash",
            "Compute versioned pHashes and group a bounded image batch.",
            _PHASH_INPUT,
            _PHASH_OUTPUT,
        ),
        "image_lsh_search": _spec(
            "capicove.image.lsh-search",
            "Search bounded pHash records using deterministic LSH and exact Hamming distance.",
            _LSH_SEARCH_INPUT,
            _LSH_SEARCH_OUTPUT,
            filesystem_read=False,
        ),
        "image_perceptual_deduplicator": _spec(
            "capicove.image.perceptual-deduplicate",
            "Group visually similar training images using pHash; not byte-exact.",
            _DEDUP_INPUT,
            _DEDUP_OUTPUT,
        ),
        "ray_image_perceptual_deduplicator": _spec(
            "capicove.image.perceptual-deduplicate-ray",
            "Distribute pHash preparation for visual training-image deduplication.",
            _DEDUP_INPUT,
            _DEDUP_OUTPUT,
        ),
        "image_ocr": _spec(
            "capicove.image.ocr",
            "Recognize structured text using a local Tesseract CLI provider.",
            _OCR_INPUT,
            _OCR_OUTPUT,
            model=True,
            subprocesses=("tesseract",),
        ),
    }
)


@dataclasses.dataclass(frozen=True, slots=True)
class _ParsedArguments:
    sources: tuple[ImageSource, ...] = ()
    records: tuple[ImageRecord, ...] = ()
    lsh_query_hashes: tuple[int, ...] = ()
    lsh_candidates: tuple[HashRecord, ...] = ()
    hash_version: str | None = None
    query_text_hash: str | None = None
    max_hamming_distance: int = 0
    top_k: int = 100
    enable_lsh: bool = False
    text: str | None = None
    areas: DifferenceAreaResult | None = None
    background_options: BackgroundRemovalOptions | None = None
    languages: tuple[str, ...] | None = None
    output_directory_fd: int | None = None
    authorized_output_directory: Path | None = None


@dataclasses.dataclass(frozen=True, slots=True)
class _AuthorizedOutputDirectory:
    descriptor: int
    path: Path


@dataclasses.dataclass(slots=True)
class _ParsingBudget:
    source_count: int = 0
    decoded_bytes: int = 0

    def add_source(self, byte_size: int) -> None:
        self.source_count += 1
        self.decoded_bytes += byte_size
        if self.source_count > _MAX_SOURCES:
            raise ValueError("image invocation exceeds the source-count limit")
        if self.decoded_bytes > _MAX_TOTAL_IMAGE_BYTES:
            raise ValueError("image invocation exceeds the cumulative-byte limit")


class _PathPolicyError(ValueError):
    """Raised when an image path escapes explicitly authorized roots."""


class ImageToolHandler:
    """Enforce the canonical image Tool boundary around one capability."""

    def __init__(
        self,
        name: str,
        capability: object,
        *,
        read_roots: tuple[Path, ...],
        write_roots: tuple[Path, ...],
        read_root_identities: Mapping[Path, tuple[int, int]] | None = None,
    ) -> None:
        if name not in IMAGE_TOOL_SPECS_BY_NAME:
            raise ValueError(f"unknown image capability: {name}")
        method_name = _capability_method(name)
        if not callable(getattr(capability, method_name, None)):
            raise TypeError(f"image capability must implement {method_name}")
        self._name = name
        self._spec = IMAGE_TOOL_SPECS_BY_NAME[name]
        self._capability = capability
        self._read_roots = tuple(
            _require_windows_local_path(root) if os.name == "nt" else root.resolve()
            for root in read_roots
        )
        self._read_root_identities: dict[Path, tuple[int, int] | None] = {}
        if os.name == "posix":
            if read_root_identities is not None:
                self._read_root_identities = dict(read_root_identities)
            else:
                for root in self._read_roots:
                    try:
                        metadata = root.stat()
                    except OSError:
                        self._read_root_identities[root] = None
                    else:
                        self._read_root_identities[root] = (
                            (metadata.st_dev, metadata.st_ino)
                            if stat.S_ISDIR(metadata.st_mode)
                            else None
                        )
        self._write_roots = tuple(
            _require_windows_local_path(root) if os.name == "nt" else root.resolve()
            for root in write_roots
        )

    def __call__(
        self,
        context: InvocationContext,
        arguments: Mapping[str, object],
        /,
    ) -> ToolResult:
        """Validate, authorize, execute, and serialize one image invocation."""
        started = datetime.now(UTC)
        missing = self._spec.required_permissions.difference(context.permissions)
        if missing:
            return self._failure(
                context,
                started,
                ToolStatus.REJECTED,
                ToolErrorCode.PERMISSION_DENIED,
                f"Missing required permissions: {', '.join(sorted(missing))}.",
            )
        if started >= context.deadline:
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.TIMEOUT,
                "The image operation deadline has expired.",
            )
        try:
            with image_invocation_deadline(context.deadline):
                authorized_output = _authorize_capability_configuration(
                    self._name,
                    self._capability,
                    self._write_roots,
                )
                try:
                    parsed = _parse_arguments(
                        self._name, arguments, self._read_roots, self._read_root_identities
                    )
                    if authorized_output is not None:
                        parsed = dataclasses.replace(
                            parsed,
                            output_directory_fd=authorized_output.descriptor,
                            authorized_output_directory=authorized_output.path,
                        )
                    processing_result = _invoke(self._name, self._capability, parsed)
                finally:
                    if authorized_output is not None:
                        os.close(authorized_output.descriptor)
            _validate_result_for_tool(self._name, processing_result)
            _authorize_derived_outputs(processing_result, self._write_roots)
            serialized, evidence_refs = _serialize_result(
                processing_result,
                self._name,
                parsed.languages,
            )
        except ImageDeadlineExceeded:
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.TIMEOUT,
                "The image operation exceeded its deadline.",
            )
        except _PathPolicyError:
            return self._failure(
                context,
                started,
                ToolStatus.REJECTED,
                ToolErrorCode.PERMISSION_DENIED,
                "The requested path is outside the image execution policy.",
            )
        except OcrProviderUnavailableError:
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.UNAVAILABLE,
                "The requested OCR provider or language data is unavailable.",
            )
        except OcrProcessingError:
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "The OCR operation failed.",
            )
        except (ValueError, TypeError, OSError, binascii.Error):
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.INVALID_INPUT,
                "Image tool arguments or media are invalid.",
            )
        except RuntimeError as error:
            unavailable = "unavailable" in str(error).casefold()
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.UNAVAILABLE if unavailable else ToolErrorCode.INTERNAL,
                "The requested image provider is unavailable."
                if unavailable
                else "The image operation failed.",
            )
        except Exception:
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.INTERNAL,
                "The image operation failed.",
            )
        if datetime.now(UTC) >= context.deadline:
            return self._failure(
                context,
                started,
                ToolStatus.FAILED,
                ToolErrorCode.TIMEOUT,
                "The image operation exceeded its deadline.",
            )
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=self._spec.tool_id,
            tool_version=self._spec.contract_version,
            status=ToolStatus.SUCCEEDED,
            result=serialized,
            evidence_refs=evidence_refs,
            started_at=started,
            completed_at=datetime.now(UTC),
        )

    def _failure(
        self,
        context: InvocationContext,
        started: datetime,
        status: ToolStatus,
        code: ToolErrorCode,
        message: str,
    ) -> ToolResult:
        return ToolResult(
            request_id=context.request_id,
            trace_id=context.trace_id,
            tool_id=self._spec.tool_id,
            tool_version=self._spec.contract_version,
            status=status,
            error=ToolError(code, message, retryable=False),
            started_at=started,
            completed_at=datetime.now(UTC),
        )


def register_image_tools(
    registry: ToolRegistry,
    capabilities: Mapping[str, object],
    *,
    read_roots: tuple[Path, ...] = (),
    write_roots: tuple[Path, ...] = (),
) -> None:
    """Register all image capabilities behind canonical security handlers."""
    missing = set(IMAGE_TOOL_SPECS_BY_NAME).difference(capabilities)
    extra = set(capabilities).difference(IMAGE_TOOL_SPECS_BY_NAME)
    if missing or extra:
        raise ValueError(
            f"image capabilities must match catalog; missing={sorted(missing)}, "
            f"extra={sorted(extra)}"
        )
    for name, spec in IMAGE_TOOL_SPECS_BY_NAME.items():
        registry.register(
            spec,
            build_image_tool_handler(
                name,
                capabilities[name],
                read_roots=read_roots,
                write_roots=write_roots,
            ),
        )


def build_image_tool_handler(
    name: str,
    capability: object,
    *,
    read_roots: tuple[Path, ...],
    write_roots: tuple[Path, ...] = (),
    read_root_identities: Mapping[Path, tuple[int, int]] | None = None,
) -> ImageToolHandler:
    """Build one canonical image Tool handler without loading optional providers."""
    return ImageToolHandler(
        name,
        capability,
        read_roots=read_roots,
        write_roots=write_roots,
        read_root_identities=read_root_identities,
    )


def _parse_arguments(
    name: str,
    arguments: Mapping[str, object],
    read_roots: tuple[Path, ...],
    read_root_identities: Mapping[Path, tuple[int, int] | None],
) -> _ParsedArguments:
    budget = _ParsingBudget()
    if name == "image_lsh_search":
        return _parse_lsh_arguments(arguments)
    if name in {
        "image_phash",
        "image_perceptual_deduplicator",
        "ray_image_perceptual_deduplicator",
    }:
        allowed = {"records"}
        if name == "image_phash":
            allowed.update(("enable_lsh", "max_hamming_distance"))
        _require_exact_keys(arguments, allowed, {"records"})
        records = _parse_records(arguments["records"], read_roots, read_root_identities, budget)
        if name == "image_phash" and not records:
            raise ValueError("image_phash records must not be empty")
        return _ParsedArguments(
            records=records,
            max_hamming_distance=_bounded_integer(
                arguments.get("max_hamming_distance", 0),
                "max_hamming_distance",
                minimum=0,
                maximum=64,
            ),
            enable_lsh=_require_boolean(arguments.get("enable_lsh", False), "enable_lsh"),
        )

    allowed = {"sources"}
    required = {"sources"}
    if name in {"image_text_matching_filter", "image_text_similarity_filter"}:
        allowed.add("text")
        required.add("text")
    if name == "image_remove_background_mapper":
        allowed.add("options")
    if name == "imgdiff_difference_caption_generator_mapper":
        allowed.add("areas")
        required.add("areas")
    if name == "image_ocr":
        allowed.add("languages")
    _require_exact_keys(arguments, allowed, required)
    sources = _parse_sources(arguments["sources"], read_roots, read_root_identities, budget)
    if (
        name
        in {
            "image_pair_similarity_filter",
            "imgdiff_difference_area_generator_mapper",
            "imgdiff_difference_caption_generator_mapper",
        }
        and len(sources) != 2
    ):
        raise ValueError("image pair tools require exactly two sources")
    text_value = arguments.get("text")
    text = (
        _require_string(text_value, "text", maximum_length=_MAX_TEXT_LENGTH)
        if text_value is not None
        else None
    )
    areas_value = arguments.get("areas")
    areas = _parse_areas(areas_value) if areas_value is not None else None
    options_value = arguments.get("options")
    options = _parse_background_options(options_value) if options_value is not None else None
    languages_value = arguments.get("languages")
    languages: tuple[str, ...] | None = None
    if languages_value is not None:
        raw_languages = _require_sequence(languages_value, "languages")
        languages = validate_ocr_languages(
            tuple(_require_string(language, "language") for language in raw_languages)
        )
    return _ParsedArguments(
        sources=sources,
        text=text,
        areas=areas,
        background_options=options,
        languages=languages,
    )


def _parse_records(
    value: object,
    read_roots: tuple[Path, ...],
    read_root_identities: Mapping[Path, tuple[int, int] | None],
    budget: _ParsingBudget,
) -> tuple[ImageRecord, ...]:
    raw_records = _require_sequence(value, "records")
    if len(raw_records) > _MAX_RECORDS:
        raise ValueError("image invocation exceeds the record-count limit")
    records: list[ImageRecord] = []
    for raw_record in raw_records:
        record = _require_mapping(raw_record, "record")
        _require_exact_keys(record, {"images", "record_id", "text"}, {"images", "record_id"})
        images = _parse_sources(record["images"], read_roots, read_root_identities, budget)
        text_value = record.get("text")
        text = (
            _require_string(text_value, "text", maximum_length=_MAX_TEXT_LENGTH)
            if text_value is not None
            else None
        )
        records.append(ImageRecord(_require_string(record["record_id"], "record_id"), images, text))
    return tuple(records)


def _parse_lsh_arguments(arguments: Mapping[str, object]) -> _ParsedArguments:
    allowed = {
        "candidates",
        "hash_version",
        "max_hamming_distance",
        "query",
        "top_k",
    }
    _require_exact_keys(arguments, allowed, {"candidates", "hash_version", "query"})
    query = _require_mapping(arguments["query"], "query")
    _require_exact_keys(query, {"hashes", "text_hash"}, {"hashes"})
    query_hashes = _parse_hashes(query["hashes"], "query.hashes")
    query_text_hash = _optional_text_hash(query.get("text_hash"))
    raw_candidates = _require_sequence(arguments["candidates"], "candidates")
    if len(raw_candidates) > _MAX_LSH_CANDIDATES:
        raise ValueError("LSH search exceeds the candidate-count limit")
    candidates: list[HashRecord] = []
    for raw_candidate in raw_candidates:
        candidate = _require_mapping(raw_candidate, "candidate")
        _require_exact_keys(
            candidate,
            {"hashes", "record_id", "text_hash"},
            {"hashes", "record_id"},
        )
        candidates.append(
            HashRecord(
                record_id=_require_string(candidate["record_id"], "record_id"),
                hashes=_parse_hashes(candidate["hashes"], "candidate.hashes"),
                text_hash=_optional_text_hash(candidate.get("text_hash")),
            )
        )
    return _ParsedArguments(
        lsh_query_hashes=query_hashes,
        lsh_candidates=tuple(candidates),
        hash_version=_require_string(arguments["hash_version"], "hash_version"),
        query_text_hash=query_text_hash,
        max_hamming_distance=_bounded_integer(
            arguments.get("max_hamming_distance", 0),
            "max_hamming_distance",
            minimum=0,
            maximum=64,
        ),
        top_k=_bounded_integer(
            arguments.get("top_k", 100),
            "top_k",
            minimum=1,
            maximum=1_000,
        ),
    )


def _parse_hashes(value: object, name: str) -> tuple[int, ...]:
    raw_hashes = _require_sequence(value, name)
    if not raw_hashes or len(raw_hashes) > _MAX_SOURCES:
        raise ValueError(f"{name} must contain between 1 and {_MAX_SOURCES} hashes")
    hashes: list[int] = []
    for raw_hash in raw_hashes:
        normalized = _require_string(raw_hash, "pHash").casefold()
        if len(normalized) != 16 or any(
            character not in "0123456789abcdef" for character in normalized
        ):
            raise ValueError("pHash must contain exactly 16 hexadecimal characters")
        hashes.append(int(normalized, 16))
    return tuple(hashes)


def _optional_text_hash(value: object) -> str | None:
    if value is None:
        return None
    normalized = _require_string(value, "text_hash")
    if len(normalized) != 64 or any(
        character not in "0123456789abcdef" for character in normalized
    ):
        raise ValueError("text_hash must be a lowercase SHA-256 hexadecimal string")
    return normalized


def _parse_sources(
    value: object,
    read_roots: tuple[Path, ...],
    read_root_identities: Mapping[Path, tuple[int, int] | None],
    budget: _ParsingBudget,
) -> tuple[ImageSource, ...]:
    sources: list[ImageSource] = []
    raw_sources = _require_sequence(value, "sources")
    if len(raw_sources) > _MAX_SOURCES:
        raise ValueError("image invocation exceeds the source-count limit")
    for raw_source in raw_sources:
        source = _require_mapping(raw_source, "source")
        if set(source) == {"path"}:
            path = _require_string(source["path"], "path")
            resolved_path = _authorize_path(Path(path), read_roots)
            data = _read_authorized_regular_file(
                resolved_path,
                read_roots,
                read_root_identities,
                maximum_bytes=_MAX_INLINE_IMAGE_BYTES,
            )
            budget.add_source(len(data))
            sources.append(ImageSource(data=data))
            continue
        if set(source).issubset({"base64", "media_type"}) and "base64" in source:
            encoded = _require_string(source["base64"], "base64")
            if len(encoded) > _MAX_BASE64_LENGTH:
                raise ValueError("inline image exceeds the encoded-length limit")
            data = base64.b64decode(encoded, validate=True)
            if len(data) > _MAX_INLINE_IMAGE_BYTES:
                raise ValueError("inline image exceeds the decoded-byte limit")
            budget.add_source(len(data))
            media_type_value = source.get("media_type")
            media_type = (
                _require_string(media_type_value, "media_type")
                if media_type_value is not None
                else None
            )
            sources.append(ImageSource(data=data, media_type=media_type))
            continue
        raise ValueError("source must contain exactly path or base64")
    return tuple(sources)


def _parse_areas(value: object) -> DifferenceAreaResult:
    areas = _require_mapping(value, "areas")
    if set(areas) == {"lineage", "regions"}:
        lineage = _require_mapping(areas["lineage"], "areas.lineage")
        normalized: dict[str, object] = {
            "boxes": areas["regions"],
            "operation_version": lineage.get("operation_version", 1),
            "parameters": lineage.get("parameters", {}),
            "provider_id": lineage.get("provider_id"),
            "provider_version": lineage.get("provider_version"),
            "source_hashes": lineage.get("source_hashes"),
        }
        return _parse_areas(normalized)
    allowed = {
        "boxes",
        "operation_version",
        "parameters",
        "provider_id",
        "provider_version",
        "source_hashes",
    }
    _require_exact_keys(areas, allowed, {"boxes", "provider_id", "source_hashes"})
    hashes = _require_sequence(areas["source_hashes"], "source_hashes")
    if len(hashes) != 2:
        raise ValueError("source_hashes must contain two values")
    raw_boxes = _require_sequence(areas["boxes"], "boxes")
    if len(raw_boxes) > _MAX_DIFFERENCE_REGIONS:
        raise ValueError("difference areas exceed the region-count limit")
    boxes: list[BoundingBox] = []
    seen_boxes: set[tuple[float, float, float, float, float]] = set()
    for raw_box in raw_boxes:
        box = _require_mapping(raw_box, "box")
        _require_exact_keys(
            box,
            {"confidence", "height", "width", "x", "y"},
            {"height", "width", "x", "y"},
        )
        parsed_box = BoundingBox(
            x=_require_number(box["x"], "x"),
            y=_require_number(box["y"], "y"),
            width=_require_number(box["width"], "width"),
            height=_require_number(box["height"], "height"),
            confidence=_require_number(box.get("confidence", 1.0), "confidence"),
        )
        identity = (
            parsed_box.x,
            parsed_box.y,
            parsed_box.width,
            parsed_box.height,
            parsed_box.confidence,
        )
        if identity not in seen_boxes:
            seen_boxes.add(identity)
            boxes.append(parsed_box)
    parameters_value = areas.get("parameters", {})
    parameters_mapping = _require_mapping(parameters_value, "parameters")
    parameters = tuple(
        sorted((str(key), _require_scalar(item)) for key, item in parameters_mapping.items())
    )
    return DifferenceAreaResult(
        source_hashes=(
            _require_string(hashes[0], "source_hash"),
            _require_string(hashes[1], "source_hash"),
        ),
        boxes=tuple(boxes),
        provider_id=_require_string(areas["provider_id"], "provider_id"),
        provider_version=(
            _require_string(areas["provider_version"], "provider_version")
            if areas.get("provider_version") is not None
            else None
        ),
        operation_version=_require_integer(areas.get("operation_version", 1), "operation_version"),
        parameters=parameters,
    )


def _parse_background_options(value: object) -> BackgroundRemovalOptions:
    options = _require_mapping(value, "options")
    allowed = {
        "alpha_matting",
        "background_color",
        "background_threshold",
        "erode_size",
        "foreground_threshold",
    }
    _require_exact_keys(options, allowed, set())
    color_value = options.get("background_color")
    color: tuple[int, int, int, int] | None = None
    if color_value is not None:
        channels = _require_sequence(color_value, "background_color")
        if len(channels) != 4:
            raise ValueError("background_color requires four channels")
        color = tuple(_require_integer(channel, "background_color") for channel in channels)  # type: ignore[assignment]
    return BackgroundRemovalOptions(
        alpha_matting=_require_boolean(options.get("alpha_matting", False), "alpha_matting"),
        foreground_threshold=_require_integer(
            options.get("foreground_threshold", 240), "foreground_threshold"
        ),
        background_threshold=_require_integer(
            options.get("background_threshold", 10), "background_threshold"
        ),
        erode_size=_require_integer(options.get("erode_size", 10), "erode_size"),
        background_color=color,
    )


def _invoke(name: str, capability: object, parsed: _ParsedArguments) -> object:
    if name == "image_remove_background_mapper":
        if not isinstance(capability, ImageRemoveBackgroundMapper):
            raise TypeError("background capability must be an ImageRemoveBackgroundMapper")
        return capability.map(
            parsed.sources,
            parsed.background_options,
            output_directory_fd=parsed.output_directory_fd,
            authorized_output_directory=parsed.authorized_output_directory,
        )
    if name == "imgdiff_difference_area_generator_mapper":
        return _call(capability, "map", parsed.sources)
    if name == "imgdiff_difference_caption_generator_mapper":
        if parsed.areas is None:
            raise ValueError("difference caption areas are required")
        return _call(capability, "map", parsed.sources, parsed.areas)
    if name in {"image_text_matching_filter", "image_text_similarity_filter"}:
        if parsed.text is None:
            raise ValueError("text is required")
        return _call(capability, "evaluate", parsed.sources, parsed.text)
    if name in {"image_perceptual_deduplicator", "ray_image_perceptual_deduplicator"}:
        return _call(capability, "group", parsed.records)
    if name == "image_phash":
        method = getattr(capability, "process", None)
        if not callable(method):
            raise TypeError("image pHash capability must implement process")
        return method(
            parsed.records,
            max_hamming_distance=parsed.max_hamming_distance,
            enable_lsh=parsed.enable_lsh,
        )
    if name == "image_lsh_search":
        if parsed.hash_version is None:
            raise ValueError("hash_version is required")
        method = getattr(capability, "search", None)
        if not callable(method):
            raise TypeError("image LSH capability must implement search")
        return method(
            parsed.lsh_query_hashes,
            parsed.lsh_candidates,
            hash_version=parsed.hash_version,
            query_text_hash=parsed.query_text_hash,
            max_hamming_distance=parsed.max_hamming_distance,
            top_k=parsed.top_k,
        )
    if name == "image_ocr":
        if not isinstance(capability, ImageOcr):
            raise TypeError("OCR capability must be an ImageOcr")
        return capability.recognize(parsed.sources, parsed.languages)
    return _call(capability, "evaluate", parsed.sources)


def _capability_method(name: str) -> str:
    if name in {
        "image_remove_background_mapper",
        "imgdiff_difference_area_generator_mapper",
        "imgdiff_difference_caption_generator_mapper",
    }:
        return "map"
    if name in {
        "image_perceptual_deduplicator",
        "ray_image_perceptual_deduplicator",
    }:
        return "group"
    if name == "image_phash":
        return "process"
    if name == "image_lsh_search":
        return "search"
    if name == "image_ocr":
        return "recognize"
    return "evaluate"


def _call(capability: object, method_name: str, *arguments: object) -> object:
    method = getattr(capability, method_name, None)
    if not callable(method):
        raise TypeError(f"image capability does not implement {method_name}")
    return method(*arguments)


def _validate_result_for_tool(name: str, result: object) -> None:
    expected: type[object]
    if name == "image_phash":
        if not isinstance(result, ImagePhashResult):
            raise RuntimeError("pHash capability returned an incompatible result")
        return
    if name == "image_lsh_search":
        if not isinstance(result, LshSearchResult):
            raise RuntimeError("LSH capability returned an incompatible result")
        return
    if name == "image_ocr":
        if not isinstance(result, tuple) or not all(
            isinstance(item, ImageOcrResult) for item in result
        ):
            raise RuntimeError("OCR capability returned an incompatible result")
        return
    if name == "image_remove_background_mapper":
        if not isinstance(result, tuple) or not all(
            isinstance(item, DerivedImage) for item in result
        ):
            raise RuntimeError("background capability returned an incompatible result")
        return
    if name == "imgdiff_difference_area_generator_mapper":
        expected = DifferenceAreaResult
    elif name == "imgdiff_difference_caption_generator_mapper":
        expected = DifferenceCaptionResult
    elif name in {
        "image_perceptual_deduplicator",
        "ray_image_perceptual_deduplicator",
    }:
        expected = PerceptualDeduplicationResult
    else:
        expected = FilterDecision
    if not isinstance(result, expected):
        raise RuntimeError("image capability returned an incompatible result")


def _serialize_result(
    result: object,
    tool_name: str,
    ocr_languages: tuple[str, ...] | None = None,
) -> tuple[dict[str, object], tuple[str, ...]]:
    if isinstance(result, ImagePhashResult):
        evidence = tuple(
            dict.fromkeys(
                source_hash for record in result.records for source_hash in record.source_hashes
            )
        )
        return (
            {
                "records": [_json_value(record) for record in result.records],
                "duplicate_groups": [_json_value(group) for group in result.groups],
                "retained_record_ids": list(result.retained_record_ids),
                "lineage": {
                    "provider_id": result.provider_id,
                    "provider_version": result.provider_version,
                    "hash_version": result.hash_version,
                    "bit_width": result.bit_width,
                    "max_hamming_distance": result.max_hamming_distance,
                    "candidate_strategy": result.candidate_strategy,
                    "candidate_count": result.candidate_count,
                    "comparison_count": result.comparison_count,
                },
            },
            evidence,
        )
    if isinstance(result, LshSearchResult):
        return (
            {
                "matches": [
                    {
                        "record_id": match.record_id,
                        "hashes": [f"{value:016x}" for value in match.hashes],
                        "distance": match.distance,
                    }
                    for match in result.matches
                ],
                "lineage": {
                    "hash_version": result.hash_version,
                    "bit_width": result.bit_width,
                    "max_hamming_distance": result.max_hamming_distance,
                    "candidate_strategy": result.candidate_strategy,
                    "candidate_count": result.candidate_count,
                    "comparison_count": result.comparison_count,
                },
            },
            (),
        )
    if tool_name == "image_ocr":
        if not isinstance(result, tuple) or not all(
            isinstance(item, ImageOcrResult) for item in result
        ):
            raise TypeError("OCR capability returned an unsupported result type")
        ocr_results = tuple(item for item in result if isinstance(item, ImageOcrResult))
        evidence = tuple(dict.fromkeys(item.source_hash for item in ocr_results))
        serialized_results = [
            {
                "source_hash": item.source_hash,
                "full_text": item.full_text,
                "words": [_json_value(word) for word in item.words],
                "provider_id": item.provider_id,
                "provider_version": item.provider_version,
                "languages": list(item.languages),
                "operation_version": item.operation_version,
                "parameters": dict(item.parameters),
            }
            for item in ocr_results
        ]
        if ocr_results:
            first_ocr_result = ocr_results[0]
            lineage = _lineage(
                tuple(item.source_hash for item in ocr_results),
                first_ocr_result.operation_version,
                first_ocr_result.parameters,
                provider_id=first_ocr_result.provider_id,
                provider_version=first_ocr_result.provider_version,
            )
        else:
            effective_languages = ocr_languages or DEFAULT_OCR_LANGUAGES
            lineage = _lineage(
                (),
                1,
                (("languages", "+".join(effective_languages)),),
            )
        return {"results": serialized_results, "lineage": lineage}, evidence
    if isinstance(result, FilterDecision):
        lineage = _lineage(result.source_hashes, result.operation_version, result.parameters)
        return (
            {
                "applicable": result.applicable,
                "keep": result.keep,
                "metrics": [_json_value(metric) for metric in result.metrics],
                "lineage": lineage,
            },
            result.source_hashes,
        )
    if isinstance(result, tuple) and all(isinstance(item, DerivedImage) for item in result):
        artifacts = tuple(item for item in result if isinstance(item, DerivedImage))
        evidence = tuple(
            dict.fromkeys((*(_hashes(artifacts)), *(item.sha256 for item in artifacts)))
        )
        if artifacts:
            first_artifact = artifacts[0]
            lineage = _lineage(
                _hashes(artifacts),
                first_artifact.operation_version,
                first_artifact.parameters,
                model_id=first_artifact.model_id,
                model_revision=first_artifact.model_revision,
                model_manifest_sha256=first_artifact.model_manifest_sha256,
                preprocess_revision=first_artifact.preprocess_revision,
                device=first_artifact.device,
                provider_id=first_artifact.provider_id,
                provider_version=first_artifact.provider_version,
            )
        else:
            lineage = _lineage((), 1, ())
        lineage["artifacts"] = [item.sha256 for item in artifacts]
        return (
            {
                "artifacts": [_json_value(item) for item in artifacts],
                "lineage": lineage,
            },
            evidence,
        )
    if isinstance(result, DifferenceAreaResult):
        return (
            {
                "regions": [_json_value(box) for box in result.boxes],
                "lineage": _lineage(
                    result.source_hashes,
                    result.operation_version,
                    result.parameters,
                    provider_id=result.provider_id,
                    provider_version=result.provider_version,
                    stages=[_json_value(stage) for stage in result.stage_evidence],
                ),
            },
            result.source_hashes,
        )
    if isinstance(result, DifferenceCaptionResult):
        return (
            {
                "regions": [_json_value(region) for region in result.regions],
                "lineage": _lineage(
                    result.source_hashes,
                    result.operation_version,
                    result.parameters,
                    provider_id=result.provider_id,
                    provider_version=result.provider_version,
                    model_id=result.model_id,
                    model_revision=result.model_revision,
                    model_manifest_sha256=result.model_manifest_sha256,
                    preprocess_revision=result.preprocess_revision,
                    device=result.device,
                ),
            },
            result.source_hashes,
        )
    if isinstance(result, PerceptualDeduplicationResult):
        value = _json_value(result)
        if not isinstance(value, dict):  # pragma: no cover - dataclass serialization invariant
            raise TypeError("deduplication result must serialize to an object")
        groups = value.pop("groups")
        retained = value.pop("retained_record_ids")
        record_hashes = result.record_source_hashes
        evidence = tuple(
            source_hash for _, source_hashes in record_hashes for source_hash in source_hashes
        )
        return (
            {
                "duplicate_groups": groups,
                "retained_record_ids": retained,
                "lineage": value,
            },
            tuple(dict.fromkeys(evidence)),
        )
    raise TypeError("image capability returned an unsupported result type")


def _lineage(
    source_hashes: tuple[str, ...],
    operation_version: int,
    parameters: tuple[tuple[str, object], ...],
    **provider: object,
) -> dict[str, object]:
    lineage: dict[str, object] = {
        "operation_version": operation_version,
        "parameters": {key: value for key, value in parameters},
        "source_hashes": list(source_hashes),
    }
    lineage.update({key: value for key, value in provider.items() if value is not None})
    return lineage


def _json_value(value: object) -> object:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _json_value(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _hashes(artifacts: tuple[DerivedImage, ...]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for artifact in artifacts for value in artifact.source_hashes))


def _authorize_derived_outputs(result: object, write_roots: tuple[Path, ...]) -> None:
    if (
        not isinstance(result, tuple)
        or not result
        or not all(isinstance(item, DerivedImage) for item in result)
    ):
        return
    if not write_roots:
        raise _PathPolicyError("no image write roots are authorized")
    for item in result:
        if isinstance(item, DerivedImage):
            _authorize_path(Path(item.path), write_roots)


def _authorize_capability_configuration(
    name: str,
    capability: object,
    write_roots: tuple[Path, ...],
) -> _AuthorizedOutputDirectory | None:
    if name != "image_remove_background_mapper":
        return None
    if not isinstance(capability, ImageRemoveBackgroundMapper):
        raise TypeError("background capability must be an ImageRemoveBackgroundMapper")
    if not write_roots:
        raise _PathPolicyError("no image write roots are authorized")
    output_directory = _authorize_path(capability.output_directory, write_roots)
    return _AuthorizedOutputDirectory(
        descriptor=_open_authorized_directory(output_directory, write_roots),
        path=output_directory,
    )


def _authorize_path(path: Path, roots: tuple[Path, ...]) -> Path:
    expanded = path.expanduser()
    resolved = (
        _require_windows_local_path(expanded) if os.name == "nt" else expanded.resolve(strict=False)
    )
    if not any(resolved.is_relative_to(root) for root in roots):
        raise _PathPolicyError(str(path))
    return resolved


def _require_windows_local_path(path: _WindowsPathT) -> _WindowsPathT:
    """Check local-drive path syntax without touching a filesystem or network."""
    if (
        not path.is_absolute()
        or re.fullmatch(r"[A-Za-z]:", path.drive) is None
        or ".." in path.parts
        or any(":" in part for part in path.parts[1:])
    ):
        raise _PathPolicyError(str(path))
    return path


def _read_authorized_regular_file(
    path: Path,
    roots: tuple[Path, ...],
    root_identities: Mapping[Path, tuple[int, int] | None],
    *,
    maximum_bytes: int,
) -> bytes:
    """Read a bounded regular file through descriptors rooted in an approved directory."""
    matching_roots = tuple(root for root in roots if path.is_relative_to(root))
    if not matching_roots:
        raise _PathPolicyError(str(path))
    root = max(matching_roots, key=lambda item: len(item.parts))
    relative = path.relative_to(root)
    if not relative.parts:
        raise ValueError("image source must identify a regular file")
    if os.name == "nt":
        from capicove._windows_files import WindowsPathPolicyError, read_windows_secure_file

        try:
            return read_windows_secure_file(
                path,
                max_bytes=maximum_bytes,
                owner_required=False,
                private_acl=False,
            )
        except WindowsPathPolicyError as error:
            raise _PathPolicyError(str(path)) from error
    _require_descriptor_relative_support(("open",))

    directory_flags = os.O_RDONLY | os.O_DIRECTORY
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(root, directory_flags | no_follow)
    try:
        metadata = os.fstat(directory_fd)
        expected = root_identities.get(root)
        if expected is None or (metadata.st_dev, metadata.st_ino) != expected:
            raise _PathPolicyError(str(path))
        for part in relative.parts[:-1]:
            child_fd = os.open(
                part,
                directory_flags | no_follow,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
        file_flags = os.O_RDONLY | no_follow | getattr(os, "O_NONBLOCK", 0)
        file_fd = os.open(relative.parts[-1], file_flags, dir_fd=directory_fd)
        try:
            return _read_bounded_regular_fd(file_fd, maximum_bytes)
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def _read_bounded_regular_fd(descriptor: int, maximum_bytes: int) -> bytes:
    metadata = os.fstat(descriptor)
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("image source must be a regular file")
    if metadata.st_size > maximum_bytes:
        raise ValueError("image file exceeds the per-source byte limit")
    content = bytearray()
    while len(content) <= maximum_bytes:
        chunk = os.read(descriptor, min(1024 * 1024, maximum_bytes + 1 - len(content)))
        if not chunk:
            break
        content.extend(chunk)
    if len(content) > maximum_bytes:
        raise ValueError("image file exceeds the per-source byte limit")
    return bytes(content)


def _open_authorized_directory(path: Path, roots: tuple[Path, ...]) -> int:
    """Open or create an approved directory without following mutable child links."""
    _require_descriptor_relative_support(("open", "mkdir", "link", "unlink"))
    matching_roots = tuple(root for root in roots if path.is_relative_to(root))
    if not matching_roots:
        raise _PathPolicyError(str(path))
    root = max(matching_roots, key=lambda item: len(item.parts))
    root.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    no_follow = getattr(os, "O_NOFOLLOW", 0)
    directory_fd = os.open(root, directory_flags | no_follow)
    try:
        for part in path.relative_to(root).parts:
            try:
                os.mkdir(part, dir_fd=directory_fd)
            except FileExistsError:
                pass
            child_fd = os.open(
                part,
                directory_flags | no_follow,
                dir_fd=directory_fd,
            )
            os.close(directory_fd)
            directory_fd = child_fd
        return directory_fd
    except Exception:
        os.close(directory_fd)
        raise


def _require_descriptor_relative_support(operations: tuple[str, ...]) -> None:
    supported_operations = {getattr(operation, "__name__", "") for operation in os.supports_dir_fd}
    if not hasattr(os, "O_DIRECTORY") or any(
        operation not in supported_operations for operation in operations
    ):
        raise RuntimeError("descriptor-relative image filesystem operations are unavailable")


def _require_exact_keys(
    value: Mapping[str, object],
    allowed: set[str],
    required: set[str],
) -> None:
    if not required.issubset(value) or not set(value).issubset(allowed):
        raise ValueError("arguments contain unknown or missing fields")


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{name} must be an object")
    if not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} keys must be strings")
    return value


def _require_sequence(value: object, name: str) -> Sequence[object]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise TypeError(f"{name} must be an array")
    return value


def _require_string(
    value: object,
    name: str,
    *,
    maximum_length: int | None = None,
) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{name} must be a nonblank string")
    if maximum_length is not None and len(value) > maximum_length:
        raise ValueError(f"{name} exceeds its maximum length")
    return value


def _require_number(value: object, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise TypeError(f"{name} must be numeric")
    return float(value)


def _require_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    return value


def _bounded_integer(
    value: object,
    name: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    parsed = _require_integer(value, name)
    if parsed < minimum or parsed > maximum:
        raise ValueError(f"{name} must be within [{minimum}, {maximum}]")
    return parsed


def _require_boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{name} must be a boolean")
    return value


def _require_scalar(value: object) -> str | int | float | bool | None:
    if not isinstance(value, (str, int, float, bool, type(None))):
        raise TypeError("operation parameters must be scalar")
    return value
