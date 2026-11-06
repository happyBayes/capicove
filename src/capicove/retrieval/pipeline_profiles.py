"""Strict, versioned query-time pipeline profiles (RFC-0034, Proposed; experimental).

Profiles hold settings that were previously hardcoded in the knowledge-base,
offline text and unified text search paths. Every default profile reproduces the
previous constants exactly, so default rankings are unchanged. Each profile is a
closed JSON document (``capicove.retrieval-pipeline-profile.v1``) with a SHA-256
fingerprint over its canonical encoding.

Post-fusion reranking is disabled by default. When enabled, a missing or failing
reranker either fails the search (``on_unavailable: "fail"``) or returns the
fused order with an explicit degraded marker (``"degrade"``); it is never
skipped silently.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, ClassVar


PROFILE_SCHEMA = "capicove.retrieval-pipeline-profile.v1"
BGE_RERANKER_MODEL_ID = "BAAI/bge-reranker-v2-m3"
MAX_RERANK_DEPTH = 128
_PROFILE_ID = re.compile(r"^[a-z][a-z0-9.-]{0,63}$")
_BINDING = re.compile(r"^[a-z][a-z0-9_.-]{0,63}$")
_MODEL_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_./-]{0,127}$")
_MAX_PROFILE_BYTES = 65_536


class ProfileError(ValueError):
    """Report one stable, path-free profile validation failure."""


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def _positive_int(name: str, value: object, maximum: int) -> None:
    if type(value) is not int or not 1 <= value <= maximum:
        raise ProfileError(f"{name} must be an integer from 1 to {maximum}")


def _weight(name: str, value: object) -> float:
    if type(value) not in (int, float):
        raise ProfileError(f"{name} must be a finite non-negative number")
    number = float(value)  # type: ignore[arg-type]
    if not math.isfinite(number) or number < 0:
        raise ProfileError(f"{name} must be a finite non-negative number")
    return number


@dataclass(frozen=True, slots=True)
class RerankPolicy:
    """Optional post-fusion cross-encoder reranking of the fused head.

    ``candidate_depth`` fused candidates are reranked; the remaining order is
    preserved after them. ``binding`` names the host's model binding for the
    reranker. ``model_id`` must equal the loaded reranker's identity.
    """

    enabled: bool = False
    model_id: str = BGE_RERANKER_MODEL_ID
    binding: str = "retrieval.reranker"
    candidate_depth: int = 20
    on_unavailable: str = "fail"

    def __post_init__(self) -> None:
        if type(self.enabled) is not bool:
            raise ProfileError("rerank.enabled must be a boolean")
        if not isinstance(self.model_id, str) or not _MODEL_ID.fullmatch(self.model_id):
            raise ProfileError("rerank.model_id must be a model identifier")
        if not isinstance(self.binding, str) or not _BINDING.fullmatch(self.binding):
            raise ProfileError("rerank.binding must be a binding name")
        _positive_int("rerank.candidate_depth", self.candidate_depth, MAX_RERANK_DEPTH)
        if self.on_unavailable not in {"fail", "degrade"}:
            raise ProfileError("rerank.on_unavailable must be fail or degrade")

    @classmethod
    def from_mapping(cls, raw: object) -> RerankPolicy:
        """Decode the closed rerank object."""
        names = {item.name for item in fields(cls)}
        if not isinstance(raw, Mapping) or set(raw) != names:
            raise ProfileError("rerank must contain exactly the documented fields")
        return cls(**{name: raw[name] for name in names})

    def to_mapping(self) -> dict[str, Any]:
        """Encode the closed rerank object."""
        return {item.name: getattr(self, item.name) for item in fields(self)}


@dataclass(frozen=True, slots=True)
class _PipelineProfile:
    """Shared closed-document behavior for stack-specific pipeline profiles."""

    stack: ClassVar[str]
    profile_id: str
    version: int
    rerank: RerankPolicy

    def _validate_common(self) -> None:
        if not isinstance(self.profile_id, str) or not _PROFILE_ID.fullmatch(self.profile_id):
            raise ProfileError("profile_id must be a lowercase identifier")
        if type(self.version) is not int or self.version != 1:
            raise ProfileError("version must be 1")
        if not isinstance(self.rerank, RerankPolicy):
            raise ProfileError("rerank must be a RerankPolicy")

    def to_mapping(self) -> dict[str, Any]:
        """Return the closed JSON document for this profile."""
        body: dict[str, Any] = {"schema": PROFILE_SCHEMA, "stack": self.stack}
        for item in fields(self):
            value = getattr(self, item.name)
            body[item.name] = value.to_mapping() if isinstance(value, RerankPolicy) else value
        return body

    @property
    def fingerprint(self) -> str:
        """Return the SHA-256 of the canonical profile document."""
        return hashlib.sha256(_canonical(self.to_mapping())).hexdigest()

    def identity(self) -> dict[str, Any]:
        """Describe this profile for run records and result pinning."""
        return {"id": self.profile_id, "version": self.version, "fingerprint": self.fingerprint}

    @classmethod
    def from_mapping(cls, raw: object) -> Any:
        """Decode one closed profile document for this stack."""
        names = {item.name for item in fields(cls)}
        if not isinstance(raw, Mapping) or set(raw) != names | {"schema", "stack"}:
            raise ProfileError("profile must contain exactly the documented fields")
        if raw["schema"] != PROFILE_SCHEMA or raw["stack"] != cls.stack:
            raise ProfileError(f"profile must be a {cls.stack} pipeline profile")
        values = {name: raw[name] for name in names}
        values["rerank"] = RerankPolicy.from_mapping(raw["rerank"])
        return cls(**values)


@dataclass(frozen=True, slots=True)
class KnowledgeSearchProfile(_PipelineProfile):
    """Native knowledge-base chunk search.

    ``rrf_k`` applies to zero-based lane ranks, as before profiles existed.
    """

    stack: ClassVar[str] = "kb"
    profile_id: str = "kb-default-v1"
    version: int = 1
    rerank: RerankPolicy = field(default_factory=RerankPolicy)
    rrf_k: int = 60
    lexical_weight: float = 1.0
    dense_weight: float = 1.0
    default_limit: int = 8
    max_limit: int = 20
    graph_context_limit: int = 8

    def __post_init__(self) -> None:
        self._validate_common()
        _positive_int("rrf_k", self.rrf_k, 10_000)
        object.__setattr__(self, "lexical_weight", _weight("lexical_weight", self.lexical_weight))
        object.__setattr__(self, "dense_weight", _weight("dense_weight", self.dense_weight))
        if self.lexical_weight == 0 and self.dense_weight == 0:
            raise ProfileError("at least one lane must have positive weight")
        _positive_int("max_limit", self.max_limit, 100)
        _positive_int("default_limit", self.default_limit, self.max_limit)
        _positive_int("graph_context_limit", self.graph_context_limit, 100)


@dataclass(frozen=True, slots=True)
class OfflineTextSearchProfile(_PipelineProfile):
    """Offline FTS5 workspace text search (formerly inline ``local-default``)."""

    stack: ClassVar[str] = "offline-text"
    profile_id: str = "offline-text-default-v1"
    version: int = 1
    rerank: RerankPolicy = field(default_factory=RerankPolicy)
    rrf_k: int = 60
    lexical_candidate_count: int = 100

    def __post_init__(self) -> None:
        self._validate_common()
        _positive_int("rrf_k", self.rrf_k, 10_000)
        _positive_int("lexical_candidate_count", self.lexical_candidate_count, 1_000)


@dataclass(frozen=True, slots=True)
class UnifiedSearchProfile(_PipelineProfile):
    """Unified shared-space search; reranking applies to text-chunk results only."""

    stack: ClassVar[str] = "unified"
    profile_id: str = "unified-default-v1"
    version: int = 1
    rerank: RerankPolicy = field(default_factory=RerankPolicy)
    rrf_k: int = 60
    candidate_depth: int = 100
    dense_weight: float = 1.0
    lexical_weight: float = 1.0
    max_lexical_tokens: int = 64

    def __post_init__(self) -> None:
        self._validate_common()
        _positive_int("rrf_k", self.rrf_k, 10_000)
        _positive_int("candidate_depth", self.candidate_depth, 10_000)
        object.__setattr__(self, "dense_weight", _weight("dense_weight", self.dense_weight))
        object.__setattr__(self, "lexical_weight", _weight("lexical_weight", self.lexical_weight))
        if self.dense_weight == 0:
            raise ProfileError("the shared dense lane must have positive weight")
        _positive_int("max_lexical_tokens", self.max_lexical_tokens, 256)


PROFILE_TYPES: dict[str, type[Any]] = {
    "kb": KnowledgeSearchProfile,
    "offline-text": OfflineTextSearchProfile,
    "unified": UnifiedSearchProfile,
}


def default_knowledge_profile() -> KnowledgeSearchProfile:
    """Return the profile equal to the pre-profile knowledge-base constants."""
    return KnowledgeSearchProfile()


def default_offline_text_profile() -> OfflineTextSearchProfile:
    """Return the profile equal to the pre-profile offline text constants."""
    return OfflineTextSearchProfile()


def default_unified_profile() -> UnifiedSearchProfile:
    """Return the profile equal to the pre-profile unified search constants."""
    return UnifiedSearchProfile()


def parse_pipeline_profile(raw: object, stack: str) -> Any:
    """Decode a closed profile document for exactly one stack."""
    try:
        profile_type = PROFILE_TYPES[stack]
    except KeyError:
        raise ProfileError("unsupported pipeline profile stack") from None
    try:
        return profile_type.from_mapping(raw)
    except TypeError:
        raise ProfileError("profile fields have invalid types") from None


def load_pipeline_profile(path: Path, stack: str) -> Any:
    """Read a bounded profile file that rejects duplicate keys and NaN."""
    payload = Path(path).read_bytes()
    if len(payload) > _MAX_PROFILE_BYTES:
        raise ProfileError("profile exceeds 64 KiB")

    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ProfileError("profile contains a duplicate key")
            result[key] = value
        return result

    def constant(value: str) -> Any:
        raise ProfileError("profile contains a nonstandard number")

    try:
        raw = json.loads(payload, object_pairs_hook=unique, parse_constant=constant)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ProfileError("profile is not valid JSON") from None
    return parse_pipeline_profile(raw, stack)


__all__ = [
    "BGE_RERANKER_MODEL_ID",
    "MAX_RERANK_DEPTH",
    "PROFILE_SCHEMA",
    "PROFILE_TYPES",
    "KnowledgeSearchProfile",
    "OfflineTextSearchProfile",
    "ProfileError",
    "RerankPolicy",
    "UnifiedSearchProfile",
    "default_knowledge_profile",
    "default_offline_text_profile",
    "default_unified_profile",
    "load_pipeline_profile",
    "parse_pipeline_profile",
]
