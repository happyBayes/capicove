"""Closed, versioned policies for deterministic graph construction."""

import json
import math
import re
from dataclasses import asdict, dataclass, field
from typing import Literal, TypeAlias

from capicove.graph_build.codec import _pairs, _record, _validate_json, canonical_bytes
from capicove.model.local_assets import LocalModelAssets


def _text(name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonblank text")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError:
        raise ValueError(f"{name} must be valid UTF-8 text") from None


def _positive_integer(name: str, value: object) -> None:
    if type(value) is not int:
        raise TypeError(f"{name} must be an integer")
    if value < 1:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class LocalGenerationProfile:
    """Explicit local runtime configuration, separate from canonical contracts.

    Revisions are either sha256:<manifest digest> or a pinned hexadecimal source
    revision recorded as _commit_hash in the corresponding verified config.
    Model/tokenizer IDs are sha256:<manifest digest>, independent of local paths.
    """

    assets: LocalModelAssets
    tokenizer_assets: LocalModelAssets
    model_revision: str
    tokenizer_revision: str
    device: str
    context_limit: int
    max_new_tokens: int

    def __post_init__(self) -> None:
        if not isinstance(self.assets, LocalModelAssets) or not isinstance(
            self.tokenizer_assets, LocalModelAssets
        ):
            raise TypeError("invalid_local_assets")
        for revision in (self.model_revision, self.tokenizer_revision):
            if (
                not isinstance(revision, str)
                or re.fullmatch(r"(?:sha256:[0-9a-f]{64}|[0-9a-f]{40}|[0-9a-f]{64})", revision)
                is None
            ):
                raise ValueError("invalid_local_revision")
        if (
            not isinstance(self.device, str)
            or re.fullmatch(r"cpu|mps|cuda:[0-9]+", self.device) is None
        ):
            raise ValueError("invalid_local_device")
        _positive_integer("context_limit", self.context_limit)
        _positive_integer("max_new_tokens", self.max_new_tokens)
        if self.max_new_tokens >= self.context_limit:
            raise ValueError("invalid_output_reservation")


@dataclass(frozen=True, slots=True)
class ChunkingProfile:
    """Soft packing target, hard core limit, and total noncore token budget.

    These are not a model-context guarantee. The inference stage must preflight
    the complete prompt, schema, separators, and reserved output separately.
    """

    target_tokens: int = 800
    max_tokens: int = 1200
    context_tokens: int = 100
    version: int = 1

    def __post_init__(self) -> None:
        for name in ("target_tokens", "max_tokens", "context_tokens", "version"):
            value = getattr(self, name)
            if type(value) is not int:
                raise TypeError("Chunking policy fields must be integers")
            if value < (0 if name == "context_tokens" else 1):
                raise ValueError("Chunking policy field is below its minimum")
        if self.target_tokens > self.max_tokens:
            raise ValueError("Target cannot exceed the hard core token limit")
        if self.version != 1:
            raise ValueError("Unsupported chunking policy version")


@dataclass(frozen=True, slots=True)
class InferencePolicy:
    """Pin model identity, decoding, concurrency, and invocation budgets."""

    provider_id: str
    model_id: str
    model_revision: str
    tokenizer_id: str
    tokenizer_revision: str
    input_role: str
    context_window_tokens: int
    max_output_tokens: int = 4096
    concurrent_units: int = 2
    total_calls: int = 3
    max_repairs: int = 1
    call_deadline_seconds: int = 120
    build_deadline_seconds: int = 3600
    temperature: float = 0.0
    version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "provider_id",
            "model_id",
            "model_revision",
            "tokenizer_id",
            "tokenizer_revision",
            "input_role",
        ):
            _text(name, getattr(self, name))
        for name in (
            "context_window_tokens",
            "max_output_tokens",
            "concurrent_units",
            "total_calls",
            "call_deadline_seconds",
            "build_deadline_seconds",
            "version",
        ):
            _positive_integer(name, getattr(self, name))
        if type(self.max_repairs) is not int or not 0 <= self.max_repairs < self.total_calls:
            raise ValueError("max_repairs must fit inside the total call budget")
        if self.total_calls > 3 or self.max_repairs > 1 or self.concurrent_units > 2:
            raise ValueError("inference policy exceeds runner hard bounds")
        if type(self.temperature) not in (int, float) or isinstance(self.temperature, bool):
            raise TypeError("temperature must be numeric")
        if not math.isfinite(float(self.temperature)) or not 0 <= self.temperature <= 2:
            raise ValueError("temperature must be finite and between zero and two")
        if self.context_window_tokens <= self.max_output_tokens:
            raise ValueError("context window must reserve space for non-output tokens")
        if self.version != 1:
            raise ValueError("Unsupported inference policy version")


@dataclass(frozen=True, slots=True)
class ExtractionLimits:
    """Bound untrusted extraction requests, responses, and accepted records."""

    max_entities: int = 64
    max_relationships: int = 128
    max_response_bytes: int = 262_144
    max_request_bytes: int = 1_048_576
    max_text_codepoints: int = 4096
    max_quote_codepoints: int = 8192
    max_spans_per_record: int = 64
    version: int = 1

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            _positive_integer(name, getattr(self, name))
        if self.max_entities > 64 or self.max_relationships > 128:
            raise ValueError("record limits exceed the packaged extraction schema")
        if self.max_text_codepoints > 4096 or self.max_quote_codepoints > 8192:
            raise ValueError("text limits exceed the packaged extraction schema")
        if self.max_spans_per_record > 64:
            raise ValueError("span limit exceeds the packaged extraction schema")
        if self.version != 1:
            raise ValueError("Unsupported extraction limits version")


@dataclass(frozen=True, slots=True)
class ResolutionProfile:
    """Bound candidate work with absolute pool and pair-batch stage limits."""

    top_k: int = 10
    pair_batch_size: int = 2000
    identity_pool_limit: int = 2000
    qualification_ref: str | None = None
    version: int = 1

    def __post_init__(self) -> None:
        for name in ("top_k", "pair_batch_size", "identity_pool_limit", "version"):
            _positive_integer(name, getattr(self, name))
        if self.identity_pool_limit > 10_000:
            raise ValueError("identity_pool_limit must not exceed 10000")
        if self.pair_batch_size > 2000:
            raise ValueError("pair_batch_size must not exceed 2000")
        if self.qualification_ref is not None:
            _text("qualification_ref", self.qualification_ref)
        if self.version != 1:
            raise ValueError("Unsupported resolution profile version")


@dataclass(frozen=True, slots=True)
class LouvainPartitionProfile:
    """Closed parameters for the explicitly selected Louvain adapter."""

    resolution: float = 1.0
    threshold: float = 1e-7
    max_level: int = 10
    seed: int = 42
    version: int = 1
    tag: Literal["louvain"] = field(default="louvain", init=False)

    def __post_init__(self) -> None:
        for name in ("max_level", "version"):
            _positive_integer(name, getattr(self, name))
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        for name in ("resolution", "threshold"):
            value = getattr(self, name)
            if type(value) not in (int, float):
                raise TypeError(f"{name} must be numeric")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
            object.__setattr__(self, name, float(value))
        if self.version != 1:
            raise ValueError("Unsupported Louvain partition profile version")


# Extend only with separately reviewed variants; arbitrary mappings are not profiles.
PartitionProfile: TypeAlias = LouvainPartitionProfile


@dataclass(frozen=True, slots=True)
class CommunityProfile:
    """Bound association hierarchy separately from partition optimization."""

    max_size: int = 50
    max_depth: int = 3
    support_cap: int = 20
    partition_profile: PartitionProfile = LouvainPartitionProfile()
    version: int = 2

    def __post_init__(self) -> None:
        for name in ("max_size", "support_cap", "version"):
            _positive_integer(name, getattr(self, name))
        if type(self.max_depth) is not int or self.max_depth < 0:
            raise ValueError("max_depth must be a nonnegative integer")
        if type(self.partition_profile) is not LouvainPartitionProfile:
            raise TypeError("partition_profile must be a supported closed variant")
        if self.version != 2:
            raise ValueError("Unsupported community profile version")


@dataclass(frozen=True, slots=True)
class GraphBuildProfile:
    """Pin every semantic and bounded operational graph-build setting."""

    profile_id: str
    ontology_version: str
    chunking: ChunkingProfile
    inference: InferencePolicy
    generation_profile_id: str
    identity_embedding_space: str
    retrieval_embedding_space: str
    extraction_limits: ExtractionLimits = ExtractionLimits()
    resolution: ResolutionProfile = ResolutionProfile()
    community: CommunityProfile = CommunityProfile()
    extraction_prompt_version: str = "extract-v1"
    extraction_schema_version: str = "extraction-v1"
    extraction_validation_version: int = 1
    report_prompt_version: str = "report-v1"
    report_schema_version: str = "report-v1"
    diagnostic_prompt_version: str = "diagnostic-v1"
    diagnostic_schema_version: str = "diagnostic-v1"
    version: int = 1

    def __post_init__(self) -> None:
        for name in (
            "profile_id",
            "ontology_version",
            "generation_profile_id",
            "identity_embedding_space",
            "retrieval_embedding_space",
            "extraction_prompt_version",
            "extraction_schema_version",
            "report_prompt_version",
            "report_schema_version",
            "diagnostic_prompt_version",
            "diagnostic_schema_version",
        ):
            _text(name, getattr(self, name))
        if not isinstance(self.chunking, ChunkingProfile):
            raise TypeError("chunking must be a ChunkingProfile")
        if not isinstance(self.inference, InferencePolicy):
            raise TypeError("inference must be an InferencePolicy")
        if not isinstance(self.extraction_limits, ExtractionLimits):
            raise TypeError("extraction_limits must be ExtractionLimits")
        if not isinstance(self.resolution, ResolutionProfile):
            raise TypeError("resolution must be a ResolutionProfile")
        if not isinstance(self.community, CommunityProfile):
            raise TypeError("community must be a CommunityProfile")
        _positive_integer("extraction_validation_version", self.extraction_validation_version)
        _positive_integer("version", self.version)
        if self.version != 1:
            raise ValueError("Unsupported graph-build profile version")


@dataclass(frozen=True, slots=True)
class StructuredRequest:
    """Canonical provider request without occurrence IDs or source coordinates."""

    role: str
    schema_version: str
    schema_payload: bytes
    input_segments: tuple[tuple[str, str, str], ...]
    section_path: tuple[str, ...]
    prompt_text: str
    prompt_version: str
    ontology_version: str
    validation_version: int
    generation_profile_id: str
    inference_policy: InferencePolicy
    profile_fingerprint: str
    max_output_tokens: int
    output_limits: ExtractionLimits
    canonical_payload: bytes
    request_fingerprint: str

    def __post_init__(self) -> None:
        for name in (
            "role",
            "schema_version",
            "prompt_text",
            "prompt_version",
            "ontology_version",
            "generation_profile_id",
        ):
            _text(name, getattr(self, name))
        _positive_integer("validation_version", self.validation_version)
        _positive_integer("max_output_tokens", self.max_output_tokens)
        if not isinstance(self.inference_policy, InferencePolicy):
            raise TypeError("inference_policy must be InferencePolicy")
        if not isinstance(self.output_limits, ExtractionLimits):
            raise TypeError("output_limits must be ExtractionLimits")
        if not isinstance(self.schema_payload, bytes) or not isinstance(
            self.canonical_payload, bytes
        ):
            raise TypeError("schema and canonical payloads must be bytes")
        import hashlib
        import re

        for name in ("profile_fingerprint", "request_fingerprint"):
            if re.fullmatch(r"[0-9a-f]{64}", getattr(self, name)) is None:
                raise ValueError(f"{name} must be a lowercase SHA-256 hash")
        if hashlib.sha256(self.canonical_payload).hexdigest() != self.request_fingerprint:
            raise ValueError("request fingerprint does not match canonical payload")
        try:
            payload = json.loads(self.canonical_payload, object_pairs_hook=_pairs)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("canonical request payload is invalid JSON") from exc
        _validate_json(payload)
        if not isinstance(payload, dict):
            raise ValueError("canonical request payload must be an object")
        try:
            schema = json.loads(self.schema_payload, object_pairs_hook=_pairs)
        except (UnicodeError, json.JSONDecodeError, RecursionError) as exc:
            raise ValueError("schema payload is invalid JSON") from exc
        exposed = {
            "role": self.role,
            "schema_version": self.schema_version,
            "schema": schema,
            "segments": [list(item) for item in self.input_segments],
            "section_path": list(self.section_path),
            "prompt": self.prompt_text,
            "prompt_version": self.prompt_version,
            "ontology_version": self.ontology_version,
            "validation_version": self.validation_version,
            "generation_profile_id": self.generation_profile_id,
            "inference": asdict(self.inference_policy),
            "profile_fingerprint": self.profile_fingerprint,
            "max_output_tokens": self.max_output_tokens,
            "output_limits": asdict(self.output_limits),
        }
        if payload != exposed:
            raise ValueError("exposed request fields do not match canonical payload")


def graph_build_profile_fingerprint(profile: GraphBuildProfile) -> str:
    """Return the stable digest of every closed graph-build policy field."""
    if not isinstance(profile, GraphBuildProfile):
        raise TypeError("Expected GraphBuildProfile")
    from capicove.graph_build.codec import fingerprint

    return fingerprint(asdict(profile))


def extraction_profile_fingerprint(profile: GraphBuildProfile) -> str:
    """Return the digest of policies that can change extraction requests."""
    if not isinstance(profile, GraphBuildProfile):
        raise TypeError("Expected GraphBuildProfile")
    from capicove.graph_build.codec import fingerprint

    return fingerprint(
        {
            "version": profile.version,
            "ontology_version": profile.ontology_version,
            "chunking": asdict(profile.chunking),
            "inference": asdict(profile.inference),
            "generation_profile_id": profile.generation_profile_id,
            "extraction_limits": asdict(profile.extraction_limits),
            "extraction_prompt_version": profile.extraction_prompt_version,
            "extraction_schema_version": profile.extraction_schema_version,
            "extraction_validation_version": profile.extraction_validation_version,
        }
    )


def graph_build_profile_to_json(profile: GraphBuildProfile) -> str:
    """Encode the complete closed graph-build profile."""
    if not isinstance(profile, GraphBuildProfile):
        raise TypeError("Expected GraphBuildProfile")
    return canonical_bytes(asdict(profile)).decode("utf-8")


def graph_build_profile_from_json(payload: str) -> GraphBuildProfile:
    """Decode a graph-build profile while rejecting every unknown nested field."""
    if not isinstance(payload, str):
        raise TypeError("Profile payload must be text")
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
    except (json.JSONDecodeError, RecursionError) as exc:
        raise ValueError("Invalid graph-build profile JSON") from exc
    _validate_json(value)
    data = _record(value, GraphBuildProfile).copy()
    data["chunking"] = ChunkingProfile(**_record(data["chunking"], ChunkingProfile))
    data["inference"] = InferencePolicy(**_record(data["inference"], InferencePolicy))
    data["extraction_limits"] = ExtractionLimits(
        **_record(data["extraction_limits"], ExtractionLimits)
    )
    data["resolution"] = ResolutionProfile(**_record(data["resolution"], ResolutionProfile))
    community = _record(data["community"], CommunityProfile).copy()
    partition = _record(community["partition_profile"], LouvainPartitionProfile).copy()
    if partition.pop("tag") != "louvain":
        raise ValueError("Unsupported partition profile tag")
    community["partition_profile"] = LouvainPartitionProfile(**partition)
    data["community"] = CommunityProfile(**community)
    return GraphBuildProfile(**data)


def chunking_profile_to_json(profile: ChunkingProfile) -> str:
    """Encode the closed versioned policy, without provider configuration."""
    if type(profile) is not ChunkingProfile:
        raise TypeError("Expected ChunkingProfile")
    return canonical_bytes(asdict(profile)).decode("utf-8")


def chunking_profile_from_json(payload: str) -> ChunkingProfile:
    """Reject duplicate, unknown, and incompatible policy fields."""
    if not isinstance(payload, str):
        raise TypeError("Policy payload must be text")
    try:
        value = json.loads(payload, object_pairs_hook=_pairs)
    except RecursionError as exc:
        raise ValueError("Policy JSON exceeds parser nesting bounds") from exc
    return ChunkingProfile(**_record(value, ChunkingProfile))
