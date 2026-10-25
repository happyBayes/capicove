"""Closed, versioned user configuration for source-preserving book chunking."""

from __future__ import annotations

import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path


CONTENT_TYPES = (
    "definition",
    "principle",
    "proof",
    "procedure",
    "example",
    "comparison",
    "code",
    "table",
    "formula",
)
PRESETS: dict[str, dict[str, object]] = {
    "textbook-balanced": {"protected_types": ("definition", "example", "procedure", "formula")},
    "technical-code": {
        "protected_types": ("code", "procedure", "table"),
        "page_boundary": "strict",
    },
    "math-heavy": {"protected_types": ("definition", "principle", "proof", "formula", "example")},
}


@dataclass(frozen=True, slots=True, init=False)
class ChunkingConfig:
    """Separate chunk-affecting policy from bounded original-context reads."""

    preset: str = "textbook-balanced"
    target_core_tokens: int = 600
    max_chunk_tokens: int = 1200
    context_before_tokens: int = 100
    context_after_tokens: int = 0
    boundary_level: str = "deepest"
    tokenizer: str = "utf8-bytes-v1"
    protected_types: tuple[str, ...] = ()
    oversize_policy: str = "split"
    page_boundary: str = "allow_continuations"
    merge_small_chunks: bool = True
    front_matter_policy: str = "navigation-v1"
    max_context_read_tokens: int = 2000

    def __init__(
        self,
        *,
        preset: str = "textbook-balanced",
        target_core_tokens: int = 600,
        max_chunk_tokens: int = 1200,
        context_before_tokens: int = 100,
        context_after_tokens: int = 0,
        boundary_level: str = "deepest",
        tokenizer: str = "utf8-bytes-v1",
        protected_types: tuple[str, ...] | None = None,
        oversize_policy: str = "split",
        page_boundary: str | None = None,
        merge_small_chunks: bool = True,
        front_matter_policy: str = "navigation-v1",
        max_context_read_tokens: int = 2000,
    ) -> None:
        if not isinstance(preset, str) or preset not in PRESETS:
            raise ValueError("unknown chunk preset")
        values = dict(locals())
        del values["self"]
        if protected_types is None:
            values["protected_types"] = PRESETS[preset]["protected_types"]
        if page_boundary is None:
            values["page_boundary"] = PRESETS[preset].get("page_boundary", "allow_continuations")
        for name, value in values.items():
            object.__setattr__(self, name, value)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not isinstance(self.preset, str) or self.preset not in PRESETS:
            raise ValueError("unknown chunk preset")
        for name in ("target_core_tokens", "max_chunk_tokens", "max_context_read_tokens"):
            integer_limit(name, getattr(self, name), minimum=1, maximum=32768)
        for name in ("context_before_tokens", "context_after_tokens"):
            integer_limit(name, getattr(self, name), minimum=0, maximum=32768)
        if self.target_core_tokens > self.max_chunk_tokens:
            raise ValueError("target_core_tokens exceeds max_chunk_tokens")
        if self.context_before_tokens + self.context_after_tokens >= self.max_chunk_tokens:
            raise ValueError("context budgets must leave room for core text")
        for name, allowed in (
            ("boundary_level", ("deepest", "section", "chapter")),
            ("oversize_policy", ("split", "error")),
            ("page_boundary", ("strict", "allow_continuations")),
            ("front_matter_policy", ("navigation-v1", "preserve")),
        ):
            if getattr(self, name) not in allowed:
                raise ValueError(f"invalid {name}")
        if not isinstance(self.merge_small_chunks, bool):
            raise TypeError("merge_small_chunks must be boolean")
        if not isinstance(self.protected_types, (tuple, list)):
            raise TypeError("protected_types must be a list of content types")
        protected = tuple(self.protected_types)
        if any(not isinstance(item, str) or item not in CONTENT_TYPES for item in protected):
            raise ValueError("unknown protected content type")
        if len(protected) != len(set(protected)):
            raise ValueError("protected_types must be unique")
        object.__setattr__(self, "protected_types", tuple(sorted(protected)))
        if not isinstance(self.tokenizer, str) or not (
            self.tokenizer == "utf8-bytes-v1"
            or (self.tokenizer.startswith("local:") and Path(self.tokenizer[6:]).is_absolute())
        ):
            raise ValueError("tokenizer must be utf8-bytes-v1 or local:/absolute/tokenizer.json")

    def chunk_values(self) -> dict[str, object]:
        """Return canonical chunk inputs, excluding execution-only read budgets."""
        values = asdict(self)
        del values["max_context_read_tokens"]
        if self.front_matter_policy == "preserve":
            del values["front_matter_policy"]
        return values


def persisted_chunk_config(values: Mapping[str, object]) -> ChunkingConfig:
    """Interpret missing policy fields using their original persisted semantics."""
    return resolve_chunk_config({"front_matter_policy": "preserve", **values})


def integer_limit(name: str, value: object, *, minimum: int, maximum: int) -> int:
    """Require an explicit integer within an inclusive service limit."""
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"{name} must be an integer in [{minimum}, {maximum}]")
    return value


def resolve_chunk_config(
    file_values: Mapping[str, object] | None = None,
    overrides: Mapping[str, object] | None = None,
) -> ChunkingConfig:
    """Resolve preset defaults, file fields, then explicit caller overrides."""
    supplied: dict[str, object] = {}
    known = {field.name for field in fields(ChunkingConfig)}
    for values in (file_values, overrides):
        if values is not None:
            if not isinstance(values, Mapping) or any(key not in known for key in values):
                raise ValueError("unknown chunking configuration field")
            supplied.update(values)
    preset = supplied.get("preset", "textbook-balanced")
    if not isinstance(preset, str) or preset not in PRESETS:
        raise ValueError("unknown chunk preset")
    resolved = {**asdict(ChunkingConfig()), **PRESETS[preset], **supplied}
    return ChunkingConfig(**resolved)


def load_chunk_config(path: Path) -> dict[str, object]:
    """Read an explicitly selected, bounded TOML file with one chunking table."""
    if path.is_symlink() or not path.is_file() or path.stat().st_size > 65536:
        raise ValueError("chunking config must be a regular TOML file at most 64 KiB")
    value = tomllib.loads(path.read_text(encoding="utf-8"))
    if set(value) != {"chunking"} or not isinstance(value["chunking"], dict):
        raise ValueError("configuration requires only a [chunking] table")
    resolve_chunk_config(value["chunking"])
    return dict(value["chunking"])
