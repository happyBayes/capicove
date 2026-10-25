"""Model-free S1 code representation: intake policy, chunks, symbols and paths (RFC-0033).

Every rule here is a versioned closed constant: changing one changes the code
policy fingerprint. Definitions come from lexical heuristics and say so; this
module makes no AST, type-resolution or call-graph claim.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath


POLICY_VERSION = "code-symbols-v1"
CHUNKER_VERSION = "capicove-code-lines-v1"
DEFINITION_METHOD = "lexical-heuristic-v1"
CLASSIFIER_VERSION = "code-query-classifier-v1"
PATH_NORMALIZATION = "posix-casefold-nfkc-v1"
CHUNK_TARGET_CHARACTERS = 1_800
MAX_CHUNK_JSON_BYTES = 2_400
OVERLAP_LINES = 2
MAX_LINE_LENGTH_FOR_SOURCE = 4_000
MAX_SOURCE_BYTES = 1024 * 1024

_LANGUAGES = {
    ".py": "python",
    ".pyi": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".rs": "rust",
    ".java": "java",
    ".kt": "kotlin",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".rb": "ruby",
    ".php": "php",
    ".swift": "swift",
    ".scala": "scala",
    ".sh": "shell",
    ".bash": "shell",
    ".sql": "sql",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".md": "markdown",
    ".markdown": "markdown",
    ".rst": "rst",
}
_LOCKFILES = frozenset(
    {
        "package-lock.json",
        "yarn.lock",
        "pnpm-lock.yaml",
        "poetry.lock",
        "uv.lock",
        "pipfile.lock",
        "cargo.lock",
        "gemfile.lock",
        "composer.lock",
        "go.sum",
    }
)
_VENDORED = frozenset(
    {"node_modules", "vendor", "third_party", "dist", "build", "__pycache__", "site-packages"}
)
_SECRETS = (
    re.compile(rb"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{34,}\b"),
)
_GENERATED = re.compile(rb"@generated|DO NOT EDIT|Code generated .* DO NOT EDIT", re.IGNORECASE)
_IDENTIFIER = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*(?:(?:\.|::)[A-Za-z_$][A-Za-z0-9_$]*)*")
_SIMPLE = re.compile(r"[A-Za-z_$][A-Za-z0-9_$]*")
_CAMEL = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|\b)|[A-Z]?[a-z]+\d*|[A-Z]+\d*|\d+")
_DEFINITIONS: dict[str, tuple[tuple[re.Pattern[str], str], ...]] = {
    "python": (
        (re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)\s*\("), "function"),
        (re.compile(r"^\s*class\s+([A-Za-z_]\w*)\s*[:(]"), "class"),
    ),
    "javascript": (
        (
            re.compile(
                r"^\s*(?:export\s+)?(?:default\s+)?(?:async\s+)?function\*?\s+([A-Za-z_$][\w$]*)\s*\("
            ),
            "function",
        ),
        (re.compile(r"^\s*(?:export\s+)?(?:default\s+)?class\s+([A-Za-z_$][\w$]*)"), "class"),
        (re.compile(r"^\s*(?:export\s+)?(?:const|let|var)\s+([A-Za-z_$][\w$]*)\s*="), "variable"),
    ),
    "go": (
        (re.compile(r"^\s*func\s+(?:\([^)]*\)\s*)?([A-Za-z_]\w*)\s*\("), "function"),
        (re.compile(r"^\s*type\s+([A-Za-z_]\w*)\s+(?:struct|interface)\b"), "type"),
    ),
    "rust": (
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+([A-Za-z_]\w*)"), "function"),
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?struct\s+([A-Za-z_]\w*)"), "struct"),
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?enum\s+([A-Za-z_]\w*)"), "enum"),
        (re.compile(r"^\s*(?:pub(?:\([^)]*\))?\s+)?trait\s+([A-Za-z_]\w*)"), "trait"),
    ),
}
_DEFINITIONS["typescript"] = _DEFINITIONS["javascript"] + (
    (re.compile(r"^\s*(?:export\s+)?interface\s+([A-Za-z_$][\w$]*)"), "interface"),
    (re.compile(r"^\s*(?:export\s+)?type\s+([A-Za-z_$][\w$]*)\s*="), "type"),
    (re.compile(r"^\s*(?:export\s+)?enum\s+([A-Za-z_$][\w$]*)"), "enum"),
)


class IntakeRejection(ValueError):
    """Report a bounded, path-free reason a source is excluded from intake."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


class QueryKind(StrEnum):
    """Classify a query to select fixed field weights and ranking classes."""

    IDENTIFIER = "identifier"
    PATH = "path"
    NATURAL_LANGUAGE = "natural_language"


@dataclass(frozen=True, slots=True)
class ScreenedSource:
    """A source accepted by the closed S1 intake policy."""

    relative_path: str
    language: str
    text: str


@dataclass(frozen=True, slots=True)
class CodeChunk:
    """One exact half-open character range with one-based inclusive lines."""

    start_char: int
    end_char: int
    start_line: int
    end_line: int


@dataclass(frozen=True, slots=True)
class Identifier:
    """One raw identifier occurrence; qualified names keep their separators."""

    raw: str
    normalized: str
    line: int


@dataclass(frozen=True, slots=True)
class Definition:
    """One heuristic definition site; never an AST-derived fact."""

    name: str
    kind: str
    line: int
    method: str = DEFINITION_METHOD
    qualified_name: str = ""

    def __post_init__(self) -> None:
        if not self.qualified_name:
            object.__setattr__(self, "qualified_name", self.name)


@dataclass(frozen=True, slots=True)
class Symbols:
    """Identifiers and definitions of one text span."""

    identifiers: tuple[Identifier, ...]
    definitions: tuple[Definition, ...]


def detect_language(relative_path: str) -> str | None:
    """Return the closed language label for an allowed suffix."""
    return _LANGUAGES.get(PurePosixPath(relative_path).suffix.lower())


def screen_source(relative_path: str, payload: bytes) -> ScreenedSource:
    """Apply the closed intake policy shared by repository discovery and single imports."""
    path = PurePosixPath(relative_path)
    parts = path.parts
    if not parts or path.is_absolute() or any(part in {"", ".", ".."} for part in parts):
        raise IntakeRejection("unsafe_path")
    if any(part.startswith(".") for part in parts):
        raise IntakeRejection("hidden")
    if any(part.lower() in _VENDORED for part in parts[:-1]):
        raise IntakeRejection("vendored")
    name = parts[-1].lower()
    if name in _LOCKFILES:
        raise IntakeRejection("lockfile")
    if name.endswith(".ipynb"):
        raise IntakeRejection("notebook_unsupported")
    if ".min." in name:
        raise IntakeRejection("generated_or_minified")
    language = detect_language(relative_path)
    if language is None:
        raise IntakeRejection("unsupported_type")
    if len(payload) > MAX_SOURCE_BYTES:
        raise IntakeRejection("too_large")
    if b"\x00" in payload:
        raise IntakeRejection("binary")
    if any(pattern.search(payload) for pattern in _SECRETS):
        raise IntakeRejection("secret_detected")
    if _GENERATED.search(b"\n".join(payload.split(b"\n", 5)[:5])):
        raise IntakeRejection("generated_or_minified")
    try:
        text = payload.decode("utf-8")
    except UnicodeDecodeError:
        raise IntakeRejection("invalid_encoding") from None
    if text.startswith("﻿"):
        raise IntakeRejection("invalid_encoding")
    if language not in {"markdown", "rst"} and any(
        len(line) > MAX_LINE_LENGTH_FOR_SOURCE for line in text.splitlines()
    ):
        raise IntakeRejection("generated_or_minified")
    if not text.strip():
        raise IntakeRejection("empty")
    return ScreenedSource(relative_path, language, text)


def _json_size(text: str) -> int:
    return len(json.dumps(text, ensure_ascii=False).encode())


def _fits(text: str) -> bool:
    return len(text) <= CHUNK_TARGET_CHARACTERS and _json_size(text) <= MAX_CHUNK_JSON_BYTES


def chunk_code(text: str) -> tuple[CodeChunk, ...]:
    """Pack complete lines at blank-line boundaries; split oversized blocks exactly."""
    line_starts = [0]
    for index, character in enumerate(text):
        if character == "\n":
            line_starts.append(index + 1)
    if line_starts[-1] == len(text):
        line_starts.pop()
    lines = [
        (start, line_starts[i + 1] if i + 1 < len(line_starts) else len(text))
        for i, start in enumerate(line_starts)
    ]
    blocks: list[list[int]] = [[]]
    for index, (start, end) in enumerate(lines):
        blocks[-1].append(index)
        if not text[start:end].strip():
            blocks.append([])
    pieces: list[tuple[int, int]] = []
    for block in (b for b in blocks if b):
        pieces.extend(_split_block(text, lines, block))
    chunks: list[tuple[int, int]] = []
    for start, end in pieces:
        if chunks and chunks[-1][1] == start and _fits(text[chunks[-1][0] : end]):
            chunks[-1] = (chunks[-1][0], end)
        else:
            chunks.append((start, end))
    return tuple(
        CodeChunk(start, end, text.count("\n", 0, start) + 1, text.count("\n", 0, end - 1) + 1)
        for start, end in chunks
    )


def _split_block(
    text: str, lines: list[tuple[int, int]], block: list[int]
) -> list[tuple[int, int]]:
    start, end = lines[block[0]][0], lines[block[-1]][1]
    if _fits(text[start:end]):
        return [(start, end)]
    pieces: list[tuple[int, int]] = []
    first = 0
    while first < len(block):
        last = first
        while last + 1 < len(block) and _fits(
            text[lines[block[first]][0] : lines[block[last + 1]][1]]
        ):
            last += 1
        span = (lines[block[first]][0], lines[block[last]][1])
        if not _fits(text[span[0] : span[1]]):
            pieces.extend(_split_line(text, *span))
        else:
            pieces.append(span)
        if last + 1 >= len(block):
            break
        first = max(first + 1, last + 1 - OVERLAP_LINES) if last > first else last + 1
    return pieces


def _split_line(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Break one overlong line into exact consecutive subranges."""
    pieces = []
    position = start
    while position < end:
        stop = min(end, position + CHUNK_TARGET_CHARACTERS)
        while stop > position + 1 and not _fits(text[position:stop]):
            stop -= max(1, (stop - position) // 8)
        pieces.append((position, stop))
        position = stop
    return pieces


def normalize_identifier(raw: str) -> str:
    """Return the versioned exact-match key: NFKC and casefold, separators kept."""
    return unicodedata.normalize("NFKC", raw).casefold()


def identifier_components(raw: str) -> tuple[str, ...]:
    """Split snake_case, camelCase and qualified names into lowercase components."""
    components: list[str] = []
    for part in _SIMPLE.findall(raw):
        for piece in part.replace("$", "_").split("_"):
            components.extend(match.casefold() for match in _CAMEL.findall(piece))
    return tuple(component for component in components if component)


def extract_symbols(language: str, text: str, *, first_line: int = 1) -> Symbols:
    """Return raw identifiers (with qualified names) and heuristic definitions."""
    identifiers: list[Identifier] = []
    definitions: list[Definition] = []
    patterns = _DEFINITIONS.get(language, ())
    # Python scope comes from indentation; other languages keep unqualified names.
    scopes: list[tuple[int, str]] = []
    for offset, line in enumerate(text.splitlines()):
        number = first_line + offset
        indent = len(line) - len(line.lstrip(" \t"))
        if language == "python" and line.strip() and not line.lstrip().startswith(("#", "@")):
            while scopes and scopes[-1][0] >= indent:
                scopes.pop()
        seen: set[str] = set()
        for match in _IDENTIFIER.finditer(line):
            qualified = match.group(0)
            names = [qualified] if ("." in qualified or "::" in qualified) else []
            names += _SIMPLE.findall(qualified)
            for raw in names:
                if raw not in seen:
                    seen.add(raw)
                    identifiers.append(Identifier(raw, normalize_identifier(raw), number))
        for pattern, kind in patterns:
            found = pattern.match(line)
            if found is not None:
                name = found.group(1)
                qualified = ".".join([scope for _, scope in scopes] + [name])
                definitions.append(Definition(name, kind, number, qualified_name=qualified))
                if language == "python":
                    scopes.append((indent, name))
                break
    return Symbols(tuple(identifiers), tuple(definitions))


def path_terms(relative_path: str) -> tuple[str, ...]:
    """Return normalized relative-path segments and their components; never absolute."""
    terms: list[str] = []
    for segment in PurePosixPath(unicodedata.normalize("NFKC", relative_path)).parts:
        folded = segment.casefold()
        terms.append(folded)
        terms.extend(identifier_components(segment))
        terms.extend(re.findall(r"[a-z0-9]+", folded))
    return tuple(dict.fromkeys(term for term in terms if term and term != "/"))


def classify_query(query: str) -> QueryKind:
    """Classify with fixed versioned rules; request text cannot choose another policy."""
    stripped = query.strip()
    if " " not in stripped and (
        "/" in stripped or re.fullmatch(r"[\w.-]+\.[A-Za-z0-9]{1,8}", stripped)
    ):
        if "/" in stripped or detect_language(stripped) is not None:
            return QueryKind.PATH
    if _IDENTIFIER.fullmatch(stripped) and (
        "_" in stripped
        or "." in stripped
        or "::" in stripped
        or re.search(r"[a-z][A-Z]|[A-Z]{2}[a-z]", stripped)
        or stripped[:1].isupper()
    ):
        return QueryKind.IDENTIFIER
    return QueryKind.NATURAL_LANGUAGE


__all__ = [
    "CHUNKER_VERSION",
    "CHUNK_TARGET_CHARACTERS",
    "CLASSIFIER_VERSION",
    "DEFINITION_METHOD",
    "MAX_CHUNK_JSON_BYTES",
    "PATH_NORMALIZATION",
    "POLICY_VERSION",
    "CodeChunk",
    "Definition",
    "Identifier",
    "IntakeRejection",
    "QueryKind",
    "ScreenedSource",
    "Symbols",
    "chunk_code",
    "classify_query",
    "detect_language",
    "extract_symbols",
    "identifier_components",
    "normalize_identifier",
    "path_terms",
    "screen_source",
]
