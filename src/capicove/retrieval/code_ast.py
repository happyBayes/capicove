"""Optional build-time Tree-sitter structure for code search (RFC-0033 S2, ``code-ast-v1``).

Tree-sitter is imported only when a parse is requested. Missing grammars, syntax
errors, budget exhaustion and an absent dependency all yield a recorded
``FALLBACK`` outcome whose chunks come from the exact S1 line chunker; fallback
facts are never labelled AST-derived. No project code is executed.
"""

from __future__ import annotations

import bisect
import importlib
import importlib.metadata
import warnings
from dataclasses import dataclass
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from capicove.retrieval.code_symbols import (
    CHUNK_TARGET_CHARACTERS,
    MAX_CHUNK_JSON_BYTES,
    chunk_code,
)


AST_POLICY_VERSION = "code-ast-v1"
AST_METHOD = "tree-sitter-v1"
MAX_PARSE_BYTES = 1024 * 1024
PARSE_TIMEOUT_MICROS = 2_000_000
MAX_SYMBOLS = 5_000
_GRAMMARS = {
    "python": ("tree_sitter_python", "language", "tree-sitter-python"),
    "javascript": ("tree_sitter_javascript", "language", "tree-sitter-javascript"),
    "jsx": ("tree_sitter_javascript", "language", "tree-sitter-javascript"),
    "typescript": ("tree_sitter_typescript", "language_typescript", "tree-sitter-typescript"),
    "tsx": ("tree_sitter_typescript", "language_tsx", "tree-sitter-typescript"),
}
# Compound statements whose blocks may still contain module- or class-level definitions.
_PYTHON_CONTAINERS = frozenset(
    {
        "if_statement",
        "try_statement",
        "with_statement",
        "block",
        "else_clause",
        "except_clause",
        "finally_clause",
    }
)
_JS_KINDS = {
    "function_declaration": "function",
    "generator_function_declaration": "function",
    "class_declaration": "class",
    "abstract_class_declaration": "class",
    "method_definition": "method",
    "interface_declaration": "interface",
    "type_alias_declaration": "type",
    "enum_declaration": "enum",
}


class ParseStatus(StrEnum):
    """Distinguish AST-derived structure from recorded line fallback."""

    PARSED = "parsed"
    FALLBACK = "fallback"


@dataclass(frozen=True, slots=True)
class AstSymbol:
    """One AST-derived definition with exact character, line and byte spans."""

    name: str
    qualified_name: str
    kind: str
    parent: str | None
    signature: str
    decorators: tuple[str, ...]
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    method: str = AST_METHOD


@dataclass(frozen=True, slots=True)
class ParseOutcome:
    """The result of one bounded parse, including its fallback reason."""

    language: str
    status: ParseStatus
    reason: str | None
    symbols: tuple[AstSymbol, ...]


@dataclass(frozen=True, slots=True)
class AstChunk:
    """One exact chunk; ``structure`` says whether AST boundaries chose it."""

    start_char: int
    end_char: int
    start_line: int
    end_line: int
    symbol: str | None
    structure: str


def ast_language(relative_path: str) -> str | None:
    """Return the grammar key for a path, or None when no grammar is selected."""
    return {
        ".py": "python",
        ".pyi": "python",
        ".js": "javascript",
        ".mjs": "javascript",
        ".cjs": "javascript",
        ".jsx": "jsx",
        ".ts": "typescript",
        ".tsx": "tsx",
    }.get(PurePosixPath(relative_path).suffix.lower())


def parser_identity() -> dict[str, Any]:
    """Return the installed parser and grammar identity sealed into generations."""

    def version(distribution: str) -> str:
        try:
            return importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            return "unavailable"

    return {
        "policy": AST_POLICY_VERSION,
        "method": AST_METHOD,
        "tree_sitter": version("tree-sitter"),
        "grammars": {
            key: f"{distribution} {version(distribution)}"
            for key, distribution in (
                ("python", "tree-sitter-python"),
                ("javascript", "tree-sitter-javascript"),
                ("typescript", "tree-sitter-typescript"),
            )
        },
        "limits": {"max_bytes": MAX_PARSE_BYTES, "timeout_micros": PARSE_TIMEOUT_MICROS},
    }


def _fallback(language: str, reason: str) -> ParseOutcome:
    return ParseOutcome(language, ParseStatus.FALLBACK, reason, ())


def parse_tree(language: str | None, payload: bytes, *, max_bytes: int = MAX_PARSE_BYTES) -> Any:
    """Return a syntax tree for relation extraction, or None for any fallback reason."""
    if language not in _GRAMMARS or len(payload) > max_bytes:
        return None
    try:
        tree_sitter: Any = importlib.import_module("tree_sitter")
        module_name, factory, _ = _GRAMMARS[language]
        grammar = getattr(importlib.import_module(module_name), factory)()
    except (ImportError, AttributeError):
        return None
    parser = tree_sitter.Parser(tree_sitter.Language(grammar))
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Use the progress_callback", DeprecationWarning)
        parser.timeout_micros = PARSE_TIMEOUT_MICROS
    tree = parser.parse(payload)
    if tree is None or tree.root_node.has_error:
        return None
    return tree


def parse_source(
    language: str | None, source: str, *, max_bytes: int = MAX_PARSE_BYTES
) -> ParseOutcome:
    """Parse one source within byte and time budgets; never raise for content."""
    key = language or "unknown"
    if language not in _GRAMMARS:
        return _fallback(key, "missing_grammar")
    payload = source.encode("utf-8")
    if len(payload) > max_bytes:
        return _fallback(key, "budget_exceeded")
    try:
        tree_sitter: Any = importlib.import_module("tree_sitter")
        module_name, factory, _ = _GRAMMARS[language]
        grammar = getattr(importlib.import_module(module_name), factory)()
    except (ImportError, AttributeError):
        return _fallback(key, "dependency_unavailable")
    parser = tree_sitter.Parser(tree_sitter.Language(grammar))
    # The replacement progress_callback requires a read callback, and that path
    # crashed (SIGSEGV) under tree-sitter 0.25.2; keep the C-level timeout instead.
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Use the progress_callback", DeprecationWarning)
        parser.timeout_micros = PARSE_TIMEOUT_MICROS
    tree = parser.parse(payload)
    if tree is None:
        return _fallback(key, "budget_exceeded")
    if tree.root_node.has_error:
        return _fallback(key, "syntax_error")
    mapper = _OffsetMapper(source)
    symbols: list[AstSymbol] = []
    walker = _python_symbols if language == "python" else _script_symbols
    walker(tree.root_node, payload, mapper, [], symbols)
    if len(symbols) > MAX_SYMBOLS:
        return _fallback(key, "budget_exceeded")
    return ParseOutcome(key, ParseStatus.PARSED, None, tuple(symbols))


class _OffsetMapper:
    """Translate UTF-8 byte offsets to code-point offsets and one-based lines."""

    def __init__(self, source: str) -> None:
        self._source = source
        self._byte_starts: list[int] = []
        position = 0
        for character in source:
            self._byte_starts.append(position)
            position += len(character.encode("utf-8"))
        self._byte_starts.append(position)
        self._newlines = [index for index, character in enumerate(source) if character == "\n"]

    def char(self, byte_offset: int) -> int:
        index = bisect.bisect_left(self._byte_starts, byte_offset)
        if self._byte_starts[index] != byte_offset:
            raise ValueError("byte offset splits a UTF-8 character")
        return index

    def line(self, char_offset: int) -> int:
        return bisect.bisect_left(self._newlines, char_offset) + 1


def _text(node: Any, payload: bytes) -> str:
    return payload[node.start_byte : node.end_byte].decode("utf-8")


def _append(
    symbols: list[AstSymbol],
    mapper: _OffsetMapper,
    span: Any,
    name: str,
    kind: str,
    scopes: list[str],
    signature: str,
    decorators: tuple[str, ...] = (),
) -> str:
    qualified = ".".join([*scopes, name])
    start, end = mapper.char(span.start_byte), mapper.char(span.end_byte)
    symbols.append(
        AstSymbol(
            name=name,
            qualified_name=qualified,
            kind=kind,
            parent=".".join(scopes) or None,
            signature=" ".join(signature.split())[:512],
            decorators=decorators,
            start_char=start,
            end_char=end,
            start_line=mapper.line(start),
            end_line=mapper.line(max(start, end - 1)),
        )
    )
    return qualified


def _python_symbols(
    node: Any, payload: bytes, mapper: _OffsetMapper, scopes: list[str], symbols: list[AstSymbol]
) -> None:
    for child in node.named_children:
        span, definition = child, child
        decorators: tuple[str, ...] = ()
        if child.type == "decorated_definition":
            definition = child.child_by_field_name("definition")
            decorators = tuple(
                _text(item, payload) for item in child.named_children if item.type == "decorator"
            )
        if definition is not None and definition.type in {
            "function_definition",
            "class_definition",
        }:
            name_node = definition.child_by_field_name("name")
            body = definition.child_by_field_name("body")
            if name_node is None or body is None:
                continue
            name = _text(name_node, payload)
            signature = (
                payload[definition.start_byte : body.start_byte]
                .decode("utf-8")
                .rstrip()
                .rstrip(":")
            )
            kind = "class" if definition.type == "class_definition" else "function"
            _append(symbols, mapper, span, name, kind, scopes, signature, decorators)
            _python_symbols(body, payload, mapper, [*scopes, name], symbols)
        elif child.type in _PYTHON_CONTAINERS:
            _python_symbols(child, payload, mapper, scopes, symbols)


def _script_symbols(
    node: Any, payload: bytes, mapper: _OffsetMapper, scopes: list[str], symbols: list[AstSymbol]
) -> None:
    for child in node.named_children:
        span = child
        target = child
        if child.type == "export_statement":
            declaration = child.child_by_field_name("declaration")
            if declaration is None:
                continue
            target = declaration
        kind = _JS_KINDS.get(target.type)
        if kind is not None:
            name_node = target.child_by_field_name("name")
            if name_node is None:
                continue
            name = _text(name_node, payload)
            body = target.child_by_field_name("body")
            end = body.start_byte if body is not None else target.end_byte
            signature = payload[target.start_byte : end].decode("utf-8").splitlines()[0]
            qualified = _append(symbols, mapper, span, name, kind, scopes, signature)
            if body is not None and kind in {"class", "function", "method"}:
                _script_symbols(body, payload, mapper, qualified.split("."), symbols)
            continue
        if target.type in {"lexical_declaration", "variable_declaration"}:
            for declarator in target.named_children:
                if declarator.type != "variable_declarator":
                    continue
                value = declarator.child_by_field_name("value")
                name_node = declarator.child_by_field_name("name")
                if (
                    value is not None
                    and name_node is not None
                    and value.type in {"arrow_function", "function_expression", "function"}
                ):
                    signature = _text(target, payload).splitlines()[0]
                    _append(
                        symbols,
                        mapper,
                        span,
                        _text(name_node, payload),
                        "function",
                        scopes,
                        signature,
                    )


def _line_bounds(source: str, start: int, end: int) -> tuple[int, int]:
    begin = source.rfind("\n", 0, start) + 1
    stop = source.find("\n", max(start, end - 1))
    return begin, len(source) if stop == -1 else stop + 1


def _line_number(source: str, char: int) -> int:
    return source.count("\n", 0, char) + 1


def _fits(text: str) -> bool:
    import json

    return (
        len(text) <= CHUNK_TARGET_CHARACTERS
        and len(json.dumps(text, ensure_ascii=False).encode()) <= MAX_CHUNK_JSON_BYTES
    )


def _lines(source: str, start: int, end: int, symbol: str | None, structure: str) -> list[AstChunk]:
    """Split an exact region with the S1 chunker, re-based to absolute offsets."""
    if start >= end:
        return []
    pieces = []
    for chunk in chunk_code(source[start:end]):
        absolute_start, absolute_end = start + chunk.start_char, start + chunk.end_char
        pieces.append(
            AstChunk(
                absolute_start,
                absolute_end,
                _line_number(source, absolute_start),
                _line_number(source, absolute_end - 1),
                symbol,
                structure,
            )
        )
    return pieces


def ast_chunks(source: str, outcome: ParseOutcome) -> tuple[AstChunk, ...]:
    """Prefer symbol boundaries; cover every non-symbol region with exact line chunks.

    Adjacent whole symbols and gaps of the same enclosing scope are packed up to the
    chunk budget; a symbol is split only when it alone exceeds the budget, and its
    pieces keep that symbol's identity.
    """
    if outcome.status is ParseStatus.FALLBACK:
        return tuple(_lines(source, 0, len(source), None, "line-fallback"))
    children: dict[str | None, list[AstSymbol]] = {}
    for symbol in outcome.symbols:
        children.setdefault(symbol.parent, []).append(symbol)

    def region(
        start: int, end: int, owner: str | None, members: list[AstSymbol]
    ) -> list[tuple[AstChunk, str | None, bool]]:
        output: list[tuple[AstChunk, str | None, bool]] = []
        structure = "ast" if owner else "ast-gap"

        def gap(begin: int, stop: int) -> None:
            output.extend(
                (chunk, owner, True) for chunk in _lines(source, begin, stop, owner, structure)
            )

        cursor = start
        for member in sorted(members, key=lambda item: item.start_char):
            begin, stop = _line_bounds(source, member.start_char, member.end_char)
            begin, stop = max(begin, cursor), min(stop, end)
            if begin < cursor or begin >= stop:
                continue
            gap(cursor, begin)
            if _fits(source[begin:stop]):
                chunk = AstChunk(
                    begin,
                    stop,
                    _line_number(source, begin),
                    _line_number(source, stop - 1),
                    member.qualified_name,
                    "ast",
                )
                output.append((chunk, owner, True))
            else:
                inner = region(
                    begin, stop, member.qualified_name, children.get(member.qualified_name, [])
                )
                # Pieces of one oversized symbol never merge with neighbours.
                output.extend((chunk, member.qualified_name, False) for chunk, _, _ in inner)
            cursor = stop
        gap(cursor, end)
        return output

    return tuple(_pack(source, region(0, len(source), None, children.get(None, []))))


def _pack(source: str, chunks: list[tuple[AstChunk, str | None, bool]]) -> list[AstChunk]:
    """Merge adjacent mergeable chunks of one scope while they fit the budget."""
    packed: list[tuple[AstChunk, str | None, bool]] = []
    for chunk, owner, mergeable in chunks:
        if packed:
            previous, previous_owner, previous_mergeable = packed[-1]
            combined = source[previous.start_char : chunk.end_char]
            if (
                mergeable
                and previous_mergeable
                and previous_owner == owner
                and previous.end_char == chunk.start_char
                and _fits(combined)
            ):
                symbol = previous.symbol if previous.symbol == chunk.symbol else owner
                structure = (
                    "ast" if "ast" in {previous.structure, chunk.structure} else chunk.structure
                )
                packed[-1] = (
                    AstChunk(
                        previous.start_char,
                        chunk.end_char,
                        previous.start_line,
                        chunk.end_line,
                        symbol,
                        structure,
                    ),
                    owner,
                    True,
                )
                continue
        packed.append((chunk, owner, mergeable))
    return [chunk for chunk, _, _ in packed]


__all__ = [
    "AST_METHOD",
    "AST_POLICY_VERSION",
    "AstChunk",
    "AstSymbol",
    "ParseOutcome",
    "ParseStatus",
    "ast_chunks",
    "ast_language",
    "parse_source",
    "parse_tree",
    "parser_identity",
]
