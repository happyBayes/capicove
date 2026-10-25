"""Pure exact-text adapters; source acquisition and authorization live in readers."""

import hashlib
import re
from dataclasses import asdict, dataclass

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.codec import fingerprint


def _nonblank(value: object) -> None:
    if not isinstance(value, str):
        raise TypeError("Expected text")
    if not value.strip():
        raise ValueError("Expected nonblank text")


def _nonnegative(value: object) -> None:
    if type(value) is not int:
        raise TypeError("Expected integer coordinates")
    if value < 0:
        raise ValueError("Coordinates cannot be negative")


@dataclass(frozen=True, slots=True)
class BlockSpan:
    """Index exact representation text, including empty rows and excluded layout.

    Empty rows retain provenance but must never become empty SourceSpan values.
    Page indices are optional zero-based provenance, not retrieval page numbers.
    """

    block_id: str
    start_char: int
    end_char: int
    page_index: int | None
    block_type: str
    section_path: tuple[str, ...]
    exclusion_reason: str | None = None

    def __post_init__(self) -> None:
        _nonblank(self.block_id)
        _nonblank(self.block_type)
        _nonnegative(self.start_char)
        _nonnegative(self.end_char)
        if self.end_char < self.start_char:
            raise ValueError("Block end precedes its start")
        if self.page_index is not None:
            _nonnegative(self.page_index)
        if not isinstance(self.section_path, (tuple, list)):
            raise TypeError("section_path must be a sequence")
        for heading in self.section_path:
            _nonblank(heading)
        object.__setattr__(self, "section_path", tuple(self.section_path))
        if self.exclusion_reason is not None:
            _nonblank(self.exclusion_reason)


@dataclass(frozen=True, slots=True)
class TextRepresentation:
    """Capability-local exact text with source lineage and total block coverage.

    These values grant no authorization. The source digest identifies supplied
    source bytes; representation identity additionally binds scope, document,
    structure, exclusions, and the adapter version. Normalized text is separate.
    """

    document_ref: str
    scope: GraphScope
    source_sha256: str
    text: str
    blocks: tuple[BlockSpan, ...]
    issue_codes: tuple[str, ...]

    def __post_init__(self) -> None:
        _nonblank(self.document_ref)
        if not isinstance(self.scope, GraphScope):
            raise TypeError("scope must be GraphScope")
        _nonblank(self.source_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", self.source_sha256) is None:
            raise ValueError("source_sha256 must be lowercase SHA-256")
        if not isinstance(self.text, str):
            raise TypeError("Representation text must be a string")
        if not isinstance(self.blocks, (tuple, list)):
            raise TypeError("blocks must be a sequence")
        blocks = tuple(self.blocks)
        cursor = 0
        seen: set[str] = set()
        for block in blocks:
            if not isinstance(block, BlockSpan):
                raise TypeError("blocks must contain BlockSpan")
            if block.block_id in seen:
                raise ValueError("Duplicate block ID")
            seen.add(block.block_id)
            if block.start_char != cursor or block.end_char > len(self.text):
                raise ValueError("Blocks must be ordered and cover text without gaps or overlap")
            cursor = block.end_char
        if cursor != len(self.text):
            raise ValueError("Blocks must cover the entire representation")
        object.__setattr__(self, "blocks", blocks)
        if not isinstance(self.issue_codes, (tuple, list)):
            raise TypeError("issue_codes must be a sequence")
        for code in self.issue_codes:
            _nonblank(code)
        if len(set(self.issue_codes)) != len(self.issue_codes):
            raise ValueError("Duplicate issue codes")
        object.__setattr__(self, "issue_codes", tuple(self.issue_codes))

    @property
    def declared_codepoints(self) -> int:
        """Count source-mapped representation code points, including exclusions."""
        return len(self.text)

    @property
    def excluded_codepoints(self) -> int:
        """Count explicitly excluded layout without deleting its text."""
        return sum(
            block.end_char - block.start_char
            for block in self.blocks
            if block.exclusion_reason is not None
        )

    @property
    def representation_sha256(self) -> str:
        """Hash exact content and interpretation with explicit adapter version."""
        return fingerprint({"schema_version": 1, "representation": asdict(self)})

    @property
    def representation_ref(self) -> str:
        """Identify this scoped representation, without granting access to it."""
        return f"representation:{self.representation_sha256}"


def join_exact_blocks(blocks: tuple[str, ...]) -> tuple[str, tuple[tuple[int, int], ...]]:
    """Join ordered exact blocks with two newlines and return code-point ranges.

    Returned ranges describe input blocks only. An adapter constructing a full
    TextRepresentation must also index inserted separators as separate blocks.
    """
    if not isinstance(blocks, (tuple, list)) or any(not isinstance(item, str) for item in blocks):
        raise TypeError("blocks must be a sequence of text values")
    spans = []
    cursor = 0
    for index, item in enumerate(blocks):
        if index:
            cursor += 2
        spans.append((cursor, cursor + len(item)))
        cursor += len(item)
    return "\n\n".join(blocks), tuple(spans)


def text_representation(document_ref: str, text: str, scope: GraphScope) -> TextRepresentation:
    """Adapt already-authorized plain text using lossless paragraph fallback.

    Markdown is retained verbatim, not rendered or inferred to have reliable
    table, heading, or code semantics. Blank runs get explicit separator ranges.
    The source digest here refers to UTF-8 text, not any upstream PDF/file bytes.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    blocks: list[BlockSpan] = []
    cursor = 0
    start = 0
    current_type: str | None = None
    for line in text.splitlines(keepends=True):
        block_type = "separator" if not line.strip() else "paragraph"
        if current_type is not None and block_type != current_type:
            blocks.append(BlockSpan(f"block:{len(blocks)}", start, cursor, None, current_type, ()))
            start = cursor
        current_type = block_type
        cursor += len(line)
    if current_type is not None:
        blocks.append(BlockSpan(f"block:{len(blocks)}", start, cursor, None, current_type, ()))
    return TextRepresentation(
        document_ref,
        scope,
        hashlib.sha256(text.encode("utf-8")).hexdigest(),
        text,
        tuple(blocks),
        ("plain_text_structure",),
    )
