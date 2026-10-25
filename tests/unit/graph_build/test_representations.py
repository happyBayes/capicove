"""Exact text representations preserve source identity and complete coordinates."""

import hashlib
from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.representations import BlockSpan, join_exact_blocks, text_representation


SCOPE = GraphScope("workspace", "private")


def test_block_join_preserves_source_codepoints() -> None:
    text, spans = join_exact_blocks(("A", "学习"))
    assert text == "A\n\n学习"
    assert spans == ((0, 1), (3, 5))


def test_empty_blocks_retain_rows_without_losing_separator_offsets() -> None:
    text, spans = join_exact_blocks(("", "中🙂", ""))
    assert text == "\n\n中🙂\n\n"
    assert spans == ((0, 0), (2, 4), (6, 6))
    assert join_exact_blocks(()) == ("", ())


@pytest.mark.parametrize("blocks", ["abc", b"abc", ("a", None), (1,)])
def test_join_rejects_non_text_blocks(blocks: Any) -> None:
    with pytest.raises(TypeError):
        join_exact_blocks(blocks)


def test_text_representation_preserves_unicode_crlf_and_paragraph_coordinates() -> None:
    source = "A中🙂\r\n\r\nB e\u0301\n"
    rep = text_representation("doc", source, SCOPE)
    assert rep.text == source
    assert rep.source_sha256 == hashlib.sha256(source.encode("utf-8")).hexdigest()
    assert [(b.start_char, b.end_char, b.block_type) for b in rep.blocks] == [
        (0, 5, "paragraph"),
        (5, 7, "separator"),
        (7, 12, "paragraph"),
    ]
    assert "".join(source[b.start_char : b.end_char] for b in rep.blocks) == source
    assert all(b.page_index is None for b in rep.blocks)
    assert rep.declared_codepoints == 12
    assert rep.excluded_codepoints == 0


def test_markdown_uses_explicit_plain_structure_fallback_without_normalizing() -> None:
    source = "# Title\n\n| a | b |\n|---|---|\n| 1 | 2 |\n"
    rep = text_representation("doc", source, SCOPE)
    assert rep.text == source
    assert rep.issue_codes == ("plain_text_structure",)
    assert all(b.section_path == () for b in rep.blocks)
    assert rep == text_representation("doc", source, SCOPE)


def test_representation_identity_binds_scope_document_structure_and_source_hash() -> None:
    original = text_representation("doc", "Exact text", SCOPE)
    variants = (
        text_representation("other", "Exact text", SCOPE),
        text_representation("doc", "Exact text", GraphScope("workspace", "other")),
        text_representation("doc", "changed", SCOPE),
        replace(original, source_sha256="a" * 64),
        replace(original, blocks=(replace(original.blocks[0], section_path=("Heading",)),)),
    )
    assert len({original.representation_ref, *(rep.representation_ref for rep in variants)}) == 6
    assert original.representation_ref == f"representation:{original.representation_sha256}"


def test_empty_text_is_valid_without_an_empty_source_span() -> None:
    rep = text_representation("doc", "", SCOPE)
    assert rep.blocks == ()
    assert rep.declared_codepoints == rep.excluded_codepoints == 0


def test_whitespace_only_text_is_retained_as_a_separator() -> None:
    rep = text_representation("doc", " \t\n\n", SCOPE)
    assert [(b.start_char, b.end_char, b.block_type) for b in rep.blocks] == [(0, 4, "separator")]


@pytest.mark.parametrize(
    "changes",
    [
        {"source_sha256": ""},
        {"text": 1},
        {"scope": "scope"},
        {"document_ref": ""},
        {"blocks": ()},
        {"issue_codes": ("x", "x")},
    ],
)
def test_representation_rejects_missing_identity_or_coverage(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(text_representation("doc", "abc", SCOPE), **changes)


@pytest.mark.parametrize(
    "blocks",
    [
        (BlockSpan("a", 1, 3, None, "paragraph", ()),),
        (BlockSpan("a", 0, 4, None, "paragraph", ()),),
        (BlockSpan("a", 0, 2, None, "paragraph", ()),),
        (BlockSpan("a", 0, 2, None, "paragraph", ()), BlockSpan("b", 1, 3, None, "paragraph", ())),
        (BlockSpan("a", 0, 1, None, "paragraph", ()), BlockSpan("a", 1, 3, None, "paragraph", ())),
    ],
)
def test_representation_rejects_gaps_overlap_bounds_and_duplicate_blocks(
    blocks: tuple[BlockSpan, ...],
) -> None:
    with pytest.raises(ValueError):
        replace(text_representation("doc", "abc", SCOPE), blocks=blocks)


def test_explicit_exclusion_retains_text_and_counts_source_coverage() -> None:
    rep = text_representation("doc", "abc", SCOPE)
    block = replace(rep.blocks[0], exclusion_reason="layout_header")
    excluded = replace(rep, blocks=(block,))
    assert excluded.text == "abc"
    assert excluded.declared_codepoints == excluded.excluded_codepoints == 3
    assert excluded.representation_ref != rep.representation_ref


def test_empty_block_coverage_rows_and_defensive_collections() -> None:
    rep = text_representation("doc", "abc", SCOPE)
    blocks = [BlockSpan("empty", 0, 0, None, "paragraph", ()), *rep.blocks]
    frozen = replace(rep, blocks=blocks)  # type: ignore[arg-type]
    blocks.clear()
    assert len(frozen.blocks) == 2
    path = ["Repeated", "Repeated"]
    block = BlockSpan("a", 0, 3, None, "paragraph", path)  # type: ignore[arg-type]
    path.clear()
    assert block.section_path == ("Repeated", "Repeated")


@pytest.mark.parametrize(
    "changes",
    [
        {"start_char": True},
        {"start_char": -1},
        {"end_char": -1},
        {"page_index": True},
        {"page_index": -1},
        {"block_id": ""},
        {"section_path": "heading"},
        {"block_type": ""},
        {"exclusion_reason": ""},
    ],
)
def test_block_rejects_invalid_coordinates_and_metadata(changes: dict[str, Any]) -> None:
    with pytest.raises((TypeError, ValueError)):
        replace(BlockSpan("a", 0, 3, None, "paragraph", ()), **changes)
