"""Token-bounded graph chunks retain exact occurrence identity and source maps."""

import json
from dataclasses import dataclass, replace
from itertools import pairwise
from typing import Any

import pytest

from capicove.contracts.graph_build import GraphScope
from capicove.graph_build.profiles import (
    ChunkingProfile,
    chunking_profile_from_json,
    chunking_profile_to_json,
)
from capicove.graph_build.representations import BlockSpan, text_representation
from capicove.graph_build.text_units import compose_text_units, occurrence_id


@dataclass(frozen=True)
class CharacterTokenizer:
    fingerprint: str = "a" * 64

    def count(self, text: str) -> int:
        return len(text)


def test_same_bytes_in_different_documents_do_not_collapse() -> None:
    left = occurrence_id("s", "doc:a", "rep:x", 0, 10, "f" * 64)
    right = occurrence_id("s", "doc:b", "rep:x", 0, 10, "f" * 64)
    assert left != right
    assert left == occurrence_id("s", "doc:a", "rep:x", 0, 10, "f" * 64)


@pytest.mark.parametrize(
    "changes",
    [
        {"target_tokens": 0},
        {"target_tokens": True},
        {"max_tokens": -1},
        {"context_tokens": -1},
        {"context_tokens": 1.5},
        {"target_tokens": 6, "max_tokens": 5},
        {"version": 2},
    ],
)
def test_profile_rejects_incompatible_or_noninteger_budgets(changes: dict[str, Any]) -> None:
    with pytest.raises((ValueError, TypeError)):
        replace(ChunkingProfile(), **changes)


def test_profile_codec_is_closed_and_versioned() -> None:
    profile = ChunkingProfile(target_tokens=4, max_tokens=6, context_tokens=0)
    assert chunking_profile_from_json(chunking_profile_to_json(profile)) == profile
    for payload in (
        '{"version":1,"version":1,"target_tokens":4,"max_tokens":6,"context_tokens":0}',
        json.dumps(
            {
                "version": 1,
                "target_tokens": 4,
                "max_tokens": 6,
                "context_tokens": 0,
                "model": "other",
            }
        ),
    ):
        with pytest.raises(ValueError):
            chunking_profile_from_json(payload)


def test_unicode_chunks_cover_exact_source_once_with_separate_preceding_context() -> None:
    rep = text_representation("doc", "学习🙂abcdefgh", GraphScope("w", "s"))
    profile = ChunkingProfile(target_tokens=4, max_tokens=4, context_tokens=2)
    units = compose_text_units(rep, CharacterTokenizer(), profile)
    assert [u.segments[-1].text for u in units] == ["学习🙂a", "bcde", "fgh"]
    assert [u.core_locator.character_range.start_char for u in units] == [0, 4, 8]
    assert [u.core_locator.character_range.end_char for u in units] == [4, 8, 11]
    assert units[1].segments[0].role == "context"
    assert units[1].segments[0].text == "🙂a"
    for unit in units:
        assert unit.token_count <= 4
        for segment in unit.segments:
            assert segment.text == "".join(
                rep.text[s.start_char : s.end_char] for s in segment.source_spans
            )
            assert all(s.representation_ref == rep.representation_ref for s in segment.source_spans)
    assert units == compose_text_units(rep, CharacterTokenizer(), profile)


def test_chunk_ids_change_with_tokenizer_profile_and_workspace() -> None:
    rep = text_representation("doc", "abc", GraphScope("w", "s"))
    first = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile())[0]
    variants = (
        compose_text_units(rep, CharacterTokenizer("b" * 64), ChunkingProfile())[0],
        compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(context_tokens=0))[0],
        compose_text_units(
            replace(rep, scope=GraphScope("other", "s")), CharacterTokenizer(), ChunkingProfile()
        )[0],
    )
    assert all(u.text_unit_id != first.text_unit_id for u in variants)


def test_paragraphs_pack_to_target_without_losing_blank_separator() -> None:
    rep = text_representation("doc", "A\n\nB\n\nC", GraphScope("w", "s"))
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(6, 8, 0))
    assert [u.segments[-1].text for u in units] == ["A\n\nB\n\n", "C"]


def test_oversized_block_prefers_sentence_boundary() -> None:
    rep = text_representation("doc", "abc。defghij", GraphScope("w", "s"))
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(5, 6, 0))
    assert units[0].segments[-1].text == "abc。"
    assert "".join(u.segments[-1].text for u in units) == rep.text


def test_nonmonotonic_token_counts_do_not_lose_a_later_fitting_prefix() -> None:
    class NonmonotonicTokenizer(CharacterTokenizer):
        def count(self, text: str) -> int:
            return {"abc": 9, "abcde": 2, "abcdef": 9}.get(text, len(text))

    rep = text_representation("doc", "abcdef", GraphScope("w", "s"))
    units = compose_text_units(rep, NonmonotonicTokenizer(), ChunkingProfile(2, 2, 0))
    assert [u.segments[-1].text for u in units] == ["abcde", "f"]


def test_emission_rechecks_hard_limit_if_tokenizer_changes_its_count() -> None:
    class UnstableTokenizer:
        fingerprint = "a" * 64

        def __init__(self) -> None:
            self.calls = 0

        def count(self, text: str) -> int:
            self.calls += 1
            return 1 if self.calls == 1 else 100

    rep = text_representation("doc", "abc", GraphScope("w", "s"))
    with pytest.raises(ValueError):
        compose_text_units(rep, UnstableTokenizer(), ChunkingProfile(2, 2, 0))


@pytest.mark.parametrize("count", [True, -1, 1.5, 0, 100])
def test_invalid_or_unencodable_tokenizer_fails_without_partial_output(count: Any) -> None:
    class InvalidTokenizer(CharacterTokenizer):
        def count(self, text: str) -> int:
            return count  # type: ignore[no-any-return]

    rep = text_representation("doc", "abc", GraphScope("w", "s"))
    with pytest.raises((TypeError, ValueError)):
        compose_text_units(rep, InvalidTokenizer(), ChunkingProfile(2, 2, 0))


def test_core_and_context_never_cross_excluded_source_or_section_boundaries() -> None:
    rep = text_representation("doc", "abcXdefghi", GraphScope("w", "s"))
    rep = replace(
        rep,
        blocks=(
            BlockSpan("a", 0, 3, None, "paragraph", ("one",)),
            BlockSpan("x", 3, 4, None, "paragraph", ("one",), "excluded"),
            BlockSpan("b", 4, 7, None, "paragraph", ("one",)),
            BlockSpan("c", 7, 10, None, "paragraph", ("two",)),
        ),
    )
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(10, 12, 4))
    assert [u.segments[-1].text for u in units] == ["abc", "def", "ghi"]
    assert all(len(u.segments) == 1 for u in units)


def test_page_provenance_is_retained_without_inventing_ocr_locators() -> None:
    rep = text_representation("doc", "abcdef", GraphScope("w", "s"))
    rep = replace(
        rep,
        blocks=(
            BlockSpan("a", 0, 3, 0, "paragraph", ()),
            BlockSpan("b", 3, 6, 1, "paragraph", ()),
        ),
    )
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(10, 12, 4))
    assert [u.segments[-1].text for u in units] == ["abc", "def"]
    assert [u.segments[-1].source_spans[0].page_index for u in units] == [0, 1]
    assert all(u.core_locator.page_number is None for u in units)


def test_source_mapped_heading_and_table_header_are_noncore_context() -> None:
    rep = text_representation("doc", "TitleH1\nrow1\nrow2", GraphScope("w", "s"))
    rep = replace(
        rep,
        blocks=(
            BlockSpan("t", 0, 5, None, "heading", ("Title",)),
            BlockSpan("h", 5, 8, None, "table_header", ("Title",)),
            BlockSpan("r1", 8, 13, None, "table_row", ("Title",)),
            BlockSpan("r2", 13, 17, None, "table_row", ("Title",)),
        ),
    )
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(5, 6, 8))
    assert [u.segments[-1].text for u in units] == ["row1\n", "row2"]
    assert [s.role for s in units[1].segments] == ["title", "table_header", "core"]
    assert [s.text for s in units[1].segments] == ["Title", "H1\n", "row2"]


def test_text_origin_acceptance_outputs_are_lossless_and_deterministic() -> None:
    """Keep the accepted text-only fixtures stable without fabricating media origins."""
    plain_text = "```py\n学习🙂\n```\n\n超长无句号文本"
    plain_representation = text_representation("doc:plain", plain_text, GraphScope("w", "s"))
    structured_text = "第一节\n甲🙂\n第二节\n乙\n表题\n列A\n😀一\n😀二\n"
    structured_representation = replace(
        text_representation("doc:structured", structured_text, GraphScope("w", "s")),
        blocks=(
            BlockSpan("h1", 0, 4, None, "heading", ("第一节",)),
            BlockSpan("p1", 4, 7, None, "paragraph", ("第一节",)),
            BlockSpan("h2", 7, 11, None, "heading", ("第二节",)),
            BlockSpan("p2", 11, 13, None, "paragraph", ("第二节",)),
            BlockSpan("h3", 13, 16, None, "heading", ("表题",)),
            BlockSpan("header", 16, 19, None, "table_header", ("表题",)),
            BlockSpan("row1", 19, 22, None, "table_row", ("表题",)),
            BlockSpan("row2", 22, 25, None, "table_row", ("表题",)),
        ),
    )
    cases = (
        (
            plain_representation,
            ChunkingProfile(5, 5, 0),
            ("```py", "\n学习🙂\n", "```\n\n", "超长无句号", "文本"),
            ((0, 5), (5, 10), (10, 15), (15, 20), (20, 22)),
            ((), (), (), (), ()),
            (("core",),) * 5,
            plain_text,
        ),
        (
            structured_representation,
            ChunkingProfile(3, 3, 6),
            ("甲🙂\n", "乙\n", "😀一\n", "😀二\n"),
            ((4, 7), (11, 13), (19, 22), (22, 25)),
            (("第一节",), ("第二节",), ("表题",), ("表题",)),
            (
                ("title", "core"),
                ("title", "core"),
                ("title", "table_header", "core"),
                ("title", "table_header", "core"),
            ),
            "甲🙂\n乙\n😀一\n😀二\n",
        ),
    )
    for (
        representation,
        profile,
        core_texts,
        core_ranges,
        section_paths,
        roles,
        eligible_text,
    ) in cases:
        first = compose_text_units(representation, CharacterTokenizer(), profile)
        second = compose_text_units(representation, CharacterTokenizer(), profile)
        assert first == second
        assert tuple(unit.segments[-1].text for unit in first) == core_texts
        assert (
            tuple(
                (
                    unit.core_locator.character_range.start_char,
                    unit.core_locator.character_range.end_char,
                )
                for unit in first
            )
            == core_ranges
        )
        assert tuple(unit.section_path for unit in first) == section_paths
        assert tuple(tuple(segment.role for segment in unit.segments) for unit in first) == roles
        assert all(unit.token_count <= profile.max_tokens for unit in first)
        assert all(unit.segments[-1].text for unit in first)
        assert "".join(unit.segments[-1].text for unit in first) == eligible_text
        assert all(left[1] <= right[0] for left, right in pairwise(core_ranges))


@pytest.mark.parametrize("source", ["", "\n\t ", "```python\nprint('hi')\n```", "x" * 40])
def test_plain_fallback_is_lossless_or_empty_without_fabricated_structure(source: str) -> None:
    rep = text_representation("doc", source, GraphScope("w", "s"))
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(5, 7, 0))
    assert "".join(u.segments[-1].text for u in units) == (source if source.strip() else "")


@pytest.mark.parametrize("source", ["a     b", "abc   ", "    abc", "a\t\t\tb"])
def test_whitespace_inside_eligible_paragraphs_is_not_silently_dropped(source: str) -> None:
    rep = text_representation("doc", source, GraphScope("w", "s"))
    units = compose_text_units(rep, CharacterTokenizer(), ChunkingProfile(2, 2, 1))
    assert "".join(u.segments[-1].text for u in units) == source


@pytest.mark.parametrize("source", ["\n\nabc", "abc   ", "a     b"])
def test_zero_token_whitespace_is_attached_to_a_positive_token_core(source: str) -> None:
    class WhitespaceIgnoringTokenizer(CharacterTokenizer):
        def count(self, text: str) -> int:
            return len(text.strip())

    rep = text_representation("doc", source, GraphScope("w", "s"))
    units = compose_text_units(rep, WhitespaceIgnoringTokenizer(), ChunkingProfile(1, 1, 0))
    assert "".join(u.segments[-1].text for u in units) == source
    assert all(u.token_count == 1 for u in units)


class WorkCountingTokenizer:
    fingerprint = "a" * 64

    def __init__(self) -> None:
        self.calls = 0
        self.codepoints = 0

    def count(self, text: str) -> int:
        self.calls += 1
        self.codepoints += len(text)
        return len(text)


def test_exhaustive_chunking_work_is_bounded_before_calling_tokenizer() -> None:
    rep = text_representation("doc", "x" * 1000, GraphScope("w", "s"))
    tokenizer = WorkCountingTokenizer()
    with pytest.raises(ValueError, match="work budget"):
        compose_text_units(rep, tokenizer, ChunkingProfile(2, 2, 1))
    assert tokenizer.calls <= 50_000
    assert tokenizer.codepoints <= 8_000_000


def test_oversized_representation_fails_before_tokenization() -> None:
    class UnexpectedTokenizer(WorkCountingTokenizer):
        def count(self, text: str) -> int:
            raise AssertionError("Oversized inputs must be rejected before tokenization")

    rep = text_representation("doc", "x" * 1_000_001, GraphScope("w", "s"))
    tokenizer = UnexpectedTokenizer()
    with pytest.raises(ValueError, match="input limit"):
        compose_text_units(rep, tokenizer, ChunkingProfile())
    assert tokenizer.calls == 0


@pytest.mark.parametrize(
    "start,end,policy", [(0, 0, "a" * 64), (-1, 1, "a" * 64), (True, 2, "a" * 64), (0, 1, "bad")]
)
def test_occurrence_identity_rejects_invalid_coordinates_or_policy(
    start: Any, end: Any, policy: str
) -> None:
    with pytest.raises((TypeError, ValueError)):
        occurrence_id("s", "doc", "rep", start, end, policy)
