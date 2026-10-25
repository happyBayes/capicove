from dataclasses import replace
from itertools import pairwise

import pytest

from capicove.books import build_book_structure
from capicove.books.chunk_config import ChunkingConfig, resolve_chunk_config
from capicove.contracts import DocumentBlockType as Kind
from tests.unit.books.test_structure import _block, _result


def _config(**values: object) -> ChunkingConfig:
    return resolve_chunk_config(
        overrides={
            "target_core_tokens": 50,
            "max_chunk_tokens": 100,
            "context_before_tokens": 0,
            **values,
        }
    )


def test_oversized_chinese_block_has_exact_disjoint_source_coverage() -> None:
    from capicove.books.chunking import build_chunk_run

    text = "位移描述位置的变化。" * 30 + "A😀e\u0301"
    result = _result(
        {0: (_block(0, 0, "Chapter 1", Kind.DOCUMENT_TITLE), _block(0, 1, text, Kind.TEXT))}
    )
    run = build_chunk_run(result, build_book_structure(result), _config())
    spans = [
        span
        for chunk in run.chunks
        for span in chunk.core_spans
        if span.block_id == "p0000:b000001"
    ]
    assert "".join(span.text(result) for span in spans) == text
    assert spans[0].start_char == 0 and spans[-1].end_char == len(text)
    assert all(a.end_char == b.start_char for a, b in pairwise(spans))
    assert all(chunk.rendered_tokens <= 100 for chunk in run.chunks)
    assert run.coverage_valid


def test_context_is_separate_and_does_not_consume_or_duplicate_core() -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result(
        {
            0: (
                _block(0, 0, "Chapter 1", Kind.DOCUMENT_TITLE),
                _block(0, 1, "A small definition. " * 5, Kind.TEXT),
                _block(0, 2, "It describes motion. " * 5, Kind.TEXT),
            )
        }
    )
    structure = build_book_structure(result)
    plain = build_chunk_run(result, structure, _config())
    contextual = build_chunk_run(
        result, structure, _config(context_before_tokens=20, context_after_tokens=10)
    )
    assert any(chunk.before_spans for chunk in contextual.chunks)
    assert any(chunk.after_spans for chunk in contextual.chunks)
    assert all(chunk.rendered_tokens <= 100 for chunk in contextual.chunks)
    assert contextual.coverage_valid and plain.coverage_valid
    assert contextual.run_id != plain.run_id


def test_page_and_section_policies_affect_packing_without_losing_text() -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result(
        {
            0: (
                _block(0, 0, "Chapter 1", Kind.DOCUMENT_TITLE),
                _block(0, 1, "A continuing", Kind.TEXT),
            ),
            1: (_block(1, 0, " explanation.", Kind.TEXT),),
            2: (
                _block(2, 0, "1.1 Scope", Kind.PARAGRAPH_TITLE),
                _block(2, 1, "A new section.", Kind.TEXT),
            ),
        }
    )
    structure = build_book_structure(result)
    strict = build_chunk_run(result, structure, _config(page_boundary="strict"))
    flowing = build_chunk_run(result, structure, _config(page_boundary="allow_continuations"))
    assert all(len({span.page_index for span in chunk.core_spans}) == 1 for chunk in strict.chunks)
    assert any(len({span.page_index for span in chunk.core_spans}) == 2 for chunk in flowing.chunks)
    assert len({chunk.parent_id for chunk in flowing.chunks}) == 2


def test_protected_definition_can_exceed_target_but_not_maximum() -> None:
    from capicove.books.chunking import build_chunk_run

    text = "Definition: A measurement describes a physical property."
    result = _result({0: (_block(0, 0, text, Kind.TEXT),)})
    run = build_chunk_run(result, build_book_structure(result), _config(target_core_tokens=20))
    assert len(run.chunks) == 1
    assert run.chunks[0].core_tokens > 20
    assert "definition" in run.chunks[0].content_types


def test_oversize_error_does_not_truncate_and_formula_is_not_sliced() -> None:
    from capicove.books.chunking import build_chunk_run

    for kind, policy in ((Kind.TEXT, "error"), (Kind.FORMULA, "split")):
        result = _result({0: (_block(0, 0, "x" * 300, kind),)})
        with pytest.raises(ValueError, match="oversize"):
            build_chunk_run(result, build_book_structure(result), _config(oversize_policy=policy))


def test_read_policy_does_not_rechunk_and_tampered_span_is_rejected() -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result({0: (_block(0, 0, "An exact source.", Kind.TEXT),)})
    structure = build_book_structure(result)
    first = build_chunk_run(result, structure, _config())
    second = build_chunk_run(result, structure, _config(max_context_read_tokens=3000))
    assert first.run_id == second.run_id
    assert first.chunks == second.chunks
    span = replace(first.chunks[0].core_spans[0], text_sha256="0" * 64)
    with pytest.raises(ValueError, match="hash"):
        span.text(result)


def test_furniture_is_explicitly_excluded() -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result(
        {
            0: (
                _block(0, 0, "Running title", Kind.HEADER),
                _block(0, 1, "Body", Kind.TEXT),
                _block(0, 2, "1", Kind.PAGE_NUMBER),
            )
        }
    )
    run = build_chunk_run(result, build_book_structure(result), _config())
    assert {item[0] for item in run.exclusions} == {"p0000:b000000", "p0000:b000002"}
    assert run.coverage_valid


def test_nonmonotonic_counter_never_accepts_an_over_budget_shorter_cut() -> None:
    from capicove.books.chunking import build_chunk_run

    class Counter:
        fingerprint = "test-nonmonotonic"
        mode = "exact"

        def count(self, text: str) -> int:
            if text == "A. ":
                return 200
            return len(text)

    result = _result({0: (_block(0, 0, "A. " + "x" * 200, Kind.TEXT),)})
    run = build_chunk_run(result, build_book_structure(result), _config(), counter=Counter())
    assert all(chunk.rendered_tokens <= 100 for chunk in run.chunks)


def test_nonmonotonic_prefix_search_reaches_short_valid_prefix() -> None:
    from capicove.books.chunking import _fitting_prefix

    class Counter:
        fingerprint = "test"
        mode = "exact"

        def count(self, text: str) -> int:
            return 1 if len(text) == 1 else 101

    config = ChunkingConfig(target_core_tokens=100, max_chunk_tokens=100, context_before_tokens=0)
    assert _fitting_prefix("x" * 40000, (), config, Counter()) == 1


@pytest.mark.parametrize(
    "title,body,preset",
    [
        (
            "Theorem: Every position has a reference.",
            "Proof: Choose the reference first.",
            "math-heavy",
        ),
        ("Example: Compute the speed.", "Solution: Divide distance by time.", "textbook-balanced"),
        (
            "Definition: A vector has direction.",
            "This means direction must be recorded.",
            "textbook-balanced",
        ),
    ],
)
def test_protected_adjacent_units_remain_together_over_soft_target(
    title: str, body: str, preset: str
) -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result({0: (_block(0, 0, title, Kind.TEXT), _block(0, 1, body, Kind.TEXT))})
    run = build_chunk_run(
        result, build_book_structure(result), _config(preset=preset, target_core_tokens=10)
    )
    assert len(run.chunks) == 1


def test_protection_forced_apart_by_hard_budget_is_reported() -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result(
        {
            0: (
                _block(0, 0, "Theorem: " + "x" * 65, Kind.TEXT),
                _block(0, 1, "Proof: " + "y" * 65, Kind.TEXT),
            )
        }
    )
    run = build_chunk_run(result, build_book_structure(result), _config(preset="math-heavy"))
    assert len(run.chunks) == 2
    assert any("protected_group_split:packing_limit" in chunk.warnings for chunk in run.chunks)


@pytest.mark.parametrize(
    "kind,text",
    [
        (Kind.TEXT, "def method():\n" + "    call()\n" * 30),
        (Kind.TABLE, "| label | value |\n" + "| item | 123 |\n" * 30),
        (Kind.TEXT, ("A paragraph.\n\n" * 30)),
    ],
)
def test_type_aware_recursion_preserves_exact_source(kind: Kind, text: str) -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result({0: (_block(0, 0, text, kind),)})
    run = build_chunk_run(result, build_book_structure(result), _config(preset="technical-code"))
    assert len(run.chunks) > 1
    assert "".join(span.text(result) for chunk in run.chunks for span in chunk.core_spans) == text
    assert all(chunk.rendered_tokens <= 100 for chunk in run.chunks)


def test_large_reference_graph_is_dependency_ordered_with_bounded_fan_in() -> None:
    from capicove.books.chunk_tasks import reference_tasks
    from capicove.books.chunking import build_chunk_run

    result = _result(
        {index: (_block(index, 0, f"Source {index}.", Kind.TEXT),) for index in range(70)}
    )
    structure = build_book_structure(result)
    run = build_chunk_run(result, structure, _config(merge_small_chunks=False))
    tasks = reference_tasks(run, structure)
    seen: set[str] = set()
    for task in tasks:
        assert len(task["dependency_task_ids"]) <= 32
        assert set(task["dependency_task_ids"]) <= seen
        seen.add(task["task_id"])
    assert any(task["task_type"] == "reduce_group" for task in tasks)


def test_disabling_small_chunk_merge_does_not_disable_protection() -> None:
    from capicove.books.chunking import build_chunk_run

    result = _result(
        {
            0: (
                _block(0, 0, "Theorem: A stated result.", Kind.TEXT),
                _block(0, 1, "Proof: A supporting proof.", Kind.TEXT),
            )
        }
    )
    run = build_chunk_run(
        result,
        build_book_structure(result),
        _config(preset="math-heavy", merge_small_chunks=False, target_core_tokens=5),
    )
    assert len(run.chunks) == 1
