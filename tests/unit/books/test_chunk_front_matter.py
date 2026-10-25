from dataclasses import replace

import pytest

from capicove.books import build_book_structure
from capicove.books.chunk_config import ChunkingConfig
from capicove.books.chunk_models import spans_text
from capicove.books.chunk_tasks import reference_tasks
from capicove.books.chunking import build_chunk_run
from capicove.contracts import DocumentBlockType as Kind
from capicove.contracts import DocumentParseResult
from tests.unit.books.test_structure import _block, _result


def front_matter_result() -> DocumentParseResult:
    """Return synthetic evidence, not text copied from a user's book."""
    return _result(
        {
            0: (
                _block(0, 0, "物理", Kind.DOCUMENT_TITLE),
                _block(0, 1, "普通高中教科书\n必修第一册\n示例出版社", Kind.TEXT),
            ),
            1: (
                _block(1, 0, "责任编辑: 甲\n出版: 示例出版社", Kind.TEXT),
                _block(1, 1, "版权所有, 未经许可不得复制。ISBN 000-0", Kind.OTHER),
            ),
            2: (_block(2, 0, "目录\n第一章 测量 10\n第二章 运动 20", Kind.TEXT),),
            3: (
                _block(3, 0, "致同学们", Kind.DOCUMENT_TITLE),
                _block(3, 1, "请先观察现象, 再设计实验", Kind.TEXT),
            ),
            4: (
                _block(4, 0, "序言", Kind.DOCUMENT_TITLE),
                _block(4, 1, "本书介绍实验方法。", Kind.TEXT),
            ),
        }
    )


def test_cover_and_copyright_never_enter_core_or_overlap() -> None:
    result = front_matter_result()
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert dict(run.exclusions) == {
        "p0000:b000000": "front_matter:cover",
        "p0000:b000001": "front_matter:cover",
        "p0001:b000000": "front_matter:copyright",
        "p0001:b000001": "front_matter:copyright",
    }
    assert all(
        span.page_index >= 2
        for chunk in run.chunks
        for span in (*chunk.core_spans, *chunk.before_spans, *chunk.after_spans)
    )
    assert run.coverage_valid


def test_toc_is_separate_navigation_not_a_knowledge_summary_task() -> None:
    result = front_matter_result()
    structure = build_book_structure(result)
    run = build_chunk_run(result, structure, ChunkingConfig(target_core_tokens=1000))
    toc = [chunk for chunk in run.chunks if chunk.core_spans[0].page_index == 2]
    assert toc and all(chunk.content_role == "navigation" for chunk in toc)
    assert all(
        span.page_index == 2
        for chunk in toc
        for span in (*chunk.core_spans, *chunk.before_spans, *chunk.after_spans)
    )
    tasks = reference_tasks(run, structure)
    assert not {chunk.chunk_id for chunk in toc} & {
        identifier for task in tasks for identifier in task["chunk_ids"]
    }
    body = [chunk for chunk in run.chunks if chunk.content_role == "content"]
    assert {chunk.chunk_id for chunk in body} <= {
        identifier for task in tasks for identifier in task["chunk_ids"]
    }
    assert tasks[-1]["task_type"] == "book_summary"


def test_guidance_preface_and_unknown_front_matter_are_retained() -> None:
    result = front_matter_result()
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    blocks = {block.block_id: block for page in result.pages for block in page.blocks}
    retained = "\n".join(spans_text(chunk.core_spans, blocks) for chunk in run.chunks)
    assert "致同学们" in retained and "序言" in retained and "实验方法" in retained
    unknown = _result({0: (_block(0, 0, "A dedication to curious readers.", Kind.TEXT),)})
    run = build_chunk_run(unknown, build_book_structure(unknown), ChunkingConfig())
    assert not run.exclusions and run.chunks


@pytest.mark.parametrize("first_page", [0, 14])
def test_body_discussing_copyright_is_not_a_copyright_page(first_page: int) -> None:
    result = _result(
        {
            first_page: (
                _block(first_page, 0, "Chapter 1 Copyright", Kind.DOCUMENT_TITLE),
                _block(first_page, 1, "Copyright and ISBN: all rights reserved.", Kind.TEXT),
            )
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert not run.exclusions


def test_copyright_phrase_in_preface_is_retained() -> None:
    result = _result(
        {
            0: (
                _block(0, 0, "Preface", Kind.DOCUMENT_TITLE),
                _block(
                    0, 1, "Copyright, all rights reserved and ISBN are explained here.", Kind.TEXT
                ),
            )
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert not run.exclusions


def test_new_policy_has_distinct_identity_and_legacy_round_trips() -> None:
    from capicove.books.chunk_codec import decode_run, encode_run

    result = front_matter_result()
    structure = build_book_structure(result)
    legacy = build_chunk_run(result, structure, ChunkingConfig(front_matter_policy="preserve"))
    current = build_chunk_run(result, structure, ChunkingConfig())
    assert "front_matter_policy" not in legacy.config_json
    assert not legacy.exclusions
    assert legacy.run_id != current.run_id
    assert encode_run(decode_run(encode_run(legacy))) == encode_run(legacy)


def test_english_title_and_copyright_pages_are_excluded() -> None:
    result = front_matter_result()
    pages = list(result.pages)
    pages[0] = replace(
        pages[0],
        blocks=(
            _block(0, 0, "Measurement", Kind.DOCUMENT_TITLE),
            _block(0, 1, "Second edition\nPublished by Example Press", Kind.TEXT),
        ),
    )
    pages[1] = replace(
        pages[1],
        blocks=(_block(1, 0, "Copyright 2026. All rights reserved. ISBN 000-0", Kind.TEXT),),
    )
    result = replace(result, pages=tuple(pages))
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert dict(run.exclusions)["p0000:b000000"] == "front_matter:cover"
    assert dict(run.exclusions)["p0001:b000000"] == "front_matter:copyright"


def test_toc_header_and_continuation_pages_share_navigation_parent() -> None:
    result = _result(
        {
            0: (
                _block(0, 0, "Contents", Kind.HEADER),
                _block(0, 1, "Chapter 1 Measurement 10\nChapter 2 Motion 20", Kind.TEXT),
            ),
            1: (_block(1, 0, "Chapter 3 Forces 30\nChapter 4 Energy 40", Kind.TEXT),),
            2: (_block(2, 0, "Preface\nExplore through experiments.", Kind.TEXT),),
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    navigation = [chunk for chunk in run.chunks if chunk.content_role == "navigation"]
    assert {span.page_index for chunk in navigation for span in chunk.core_spans} == {0, 1}
    assert len({chunk.parent_id for chunk in navigation}) == 1
    assert any(chunk.content_role == "content" for chunk in run.chunks)


def test_contents_example_inside_a_chapter_is_not_front_matter() -> None:
    result = _result(
        {
            0: (_block(0, 0, "Chapter 1 Document Design", Kind.DOCUMENT_TITLE),),
            1: (
                _block(
                    1, 0, "Contents\nChapter 1 Introduction 10\nChapter 2 Results 20", Kind.TEXT
                ),
            ),
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert all(chunk.content_role == "content" for chunk in run.chunks)


def test_numbered_preface_after_toc_is_not_a_toc_continuation() -> None:
    result = _result(
        {
            0: (
                _block(0, 0, "Contents\nChapter 1 Measurement 10\nChapter 2 Motion 20", Kind.TEXT),
            ),
            1: (
                _block(1, 0, "Preface", Kind.DOCUMENT_TITLE),
                _block(1, 1, "Start with experiment 1\nThen complete experiment 2", Kind.TEXT),
            ),
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    content = [chunk for chunk in run.chunks if chunk.content_role == "content"]
    assert {span.page_index for chunk in content for span in chunk.core_spans} == {1}


@pytest.mark.parametrize(
    "title,text",
    [
        (
            "How to use this textbook",
            "Published by Example Press\nRead each example before attempting the exercises.",
        ),
        (
            "Using this book",
            "Read each example before attempting the exercises.\nCopyright 2026. All rights reserved. ISBN 000-0",
        ),
        (
            "Reading notes",
            "Read each example before attempting the exercises.\nCopyright 2026. All rights reserved. ISBN 000-0",
        ),
    ],
)
def test_english_guidance_with_imprint_is_not_excluded(title: str, text: str) -> None:
    result = _result(
        {
            0: (
                _block(0, 0, title, Kind.DOCUMENT_TITLE),
                _block(0, 1, text, Kind.TEXT),
            )
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert not run.exclusions and run.chunks


def test_parsed_empty_first_page_does_not_hide_following_front_matter() -> None:
    result = _result(
        {
            0: (),
            1: (
                _block(1, 0, "Contents\nChapter 1 Measurement 10\nChapter 2 Motion 20", Kind.TEXT),
            ),
        }
    )
    run = build_chunk_run(result, build_book_structure(result), ChunkingConfig())
    assert run.chunks and all(chunk.content_role == "navigation" for chunk in run.chunks)
