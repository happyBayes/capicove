"""Frozen bilingual labels precede lexical policy implementation."""

import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from capicove.wiki.maintenance.discovery import DiscoveryPage, build_discovery, discover
from capicove.wiki.maintenance.impact import affected_pages


SOURCE = "source:sha256:" + "a" * 64


def page(
    identity: str,
    title: str,
    body: str,
    *,
    aliases: tuple[str, ...] = (),
    links: tuple[str, ...] = (),
    sources: tuple[str, ...] = (),
) -> DiscoveryPage:
    return DiscoveryPage(
        identity,
        "pages/" + identity.replace(":", "-") + ".md",
        title,
        body,
        hashlib.sha256(body.encode()).hexdigest(),
        aliases,
        links,
        sources,
        (),
    )


def corpus() -> tuple[DiscoveryPage, ...]:
    return (
        page(
            "page:ZH", "光合作用", "植物利用光能形成有机物", aliases=("碳同化",), sources=(SOURCE,)
        ),
        page("page:EN", "Photosynthesis", "Plants convert light energy into chemical energy"),
        page("page:MIX", "ATP 与能量", "ATP stores chemical energy", links=("page:ZH",)),
        page("page:OTHER", "河流", "侵蚀沉积"),
    )


@pytest.mark.parametrize(
    "query,expected",
    [
        ("光能", "page:ZH"),
        ("碳同化", "page:ZH"),
        ("Photosynthesis", "page:EN"),
        ("ATP 能量", "page:MIX"),
    ],
)
def test_frozen_bilingual_recall(tmp_path: Path, query: str, expected: str) -> None:
    pages = corpus()
    index = tmp_path / "discovery.sqlite"
    build_discovery(index, pages)
    result = discover(index, pages, query)
    assert expected in [item.page_id for item in result.candidates]
    assert result.scanned_pages == 4
    assert result.coverage == "complete_inventory"
    assert result.policy == "wiki-lexical-v1-rrf60"
    assert all(item.lane_ranks for item in result.candidates)


def test_empty_lane_is_not_semantic_absence_and_ties_are_stable(tmp_path: Path) -> None:
    pages = tuple(page(f"page:{i:03}", "Shared", "same") for i in reversed(range(25)))
    index = tmp_path / "discovery.sqlite"
    build_discovery(index, pages)
    result = discover(index, pages, "Shared")
    assert [item.page_id for item in result.candidates] == [f"page:{i:03}" for i in range(20)]
    assert result.coverage == "partial"
    assert result.omitted_candidates == 5
    empty = discover(index, pages, "unknown")
    assert empty.candidates == ()
    assert empty.semantic_recall == "unknown"


def test_backlinks_shared_sources_and_expansion_authorization(tmp_path: Path) -> None:
    pages = corpus()
    index = tmp_path / "discovery.sqlite"
    build_discovery(index, pages)
    result = discover(index, pages, "irrelevant", source_ids=(SOURCE,), page_ids=("page:ZH",))
    assert {item.page_id for item in result.candidates} == {"page:ZH", "page:MIX"}
    assert affected_pages(pages, source_ids=(SOURCE,), page_ids=("page:ZH",)).page_ids == (
        "page:MIX",
        "page:ZH",
    )
    with pytest.raises(ValueError, match="scope"):
        discover(index, pages, "x", page_ids=("page:OUTSIDE",))


def test_index_missing_stale_and_invalid_bounds_fail_closed(tmp_path: Path) -> None:
    pages = corpus()
    index = tmp_path / "discovery.sqlite"
    with pytest.raises(ValueError, match="rebuild"):
        discover(index, pages, "light")
    build_discovery(index, pages)
    changed = (replace(pages[0], aliases=("changed",)), *pages[1:])
    with pytest.raises(ValueError, match="stale"):
        discover(index, changed, "light")
    with pytest.raises(ValueError, match="capacity"):
        discover(index, pages, "light", limit=21)
    with pytest.raises(ValueError, match="duplicate"):
        build_discovery(index, (pages[0], pages[0]))


def test_impact_overflow_is_explicit_follow_up() -> None:
    pages = tuple(page(f"page:{i:03}", "Shared", "same", sources=(SOURCE,)) for i in range(105))
    result = affected_pages(pages, source_ids=(SOURCE,))
    assert len(result.page_ids) == 100
    assert len(result.follow_up_page_ids) == 5
    assert result.coverage == "partial"
