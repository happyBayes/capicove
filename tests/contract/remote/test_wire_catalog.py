"""Catalog cursors bind exact schemas, permissions, authority and session."""

import importlib
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.unit.runtime.remote.test_remote_dispatch import dispatcher


def fixture(tmp_path: Path, count: int = 51) -> tuple[Any, Any, tuple[Any, ...]]:
    assert importlib.util.find_spec("capicove.mcp.remote_catalog") is not None
    module = importlib.import_module("capicove.mcp.remote_catalog")
    dispatch, _, _ = dispatcher(tmp_path)
    base = dispatch._bindings["capicove_test_echo"]
    bindings = tuple(
        replace(
            base,
            name=f"capicove_test_item_{index:03}",
            spec=replace(base.spec, tool_id=f"capicove.test.item_{index:03}"),
        )
        for index in range(count)
    )
    pager = module.CatalogPager(b"fixture-catalog-signing-key-32-bytes")
    return pager, dispatch.authority, bindings


def test_catalog_has_at_most_fifty_entries_per_page(tmp_path: Path) -> None:
    pager, authority, bindings = fixture(tmp_path)
    first = pager.page(
        bindings, authority=authority, session_id="session", permissions=frozenset({"read"})
    )
    assert len(first.bindings) == 50
    assert first.next_cursor
    second = pager.page(
        bindings,
        authority=authority,
        session_id="session",
        permissions=frozenset({"read"}),
        cursor=first.next_cursor,
    )
    assert len(second.bindings) == 1
    assert second.next_cursor is None
    assert first.catalog_digest == second.catalog_digest


@pytest.mark.parametrize("mutation", ["session", "grant", "scope", "schema", "signature"])
def test_foreign_or_stale_cursor_is_rejected(tmp_path: Path, mutation: str) -> None:
    pager, authority, bindings = fixture(tmp_path)
    first = pager.page(
        bindings, authority=authority, session_id="session", permissions=frozenset({"read"})
    )
    session_id, permissions, cursor = "session", frozenset({"read"}), first.next_cursor
    if mutation == "session":
        session_id = "other"
    elif mutation == "grant":
        authority = replace(authority, grant_id="other")
    elif mutation == "scope":
        permissions = frozenset()
    elif mutation == "schema":
        bindings = (
            replace(
                bindings[0], spec=replace(bindings[0].spec, description="Changed schema contract")
            ),
            *bindings[1:],
        )
    elif mutation == "signature":
        cursor = "x" + cursor[1:]
    with pytest.raises(ValueError):
        pager.page(
            bindings,
            authority=authority,
            session_id=session_id,
            permissions=permissions,
            cursor=cursor,
        )


def test_duplicate_wire_catalog_is_rejected(tmp_path: Path) -> None:
    pager, authority, bindings = fixture(tmp_path, 1)
    with pytest.raises(ValueError):
        pager.page(
            (bindings[0], bindings[0]),
            authority=authority,
            session_id="session",
            permissions=frozenset({"read"}),
        )


def test_catalog_byte_budget_splits_small_entry_counts(tmp_path: Path) -> None:
    pager, authority, bindings = fixture(tmp_path, 2)
    bindings = tuple(
        replace(item, spec=replace(item.spec, description="x" * 300000)) for item in bindings
    )
    page = pager.page(
        bindings, authority=authority, session_id="session", permissions=frozenset({"read"})
    )
    assert len(page.bindings) == 1
    assert page.next_cursor


def test_expired_catalog_cursor_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pager, authority, bindings = fixture(tmp_path)
    module = importlib.import_module("capicove.mcp.remote_catalog")
    monkeypatch.setattr(module.time, "time", lambda: 1000.0)
    first = pager.page(
        bindings, authority=authority, session_id="session", permissions=frozenset({"read"})
    )
    monkeypatch.setattr(module.time, "time", lambda: 1301.0)
    with pytest.raises(ValueError):
        pager.page(
            bindings,
            authority=authority,
            session_id="session",
            permissions=frozenset({"read"}),
            cursor=first.next_cursor,
        )


def test_single_entry_cannot_consume_the_response_envelope_budget(tmp_path: Path) -> None:
    import json

    pager, authority, bindings = fixture(tmp_path, 1)
    module = importlib.import_module("capicove.mcp.remote_catalog")
    entry = module.catalog_entry(bindings[0])
    size = len(json.dumps(entry, sort_keys=True, separators=(",", ":")).encode())
    description = bindings[0].spec.description + "x" * (module.MAX_PAGE_BYTES - size)
    binding = replace(bindings[0], spec=replace(bindings[0].spec, description=description))
    with pytest.raises(ValueError):
        pager.page(
            (binding,), authority=authority, session_id="session", permissions=frozenset({"read"})
        )
