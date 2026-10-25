"""Inventory claims must match existing contracts without inventing acceptance."""

from pathlib import Path

from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME
from capicove.tools.memory import MEMORY_SPECS


def test_memory_contracts_are_accepted_for_product_composition() -> None:
    for path in (
        Path("docs/rfcs/0009-temporal-memory-contracts.md"),
        Path("docs/adr/0015-temporal-memory-sqlite-boundaries.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "- Status: Accepted" in text
        assert "2026-09-28" in text


def test_inventory_names_all_current_image_and_memory_tools() -> None:
    text = Path("docs/development/cli-mcp-capability-catalog.md").read_text()
    for spec in (*IMAGE_TOOL_SPECS_BY_NAME.values(), *MEMORY_SPECS.values()):
        assert f"`{spec.tool_id}`" in text
    assert len(IMAGE_TOOL_SPECS_BY_NAME) == 18
    assert len(MEMORY_SPECS) == 11
    assert "not a maintainer-approved expansion" in text


def test_operator_guide_documents_supported_commands_and_exclusions() -> None:
    text = Path("docs/site/guides/mcp-stdio.md").read_text()
    for command in ("capicove mcp serve", "capicove mcp catalog"):
        assert command in text
    for field in ("schema_version", "enabled_tools", "permissions", "trace_store", "read_roots"):
        assert f'"{field}"' in text
    assert "Deadlines are cooperative" in text
