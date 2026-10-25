"""Prevent transport reach-through and oversized implementation reunification."""

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3] / "src" / "capicove"


def test_memory_transports_do_not_import_backend_or_reach_through_agent() -> None:
    for relative in (
        "tools/memory_tools.py",
        "cli/commands/memory_commands.py",
        "mcp/memory_transport.py",
    ):
        tree = ast.parse((ROOT / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                assert not module.startswith(("sqlite3", "capicove.knowledge", "capicove.source"))
            if isinstance(node, ast.Import):
                assert all(
                    not name.name.startswith(("sqlite3", "capicove.knowledge._adapters"))
                    for name in node.names
                )
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"knowledge", "repository", "_db"}


def test_required_memory_responsibilities_have_implementation_modules() -> None:
    for relative in (
        "knowledge/temporal.py",
        "knowledge/entity_resolution.py",
        "knowledge/extraction.py",
        "knowledge/diagnostics.py",
        "knowledge/_adapters/sqlite_schema.py",
        "knowledge/_adapters/sqlite_codec.py",
        "agent/memory_policy.py",
        "agent/context_assembly.py",
        "source/models.py",
        "tools/memory_tools.py",
    ):
        assert (ROOT / relative).is_file()


def test_main_memory_orchestrators_remain_reviewable() -> None:
    for relative in ("knowledge/service.py", "knowledge/_adapters/sqlite_memory.py"):
        assert len((ROOT / relative).read_text().splitlines()) < 650
