"""Explicit single-backend local memory composition; never silently fall back."""

import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import cast

from capicove.agent import AgentMemoryService
from capicove.contracts import InvocationContext, ToolHandler
from capicove.contracts.knowledge import Predicate
from capicove.contracts.memory import MemoryQuery
from capicove.knowledge import SQLiteMemoryRepository, TemporalKnowledgeService
from capicove.knowledge.ports import EntityFactExtractor
from capicove.source import SQLiteTextEvidenceStore
from capicove.tools.memory_tools import MEMORY_SPECS, MemoryToolHandler


def build_memory_runtime(
    root: Path,
    *,
    workspace: str,
    scopes: frozenset[str],
    ontology_version: str,
    predicates: tuple[Predicate, ...],
    backend: str = "sqlite",
    extractor: EntityFactExtractor | None = None,
    dense: Callable[[MemoryQuery], tuple[str, ...]] | None = None,
    authorize_scope: Callable[[InvocationContext, str], bool] | None = None,
) -> dict[str, ToolHandler]:
    if backend != "sqlite":
        raise RuntimeError("backend_unavailable")
    if any(parent.is_symlink() for parent in (root, *root.parents)):
        raise ValueError("unsafe memory root")
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    if root.stat().st_mode & 0o077:
        if root.stat().st_uid == os.getuid() and not any(root.iterdir()):
            root.chmod(0o700)
        else:
            raise ValueError("memory root must be private")
    repository = SQLiteMemoryRepository(root / "memory.sqlite3")
    try:
        service = AgentMemoryService(
            TemporalKnowledgeService(
                repository,
                SQLiteTextEvidenceStore(
                    root / "evidence.sqlite3",
                    root / "privacy-authority",
                    authorize=lambda w, s: w == workspace and s in scopes,
                ),
                ontology_version=ontology_version,
                predicates=predicates,
                extractor=extractor,
                dense=dense,
            )
        )

        def authorized(context: InvocationContext, scope: str) -> bool:
            return scope in scopes and (authorize_scope is None or authorize_scope(context, scope))

        return {
            name: MemoryToolHandler(name, service, workspace=workspace, authorize_scope=authorized)
            for name in MEMORY_SPECS
        }
    except BaseException:
        repository.close()
        raise


def close_memory_runtime(handlers: Mapping[str, ToolHandler]) -> None:
    """Close storage owned by one successfully composed Memory runtime."""
    owner = handlers["doctor"]
    assert isinstance(owner, MemoryToolHandler)
    repository = cast(SQLiteMemoryRepository, owner.service.knowledge.repository)
    repository.close()
