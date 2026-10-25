"""Reference-only caller task DAGs; separate from legacy summary submissions."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from capicove.books.chunk_models import ChunkRun, digest
from capicove.books.models import BookStructure


def reference_tasks(run: ChunkRun, structure: BookStructure) -> tuple[dict[str, Any], ...]:
    """Export chunk -> section -> book tasks with bounded reduction fan-in."""
    tasks: list[dict[str, Any]] = []
    by_parent: dict[str, list[str]] = defaultdict(list)

    def add(kind: str, parent: str, dependencies: list[str], chunk_ids: list[str]) -> str:
        identifier = (
            "chunk-task:" + digest([run.run_id, kind, parent, dependencies, chunk_ids])[:24]
        )
        tasks.append(
            {
                "schema_version": 1,
                "task_id": identifier,
                "task_type": kind,
                "run_id": run.run_id,
                "parent_id": parent,
                "dependency_task_ids": dependencies,
                "chunk_ids": chunk_ids,
                "execution": "caller_owned",
                "submission_contract": None,
            }
        )
        return identifier

    def reduce(kind: str, parent: str, dependencies: list[str]) -> str:
        while len(dependencies) > 32:
            dependencies = [
                add("reduce_group", parent, dependencies[index : index + 32], [])
                for index in range(0, len(dependencies), 32)
            ]
        return add(kind, parent, dependencies, [])

    for chunk in run.chunks:
        if chunk.content_role == "navigation":
            continue
        by_parent[chunk.parent_id].append(
            add("chunk_summary", chunk.parent_id, [], [chunk.chunk_id])
        )
    summaries: dict[str, str] = {}
    for section in sorted(structure.sections, key=lambda section: section.level, reverse=True):
        dependencies = by_parent[section.section_id] + [
            summaries[key] for key in section.children_ids if key in summaries
        ]
        if dependencies:
            summaries[section.section_id] = reduce(
                "section_summary", section.section_id, dependencies
            )
    roots = [summaries[key] for key in structure.root_ids if key in summaries]
    roots.extend(by_parent["front-matter"])
    if roots:
        reduce("book_summary", "book", roots)
    return tuple(tasks)
