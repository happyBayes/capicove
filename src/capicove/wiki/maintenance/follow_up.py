"""Activate committed deferred work against the current publication view."""

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

from capicove.contracts.wiki_maintenance import CompilationTask
from capicove.wiki.maintenance.ownership import stale_pages
from capicove.wiki.maintenance.sources import committed_bytes
from capicove.wiki.maintenance.store import record_path, write_object
from capicove.wiki.maintenance.tasks import create_task


if TYPE_CHECKING:
    from capicove.wiki.maintenance.service import MaintenanceService


@dataclass(frozen=True, slots=True)
class FollowUpActivation:
    """A real compilation context and explicitly unselected deferred pages."""

    task: CompilationTask
    remaining_page_ids: tuple[str, ...]


def activate(
    service: "MaintenanceService", identity: str, page_ids: tuple[str, ...]
) -> FollowUpActivation:
    """Select at most 20 stale pages without claiming the remainder is resolved."""
    service._admit()
    path = record_path(service.wiki, "follow-ups", identity, "task")
    value = json.loads(committed_bytes(service, path.relative_to(service.wiki).as_posix()))
    if (
        not isinstance(value, dict)
        or value.get("task_id") != identity
        or not isinstance(value.get("page_ids"), list)
        or not isinstance(value.get("source_ids"), list)
        or not 1 <= len(value["page_ids"]) <= 100
        or not 1 <= len(value["source_ids"]) <= 8
        or any(not isinstance(item, str) for item in [*value["page_ids"], *value["source_ids"]])
    ):
        raise ValueError("invalid follow-up record")
    stale = stale_pages(service)
    remaining = tuple(page for page in value["page_ids"] if page in stale)
    selected = page_ids or remaining[:20]
    if not selected or not set(selected) <= set(remaining):
        raise ValueError("follow_up_resolved_or_out_of_scope")
    pages = {page.page_id: page for page in service._inventory()}
    task = create_task(
        service, tuple(value["source_ids"]), pages[selected[0]].title, None, None, selected
    )
    write_object(
        record_path(service.wiki, "follow-up-activations", task.task_id, "task"),
        {
            "schema_version": 1,
            "follow_up_id": identity,
            "task_id": task.task_id,
            "wiki_revision": task.wiki_revision,
            "page_ids": list(selected),
        },
        service.wiki,
    )
    return FollowUpActivation(task, tuple(page for page in remaining if page not in selected))
