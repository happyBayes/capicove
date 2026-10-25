"""Provider-neutral operational health without source payloads."""

from typing import Any

from capicove.knowledge.ports import TemporalMemoryRepository


def diagnose(
    repository: TemporalMemoryRepository, *, extractor_configured: bool, dense_configured: bool
) -> dict[str, Any]:
    return dict(
        repository.doctor(),
        extractor="configured" if extractor_configured else "deferred",
        dense="configured" if dense_configured else "disabled",
        worker="explicit_work_command",
    )
