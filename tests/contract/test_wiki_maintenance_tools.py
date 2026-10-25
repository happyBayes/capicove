"""Host-bound Tools deny unknown fields, forged identities and default writes."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from capicove.contracts import InvocationContext, ToolStatus
from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


def test_default_catalog_and_denied_requests(tmp_path: Path) -> None:
    from capicove.tools.wiki_maintenance import MaintenanceToolHandler, maintenance_specs

    service, _, _ = prepared(tmp_path)
    assert all(spec.side_effect.value == "read" for spec in maintenance_specs())
    context = InvocationContext(
        "r",
        "t",
        "local",
        "operator",
        "test",
        datetime.now(UTC) + timedelta(seconds=60),
        frozenset({"wiki:read"}),
    )
    before = snapshot(service.wiki)
    handler = MaintenanceToolHandler("capture", {"local": service})
    result = handler(context, {"workspace_id": "local", "source": str(tmp_path / "a.md")})
    assert result.status is not ToolStatus.SUCCEEDED
    status = MaintenanceToolHandler("status", {"local": service})
    for arguments in (
        {"workspace_id": "../../escape"},
        {"workspace_id": "local", "reviewer_id": "admin"},
    ):
        assert status(context, arguments).status is not ToolStatus.SUCCEEDED
    assert snapshot(service.wiki) == before


def test_host_actor_is_review_authority(tmp_path: Path) -> None:
    from capicove.tools.wiki_maintenance import MaintenanceToolHandler

    service, proposal, _ = prepared(tmp_path)
    context = InvocationContext(
        "r",
        "t",
        "local",
        "real-operator",
        "test",
        datetime.now(UTC) + timedelta(seconds=60),
        frozenset({"wiki:review"}),
    )
    handler = MaintenanceToolHandler("review", {"local": service}, enable_mutations=True)
    decisions = [
        {"subject_id": subject, "verdict": "supported", "reason": "Synthetic test."}
        for subject in service.preview(proposal.proposal_id).review_subjects
    ]
    result = handler(
        context,
        {
            "workspace_id": "local",
            "proposal_id": proposal.proposal_id,
            "accept": True,
            "decisions": decisions,
        },
    )
    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None and result.result["reviewer_id"] == "real-operator"


def test_context_expansion_and_compensation_are_in_mutation_catalog() -> None:
    from capicove.tools.wiki_maintenance import maintenance_specs

    specs = {spec.tool_id: spec for spec in maintenance_specs(enable_mutations=True)}
    assert "capicove.wiki.maintenance-follow-up" in specs
    assert "capicove.wiki.maintenance-rollback" in specs
    context: Any = specs["capicove.wiki.maintenance-context"].input_schema
    assert "intervals" in context["properties"]
    assert "page_ids" in context["properties"]
    assert "intervals" not in context["required"]
