"""Bounded readiness Tool contract of the authenticated remote MCP endpoint (R2.04)."""

from __future__ import annotations

from capicove.contracts import PrivacyClass, SideEffect, ToolSpec


READINESS_SPEC = ToolSpec(
    tool_id="capicove.remote.readiness",
    contract_version=1,
    description="Report bounded readiness of this remote MCP endpoint without invoking models.",
    input_schema={"type": "object", "additionalProperties": False, "properties": {}},
    output_schema={"type": "object"},
    modalities=("text",),
    required_permissions=frozenset({"remote:readiness"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.READ,
    network_access=False,
    timeout_seconds=5,
)


__all__ = ["READINESS_SPEC"]
