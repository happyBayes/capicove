"""Canonical operator model-management Tool contracts (RFC-0016, amendment 1).

Requests name an installed plugin, a profile, and a configured destination
alias. Installation accepts only a plan digest that the startup configuration
allowlists; no argument can supply a URL, path, credential, or confirmation.
"""

from __future__ import annotations

from capicove.contracts import PrivacyClass, SideEffect, ToolResources, ToolSpec


_ALIAS: dict[str, object] = {"type": "string", "pattern": "^[a-z][a-z0-9_-]{0,63}$"}
_SHA256: dict[str, object] = {"type": "string", "pattern": "^[0-9a-f]{64}$"}
_COUNT: dict[str, object] = {"type": "integer", "minimum": 0}
_SELECTORS: dict[str, object] = {
    "plugin": {"type": "string", "pattern": "^[a-z0-9][a-z0-9.-]{0,63}$"},
    "profile": {"type": "string", "pattern": "^[a-z0-9][a-z0-9._-]{0,63}$"},
    "destination": _ALIAS,
}
MODEL_SELECT_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["plugin", "profile", "destination"],
    "properties": _SELECTORS,
}
MODEL_INSTALL_INPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["plan_digest", "destination"],
    "properties": {"plan_digest": _SHA256, "destination": _ALIAS},
}
MODEL_PLAN_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "plan_digest",
        "plugin",
        "profile",
        "profile_version",
        "destination",
        "manifest_sha256",
        "models",
        "source_hosts",
        "total_size_bytes",
        "install_allowlisted",
    ],
    "properties": {
        "plan_digest": _SHA256,
        "plugin": {"type": "string"},
        "profile": {"type": "string"},
        "profile_version": {"type": "integer"},
        "destination": _ALIAS,
        "manifest_sha256": {"type": "string", "maxLength": 128},
        "models": {
            "type": "array",
            "maxItems": 64,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["key", "repository", "revision", "size_bytes", "files"],
                "properties": {
                    "key": {"type": "string"},
                    "repository": {"type": "string"},
                    "revision": {"type": "string"},
                    "size_bytes": _COUNT,
                    "files": {
                        "type": "array",
                        "maxItems": 10_000,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["path", "size_bytes", "sha256"],
                            "properties": {
                                "path": {"type": "string"},
                                "size_bytes": _COUNT,
                                "sha256": _SHA256,
                            },
                        },
                    },
                },
            },
        },
        "source_hosts": {"type": "array", "items": {"type": "string"}},
        "total_size_bytes": _COUNT,
        "install_allowlisted": {"type": "boolean"},
    },
}
MODEL_VERIFY_OUTPUT_SCHEMA: dict[str, object] = {
    "type": "object",
    "additionalProperties": False,
    "required": [
        "plan_digest",
        "destination",
        "manifest_sha256",
        "verified_files",
        "missing_files",
        "mismatched_files",
        "complete",
    ],
    "properties": {
        "plan_digest": _SHA256,
        "destination": _ALIAS,
        "manifest_sha256": {"type": "string", "maxLength": 128},
        "verified_files": _COUNT,
        "missing_files": _COUNT,
        "mismatched_files": _COUNT,
        "complete": {"type": "boolean"},
    },
}

MODEL_PLAN_SPEC = ToolSpec(
    tool_id="capicove.model.plan",
    contract_version=1,
    description="Return the pinned, immutable installation plan for one parser model profile.",
    input_schema=MODEL_SELECT_INPUT_SCHEMA,
    output_schema=MODEL_PLAN_OUTPUT_SCHEMA,
    required_permissions=frozenset({"model:read"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.READ,
    timeout_seconds=60.0,
)
MODEL_VERIFY_SPEC = ToolSpec(
    tool_id="capicove.model.verify",
    contract_version=1,
    description="Hash an installed model profile against its plan without downloading.",
    input_schema=MODEL_SELECT_INPUT_SCHEMA,
    output_schema=MODEL_VERIFY_OUTPUT_SCHEMA,
    required_permissions=frozenset({"model:read"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.READ,
    timeout_seconds=1800.0,
    resources=ToolResources(filesystem_read=True),
)
MODEL_INSTALL_SPEC = ToolSpec(
    tool_id="capicove.model.install",
    contract_version=1,
    description="Install one startup-allowlisted model plan into its configured destination.",
    input_schema=MODEL_INSTALL_INPUT_SCHEMA,
    output_schema=MODEL_VERIFY_OUTPUT_SCHEMA,
    required_permissions=frozenset({"model:install"}),
    privacy_class=PrivacyClass.INTERNAL,
    side_effect=SideEffect.EXTERNAL,
    network_access=True,
    deterministic=False,
    timeout_seconds=7200.0,
    resources=ToolResources(filesystem_read=True, filesystem_write=True),
)
MODEL_SPECS = {
    spec.tool_id: spec for spec in (MODEL_PLAN_SPEC, MODEL_VERIFY_SPEC, MODEL_INSTALL_SPEC)
}

__all__ = ["MODEL_INSTALL_SPEC", "MODEL_PLAN_SPEC", "MODEL_SPECS", "MODEL_VERIFY_SPEC"]
