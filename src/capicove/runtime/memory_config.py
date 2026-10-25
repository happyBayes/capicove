"""Closed host configuration and safe, actionable startup diagnostics."""

from typing import Any

from capicove.contracts.knowledge import Predicate


DEFAULT_PREDICATES = (
    Predicate("city", ("person",), "literal", ("string",), "one", "interval"),
    Predicate("employer", ("person",), "literal", ("string",), "one", "interval"),
    Predicate("preference", ("person",), "literal", ("string",), "many", "interval"),
    Predicate("knows", ("person",), "entity", ("person",), "many", "interval"),
)


class MemoryConfigurationError(ValueError):
    def __init__(self, field: str, hint: str) -> None:
        self.field, self.hint = field, hint
        super().__init__("invalid_memory_configuration")


def validate_config(config: Any) -> dict[str, Any]:
    allowed = {
        "root",
        "workspace",
        "scopes",
        "ontology_version",
        "predicates",
        "backend",
        "extractor",
        "provider_factory",
        "allow_provider_code",
    }
    if not isinstance(config, dict) or set(config) - allowed:
        raise MemoryConfigurationError("config", "Use only documented configuration fields.")
    for field in ("root", "workspace"):
        if not isinstance(config.get(field), str) or not config[field].strip():
            raise MemoryConfigurationError(
                field, "This field is required and must be a nonempty string."
            )
    if (
        not isinstance(config.get("scopes"), list)
        or not config["scopes"]
        or any(not isinstance(s, str) or not s.strip() for s in config["scopes"])
    ):
        raise MemoryConfigurationError(
            "scopes", "A nonempty list of authorized scopes is required."
        )
    if config.get("extractor", "none") not in {"none", "structured-json-v1"}:
        raise MemoryConfigurationError(
            "extractor",
            "Choose none or structured-json-v1, or configure a trusted provider factory.",
        )
    return config


def ontology(config: dict[str, Any]) -> tuple[str, tuple[Predicate, ...]]:
    if "predicates" not in config:
        if "ontology_version" in config:
            raise MemoryConfigurationError(
                "predicates", "Custom ontology_version requires explicit predicate definitions."
            )
        return "personal-v1", DEFAULT_PREDICATES
    try:
        if not isinstance(config["predicates"], list):
            raise ValueError
        predicates = []
        for p in config["predicates"]:
            if set(p) != {
                "name",
                "subject_types",
                "object_kind",
                "object_types",
                "cardinality",
                "temporal_kind",
            }:
                raise ValueError
            predicates.append(
                Predicate(
                    p["name"],
                    tuple(p["subject_types"]),
                    p["object_kind"],
                    tuple(p["object_types"]),
                    p["cardinality"],
                    p["temporal_kind"],
                )
            )
        version = config["ontology_version"]
        if not isinstance(version, str) or not version.strip():
            raise ValueError
        return version, tuple(predicates)
    except (ValueError, KeyError, TypeError):
        raise MemoryConfigurationError(
            "predicates",
            "Provide a version and closed predicate definitions with valid domain, range, cardinality and temporal_kind.",
        ) from None


def startup_error(error: Exception, phase: str) -> dict[str, Any]:
    if isinstance(error, MemoryConfigurationError):
        code, field, hint = "invalid_memory_configuration", error.field, error.hint
    elif isinstance(error, FileNotFoundError):
        code, field, hint = (
            "input_not_found",
            phase,
            "Check that the configuration/request file and its parent directories exist.",
        )
    elif isinstance(error, PermissionError):
        code, field, hint = (
            "permission_denied",
            phase,
            "Grant the current user access to the input file and memory root.",
        )
    elif str(error) == "memory root must be private":
        code, field, hint = (
            "unsafe_memory_root",
            "root",
            "Use an empty owned directory, or explicitly chmod 700 the dedicated existing memory root.",
        )
    elif str(error) == "unsafe memory root":
        code, field, hint = (
            "unsafe_memory_root",
            "root",
            "Choose a real dedicated directory, not a symbolic link.",
        )
    elif str(error) == "backend_unavailable":
        code, field, hint = (
            "backend_unavailable",
            "backend",
            "This runtime supports sqlite; configure an installed backend explicitly.",
        )
    elif str(error) == "memory_backend_unavailable":
        code, field, hint = (
            "backend_unavailable",
            "root",
            "The memory database could not be opened or read. Check access and restore a verified consistent backup if corrupt.",
        )
    elif str(error) in {"evidence_unavailable", "privacy_authority_unavailable"}:
        code, field, hint = (
            str(error),
            "root",
            "Check the Evidence database and retain the current privacy-authority directory; do not replace it with an older backup.",
        )
    else:
        code, field, hint = (
            "invalid_request" if phase == "request" else "memory_startup_failed",
            phase,
            "Check documented JSON fields, types, UTC times and private directory permissions.",
        )
    return {"status": "failed", "error": {"code": code, "field": field, "hint": hint}}
