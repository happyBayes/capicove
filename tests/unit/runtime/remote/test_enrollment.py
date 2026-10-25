"""Enrollment accepts only explicit, server-owned identity mappings."""

import importlib
import json
from typing import Any

import pytest

from capicove.runtime.remote_config import IdentityEnrollment, decode_enrollment


def decode(value: str) -> IdentityEnrollment:
    assert importlib.util.find_spec("capicove.runtime.remote_config") is not None
    return decode_enrollment(value)


def document() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "issuer": "https://issuer.test",
        "audience": "https://mcp.test/mcp",
        "applications": [{"client_id": "client", "tenant_id": "tenant", "application_id": "app"}],
    }


def test_enrollment_owns_tenant_and_application() -> None:
    config = decode(json.dumps(document()))
    principal = config.map_subject("client", "human", "alice")
    assert (principal.tenant_id, principal.application_id, principal.subject) == (
        "tenant",
        "app",
        "alice",
    )
    with pytest.raises(ValueError):
        config.map_subject("unenrolled", "human", "alice")


@pytest.mark.parametrize("field", ["token", "tenant_claim", "client_secret", "unknown"])
def test_unknown_enrollment_fields_rejected(field: str) -> None:
    value = document()
    value[field] = "untrusted"
    with pytest.raises(ValueError):
        decode(json.dumps(value))


def test_duplicate_json_fields_rejected() -> None:
    with pytest.raises(ValueError):
        decode('{"schema_version":1,"schema_version":1}')


def test_duplicate_clients_rejected() -> None:
    value = document()
    value["applications"] *= 2
    with pytest.raises(ValueError):
        decode(json.dumps(value))


@pytest.mark.parametrize(
    "issuer",
    [
        "http://issuer.test",
        # Synthetic URL for rejecting inline credentials.
        "https://user:secret@issuer.test",  # pragma: allowlist secret
        "https://issuer.test/#fragment",
    ],
)
def test_unsafe_issuer_rejected(issuer: str) -> None:
    value = document()
    value["issuer"] = issuer
    with pytest.raises(ValueError):
        decode(json.dumps(value))


def test_constructor_snapshots_and_validates_applications() -> None:
    module = importlib.import_module("capicove.runtime.remote_config")
    applications = [module.EnrolledApplication("client", "tenant", "app")]
    config = module.IdentityEnrollment("https://issuer.test", "https://mcp.test/mcp", applications)
    applications[0] = module.EnrolledApplication("client", "other", "app")
    assert config.map_subject("client", "human", "alice").tenant_id == "tenant"
    with pytest.raises(ValueError):
        module.IdentityEnrollment("http://issuer.test", "https://mcp.test/mcp", applications)
    with pytest.raises(ValueError):
        module.EnrolledApplication("", "tenant", "app")


def test_malformed_port_is_rejected() -> None:
    value = document()
    value["issuer"] = "https://issuer.test:invalid"
    with pytest.raises(ValueError):
        decode(json.dumps(value))
