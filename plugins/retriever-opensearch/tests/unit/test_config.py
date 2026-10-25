"""Tests for closed OpenSearch runtime configuration."""

import pytest
from capicove_retriever_opensearch.config import OpenSearchConfiguration


def test_configuration_accepts_explicit_local_http_and_hides_password() -> None:
    value = OpenSearchConfiguration.from_mapping(
        {
            "hosts": ["http://127.0.0.1:9200"],
            "username": "admin",
            "password": "VERY_123",  # pragma: allowlist secret
            "verify_certs": False,
            "index_prefix": "capicove-test",
            "active_alias": "capicove-test-active",
        }
    )

    assert value.hosts == ("http://127.0.0.1:9200",)
    assert value.password == "VERY_123"  # pragma: allowlist secret
    assert "VERY_123" not in repr(value)


def test_configuration_rejects_unknown_keys_and_insecure_remote_http() -> None:
    with pytest.raises(ValueError, match="unsupported OpenSearch configuration key"):
        OpenSearchConfiguration.from_mapping({"hosts": ["https://localhost:9200"], "raw_dsl": {}})
    with pytest.raises(ValueError, match="loopback"):
        OpenSearchConfiguration(hosts=("http://example.com:9200",), verify_certs=False)


@pytest.mark.parametrize("timeout", [0, 121, True])
def test_configuration_bounds_request_timeout(timeout: object) -> None:
    with pytest.raises(ValueError, match="request_timeout"):
        OpenSearchConfiguration(
            hosts=("https://localhost:9200",),
            request_timeout=timeout,  # type: ignore[arg-type]
        )
