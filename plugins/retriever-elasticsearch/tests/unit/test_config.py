"""Tests for closed Elasticsearch runtime configuration."""

from __future__ import annotations

import pytest
from capicove_retriever_elasticsearch.config import ElasticsearchConfiguration


def test_configuration_normalizes_reviewed_secure_values_without_exposing_secret() -> None:
    configuration = ElasticsearchConfiguration.from_mapping(
        {
            "hosts": ["https://search.example.com:9243"],
            "api_key": "secret-api-key",  # pragma: allowlist secret
            "ca_certs": "/etc/capicove/elasticsearch-ca.pem",
            "verify_certs": True,
            "request_timeout": 15,
            "index_prefix": "capicove-text",
            "active_alias": "capicove-text-active",
            "knn_candidate_multiplier": 12,
        }
    )

    assert configuration.hosts == ("https://search.example.com:9243",)
    assert configuration.api_key == "secret-api-key"  # pragma: allowlist secret
    assert configuration.ca_certs == "/etc/capicove/elasticsearch-ca.pem"
    assert configuration.request_timeout == 15.0
    assert configuration.index_prefix == "capicove-text"
    assert configuration.active_alias == "capicove-text-active"
    assert configuration.knn_candidate_multiplier == 12
    assert "secret-api-key" not in repr(configuration)  # pragma: allowlist secret


def test_configuration_supplies_bounded_defaults() -> None:
    configuration = ElasticsearchConfiguration.from_mapping({"hosts": ("https://localhost:9200",)})

    assert configuration.api_key is None
    assert configuration.ca_certs is None
    assert configuration.verify_certs is True
    assert configuration.request_timeout == 10.0
    assert configuration.index_prefix == "capicove-text"
    assert configuration.active_alias == "capicove-text-active"
    assert configuration.knn_candidate_multiplier == 10


def test_configuration_rejects_unknown_provider_knobs() -> None:
    with pytest.raises(ValueError, match="unsupported Elasticsearch configuration key"):
        ElasticsearchConfiguration.from_mapping(
            {
                "hosts": ("https://localhost:9200",),
                "raw_dsl": {"script": "unreviewed"},
            }
        )


@pytest.mark.parametrize(
    "hosts",
    [
        (),
        "https://localhost:9200",
        ("http://localhost:9200",),
        ("https://user:password@localhost:9200",),  # pragma: allowlist secret
        ("https://localhost:9200/unreviewed-path",),
        ("https://localhost:9200?pretty=true",),
        ("https://localhost:9200#fragment",),
        ("not-a-url",),
        tuple(f"https://node-{index}.example.com:9200" for index in range(9)),
        ("https://localhost:9200", "https://localhost:9200"),
    ],
)
def test_configuration_rejects_unbounded_or_insecure_hosts(hosts: object) -> None:
    with pytest.raises((TypeError, ValueError), match="hosts"):
        ElasticsearchConfiguration.from_mapping({"hosts": hosts})


@pytest.mark.parametrize("verify_certs", [False, 1, "true"])
def test_configuration_requires_certificate_verification(verify_certs: object) -> None:
    with pytest.raises(ValueError, match="verify_certs"):
        ElasticsearchConfiguration.from_mapping(
            {
                "hosts": ("https://localhost:9200",),
                "verify_certs": verify_certs,
            }
        )


@pytest.mark.parametrize("request_timeout", [True, 0, -1, 120.1, "10"])
def test_configuration_rejects_unbounded_request_timeouts(
    request_timeout: object,
) -> None:
    with pytest.raises(ValueError, match="request_timeout"):
        ElasticsearchConfiguration.from_mapping(
            {
                "hosts": ("https://localhost:9200",),
                "request_timeout": request_timeout,
            }
        )


@pytest.mark.parametrize("multiplier", [True, 0, 101, 1.5])
def test_configuration_rejects_unbounded_knn_candidate_multiplier(
    multiplier: object,
) -> None:
    with pytest.raises(ValueError, match="knn_candidate_multiplier"):
        ElasticsearchConfiguration.from_mapping(
            {
                "hosts": ("https://localhost:9200",),
                "knn_candidate_multiplier": multiplier,
            }
        )


@pytest.mark.parametrize(
    "index_prefix",
    ["", "Capicove", "capicove_text", "-capicove", "a" * 41],
)
def test_configuration_rejects_unsafe_index_prefixes(index_prefix: str) -> None:
    with pytest.raises(ValueError, match="index_prefix"):
        ElasticsearchConfiguration.from_mapping(
            {
                "hosts": ("https://localhost:9200",),
                "index_prefix": index_prefix,
            }
        )


@pytest.mark.parametrize(
    "active_alias",
    [
        "",
        "Capicove-Active",
        "capicove_active",
        "-active",
        "a" * 81,
        "capicove-text-g-deadbeef",
    ],
)
def test_configuration_rejects_unsafe_active_aliases(active_alias: str) -> None:
    with pytest.raises(ValueError, match="active_alias"):
        ElasticsearchConfiguration.from_mapping(
            {
                "hosts": ("https://localhost:9200",),
                "active_alias": active_alias,
            }
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [("api_key", " "), ("api_key", 1), ("ca_certs", ""), ("ca_certs", 1)],
)
def test_configuration_rejects_invalid_optional_strings(
    field: str,
    value: object,
) -> None:
    with pytest.raises((TypeError, ValueError), match=field):
        ElasticsearchConfiguration.from_mapping(
            {"hosts": ("https://localhost:9200",), field: value}
        )
