"""Closed secure configuration for the Elasticsearch text retriever."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit


_CONFIGURATION_KEYS = frozenset(
    {
        "hosts",
        "api_key",
        "ca_certs",
        "verify_certs",
        "request_timeout",
        "index_prefix",
        "active_alias",
        "knn_candidate_multiplier",
    }
)
_INDEX_PREFIX = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,39})$")
_ACTIVE_ALIAS = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,79})$")


@dataclass(frozen=True, slots=True)
class ElasticsearchConfiguration:
    """Contain only reviewed Elasticsearch startup settings."""

    hosts: tuple[str, ...]
    api_key: str | None = field(default=None, repr=False)
    ca_certs: str | None = None
    verify_certs: bool = True
    request_timeout: float = 10.0
    index_prefix: str = "capicove-text"
    active_alias: str = "capicove-text-active"
    knn_candidate_multiplier: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "hosts", _normalize_hosts(self.hosts))
        for field_name in ("api_key", "ca_certs"):
            value = getattr(self, field_name)
            if value is not None:
                _require_bounded_string(field_name, value, maximum=4_096)
        if self.verify_certs is not True:
            raise ValueError("verify_certs must be true")
        if (
            isinstance(self.request_timeout, bool)
            or not isinstance(self.request_timeout, (int, float))
            or not 0 < self.request_timeout <= 120
        ):
            raise ValueError("request_timeout must be within (0, 120]")
        object.__setattr__(self, "request_timeout", float(self.request_timeout))
        if (
            not isinstance(self.index_prefix, str)
            or _INDEX_PREFIX.fullmatch(self.index_prefix) is None
        ):
            raise ValueError("index_prefix must be a safe lowercase index prefix")
        if (
            not isinstance(self.active_alias, str)
            or _ACTIVE_ALIAS.fullmatch(self.active_alias) is None
            or self.active_alias.startswith(f"{self.index_prefix}-g-")
        ):
            raise ValueError("active_alias must be a safe lowercase index alias")
        if (
            isinstance(self.knn_candidate_multiplier, bool)
            or not isinstance(self.knn_candidate_multiplier, int)
            or not 1 <= self.knn_candidate_multiplier <= 100
        ):
            raise ValueError("knn_candidate_multiplier must be an integer from 1 to 100")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> ElasticsearchConfiguration:
        """Validate and normalize a closed startup configuration mapping."""
        unknown = set(value) - _CONFIGURATION_KEYS
        if unknown:
            raise ValueError(f"unsupported Elasticsearch configuration key: {sorted(unknown)[0]}")
        if "hosts" not in value:
            raise ValueError("hosts must be provided")
        normalized = dict(value)
        normalized["hosts"] = _normalize_hosts(normalized["hosts"])
        return cls(**normalized)  # type: ignore[arg-type]


def _normalize_hosts(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("hosts must be an iterable of HTTPS URLs")
    hosts = tuple(value)
    if not 1 <= len(hosts) <= 8:
        raise ValueError("hosts must contain from 1 to 8 HTTPS URLs")
    if len(set(hosts)) != len(hosts):
        raise ValueError("hosts must not contain duplicates")
    for host in hosts:
        if not isinstance(host, str):
            raise TypeError("hosts must contain strings")
        try:
            parsed = urlsplit(host)
            port = parsed.port
        except ValueError as error:
            raise ValueError("hosts must contain valid HTTPS URLs") from error
        if (
            parsed.scheme != "https"
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or port is None
        ):
            raise ValueError("hosts must contain secure HTTPS node URLs with ports")
    return hosts


def _require_bounded_string(field_name: str, value: object, *, maximum: int) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string")
    if not value.strip() or len(value) > maximum:
        raise ValueError(f"{field_name} must be non-blank and bounded")


__all__ = ["ElasticsearchConfiguration"]
