"""Closed secure configuration for the OpenSearch text retriever."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from urllib.parse import urlsplit


_KEYS = frozenset(
    {
        "hosts",
        "username",
        "password",
        "ca_certs",
        "verify_certs",
        "request_timeout",
        "max_retries",
        "index_prefix",
        "active_alias",
        "knn_candidate_multiplier",
    }
)
_SAFE_NAME = re.compile(r"^[a-z0-9](?:[a-z0-9-]{0,79})$")


@dataclass(frozen=True, slots=True)
class OpenSearchConfiguration:
    """Contain only reviewed OpenSearch startup settings."""

    hosts: tuple[str, ...]
    username: str | None = None
    password: str | None = field(default=None, repr=False)
    ca_certs: str | None = None
    verify_certs: bool = True
    request_timeout: float = 10.0
    max_retries: int = 2
    index_prefix: str = "capicove-text"
    active_alias: str = "capicove-text-active"
    knn_candidate_multiplier: int = 10

    def __post_init__(self) -> None:
        object.__setattr__(self, "hosts", _normalize_hosts(self.hosts, self.verify_certs))
        if not isinstance(self.verify_certs, bool):
            raise TypeError("verify_certs must be a boolean")
        if (self.username is None) != (self.password is None):
            raise ValueError("username and password must be provided together")
        for name in ("username", "password", "ca_certs"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not value.strip() or len(value) > 4_096
            ):
                raise ValueError(f"{name} must be a bounded non-blank string")
        if (
            isinstance(self.request_timeout, bool)
            or not isinstance(self.request_timeout, (int, float))
            or not 0 < self.request_timeout <= 120
        ):
            raise ValueError("request_timeout must be within (0, 120]")
        object.__setattr__(self, "request_timeout", float(self.request_timeout))
        for name, maximum in (("max_retries", 10), ("knn_candidate_multiplier", 100)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= maximum:
                raise ValueError(f"{name} must be an integer from 0 to {maximum}")
        for name in ("index_prefix", "active_alias"):
            value = getattr(self, name)
            if not isinstance(value, str) or _SAFE_NAME.fullmatch(value) is None:
                raise ValueError(f"{name} must be a safe lowercase OpenSearch name")
        if self.active_alias.startswith(f"{self.index_prefix}-g-"):
            raise ValueError("active_alias must not overlap physical generation names")

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> OpenSearchConfiguration:
        """Validate one closed startup mapping."""
        unknown = set(value) - _KEYS
        if unknown:
            raise ValueError(f"unsupported OpenSearch configuration key: {sorted(unknown)[0]}")
        if "hosts" not in value:
            raise ValueError("hosts must be provided")
        return cls(**dict(value))  # type: ignore[arg-type]


def _normalize_hosts(value: object, verify_certs: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Iterable):
        raise TypeError("hosts must be an iterable of URLs")
    hosts = tuple(value)
    if not 1 <= len(hosts) <= 8 or len(set(hosts)) != len(hosts):
        raise ValueError("hosts must contain from 1 to 8 unique URLs")
    for host in hosts:
        if not isinstance(host, str):
            raise TypeError("hosts must contain strings")
        try:
            parsed = urlsplit(host)
            port = parsed.port
        except ValueError as error:
            raise ValueError("hosts must contain valid URLs") from error
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or port is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("hosts must contain node URLs with explicit ports")
        if parsed.scheme == "http" and not _is_loopback(parsed.hostname):
            raise ValueError("plain HTTP hosts must use loopback addresses")
        if parsed.scheme == "http" and verify_certs is not False:
            raise ValueError("loopback HTTP requires verify_certs=false")
    return hosts


def _is_loopback(hostname: str) -> bool:
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


__all__ = ["OpenSearchConfiguration"]
