"""Explicit endpoint profiles without local asset or GPU requirements."""

from __future__ import annotations

import hashlib
import ipaddress
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from urllib.parse import urlsplit

from capicove.model.configuration import ModelCapability, OptionValue


@dataclass(frozen=True, slots=True)
class HttpModelProfile:
    """Register any compatible VLM; revisions are operator declarations only."""

    name: str
    provider: str
    capability: ModelCapability
    endpoint: str
    model: str
    allow_network: bool = False
    api_key_env: str | None = None
    model_revision: str | None = None
    modalities: tuple[str, ...] = ("image",)
    timeout_seconds: float = 120.0
    max_images: int = 8
    max_videos: int = 1
    max_output_tokens: int = 4096
    token_counting: str | None = None
    chat_template_kwargs: Mapping[str, OptionValue] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.token_counting not in (None, "vllm"):
            raise ValueError("unsupported token counting backend")
        if any(
            not isinstance(v, str) or not re.fullmatch(r"[a-z][a-z0-9_.-]{0,63}", v)
            for v in (self.name, self.provider)
        ):
            raise ValueError("invalid profile or provider name")
        if (
            not isinstance(self.capability, ModelCapability)
            or self.capability != ModelCapability.MULTIMODAL_GENERATION
        ):
            raise ValueError("HTTP profiles currently require multimodal_generation")
        if self.allow_network is not True:
            raise ValueError("HTTP inference requires allow_network=true")
        if (
            not isinstance(self.endpoint, str)
            or len(self.endpoint) > 2048
            or any(ord(c) < 33 or ord(c) > 126 for c in self.endpoint)
        ):
            raise ValueError("invalid endpoint")
        parsed = urlsplit(self.endpoint)
        try:
            loopback = (
                parsed.hostname == "localhost"
                or ipaddress.ip_address(parsed.hostname or "").is_loopback
            )
        except ValueError:
            loopback = False
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path.rstrip("/") != "/v1"
            or (parsed.scheme == "http" and not loopback)
            or (parsed.port is not None and not 1 <= parsed.port <= 65535)
        ):
            raise ValueError("endpoint must be HTTPS or loopback HTTP ending in /v1")
        for identity_value in (self.model, self.model_revision):
            if identity_value is not None and (
                not isinstance(identity_value, str)
                or not identity_value.strip()
                or len(identity_value) > 256
                or any(ord(c) < 32 for c in identity_value)
            ):
                raise ValueError("invalid model identity")
        if not isinstance(self.model, str) or not self.model:
            raise ValueError("model name is required")
        if self.api_key_env is not None and (
            not isinstance(self.api_key_env, str)
            or not re.fullmatch(r"CAPICOVE_[A-Z0-9_]{1,100}", self.api_key_env)
        ):
            raise ValueError("API key must reference a CAPICOVE_ environment variable")
        if (
            not isinstance(self.modalities, tuple)
            or not self.modalities
            or any(not isinstance(modality, str) for modality in self.modalities)
            or len(set(self.modalities)) != len(self.modalities)
            or set(self.modalities) - {"image", "video"}
        ):
            raise ValueError("modalities must contain image and/or video")
        if (
            type(self.timeout_seconds) not in (int, float)
            or not math.isfinite(self.timeout_seconds)
            or not 0 < self.timeout_seconds <= 600
        ):
            raise ValueError("timeout_seconds must be positive and at most 600")
        for limit_value, maximum in (
            (self.max_images, 32),
            (self.max_videos, 4),
            (self.max_output_tokens, 32768),
        ):
            if type(limit_value) is not int or not 1 <= limit_value <= maximum:
                raise ValueError("invalid VLM limit")
        if (
            not isinstance(self.chat_template_kwargs, Mapping)
            or len(self.chat_template_kwargs) > 16
        ):
            raise ValueError("invalid template options")
        for key, value in self.chat_template_kwargs.items():
            if (
                not isinstance(key, str)
                or not re.fullmatch(r"[a-z][a-z0-9_]{0,63}", key)
                or type(value) not in (str, int, float, bool)
                or (isinstance(value, float) and not math.isfinite(value))
                or (isinstance(value, str) and len(value) > 1024)
            ):
                raise ValueError("template options must contain bounded finite scalars")
        object.__setattr__(
            self, "chat_template_kwargs", MappingProxyType(dict(self.chat_template_kwargs))
        )

    @property
    def identity(self) -> str:
        """Hash endpoint and effective policy, excluding aliases and key values."""
        body = {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
            if name not in {"name", "chat_template_kwargs"}
        }
        body["chat_template_kwargs"] = dict(self.chat_template_kwargs)
        return hashlib.sha256(
            json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
