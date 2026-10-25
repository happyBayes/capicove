"""Explicit local path policy for audio evidence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from capicove.processing.audio.ports import AudioPathPolicyError


class ExplicitAudioPathAuthorizer:
    """Authorize resolved audio paths under configured local read roots."""

    def __init__(self, policies: Mapping[str, tuple[Path, ...]]) -> None:
        normalized: dict[str, tuple[Path, ...]] = {}
        for policy_ref, roots in policies.items():
            if not policy_ref.strip():
                raise ValueError("policy reference must not be blank")
            if any(not root.is_absolute() for root in roots):
                raise ValueError("path policy roots must be absolute")
            normalized[policy_ref] = tuple(root.resolve() for root in roots)
        self._policies = normalized

    def authorize_read(self, policy_ref: str, path: Path) -> Path:
        """Authorize one resolved path without exposing it in failures."""
        try:
            roots = self._policies[policy_ref]
        except KeyError as error:
            raise AudioPathPolicyError("execution policy is not configured") from error
        resolved = path.resolve()
        if not any(resolved == root or resolved.is_relative_to(root) for root in roots):
            raise AudioPathPolicyError("path is outside configured read roots")
        return resolved
