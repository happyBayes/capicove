"""Public configuration catalog shared by VLM command and HTTP composition."""

from capicove.model.configuration import ModelConfiguration
from capicove.model.http_profile import HttpModelProfile


def registered_vlms(configuration: ModelConfiguration) -> dict[str, tuple[str, HttpModelProfile]]:
    """Select bound HTTP profiles; each profile name is its stable service alias."""
    result: dict[str, tuple[str, HttpModelProfile]] = {}
    for binding, name in configuration.bindings.items():
        profile = configuration.profiles[name]
        if isinstance(profile, HttpModelProfile):
            result.setdefault(name, (binding, profile))
    if not result:
        raise ValueError("no bound VLM profiles configured")
    return result


def model_info(profile: HttpModelProfile) -> dict[str, object]:
    """Describe configured identity without claiming to attest the upstream GPU."""
    return {
        "id": profile.name,
        "object": "model",
        "created": 0,
        "owned_by": "capicove",
        "upstream_model": profile.model,
        "modalities": ["text", *profile.modalities],
        "declared_model_revision": profile.model_revision,
        "profile_identity": profile.identity,
        "tools": True,
        "streaming": False,
        "token_counting": profile.token_counting,
        "max_output_tokens": profile.max_output_tokens,
    }
