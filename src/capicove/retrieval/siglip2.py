"""Public offline SigLIP2 provider composition for model factories."""

from pathlib import Path

from capicove.model import load_packaged_manifest, verify_model_installation
from capicove.retrieval._adapters.siglip2 import HuggingFaceSiglip2Provider


def create_siglip2_provider(
    root: Path, *, expected_manifest_sha256: str, device: str = "cpu"
) -> HuggingFaceSiglip2Provider:
    """Compose a lazy provider from a pinned, operator-trusted installation."""
    manifest = load_packaged_manifest()
    spec = next(model for model in manifest.models if model.key == "siglip2-base-patch16-224")
    installation = verify_model_installation(
        root,
        spec,
        manifest_sha256=manifest.source_sha256,
        expected_model_manifest_sha256=expected_manifest_sha256,
    )
    return HuggingFaceSiglip2Provider(installation, device=device)


__all__ = ["create_siglip2_provider"]
