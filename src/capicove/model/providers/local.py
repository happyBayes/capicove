"""Compose existing public capability adapters without private cross-imports."""

from capicove.model.configuration import ModelProfile


def validate_bge(profile: ModelProfile) -> None:
    """Bound only the qualified dense embedding settings."""
    if set(profile.options) - {"batch_size", "max_tokens"}:
        raise ValueError("unsupported BGE model options")
    for key, default, maximum in (("batch_size", 1, 64), ("max_tokens", 512, 8192)):
        value = profile.options.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("invalid BGE model options")


def validate_siglip2(profile: ModelProfile) -> None:
    """Keep preprocessing fixed by the locked SigLIP2 adapter."""
    if profile.options:
        raise ValueError("unsupported SigLIP2 model options")
    if profile.device not in {"cpu", "cuda", "mps"}:
        raise ValueError("unsupported SigLIP2 device")


def validate_paddle(profile: ModelProfile) -> None:
    """Retain the qualified CPU three-model policy and bounded options."""
    if profile.device != "cpu":
        raise ValueError("Paddle factory currently requires CPU")
    if (
        set(profile.options) - {"enable_mkldnn"}
        or type(profile.options.get("enable_mkldnn", False)) is not bool
    ):
        raise ValueError("unsupported Paddle model options")


def create_bge(profile: ModelProfile) -> object:
    """Create the existing text encoder from a verified safetensors inventory."""
    from capicove.retrieval.bge import BgeTextBatchEncoder, load_bge_m3_profile

    embedding = load_bge_m3_profile(
        profile.root,
        expected_manifest_sha256=profile.manifest_sha256,
        max_tokens=int(profile.options.get("max_tokens", 512)),
    )
    return BgeTextBatchEncoder(
        embedding, device=profile.device, batch_size=int(profile.options.get("batch_size", 1))
    )


def create_siglip2(profile: ModelProfile) -> object:
    """Create a lazy image/text encoder using the public retrieval entry point."""
    from capicove.retrieval.siglip2 import create_siglip2_provider

    return create_siglip2_provider(
        profile.root, expected_manifest_sha256=profile.manifest_sha256, device=profile.device
    )


def validate_reranker(profile: ModelProfile) -> None:
    """Accept only the pinned BGE reranker inventory and bounded runtime options."""
    from capicove.model.reranker_assets import MANIFEST_SHA256

    if profile.manifest_sha256 != MANIFEST_SHA256:
        raise ValueError("reranker trusted manifest mismatch")
    if set(profile.options) - {"batch_size", "max_tokens"}:
        raise ValueError("unsupported reranker options")
    for key, default, maximum in (("batch_size", 8, 128), ("max_tokens", 512, 8192)):
        value = profile.options.get(key, default)
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("invalid reranker options")


def create_reranker(profile: ModelProfile) -> object:
    """Create the public offline reranker adapter without implicit downloads."""
    from capicove.retrieval.bge_reranker import BgeReranker

    return BgeReranker(
        profile.root,
        device=profile.device,
        batch_size=int(profile.options.get("batch_size", 8)),
        max_tokens=int(profile.options.get("max_tokens", 512)),
    )


def _paddle(profile: ModelProfile) -> object:
    from capicove.model.installers import load_model_installer

    installer = load_model_installer("paddleocr")
    plan = installer.model_plan("book-balanced", profile.root)
    if getattr(plan, "manifest_sha256", None) != profile.manifest_sha256:
        raise ValueError("Paddle trusted manifest mismatch")
    verify = getattr(installer, "verify_models", None)
    if not callable(verify):
        raise TypeError("Paddle installer requires local format verification")
    verify("book-balanced", profile.root)
    return installer


def create_paddle_ocr(profile: ModelProfile) -> object:
    """Reuse the complete locked Paddle inventory with its public image port."""
    _paddle(profile)
    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration

    return ImageOcrConfiguration(
        "paddleocr", profile.root, profile.device, bool(profile.options.get("enable_mkldnn", False))
    ).create_provider()


def create_paddle_parser(profile: ModelProfile) -> object:
    """Use the same assets through the plugin's public parser factory."""
    _paddle(profile)
    from capicove.plugins import load_document_parser_plugin

    plugin = load_document_parser_plugin("paddleocr")
    return plugin.create(
        {
            "model_root": str(profile.root),
            "device": profile.device,
            "enable_mkldnn": profile.options.get("enable_mkldnn", False),
        }
    )
