"""Builtin factories delegate to verified public composition entry points."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from capicove.model import ModelCapability, ModelConfiguration, ModelFactory, ModelProfile
from capicove.model.providers.local import validate_bge, validate_paddle, validate_siglip2
from tests.unit.model.test_factory import Encoder


def runtime(
    root: Path, provider: str, capability: ModelCapability, **options: object
) -> ModelFactory:
    profile = ModelProfile("local", provider, capability, root, "a" * 64, options=options)  # type: ignore[arg-type]
    return ModelFactory(ModelConfiguration({"local": profile}, {"selected": "local"}))


def test_bge_uses_trusted_assets_and_explicit_inference_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove.retrieval.bge as bge

    seen: list[object] = []

    def load_profile(root: Path, **kwargs: object) -> str:
        seen.append((root, kwargs))
        return "verified"

    def create_encoder(profile: object, **kwargs: object) -> Encoder:
        seen.append((profile, kwargs))
        return Encoder()

    monkeypatch.setattr(bge, "load_bge_m3_profile", load_profile)
    monkeypatch.setattr(bge, "BgeTextBatchEncoder", create_encoder)
    with runtime(
        tmp_path, "bge-m3", ModelCapability.TEXT_EMBEDDING, max_tokens=128, batch_size=2
    ) as factory:
        with factory.text_encoder("selected") as encoder:
            assert encoder.space.dimension == 2
    assert seen == [
        (tmp_path, {"expected_manifest_sha256": "a" * 64, "max_tokens": 128}),
        ("verified", {"device": "cpu", "batch_size": 2}),
    ]


def test_siglip2_uses_public_offline_provider_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import capicove.retrieval.siglip2 as siglip

    seen: list[object] = []
    fake = SimpleNamespace(
        model_id="siglip2",
        model_revision="1",
        device="cpu",
        rank=lambda *a: (),
        embed_images=lambda *a: (),
        embed_texts=lambda *a: (),
    )

    def create_visual(root: Path, **kwargs: object) -> SimpleNamespace:
        seen.append((root, kwargs))
        return fake

    monkeypatch.setattr(siglip, "create_siglip2_provider", create_visual)
    with runtime(tmp_path, "siglip2", ModelCapability.IMAGE_TEXT_EMBEDDING) as factory:
        with factory.image_text_encoder("selected") as encoder:
            assert encoder.device == "cpu"
    assert seen == [(tmp_path, {"expected_manifest_sha256": "a" * 64, "device": "cpu"})]


@pytest.mark.parametrize(
    "capability,provider",
    [
        (ModelCapability.IMAGE_OCR, "paddleocr"),
        (ModelCapability.DOCUMENT_PARSER, "paddle-document"),
    ],
)
def test_paddle_verifies_complete_format_inventory_before_composition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capability: ModelCapability, provider: str
) -> None:
    from capicove.contracts import ProcessingFingerprint
    from capicove.processing.image.ocr_configuration import ImageOcrConfiguration

    calls: list[object] = []
    installer = SimpleNamespace(
        model_plan=lambda *args: SimpleNamespace(manifest_sha256="a" * 64),
        verify_models=lambda *args: calls.append(args),
    )
    monkeypatch.setattr("capicove.model.installers.load_model_installer", lambda name: installer)
    fake = SimpleNamespace(
        provider_id="paddleocr",
        provider_version="fake",
        fingerprint=ProcessingFingerprint("a" * 64),
        parser_id="paddleocr",
        recognize=lambda *a: (),
        parse=lambda *a: (),
    )
    monkeypatch.setattr(ImageOcrConfiguration, "create_provider", lambda self: fake)
    monkeypatch.setattr(
        "capicove.plugins.load_document_parser_plugin",
        lambda name: SimpleNamespace(create=lambda config: fake),
    )
    with runtime(tmp_path, provider, capability) as factory:
        with factory.acquire("selected", capability) as result:
            assert result is fake
    assert calls == [("book-balanced", tmp_path)]
    installer.model_plan = lambda *args: SimpleNamespace(manifest_sha256="b" * 64)
    with pytest.raises(ValueError, match="manifest mismatch"):
        with runtime(tmp_path, provider, capability).acquire("selected", capability):
            pass


@pytest.mark.parametrize(
    "provider,capability,options,device",
    [
        ("bge-m3", ModelCapability.TEXT_EMBEDDING, {"max_tokens": True}, "cpu"),
        ("bge-m3", ModelCapability.TEXT_EMBEDDING, {"batch_size": 65}, "cpu"),
        ("siglip2", ModelCapability.IMAGE_TEXT_EMBEDDING, {"batch_size": 1}, "cpu"),
        ("siglip2", ModelCapability.IMAGE_TEXT_EMBEDDING, {}, "cuda:1"),
        ("paddleocr", ModelCapability.IMAGE_OCR, {}, "cuda"),
        ("paddleocr", ModelCapability.IMAGE_OCR, {"enable_mkldnn": 1}, "cpu"),
    ],
)
def test_builtin_options_fail_before_assets_are_loaded(
    tmp_path: Path,
    provider: str,
    capability: ModelCapability,
    options: dict[str, object],
    device: str,
) -> None:
    profile = ModelProfile("local", provider, capability, tmp_path, "a" * 64, device, options)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        {"bge-m3": validate_bge, "siglip2": validate_siglip2, "paddleocr": validate_paddle}[
            provider
        ](profile)
