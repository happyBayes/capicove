"""Unified model selection, isolation and HTTP routing."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from capicove.api.model_server import create_app
from capicove.model import ModelCapability, ModelConfiguration, ProviderDefinition, ProviderRegistry
from tests.unit.api.test_vlm_api import FakeVlm
from tests.unit.model.test_vlm import configuration


def test_registered_alias_dispatch_and_catalog() -> None:
    raw = configuration()
    config = ModelConfiguration.from_mapping(raw)
    provider = FakeVlm()
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, lambda profile: provider
        )
    )
    alias = next(iter(config.profiles))
    with TestClient(create_app(config, models=(alias,), registry=registry)) as client:
        assert client.get("/v1/models").json()["data"][0]["id"] == alias
        result = client.post(
            "/v1/chat/completions",
            json={"model": alias, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert result.status_code == 200, result.text
        assert result.json()["model"] == alias
        assert client.post("/v1/chat/completions", json={"model": "missing"}).status_code == 404
        assert client.post("/v1/embeddings", json={"model": alias, "input": "x"}).status_code == 404
        assert (
            client.post("/v1/chat/completions", content='{"model":"x","model":"y"}').status_code
            == 400
        )


def test_unified_cli_parse() -> None:
    from capicove.cli.app import _build_parser

    args = _build_parser().parse_args(
        ["models", "serve", "--config", "/tmp/models.toml", "--model", "speech"]
    )
    assert args.model == ["speech"]


def test_unified_cli_preserves_ocr_limits(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from typing import Any

    from capicove.api.ocr import OcrHttpSettings
    from capicove.cli.app import _build_parser
    from capicove.cli.commands import model_server

    captured: dict[str, Any] = {}

    def application(*args: Any, **kwargs: Any) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        model_server,
        "load_configuration",
        lambda path: ModelConfiguration.from_mapping(configuration()),
    )
    monkeypatch.setattr("capicove.api.model_server.create_app", application)
    monkeypatch.setattr("uvicorn.run", lambda *args, **kwargs: None)
    args = _build_parser().parse_args(
        [
            "models",
            "serve",
            "--config",
            str(tmp_path / "models.toml"),
            "--all",
            "--max-body-bytes",
            "1000",
            "--max-pixels",
            "100",
            "--max-document-bytes",
            "2000",
            "--max-document-pages",
            "2",
            "--max-document-page-points",
            "500",
            "--document-temp-directory",
            str(tmp_path),
        ]
    )
    assert model_server.run(args) == 0
    assert captured.get("ocr_settings") == OcrHttpSettings(
        max_body_bytes=1000,
        max_pixels=100,
        max_document_bytes=2000,
        max_document_pages=2,
        max_document_page_points=500,
        document_temp_directory=tmp_path,
    )


@pytest.mark.parametrize(
    "capability,provider,route",
    [
        (ModelCapability.IMAGE_OCR, "paddleocr", "/v1/ocr?model=local"),
        (ModelCapability.DOCUMENT_PARSER, "paddle-document", "/v1/documents/parse?model=local"),
    ],
)
def test_unified_children_enforce_custom_body_limits(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capability: ModelCapability,
    provider: str,
    route: str,
) -> None:
    from types import SimpleNamespace

    from capicove.api.ocr import OcrHttpSettings
    from capicove.model import ModelProfile
    from tests.unit.api.test_document_api import Parser
    from tests.unit.api.test_ocr_api import Provider

    instance = Provider() if capability == ModelCapability.IMAGE_OCR else Parser()
    registry = ProviderRegistry()
    registry.register(ProviderDefinition(provider, "1", capability, lambda profile: instance))
    profile = ModelProfile("local", provider, capability, tmp_path, "a" * 64)
    plan = SimpleNamespace(profile_id="book-balanced", profile_version=7, manifest_sha256="a" * 64)
    monkeypatch.setattr(
        "capicove.model.installers.load_model_installer",
        lambda name: SimpleNamespace(model_plan=lambda *args: plan),
    )
    config = ModelConfiguration({"local": profile}, {"local": "local"})
    with TestClient(
        create_app(
            config,
            registry=registry,
            ocr_settings=OcrHttpSettings(
                max_body_bytes=8,
                max_document_bytes=8,
            ),
        )
    ) as client:
        response = client.post(
            route,
            content=b"x" * 9,
            headers={
                "Content-Type": "application/pdf"
                if capability == ModelCapability.DOCUMENT_PARSER
                else "image/png"
            },
        )
        assert response.status_code == 413


def test_two_models_are_isolated_and_busy_is_rejected() -> None:
    from concurrent.futures import ThreadPoolExecutor
    from typing import Any

    raw = configuration()
    profile = raw["profiles"]["vision"]
    raw["profiles"]["other"] = dict(profile, model="other-model")
    raw["bindings"]["other"] = "other"
    raw["max_instances"] = 2
    providers = {name: FakeVlm() for name in ("vision", "other")}
    providers["vision"].release.clear()
    providers["other"].model_id = "other-model"
    registry = ProviderRegistry()

    def create(profile: Any) -> FakeVlm:
        return providers[profile.name]

    registry.register(
        ProviderDefinition("openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, create)
    )
    with TestClient(create_app(ModelConfiguration.from_mapping(raw), registry=registry)) as client:
        with ThreadPoolExecutor() as pool:
            body = {"model": "vision", "messages": [{"role": "user", "content": "hi"}]}
            pending = pool.submit(client.post, "/v1/chat/completions", json=body)
            try:
                assert providers["vision"].entered.wait(3)
                assert client.get("/health").status_code == 200
                assert client.post("/v1/chat/completions", json=body).status_code == 429
                assert (
                    client.post("/v1/chat/completions", json=dict(body, model="other")).status_code
                    == 200
                )
            finally:
                providers["vision"].release.set()
            assert pending.result().status_code == 200
    assert all(p.closed for p in providers.values())


def test_startup_failure_closes_previously_created_provider() -> None:
    from typing import Any

    import pytest

    raw = configuration()
    raw["profiles"]["other"] = dict(raw["profiles"]["vision"], model="other")
    raw["bindings"]["other"] = "other"
    raw["max_instances"] = 2
    provider = FakeVlm()

    def create(profile: Any) -> FakeVlm:
        if profile.name == "other":
            raise ValueError("deliberate startup failure")
        return provider

    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition("openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, create)
    )
    with pytest.raises(ValueError, match="deliberate"):
        with TestClient(create_app(ModelConfiguration.from_mapping(raw), registry=registry)):
            pass
    assert provider.closed


def test_selection_rejects_unbound_and_excess_capacity() -> None:
    import pytest

    from capicove.model.serving import select_models

    raw = configuration()
    raw["profiles"]["other"] = dict(raw["profiles"]["vision"])
    with pytest.raises(ValueError, match="bound"):
        select_models(ModelConfiguration.from_mapping(raw), ("other",))
    raw["bindings"]["other"] = "other"
    with pytest.raises(ValueError, match="max_instances"):
        select_models(ModelConfiguration.from_mapping(raw), None)


def test_document_binding_uses_installer_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    from capicove.model import ModelProfile
    from capicove.model.serving import document_binding

    profile = ModelProfile(
        "pdf", "paddle-document", ModelCapability.DOCUMENT_PARSER, tmp_path, "a" * 64
    )
    plan = SimpleNamespace(profile_id="book-balanced", profile_version=7, manifest_sha256="a" * 64)
    monkeypatch.setattr(
        "capicove.model.installers.load_model_installer",
        lambda name: SimpleNamespace(model_plan=lambda *args: plan),
    )
    backend = SimpleNamespace(parser_id="test", parse=lambda request: None)
    binding = document_binding(profile, backend)
    assert binding.profile_version == 7
    plan.manifest_sha256 = "b" * 64
    with pytest.raises(ValueError, match="manifest"):
        document_binding(profile, backend)


def test_all_local_capabilities_share_listener_and_preserve_aliases(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace
    from typing import Any

    from capicove.model import ModelProfile
    from tests.unit.api.test_audio_api import Provider as Audio
    from tests.unit.api.test_bge_m3_api import Encoder
    from tests.unit.api.test_document_api import Parser, pdf
    from tests.unit.api.test_ocr_api import Provider as Ocr
    from tests.unit.api.test_ocr_api import picture
    from tests.unit.api.test_reranker_api import Ranker
    from tests.unit.api.test_siglip2_api import ApiProvider, _png

    entries = {
        "embedding": ("bge-m3", ModelCapability.TEXT_EMBEDDING, Encoder()),
        "reranking": ("bge-reranker-v2-m3", ModelCapability.TEXT_RERANKING, Ranker()),
        "vision": ("siglip2", ModelCapability.IMAGE_TEXT_EMBEDDING, ApiProvider()),
        "speech": ("whisper-cpp", ModelCapability.AUDIO_TRANSCRIPTION, Audio()),
        "ocr": ("paddleocr", ModelCapability.IMAGE_OCR, Ocr()),
        "pdf": ("paddle-document", ModelCapability.DOCUMENT_PARSER, Parser()),
    }
    registry = ProviderRegistry()
    profiles = {}
    for name, (identity, capability, instance) in entries.items():

        def make(profile: ModelProfile, value: Any = instance) -> Any:
            return value

        registry.register(ProviderDefinition(identity, "1", capability, make))
        profiles[name] = ModelProfile(name, identity, capability, tmp_path, "a" * 64)
    plan = SimpleNamespace(profile_id="book-balanced", profile_version=7, manifest_sha256="a" * 64)
    monkeypatch.setattr(
        "capicove.model.installers.load_model_installer",
        lambda name: SimpleNamespace(model_plan=lambda *args: plan),
    )
    configuration = ModelConfiguration(profiles, {name: name for name in profiles}, 6)
    key = "test-key"  # pragma: allowlist secret
    with TestClient(create_app(configuration, registry=registry, api_key=key)) as client:
        assert client.get("/v1/models").status_code == 401
        client.headers["Authorization"] = "Bearer test-key"
        response = client.post("/v1/embeddings", json={"model": "embedding", "input": "test"})
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "embedding"
        response = client.post(
            "/v1/rerank",
            json={"model": "reranking", "query": "panda", "documents": ["panda", "other"]},
        )
        assert response.status_code == 200, response.text
        assert response.json()["model"] == "reranking"
        assert (
            client.post(
                "/v1/embeddings/text", json={"model": "vision", "texts": ["test"]}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/embeddings/image?model=vision", files={"image": ("x.png", _png(), "image/png")}
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/audio/transcriptions",
                files={"file": ("x.wav", b"audio")},
                data={"model": "speech"},
            ).status_code
            == 200
        )
        assert (
            client.post(
                "/v1/ocr?model=ocr", content=picture(), headers={"Content-Type": "image/png"}
            ).status_code
            == 200
        )
        response = client.post(
            "/v1/documents/parse?model=pdf",
            content=pdf(),
            headers={"Content-Type": "application/pdf"},
        )
        assert response.status_code == 200, response.text
        parser = entries["pdf"][2]
        assert isinstance(parser, Parser)
        assert parser.requests[0].profile_version == 7
        assert (
            client.post(
                "/v1/documents/parse?model=pdf&end_page_index=0",
                content=pdf(),
                headers={"Content-Type": "application/pdf"},
            ).status_code
            == 400
        )
        schema = client.get("/openapi.json").json()
        assert len(schema["paths"]) >= 8
        assert (
            schema["paths"]["/v1/audio/transcriptions"]["post"]["requestBody"]["content"][
                "multipart/form-data"
            ]["schema"]["properties"]["file"]["format"]
            == "binary"
        )


@pytest.mark.parametrize("error_type", [ValueError, TimeoutError])
def test_child_failure_does_not_send_a_second_response(error_type: type[Exception]) -> None:
    import asyncio
    import json

    import httpx
    from fastapi import FastAPI
    from starlette.types import Message, Receive, Scope, Send

    from capicove.api.model_server import ModelDispatch

    child = FastAPI()

    @child.post("/v1/chat/completions")
    async def fail() -> None:
        raise error_type("private inference diagnostic")

    starts: list[int] = []
    dispatch = ModelDispatch(ModelConfiguration.from_mapping(configuration()), {"vision": child})

    async def observed(scope: Scope, receive: Receive, send: Send) -> None:
        async def record(message: Message) -> None:
            if message["type"] == "http.response.start":
                starts.append(message["status"])
            await send(message)

        await dispatch(scope, receive, record)

    async def exercise() -> None:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=observed, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            result = await client.post(
                "/v1/chat/completions", content=json.dumps({"model": "vision"})
            )
            assert result.status_code == 500
            assert "private" not in result.text
        assert starts == [500]

    asyncio.run(exercise())


def test_upstream_model_name_remains_supported() -> None:
    config = ModelConfiguration.from_mapping(configuration())
    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, lambda profile: FakeVlm()
        )
    )
    with TestClient(create_app(config, registry=registry)) as client:
        name = "example/vision-model"
        response = client.post(
            "/v1/chat/completions",
            json={"model": name, "messages": [{"role": "user", "content": "hello"}]},
        )
        assert response.status_code == 200
        assert response.json()["model"] == name
        assert client.get(f"/v1/models/{name}").status_code == 200


def test_cancelled_startup_disposes_completed_instance(tmp_path: Path) -> None:
    import asyncio
    from contextlib import AsyncExitStack
    from threading import Event

    from capicove.api.model_server import _application
    from capicove.model import ModelProfile

    entered, release = Event(), Event()
    disposed: list[object] = []
    instance = object()

    def construct(profile: ModelProfile) -> object:
        entered.set()
        assert release.wait(5)
        return instance

    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition(
            "probe", "1", ModelCapability.TEXT_EMBEDDING, construct, dispose=disposed.append
        )
    )
    profile = ModelProfile("local", "probe", ModelCapability.TEXT_EMBEDDING, tmp_path, "a" * 64)
    config = ModelConfiguration({"local": profile}, {"embedding": "local"})

    async def exercise() -> None:
        async with AsyncExitStack() as stack:
            from capicove.api.ocr import OcrHttpSettings

            task = asyncio.create_task(
                _application(config, "local", registry, stack, OcrHttpSettings())
            )
            try:
                assert await asyncio.to_thread(entered.wait, 5)
                task.cancel()
            finally:
                release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
        assert disposed == [instance]

    asyncio.run(exercise())


def test_ambiguous_upstream_name_is_rejected_and_registered_names_win() -> None:
    from capicove.api.model_server import _aliases

    raw = configuration("shared")
    raw["profiles"]["other"] = dict(raw["profiles"]["vision"])
    raw["bindings"]["other"] = "other"
    config = ModelConfiguration.from_mapping(raw)
    assert "shared" not in _aliases(config)
    raw["profiles"]["other"]["model"] = "vision"
    aliases = _aliases(ModelConfiguration.from_mapping(raw))
    assert aliases["vision"] == "vision"
    assert aliases["shared"] == "vision"
