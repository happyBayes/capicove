"""HTTP routing, authentication, bounded admission and failure recovery."""

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import pytest
from fastapi.testclient import TestClient

from capicove.api.vlm import create_app
from capicove.model import (
    HttpModelProfile,
    ModelCapability,
    ModelConfiguration,
    ModelFactory,
    ProviderDefinition,
    ProviderRegistry,
)
from capicove.model.providers.openai_vlm import VlmError
from capicove.model.vlm import VlmRequest, VlmResult, VlmUsage
from tests.unit.model.test_vlm import configuration


class FakeVlm:
    model_id = "example/vision-model"

    def __init__(self) -> None:
        self.closed = False
        self.entered = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.error: Exception | None = None

    def generate(self, request: VlmRequest) -> VlmResult:
        self.entered.set()
        assert self.release.wait(5)
        if self.error:
            raise self.error
        return VlmResult(
            self.model_id,
            "red square",
            "stop",
            VlmUsage(10, 2),
            "a" * 64,
            "b" * 64,
            request.media_sha256,
        )

    def close(self) -> None:
        self.closed = True


def factory(provider: FakeVlm, raw: dict[str, Any] | None = None) -> ModelFactory:
    registry = ProviderRegistry()

    def create(profile: HttpModelProfile) -> FakeVlm:
        return provider

    registry.register(
        ProviderDefinition("openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, create)
    )
    return ModelFactory(ModelConfiguration.from_mapping(raw or configuration()), registry=registry)


def body() -> dict[str, object]:
    return {"model": "vision", "messages": [{"role": "user", "content": "describe"}]}


def test_models_auth_generation_and_owned_cleanup() -> None:
    provider = FakeVlm()
    # Synthetic credential for the in-process test listener.
    key = "synthetic-key"  # pragma: allowlist secret
    with TestClient(create_app(factory(provider), api_key=key)) as client:
        assert client.get("/health").status_code == 200
        assert client.get("/v1/models").status_code == 401
        headers = {"Authorization": "Bearer synthetic-key"}
        assert client.get("/v1/models", headers=headers).json()["data"][0]["id"] == "vision"
        response = client.post("/v1/chat/completions", json=body(), headers=headers)
        assert response.status_code == 200
        assert response.json()["model"] == "vision"
        assert response.json()["choices"][0]["message"]["content"] == "red square"
    assert provider.closed


@pytest.mark.parametrize(
    "change,status",
    [
        ({"model": "missing"}, 404),
        ({"stream": True}, 400),
        ({"max_tokens": True}, 400),
        ({"tools": [{}]}, 400),
    ],
)
def test_invalid_requests_rejected_before_provider_creation(
    change: dict[str, object], status: int
) -> None:
    provider = FakeVlm()
    with TestClient(create_app(factory(provider))) as client:
        assert client.post("/v1/chat/completions", json={**body(), **change}).status_code == status
        assert not provider.entered.is_set()


def test_health_responsive_and_overload_rejected_during_generation() -> None:
    provider = FakeVlm()
    provider.release.clear()
    with TestClient(create_app(factory(provider))) as client, ThreadPoolExecutor() as executor:
        pending = executor.submit(client.post, "/v1/chat/completions", json=body())
        try:
            assert provider.entered.wait(2)
            assert client.get("/health").status_code == 200
            assert client.post("/v1/chat/completions", json=body()).status_code == 429
        finally:
            provider.release.set()
        assert pending.result(timeout=2).status_code == 200
        assert client.post("/v1/chat/completions", json=body()).status_code == 200


def test_failed_submission_does_not_leak_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    import capicove.api.vlm as api

    async def failed(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("worker submission failed")

    with TestClient(create_app(factory(FakeVlm()))) as client:
        with monkeypatch.context() as patch:
            patch.setattr(api, "run_in_threadpool", failed)
            assert client.post("/v1/chat/completions", json=body()).status_code == 500
        assert client.post("/v1/chat/completions", json=body()).status_code == 200


def test_transport_failure_is_sanitized_and_slot_recovers() -> None:
    provider = FakeVlm()
    with TestClient(create_app(factory(provider))) as client:
        provider.error = VlmError("upstream_timeout", 504)
        response = client.post("/v1/chat/completions", json=body())
        assert response.status_code == 504
        assert response.json()["error"]["code"] == "upstream_timeout"
        provider.error = None
        assert client.post("/v1/chat/completions", json=body()).status_code == 200


def test_late_worker_cannot_release_a_new_request_slot(monkeypatch: pytest.MonkeyPatch) -> None:
    import capicove.api.vlm as api

    queued: list[tuple[Any, tuple[Any, ...]]] = []

    async def failed(function: Any, *args: Any, **kwargs: Any) -> Any:
        queued.append((function, args))
        raise RuntimeError("failed after scheduling")

    provider = FakeVlm()
    with TestClient(create_app(factory(provider))) as client, ThreadPoolExecutor() as executor:
        with monkeypatch.context() as patch:
            patch.setattr(api, "run_in_threadpool", failed)
            assert client.post("/v1/chat/completions", json=body()).status_code == 500
        provider.release.clear()
        pending = executor.submit(client.post, "/v1/chat/completions", json=body())
        try:
            assert provider.entered.wait(2)
            function, args = queued[0]
            with pytest.raises(RuntimeError, match="cancelled_before_start"):
                function(*args)
            assert client.post("/v1/chat/completions", json=body()).status_code == 429
        finally:
            provider.release.set()
        assert pending.result(timeout=2).status_code == 200


@pytest.mark.parametrize("started", [False, True])
def test_cancelled_waiter_preserves_atomic_worker_ownership(
    monkeypatch: pytest.MonkeyPatch, started: bool
) -> None:
    import asyncio

    from fastapi.routing import APIRoute
    from starlette.requests import Request

    import capicove.api.vlm as api

    async def exercise() -> None:
        provider = FakeVlm()
        runtime = factory(provider)
        application = create_app(runtime)
        endpoint = next(
            route.endpoint
            for route in application.routes
            if isinstance(route, APIRoute) and route.path == "/v1/chat/completions"
        )
        scheduled = asyncio.Event()
        worker: threading.Thread | None = None

        async def dispatch(function: Any, *args: Any, **kwargs: Any) -> Any:
            nonlocal worker
            if started:
                provider.release.clear()
                worker = threading.Thread(target=function, args=args)
                worker.start()
                assert await asyncio.to_thread(provider.entered.wait, 2)
            scheduled.set()
            await asyncio.Event().wait()

        def request() -> Request:
            async def receive() -> dict[str, Any]:
                return {
                    "type": "http.request",
                    "body": json.dumps(body()).encode(),
                    "more_body": False,
                }

            return Request(
                {"type": "http", "method": "POST", "path": "/v1/chat/completions", "headers": []},
                receive,
            )

        with monkeypatch.context() as patch:
            patch.setattr(api, "run_in_threadpool", dispatch)
            pending = asyncio.create_task(endpoint(request()))
            await asyncio.wait_for(scheduled.wait(), 3)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        if started:
            assert (await endpoint(request())).status_code == 429
            provider.release.set()
            assert worker is not None
            await asyncio.to_thread(worker.join, 3)
        assert (await endpoint(request())).status_code == 200
        runtime.close()

    asyncio.run(exercise())


def test_multiple_models_route_by_profile_with_idle_factory_eviction() -> None:
    raw = configuration()
    raw["profiles"]["second"] = {**raw["profiles"]["vision"], "model": "other/vendor-vision"}
    raw["bindings"]["another.vision"] = "second"
    made: list[FakeVlm] = []

    def create(profile: HttpModelProfile) -> FakeVlm:
        provider = FakeVlm()
        provider.model_id = profile.model
        made.append(provider)
        return provider

    registry = ProviderRegistry()
    registry.register(
        ProviderDefinition("openai-vlm", "1", ModelCapability.MULTIMODAL_GENERATION, create)
    )
    with TestClient(
        create_app(ModelFactory(ModelConfiguration.from_mapping(raw), registry=registry))
    ) as client:
        assert len(client.get("/v1/models").json()["data"]) == 2
        for name in ("vision", "second", "other/vendor-vision", "vision"):
            response = client.post("/v1/chat/completions", json={**body(), "model": name})
            assert response.status_code == 200
            assert response.json()["model"] == name
        assert len(made) == 3
        assert made[0].closed and made[1].closed
    assert made[2].closed


def test_duplicate_served_names_require_distinct_profile_aliases() -> None:
    raw = configuration()
    raw["profiles"]["second"] = dict(raw["profiles"]["vision"])
    raw["bindings"]["another.vision"] = "second"
    with TestClient(create_app(factory(FakeVlm(), raw))) as client:
        response = client.post(
            "/v1/chat/completions", json={**body(), "model": "example/vision-model"}
        )
        assert response.status_code == 404
        assert (
            client.post("/v1/chat/completions", json={**body(), "model": "second"}).status_code
            == 200
        )


def test_slow_request_parsing_keeps_health_and_admission_responsive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import capicove.api.vlm as api
    from capicove.model.vlm import parse_chat_request

    entered = threading.Event()
    release = threading.Event()

    def parse(raw: dict[str, object]) -> VlmRequest:
        entered.set()
        assert release.wait(5)
        return parse_chat_request(raw)

    monkeypatch.setattr(api, "parse_chat_request", parse)
    with TestClient(create_app(factory(FakeVlm()))) as client, ThreadPoolExecutor() as executor:
        pending = executor.submit(client.post, "/v1/chat/completions", json=body())
        try:
            assert entered.wait(2)
            health = executor.submit(client.get, "/health")
            assert health.result(timeout=1).status_code == 200
            busy = executor.submit(client.post, "/v1/chat/completions", json=body())
            assert busy.result(timeout=1).status_code == 429
        finally:
            release.set()
        assert pending.result(timeout=2).status_code == 200


def test_count_routes_alias_and_preserves_authentication() -> None:
    from capicove.model.vlm import VlmTokenCount

    class CountingVlm(FakeVlm):
        def count_tokens(self, request: VlmRequest) -> VlmTokenCount:
            assert request.messages[0].role == "user"
            return VlmTokenCount(self.model_id, 12, 4096, "a" * 64, "b" * 64)

    provider = CountingVlm()
    # Synthetic credential used only by the in-process test client.
    key = "synthetic"  # pragma: allowlist secret
    with TestClient(create_app(factory(provider), api_key=key)) as client:
        assert client.post("/v1/count_tokens", json=body()).status_code == 401
        response = client.post(
            "/v1/count_tokens", json=body(), headers={"Authorization": "Bearer synthetic"}
        )
        assert response.status_code == 200
        assert response.json()["input_tokens"] == 12
        assert response.json()["model"] == "vision"
        assert not provider.entered.is_set()
