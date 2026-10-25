"""Standalone reranker HTTP admission, validation and model identity."""

from __future__ import annotations

import json
import threading
from typing import Any

import pytest
from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from capicove.api.reranker import create_app
from capicove.retrieval.reranking import RerankResult, rank_scores


class Ranker:
    model_id = "BAAI/bge-reranker-v2-m3"
    model_revision = "sha256:" + "a" * 64

    def __init__(self) -> None:
        self.calls = 0
        self.closed = False
        self.failure: Exception | None = None
        self.entered = threading.Event()
        self.release: threading.Event | None = None

    def rerank(
        self, query: str, documents: tuple[str, ...], *, top_n: int | None = None
    ) -> RerankResult:
        self.calls += 1
        self.entered.set()
        if self.release:
            assert self.release.wait(5)
        if self.failure:
            raise self.failure
        scores = tuple(1.0 if "panda" in doc else -1.0 for doc in documents)
        return RerankResult(
            self.model_id, self.model_revision, rank_scores(scores, top_n=top_n or len(scores)), 12
        )

    def close(self) -> None:
        self.closed = True


def test_http_reranks_documents_and_preserves_indices() -> None:
    ranker = Ranker()
    with TestClient(create_app(ranker)) as client:
        response = client.post(
            "/v1/rerank",
            json={
                "model": "bge-reranker-v2-m3",
                "query": "what is panda?",
                "documents": ["hi", "panda bear", "panda bear"],
                "top_n": 2,
                "return_documents": True,
            },
        )
        assert response.status_code == 200
        body = response.json()
        assert body["model"] == ranker.model_id and body["model_revision"] == ranker.model_revision
        assert [r["index"] for r in body["results"]] == [1, 2]
        assert body["results"][0]["document"] == {"text": "panda bear"}
        assert body["usage"] == {"total_tokens": 12}
        assert body["score_type"] == "raw_logit"
        assert client.get("/health").json()["status"] == "ok"
        assert client.get("/v1/models").json()["data"][0]["id"] == ranker.model_id
    assert ranker.closed


@pytest.mark.parametrize(
    "change",
    [
        {"query": " "},
        {"documents": []},
        {"documents": [1]},
        {"documents": ["a"] * 129},
        {"top_n": 0},
        {"top_n": True},
        {"top_n": 3},
        {"return_documents": 1},
        {"extra": 1},
        {"query": "\ud800"},
    ],
)
def test_invalid_requests_do_not_run_inference(change: dict[str, Any]) -> None:
    ranker = Ranker()
    with TestClient(create_app(ranker)) as client:
        payload = {"model": ranker.model_id, "query": "q", "documents": ["a", "b"], **change}
        assert client.post("/v1/rerank", content=json.dumps(payload)).status_code == 400
    assert ranker.calls == 0


def test_invalid_json_model_and_body_limits() -> None:
    ranker = Ranker()
    with TestClient(create_app(ranker)) as client:
        for value in [b"{", b"[]", b'{"query":"a","query":"b"}', b'{"query":NaN}']:
            assert client.post("/v1/rerank", content=value).status_code == 400
        assert client.post("/v1/rerank", content=b"x" * (2 * 1024 * 1024 + 1)).status_code == 413
        assert (
            client.post(
                "/v1/rerank", json={"model": "missing", "query": "q", "documents": ["d"]}
            ).status_code
            == 404
        )
    assert ranker.calls == 0


def test_authentication_is_checked_before_inference() -> None:
    ranker = Ranker()
    # Synthetic test credential for an in-process listener.
    key = "test-reranker-key"  # pragma: allowlist secret
    with TestClient(create_app(ranker, api_key=key)) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/rerank", content=b"{").status_code == 401
        assert (
            client.get("/v1/models", headers={"Authorization": "Bearer " + key}).status_code == 200
        )
    assert ranker.calls == 0


@pytest.mark.parametrize(
    "failure,status",
    [(ValueError("rerank_input_too_long"), 400), (RuntimeError("private/path"), 500)],
)
def test_errors_are_safe_and_release_admission(failure: Exception, status: int) -> None:
    ranker = Ranker()
    with TestClient(create_app(ranker)) as client:
        ranker.failure = failure
        request = {"model": ranker.model_id, "query": "q", "documents": ["d"]}
        response = client.post("/v1/rerank", json=request)
        assert response.status_code == status and "private" not in response.text
        ranker.failure = None
        assert client.post("/v1/rerank", json=request).status_code == 200


def test_overload_returns_429_while_health_remains_available() -> None:
    ranker = Ranker()
    ranker.release = threading.Event()
    request = {"model": ranker.model_id, "query": "q", "documents": ["d"]}
    with TestClient(create_app(ranker)) as client:
        responses: list[int] = []
        thread = threading.Thread(
            target=lambda: responses.append(client.post("/v1/rerank", json=request).status_code)
        )
        thread.start()
        try:
            assert ranker.entered.wait(5)
            assert client.get("/health").status_code == 200
            assert client.post("/v1/rerank", json=request).status_code == 429
        finally:
            ranker.release.set()
            thread.join(5)
        assert responses == [200]


def test_cancelled_http_wait_keeps_the_slot_until_native_work_finishes() -> None:
    import asyncio

    import httpx

    async def exercise() -> None:
        ranker = Ranker()
        ranker.release = threading.Event()
        application = create_app(ranker)
        request = {"model": ranker.model_id, "query": "q", "documents": ["d"]}
        async with application.router.lifespan_context(application):
            async with httpx.AsyncClient(
                transport=httpx.ASGITransport(app=application), base_url="http://local-test"
            ) as client:
                pending = asyncio.create_task(client.post("/v1/rerank", json=request))
                try:
                    assert await asyncio.to_thread(ranker.entered.wait, 3)
                    pending.cancel()
                    await asyncio.sleep(0)
                    assert (await client.get("/health")).status_code == 200
                    assert (await client.post("/v1/rerank", json=request)).status_code == 429
                    assert ranker.calls == 1
                finally:
                    ranker.release.set()
                with pytest.raises(asyncio.CancelledError):
                    await pending
                for _ in range(100):
                    response = await client.post("/v1/rerank", json=request)
                    if response.status_code == 200:
                        break
                    assert response.status_code == 429
                    await asyncio.sleep(0.01)
                assert response.status_code == 200
        assert ranker.closed

    asyncio.run(exercise())


def test_failed_thread_dispatch_releases_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    from capicove.api import reranker

    ranker = Ranker()
    with TestClient(create_app(ranker)) as client:
        request = {"model": ranker.model_id, "query": "q", "documents": ["d"]}

        async def fail(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("cannot start new thread")

        with monkeypatch.context() as context:
            context.setattr(reranker, "run_in_threadpool", fail)
            assert client.post("/v1/rerank", json=request).status_code == 500
        assert client.post("/v1/rerank", json=request).status_code == 200
        assert ranker.calls == 1


def test_late_worker_after_failed_dispatch_cannot_run_or_release_new_slot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from capicove.api import reranker

    ranker = Ranker()
    delayed: list[tuple[Any, tuple[Any, ...]]] = []

    async def fail(function: Any, *args: Any, **kwargs: Any) -> Any:
        delayed.append((function, args))
        raise RuntimeError("dispatch failed after scheduling")

    with TestClient(create_app(ranker)) as client:
        request = {"model": ranker.model_id, "query": "q", "documents": ["d"]}
        with monkeypatch.context() as context:
            context.setattr(reranker, "run_in_threadpool", fail)
            assert client.post("/v1/rerank", json=request).status_code == 500
        ranker.release = threading.Event()
        responses: list[int] = []
        worker = threading.Thread(
            target=lambda: responses.append(client.post("/v1/rerank", json=request).status_code)
        )
        worker.start()
        try:
            assert ranker.entered.wait(3)
            function, args = delayed[0]
            try:
                function(*args)
            except RuntimeError:
                pass
            assert ranker.calls == 1
            assert client.post("/v1/rerank", json=request).status_code == 429
        finally:
            ranker.release.set()
            worker.join(5)
        assert responses == [200]


def test_cancel_before_thread_start_releases_admission(monkeypatch: pytest.MonkeyPatch) -> None:
    import asyncio

    from starlette.requests import Request

    from capicove.api import reranker

    async def exercise() -> None:
        ranker = Ranker()
        application = create_app(ranker)
        endpoint = next(
            route.endpoint
            for route in application.routes
            if isinstance(route, APIRoute) and route.path == "/v1/rerank"
        )
        queued = asyncio.Event()

        async def wait(function: Any, *args: Any, **kwargs: Any) -> Any:
            queued.set()
            await asyncio.Event().wait()

        def request() -> Request:
            async def receive() -> dict[str, Any]:
                return {
                    "type": "http.request",
                    "body": json.dumps(
                        {"model": ranker.model_id, "query": "q", "documents": ["d"]}
                    ).encode(),
                    "more_body": False,
                }

            return Request(
                {"type": "http", "method": "POST", "path": "/v1/rerank", "headers": []}, receive
            )

        with monkeypatch.context() as context:
            context.setattr(reranker, "run_in_threadpool", wait)
            pending = asyncio.create_task(endpoint(request()))
            await asyncio.wait_for(queued.wait(), 3)
            pending.cancel()
            with pytest.raises(asyncio.CancelledError):
                await pending
        assert (await endpoint(request())).status_code == 200
        assert ranker.calls == 1

    asyncio.run(exercise())
