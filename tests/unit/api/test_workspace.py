"""RFC-0020 loopback HTTP application: authentication, routing, bounds, and admission."""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator, Mapping
from typing import Any

import httpx
import pytest

from capicove.api.workspace import (
    MAX_BODY_BYTES,
    WorkerTimeout,
    WorkerUnavailable,
    create_workspace_app,
)
from capicove.mcp.bindings import InvalidToolArguments


TOKEN = "a" * 64
AUTH = {"Authorization": f"Bearer {TOKEN}"}
CATALOG = {
    "schema_version": 2,
    "server_name": "capicove-local",
    "capabilities": ["retrieval"],
    "tools": [
        {"name": "capicove_search", "tool_id": "capicove.search", "tool_version": 3},
    ],
}


class FakeInvoker:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.release = threading.Event()
        self.release.set()
        self.started = threading.Event()
        self.error: Exception | None = None

    def invoke(
        self, tool_id: str, arguments: Mapping[str, object], *, timeout: float
    ) -> dict[str, object]:
        self.calls.append((tool_id, dict(arguments)))
        self.started.set()
        self.release.wait(10)
        if self.error is not None:
            raise self.error
        return {"tool_id": tool_id, "status": "succeeded", "result": {"echo": dict(arguments)}}


def client(invoker: FakeInvoker, catalog: Mapping[str, object] = CATALOG) -> httpx.AsyncClient:
    app = create_workspace_app(catalog=catalog, token=TOKEN, invoker=invoker)
    return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")


def run(coroutine: object) -> object:
    return asyncio.run(coroutine)  # type: ignore[arg-type]


def test_health_needs_no_token_and_reveals_only_status() -> None:
    async def scenario() -> httpx.Response:
        async with client(FakeInvoker()) as http:
            return await http.get("/v1/health")

    response = run(scenario())
    assert isinstance(response, httpx.Response)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": TOKEN}]
)
def test_every_other_route_requires_the_bearer_token(headers: dict[str, str]) -> None:
    invoker = FakeInvoker()

    async def scenario() -> list[int]:
        async with client(invoker) as http:
            return [
                (await http.get("/v1/tools", headers=headers)).status_code,
                (
                    await http.post("/v1/tools/capicove.search", headers=headers, json={})
                ).status_code,
                (await http.post("/v1/search", headers=headers, json={})).status_code,
            ]

    assert run(scenario()) == [401, 401, 401]
    assert invoker.calls == []


def test_catalog_matches_the_composed_catalog() -> None:
    async def scenario() -> httpx.Response:
        async with client(FakeInvoker()) as http:
            return await http.get("/v1/tools", headers=AUTH)

    response = run(scenario())
    assert isinstance(response, httpx.Response)
    assert response.json() == CATALOG
    assert "access-control-allow-origin" not in response.headers


def test_selected_tool_and_alias_reach_the_invoker() -> None:
    invoker = FakeInvoker()

    async def scenario() -> list[Any]:
        async with client(invoker) as http:
            direct = await http.post("/v1/tools/capicove.search", headers=AUTH, json={"query": "a"})
            alias = await http.post("/v1/search", headers=AUTH, json={"query": "b"})
            return [direct.json(), alias.json()]

    direct, alias = asyncio.run(scenario())
    assert direct["result"] == {"echo": {"query": "a"}}
    assert alias["result"] == {"echo": {"query": "b"}}
    assert invoker.calls == [
        ("capicove.search", {"query": "a"}),
        ("capicove.search", {"query": "b"}),
    ]


def test_unselected_tools_and_aliases_are_not_found() -> None:
    invoker = FakeInvoker()

    async def scenario() -> list[int]:
        async with client(invoker) as http:
            return [
                (
                    await http.post("/v1/tools/capicove.get_evidence", headers=AUTH, json={})
                ).status_code,
                (await http.post("/v1/evidence", headers=AUTH, json={})).status_code,
                (await http.post("/v1/tools/unknown", headers=AUTH, json={})).status_code,
            ]

    assert run(scenario()) == [404, 404, 404]
    assert invoker.calls == []


@pytest.mark.parametrize(
    "body",
    [
        b"{",
        b"[]",
        b"\xff",
        pytest.param(
            b'{"query":' + b"[" * 2000 + b"0" + b"]" * 2000 + b"}", id="excessive-nesting"
        ),
    ],
)
def test_bodies_that_are_not_json_objects_are_rejected(body: bytes) -> None:
    invoker = FakeInvoker()

    async def scenario() -> httpx.Response:
        async with client(invoker) as http:
            return await http.post("/v1/tools/capicove.search", headers=AUTH, content=body)

    response = run(scenario())
    assert isinstance(response, httpx.Response)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_request"
    assert invoker.calls == []


def test_schema_invalid_arguments_map_to_400_without_echoing_input() -> None:
    invoker = FakeInvoker()
    invoker.error = InvalidToolArguments("Invalid Tool arguments")

    async def scenario() -> httpx.Response:
        async with client(invoker) as http:
            return await http.post(
                "/v1/tools/capicove.search", headers=AUTH, json={"secret-field": "s3cr3t"}
            )

    response = run(scenario())
    assert isinstance(response, httpx.Response)
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "invalid_arguments"
    assert "s3cr3t" not in response.text


def test_oversized_bodies_are_rejected_before_invocation() -> None:
    invoker = FakeInvoker()
    oversized = b'{"query": "' + b"x" * MAX_BODY_BYTES + b'"}'

    async def chunks() -> AsyncIterator[bytes]:
        for offset in range(0, len(oversized), 65536):
            yield oversized[offset : offset + 65536]

    async def scenario() -> list[int]:
        async with client(invoker) as http:
            declared = await http.post("/v1/tools/capicove.search", headers=AUTH, content=oversized)
            streamed = await http.post("/v1/tools/capicove.search", headers=AUTH, content=chunks())
            return [declared.status_code, streamed.status_code]

    assert run(scenario()) == [413, 413]
    assert invoker.calls == []


def test_one_invocation_is_admitted_and_health_stays_responsive() -> None:
    invoker = FakeInvoker()
    invoker.release.clear()

    async def scenario() -> tuple[int, int, int]:
        async with client(invoker) as http:
            first = asyncio.create_task(
                http.post("/v1/tools/capicove.search", headers=AUTH, json={"query": "slow"})
            )
            while not invoker.started.is_set():
                await asyncio.sleep(0.01)
            busy = await http.post("/v1/search", headers=AUTH, json={"query": "second"})
            health = await http.get("/v1/health")
            invoker.release.set()
            return (await first).status_code, busy.status_code, health.status_code

    assert run(scenario()) == (200, 429, 200)
    assert [call[1] for call in invoker.calls] == [{"query": "slow"}]


@pytest.mark.parametrize(
    ("error", "status", "code"),
    [
        (WorkerTimeout(), 504, "deadline_exceeded"),
        (WorkerUnavailable(), 503, "unavailable"),
    ],
)
def test_worker_failures_map_to_stable_errors(error: Exception, status: int, code: str) -> None:
    invoker = FakeInvoker()
    invoker.error = error

    async def scenario() -> httpx.Response:
        async with client(invoker) as http:
            return await http.post("/v1/search", headers=AUTH, json={"query": "a"})

    response = run(scenario())
    assert isinstance(response, httpx.Response)
    assert response.status_code == status
    assert response.json() == {
        "error": {"code": code, "message": response.json()["error"]["message"]}
    }


def test_closing_server_admits_no_new_invocations() -> None:
    invoker = FakeInvoker()
    app = create_workspace_app(catalog=CATALOG, token=TOKEN, invoker=invoker)
    app.state.closing.set()

    async def scenario() -> list[int]:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1") as http:
            return [
                (await http.post("/v1/search", headers=AUTH, json={})).status_code,
                (await http.get("/v1/health")).status_code,
            ]

    assert run(scenario()) == [503, 200]
    assert invoker.calls == []
    assert json.dumps(CATALOG)
