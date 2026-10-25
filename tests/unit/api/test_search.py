"""Tests for optional search HTTP parity."""

from fastapi.testclient import TestClient

from capicove.api.search import create_search_app
from tests.unit.tools.test_search import SearchOperation


def test_http_search_returns_the_same_canonical_tool_payload() -> None:
    client = TestClient(create_search_app(SearchOperation()))

    response = client.post("/v1/search", json={"query": "matching", "top_k": 5})

    assert response.status_code == 200
    assert response.json()["result"]["query_id"] == "query-1"
    assert response.json()["result"]["hits"][0]["id"] == "unit-1"
