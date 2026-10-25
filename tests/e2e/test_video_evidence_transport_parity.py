"""Evidence transport parity uses the same bounded canonical handler."""

import json
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from capicove.api.search import create_search_app
from capicove.cli.app import main
from capicove.contracts import InvocationContext
from capicove.mcp.evidence import register_evidence_mcp
from capicove.tools.evidence import EvidenceToolHandler
from tests.unit.mcp.test_search import Server
from tests.unit.tools.test_search import SearchOperation, context


class Reader:
    def read(self, **kwargs: object) -> dict[str, object]:
        return {"metadata": [{"unit_id": "k"}]}


def test_evidence_tool_cli_mcp_http_match(capsys: pytest.CaptureFixture[str]) -> None:
    handler = EvidenceToolHandler(Reader())

    def factory() -> InvocationContext:
        return replace(context(), permissions=frozenset({"evidence:read"}))

    arguments = {"evidence_id": "evidence:k", "sections": ["metadata"]}
    expected = handler(factory(), arguments).result
    server = Server()
    callback = register_evidence_mcp(server, handler, context_factory=factory)  # type: ignore[arg-type]
    assert callback(arguments)["result"] == expected
    app = create_search_app(
        SearchOperation(), evidence_handler=handler, evidence_context_factory=factory
    )
    with TestClient(app) as client:
        assert client.post("/v1/evidence", json=arguments).json()["result"] == expected
    assert (
        main(["get-evidence", "evidence:k", "--section", "metadata"], evidence_handler=handler) == 0
    )
    assert json.loads(capsys.readouterr().out)["result"] == expected


def test_http_evidence_requires_explicit_context_factory() -> None:
    with pytest.raises(ValueError, match="context"):
        create_search_app(SearchOperation(), evidence_handler=EvidenceToolHandler(Reader()))
