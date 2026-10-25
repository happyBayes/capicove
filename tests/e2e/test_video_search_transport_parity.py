"""Compare real search orchestration across transports using fixture providers.

This checks transport semantics, not model quality or index qualification.
"""

import json
from concurrent.futures import ThreadPoolExecutor
from threading import BoundedSemaphore

import pytest
from fastapi.testclient import TestClient

from capicove.api.search import create_search_app
from capicove.cli.app import main
from capicove.mcp.search import register_search_mcp
from capicove.retrieval.video_profiles import StaticVideoRetrievalProfileRegistry
from capicove.retrieval.video_search import VideoSearchService
from capicove.tools.search import SearchToolHandler
from tests.unit.mcp.test_search import Server
from tests.unit.retrieval.test_video_search_text import Catalog, Encoder, Generations, Visual
from tests.unit.tools.test_search import context


@pytest.mark.parametrize("image_query", [False, True])
def test_video_hits_match_tool_cli_mcp_http(
    image_query: bool, capsys: pytest.CaptureFixture[str]
) -> None:
    with ThreadPoolExecutor(max_workers=4) as executor:
        service = VideoSearchService(
            workspace_ref="w",
            generation_sets=Generations(),
            profiles=StaticVideoRetrievalProfileRegistry(),
            catalog_resolver=lambda pin: Catalog(),
            visual_retriever=Visual(),
            query_encoder=Encoder(),
            executor=executor,
            concurrency=BoundedSemaphore(4),
        )
        handler = SearchToolHandler(service)
        query = (
            {"type": "image", "source": {"type": "local_path", "path": "/fixture.png"}}
            if image_query
            else {"type": "text", "text": "slide"}
        )
        arguments = {"query": query, "targets": ["keyframe", "video_segment"]}
        direct = handler(context(), arguments)
        assert direct.result is not None
        server = Server()
        callback = register_search_mcp(server, handler)  # type: ignore[arg-type]
        mcp = callback(arguments)
        with TestClient(create_search_app(service)) as client:
            http = client.post("/v1/search", json=arguments).json()
        argv = ["search", "--image", "/fixture.png"] if image_query else ["search", "slide"]
        assert (
            main(
                [*argv, "--target", "keyframe", "--target", "video_segment"], search_handler=handler
            )
            == 0
        )
        cli = json.loads(capsys.readouterr().out)
        for transport in (mcp, http, cli):
            assert transport["status"] == "succeeded"
            payload = transport["result"]
            assert isinstance(payload, dict)
            for field in ("hits", "degraded", "warning_codes", "status"):
                assert payload[field] == direct.result[field]
