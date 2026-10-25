"""Exercise all configured deterministic providers and real opt-in trace records."""

import asyncio
import json
import os
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from capicove import _windows_files
from capicove.mcp.server import create_bound_mcp_server
from capicove.runtime.mcp_config import load_mcp_host_config
from capicove.runtime.mcp_host import compose_mcp_host
from capicove.traces._adapters.sqlite import SqliteTraceStore
from tests.integration.mcp.test_catalog_server import call
from tests.unit.runtime.test_mcp_config import config_body


@pytest.mark.slow
@pytest.mark.parametrize("concurrency", [1, 4])
def test_every_supported_provider_executes_and_records_one_episode(
    tmp_path: Path, concurrency: int
) -> None:
    image = tmp_path / "sample.png"
    Image.new("RGB", (32, 24), "red").save(image)
    sources = [{"path": str(image)}]
    cases: dict[str, dict[str, object]] = {
        "capicove.image.filter.aspect-ratio": {"sources": sources},
        "capicove.image.filter.shape": {"sources": sources},
        "capicove.image.filter.size": {"sources": sources},
        "capicove.image.filter.subplot": {"sources": sources},
        "capicove.image.phash": {"records": [{"record_id": "sample", "images": sources}]},
        "capicove.image.lsh-search": {
            "query": {"hashes": ["0000000000000000"]},
            "candidates": [{"record_id": "sample", "hashes": ["0000000000000000"]}],
            "hash_version": "fixture-v1",
        },
    }
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir(mode=0o700)
    if os.name == "nt":
        kernel32, advapi32 = _windows_files._windows_apis()
        sid = _windows_files._windows_current_sid(kernel32, advapi32)
        for options in (["/grant:r", f"*{sid}:(OI)(CI)F"], ["/inheritance:r"]):
            result = subprocess.run(
                ["icacls", str(trace_directory), *options], capture_output=True, text=True
            )
            assert result.returncode == 0, result.stderr
        _windows_files.validate_windows_path(trace_directory, directory=True, private_acl=True)
    database = trace_directory / "trace.sqlite"
    body = config_body(tmp_path)
    body["enabled_tools"] = list(cases)
    body["trace_store"] = str(database)
    path = tmp_path / "host.json"
    path.write_text(json.dumps(body))

    async def exercise() -> None:
        with compose_mcp_host(load_mcp_host_config(path)) as bindings:
            assert len(bindings) == len(cases)
            server = create_bound_mcp_server(bindings, server_name="test")
            for item in bindings:
                responses = await asyncio.gather(
                    *(call(server, item.name, cases[item.spec.tool_id]) for _ in range(concurrency))
                )
                trace_ids = set()
                results = []
                for response in responses:
                    assert not response.root.isError, response
                    result = response.root.structuredContent
                    assert result["status"] == "succeeded", result
                    trace_ids.add(result["trace_id"])
                    results.append(result["result"])
                    records = SqliteTraceStore(database).get_episode(result["trace_id"])
                    assert [record.record_type.value for record in records] == [
                        "episode.started",
                        "tool.call",
                        "tool.result",
                        "episode.completed",
                    ]
                    assert {record.trace_id for record in records} == {result["trace_id"]}
                assert len(trace_ids) == concurrency
                assert all(result == results[0] for result in results)
                assert str(image) not in database.read_bytes().decode("latin-1")

    asyncio.run(exercise())
