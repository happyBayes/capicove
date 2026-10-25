"""Memory must run in the base wheel without image, graph, model, or MCP extras."""

import json
import subprocess
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = pytest.mark.e2e


@pytest.mark.slow
def test_base_wheel_runs_memory_and_preserves_existing_cli(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    installed = wheel_cache.install(dependencies=False)
    environment = installed.environment
    python = Path(installed.python)
    script = """
from datetime import UTC, datetime, timedelta
from pathlib import Path
from capicove.contracts import InvocationContext, ToolStatus
from capicove.runtime.memory import build_memory_runtime
from capicove.mcp.memory import register_memory_mcp
import sys
root = Path(sys.argv[1])
handlers = build_memory_runtime(root, workspace="w", scopes=frozenset({"s"}), ontology_version="v1", predicates=())
context = InvocationContext("r", "t", "w", "owner", "local", datetime.now(UTC)+timedelta(seconds=10), frozenset({"memory:read", "memory:write"}))
write = handlers["remember_episode"](context, {"scope":"s", "content":"wheel memory", "event_time":datetime.now(UTC).isoformat(), "idempotency_key":"one"})
assert write.status is ToolStatus.SUCCEEDED, write.error
read = handlers["recall"](context, {"scope":"s", "text":"wheel"})
assert read.status is ToolStatus.SUCCEEDED, read.error
assert read.result["hits"]
"""
    result = subprocess.run(
        [str(python), "-c", script, str(tmp_path / "runtime")], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
    for arguments in (("doctor",), ("video", "filter", "--help"), ("memory", "--help")):
        result = subprocess.run(
            [str(environment / "bin" / "capicove"), *arguments], capture_output=True, text=True
        )
        assert result.returncode == 0, result.stderr
    config = tmp_path / "memory-config.json"
    config.write_text(
        json.dumps(
            {
                "root": str(tmp_path / "cli-runtime"),
                "workspace": "w",
                "scopes": ["s"],
                "extractor": "structured-json-v1",
            }
        )
    )
    request = tmp_path / "request.json"
    content = json.dumps(
        {
            "text": "Alice reported moving to Paris.",
            "facts": [
                {
                    "subject_name": "Alice",
                    "subject_type": "person",
                    "predicate": "city",
                    "object": {"kind": "string", "value": "Paris"},
                }
            ],
        }
    )
    operations = [
        (
            "remember",
            {
                "scope": "s",
                "content": content,
                "episode_type": "json",
                "extraction": True,
                "event_time": "2020-01-01T00:00:00Z",
                "idempotency_key": "wheel-extraction",
            },
        ),
        ("work", {"scope": "s"}),
        (
            "recall",
            {"scope": "s", "text": "Alice city", "types": ["semantic"], "max_tokens": 10000},
        ),
    ]
    outputs = []
    for operation, body in operations:
        request.write_text(json.dumps(body))
        completed = subprocess.run(
            [
                str(environment / "bin" / "capicove"),
                "memory",
                "--config",
                str(config),
                operation,
                "--request",
                str(request),
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.append(json.loads(completed.stdout))
    assert outputs[1]["result"]["outcomes"][0]["state"] == "completed"
    assert "Alice reported moving to Paris." in outputs[2]["result"]["hits"][0]["text"]
    assert outputs[2]["result"]["hits"][0]["citations"]
