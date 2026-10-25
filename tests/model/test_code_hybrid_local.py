"""Opt-in S3 real-model smoke replay with the pinned EmbeddingGemma 2 code encoder.

``CAPICOVE_EMBEDDINGGEMMA2_CODE_MODEL_CONFIG`` selects the ``text_embedding`` model
configuration written by ``scripts/embeddinggemma_cpu.py receipt
--code-model-config-output``. Labels below were frozen before the first run. A pass
establishes real inference for these inputs only, not representative quality.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import SearchRequest, SearchStatus, TargetType, TextQuery
from capicove.runtime import code_retrieval as runtime
from capicove.runtime.mcp_config import WorkspaceHostConfiguration, parse_mcp_host_config


pytestmark = pytest.mark.model_small
GIT = shutil.which("git")
CORPUS = {
    "src/sort_stable.py": (
        "def stable_merge_sort(items):\n"
        '    """Return a new list sorted ascending; equal items keep their original order."""\n'
        "    if len(items) <= 1:\n        return list(items)\n"
        "    middle = len(items) // 2\n"
        "    left, right = stable_merge_sort(items[:middle]), stable_merge_sort(items[middle:])\n"
        "    merged = []\n"
        "    while left and right:\n"
        "        merged.append(left.pop(0) if left[0] <= right[0] else right.pop(0))\n"
        "    return merged + left + right\n"
    ),
    "src/sort_fast.py": (
        "def quick_partition(items, pivot):\n"
        '    """Split items around a pivot in place; the order of equal items is not kept."""\n'
        "    low = [x for x in items if x < pivot]\n"
        "    high = [x for x in items if x >= pivot]\n"
        "    items[:] = low + high\n"
        "    return len(low)\n"
    ),
    "src/cache.py": (
        "class LruCache:\n"
        '    """Keep the most recently used entries in memory and evict the oldest."""\n'
        "    def __init__(self, capacity):\n        self.capacity = capacity\n        self.entries = {}\n"
        "    def get(self, key):\n        value = self.entries.pop(key)\n        self.entries[key] = value\n        return value\n"
    ),
    "docs/notes_en.md": "# Deployment notes\n\nThe service reads its port from the PORT environment variable.\n",
    "docs/notes_zh.md": "# 部署说明\n\n服务启动时从环境变量读取监听端口。\n",
}
QUERIES = (
    (
        "en-stable",
        "sort a list so that equal elements keep their original order",
        "src/sort_stable.py",
        1,
    ),
    ("en-cache", "evict the least recently used entry when memory is full", "src/cache.py", 1),
    ("zh-cache", "把最近使用的数据保存在内存里并淘汰最旧的", "src/cache.py", 3),
    ("zh-port", "服务从哪里读取端口号", "docs/notes_zh.md", 3),
    ("id-exact", "quick_partition", "src/sort_fast.py", 1),
)


@pytest.fixture(scope="module")
def model_config() -> Path:
    selected = os.environ.get("CAPICOVE_EMBEDDINGGEMMA2_CODE_MODEL_CONFIG")
    if not selected:
        pytest.skip("Set CAPICOVE_EMBEDDINGGEMMA2_CODE_MODEL_CONFIG to replay the real model")
    return Path(selected)


def _repository(root: Path) -> Path:
    repo = root / "repo"
    for relative, text in CORPUS.items():
        path = repo / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    assert GIT
    environment = {"PATH": "/usr/bin:/bin", "HOME": str(root), "GIT_CONFIG_NOSYSTEM": "1"}
    subprocess.run([GIT, "-C", str(repo), "init", "-q"], check=True, env=environment)
    subprocess.run([GIT, "-C", str(repo), "add", "-A"], check=True, env=environment)
    return repo


def test_real_code_hybrid_bilingual_smoke(model_config: Path, tmp_path: Path) -> None:
    root = (tmp_path / "workspace").resolve()
    root.mkdir()
    repo = _repository(root)
    body = {
        "schema_version": 2,
        "server_name": "capicove-local",
        "workspace_id": "workspace",
        "actor_id": "owner",
        "environment_id": "local",
        "workspace_root": str(root),
        "profile": "write",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": ["capicove.get_evidence", "capicove.search"],
        "trace_store": None,
        "capabilities": {
            "retrieval": {
                "schema_version": 2,
                "source_roots": {"repo": str(repo)},
                "index_root": str(root / "index"),
                "corpus_ref": "corpus:smoke",
                "retrieval_profile_ref": "local-code-hybrid-v1",
                "max_source_bytes": 1024 * 1024,
                "model_config_path": str(model_config),
                "model_config_sha256": hashlib.sha256(model_config.read_bytes()).hexdigest(),
                "model_binding": "code",
                "max_units": 10_000,
                "max_vector_bytes": 16 * 1024 * 1024,
                "code_policy_ref": "code-hybrid-v1",
            }
        },
        "network": {"enabled": False, "source_hosts": []},
    }
    host = parse_mcp_host_config(json.dumps(body).encode())
    assert isinstance(host, WorkspaceHostConfiguration)
    assert GIT
    runtime.import_repository(host, "repo", git=GIT)
    built = runtime.build_index(host)
    assert built["vectors"] == built["units"]
    context = InvocationContext(
        request_id="r",
        trace_id="t",
        workspace_id="workspace",
        actor_id="owner",
        environment_id="local",
        deadline=datetime.now(UTC) + timedelta(minutes=5),
        permissions=frozenset({"evidence:read"}),
    )
    service = runtime.open_code_retrieval(host)
    observed: dict[str, Any] = {}
    try:
        for identifier, query, expected, within in QUERIES:
            result = service.search(
                SearchRequest(
                    TextQuery(query),
                    frozenset({TargetType.TEXT_CHUNK}),
                    top_k=5,
                    retrieval_profile_ref="local-default",
                )
            )
            assert result.status is SearchStatus.SUCCEEDED and not result.degraded
            paths = [
                service.read_evidence(context, hit.evidence_id, ("metadata",))["metadata"][
                    "relative_path"
                ]
                for hit in result.hits
            ]
            observed[identifier] = paths
            assert expected in paths[:within], (identifier, paths)
    finally:
        service.close()
    print(json.dumps(observed, ensure_ascii=False))
