"""Generate a least-privilege transport configuration for one knowledge workspace."""

from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from capicove.runtime.knowledge_config import KnowledgeConfiguration


def host_configuration(config: KnowledgeConfiguration, path: Path) -> dict[str, Any]:
    """Reuse the version-two host and require an exact generation destination."""
    hosts = []
    if config.generation:
        from capicove.graph_build.remote_generation import load_remote_generation_profile

        hosts = [urlsplit(load_remote_generation_profile(config.generation).endpoint).hostname]
    return {
        "schema_version": 2,
        "server_name": "capicove-knowledge",
        "workspace_id": config.workspace_id,
        "actor_id": "knowledge-owner",
        "environment_id": "local",
        "workspace_root": str(path.parent),
        "profile": "admin" if config.generation else "read",
        "permissions": ["evidence:read", "evidence:search"],
        "enabled_tools": [
            f"capicove.knowledge.{name}" for name in ("ask", "evidence", "search", "status")
        ],
        "trace_store": None,
        "capabilities": {"knowledge": {"schema_version": 1, "config": str(path)}},
        "network": {"enabled": bool(config.generation), "source_hosts": hosts},
    }
