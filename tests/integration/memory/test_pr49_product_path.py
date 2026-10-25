"""Exercise the same files and CLI operations a user actually runs."""

import json
from pathlib import Path

import pytest

from capicove.cli import main


def invoke(
    root: Path,
    config: dict[str, object],
    request: dict[str, object],
    operation: str,
    capsys: pytest.CaptureFixture[str],
) -> tuple[int, dict[str, object]]:
    cfg, req = root / "config.json", root / "request.json"
    cfg.write_text(json.dumps(config))
    req.write_text(json.dumps(request))
    status = main(["memory", "--config", str(cfg), operation, "--request", str(req)])
    return status, json.loads(capsys.readouterr().out)


def test_minimal_config_and_ordinary_empty_root_work(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "memory"
    root.mkdir(mode=0o755)
    status, result = invoke(
        tmp_path,
        {"root": str(root), "workspace": "w", "scopes": ["s"]},
        {
            "scope": "s",
            "content": "Alice reported moving to Paris.",
            "event_time": "2026-01-01T00:00:00Z",
            "idempotency_key": "a",
        },
        "remember",
        capsys,
    )
    assert status == 0, result
    assert root.stat().st_mode & 0o077 == 0


def test_config_errors_are_actionable_and_do_not_echo_values(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    status, result = invoke(
        tmp_path, {"workspace": "SECRET_VALUE", "scopes": ["s"]}, {"scope": "s"}, "recall", capsys
    )
    assert status == 1
    rendered = json.dumps(result)
    assert "root" in rendered and "required" in rendered
    assert "SECRET_VALUE" not in rendered


def test_configured_extraction_worker_and_cited_recall(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config: dict[str, object] = {
        "root": str(tmp_path / "memory"),
        "workspace": "w",
        "scopes": ["s"],
        "extractor": "structured-json-v1",
    }
    content = json.dumps(
        {
            "facts": [
                {
                    "subject_name": "Alice",
                    "subject_type": "person",
                    "predicate": "city",
                    "object": {"kind": "string", "value": "Paris"},
                    "valid_from": "2026-01-01T00:00:00Z",
                }
            ],
            "text": "Alice reported moving to Paris.",
        }
    )
    status, result = invoke(
        tmp_path,
        config,
        {
            "scope": "s",
            "content": content,
            "episode_type": "json",
            "extraction": True,
            "event_time": "2026-01-01T00:00:00Z",
            "idempotency_key": "a",
        },
        "remember",
        capsys,
    )
    assert status == 0, result
    status, result = invoke(tmp_path, config, {"scope": "s", "limit": 10}, "work", capsys)
    assert status == 0, result
    assert '"completed"' in json.dumps(result)
    status, result = invoke(
        tmp_path,
        config,
        {"scope": "s", "text": "Paris", "types": ["semantic"], "max_tokens": 10000},
        "recall",
        capsys,
    )
    assert status == 0, result
    assert "Alice reported moving to Paris." in json.dumps(result)
    assert '"citations"' in json.dumps(result)


@pytest.mark.parametrize(
    ("change", "expected"),
    [
        ({"event_time": "2026-01-01T00:00:00+08:00"}, "UTC"),
        ({"predicate": "unknown-private-predicate"}, "ontology"),
    ],
)
def test_cli_request_errors_distinguish_time_and_ontology(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], change: dict[str, str], expected: str
) -> None:
    config: dict[str, object] = {
        "root": str(tmp_path / "memory"),
        "workspace": "w",
        "scopes": ["s"],
    }
    request: dict[str, object] = {
        "scope": "s",
        "subject_name": "Private Person",
        "subject_type": "person",
        "predicate": "city",
        "object": {"kind": "string", "value": "Private City"},
        "event_time": "2026-01-01T00:00:00Z",
        "evidence_text": "private source text",
        "idempotency_key": "a",
        **change,
    }
    status, result = invoke(tmp_path, config, request, "assert-fact", capsys)
    assert status == 1
    message = json.dumps(result)
    assert expected in message
    assert not any(
        value in message
        for value in (
            "Private Person",
            "Private City",
            "private source text",
            "unknown-private-predicate",
        )
    )


def test_nonempty_shared_root_is_not_silently_chmodded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "shared"
    root.mkdir(mode=0o755)
    (root / "unrelated.txt").write_text("user file")
    status, result = invoke(
        tmp_path,
        {"root": str(root), "workspace": "w", "scopes": ["s"]},
        {"scope": "s"},
        "doctor",
        capsys,
    )
    assert status == 1
    assert "unsafe_memory_root" in json.dumps(result) and "chmod 700" in json.dumps(result)
    assert root.stat().st_mode & 0o077


def test_corrupt_backend_returns_a_stable_cli_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "memory"
    root.mkdir(mode=0o700)
    store = root / "memory.sqlite3"
    store.write_text("this is not a sqlite database")
    store.chmod(0o600)
    status, result = invoke(
        tmp_path,
        {"root": str(root), "workspace": "w", "scopes": ["s"]},
        {"scope": "s"},
        "doctor",
        capsys,
    )
    assert status == 1
    assert "backend_unavailable" in json.dumps(result)
    assert "this is not" not in json.dumps(result)


def test_host_factory_is_explicitly_trusted_and_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    from types import ModuleType

    from capicove.runtime.memory_config import MemoryConfigurationError
    from capicove.runtime.memory_providers import MemoryProviders, configured_providers

    module = ModuleType("authored_memory_provider")
    calls: list[str] = []

    def factory() -> MemoryProviders:
        calls.append("loaded")
        return MemoryProviders()

    module.factory = factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, module.__name__, module)
    config: dict[str, object] = {"provider_factory": "authored_memory_provider:factory"}
    with pytest.raises(MemoryConfigurationError, match="invalid_memory_configuration"):
        configured_providers(config)
    assert calls == []
    config["allow_provider_code"] = True
    assert configured_providers(config) == MemoryProviders()
    assert calls == ["loaded"]
    config["provider_factory"] = "authored_memory_provider:missing"
    with pytest.raises(MemoryConfigurationError):
        configured_providers(config)


def test_mcp_worker_shares_admission_and_readable_context(tmp_path: Path) -> None:
    from collections.abc import Callable, Mapping

    from capicove.knowledge._adapters.structured_json import StructuredJsonExtractor
    from capicove.mcp.memory_transport import register_memory_mcp
    from capicove.runtime.memory_config import DEFAULT_PREDICATES
    from capicove.runtime.memory_runtime import build_memory_runtime
    from tests.integration.memory.test_transports import context

    callbacks: dict[str, Callable[[Mapping[str, object]], Mapping[str, object]]] = {}

    class Server:
        def tool(
            self, *, name: str, description: str, input_schema: Mapping[str, object]
        ) -> Callable[
            [Callable[[Mapping[str, object]], Mapping[str, object]]],
            Callable[[Mapping[str, object]], Mapping[str, object]],
        ]:
            def register(
                callback: Callable[[Mapping[str, object]], Mapping[str, object]],
            ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
                callbacks[name] = callback
                return callback

            return register

    handlers = build_memory_runtime(
        tmp_path / "mcp",
        workspace="w",
        scopes=frozenset({"s"}),
        ontology_version="personal-v1",
        predicates=DEFAULT_PREDICATES,
        extractor=StructuredJsonExtractor(),
    )
    register_memory_mcp(Server(), handlers, context_factory=context)
    content = json.dumps(
        {
            "facts": [
                {
                    "subject_name": "Alice",
                    "subject_type": "person",
                    "predicate": "city",
                    "object": {"kind": "string", "value": "Paris"},
                }
            ],
            "text": "Alice reported moving to Paris.",
        }
    )
    assert (
        callbacks["capicove_memory_remember_episode"](
            {
                "scope": "s",
                "content": content,
                "episode_type": "json",
                "extraction": True,
                "event_time": "2020-01-01T00:00:00Z",
                "idempotency_key": "mcp",
            }
        )["status"]
        == "succeeded"
    )
    worked = callbacks["capicove_memory_work_extractions"]({"scope": "s"})
    assert "completed" in json.dumps(worked)
    recalled = callbacks["capicove_memory_recall"](
        {"scope": "s", "text": "Alice city", "types": ["semantic"], "max_tokens": 10000}
    )
    assert "Alice reported moving to Paris." in json.dumps(recalled)
