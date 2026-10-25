import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast

import pytest

import capicove.runtime.memory_runtime as memory_runtime
from capicove.contracts import InvocationContext, ToolError, ToolErrorCode, ToolStatus
from capicove.knowledge import SQLiteMemoryRepository
from capicove.runtime.memory import build_memory_runtime


def context(
    workspace: str = "w", permissions: frozenset[str] = frozenset({"memory:write", "memory:read"})
) -> InvocationContext:
    return InvocationContext(
        "r",
        "t",
        workspace,
        "owner",
        "local",
        datetime.now(UTC) + timedelta(seconds=30),
        permissions,
    )


def test_tools_fail_closed_on_scope_workspace_and_provider_fields(tmp_path: Path) -> None:
    runtime = build_memory_runtime(
        tmp_path / "runtime",
        workspace="w",
        scopes=frozenset({"s"}),
        ontology_version="v1",
        predicates=(),
    )
    handler = runtime["remember_episode"]
    args = {
        "scope": "s",
        "content": "hello",
        "event_time": datetime.now(UTC).isoformat(),
        "idempotency_key": "a",
    }
    assert (
        cast(ToolError, handler(context("other"), args).error).code
        is ToolErrorCode.PERMISSION_DENIED
    )
    assert (
        cast(ToolError, handler(context(), dict(args, scope="other")).error).code
        is ToolErrorCode.PERMISSION_DENIED
    )
    assert (
        cast(ToolError, handler(context(), dict(args, sql="DROP TABLE")).error).code
        is ToolErrorCode.INVALID_INPUT
    )
    assert handler(context(), args).status is ToolStatus.SUCCEEDED
    assert "admit_extracted_fact" not in runtime


def test_unavailable_backend_does_not_create_sqlite_fallback(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(RuntimeError, match="backend_unavailable"):
        build_memory_runtime(
            tmp_path / "runtime",
            workspace="w",
            scopes=frozenset({"s"}),
            ontology_version="v1",
            predicates=(),
            backend="neo4j",
        )
    assert not (tmp_path / "runtime").exists()


def test_memory_runtime_rejects_symlinked_parent_before_creating_storage(
    tmp_path: Path,
) -> None:
    foreign = tmp_path / "foreign"
    foreign.mkdir(mode=0o700)
    alias = tmp_path / "alias"
    try:
        alias.symlink_to(foreign, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are unavailable")

    with pytest.raises(ValueError, match="unsafe memory root"):
        build_memory_runtime(
            alias / "memory",
            workspace="w",
            scopes=frozenset({"s"}),
            ontology_version="v1",
            predicates=(),
        )

    assert not (foreign / "memory").exists()


def test_cli_memory_uses_canonical_tool_and_keeps_existing_commands(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from capicove.cli import main

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "root": str(tmp_path / "runtime"),
                "workspace": "w",
                "scopes": ["s"],
                "ontology_version": "v1",
                "predicates": [],
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "scope": "s",
                "content": "hello",
                "event_time": datetime.now(UTC).isoformat(),
                "idempotency_key": "remember",
            }
        )
    )
    assert main(["memory", "--config", str(config), "remember", "--request", str(request)]) == 0
    assert json.loads(capsys.readouterr().out)["tool_id"] == "capicove.memory.remember_episode"
    assert main(["doctor"]) == 0


def test_cli_memory_closes_repository_after_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from capicove.cli import main

    repositories: list[SQLiteMemoryRepository] = []

    def capture_repository(path: Path) -> SQLiteMemoryRepository:
        repository = SQLiteMemoryRepository(path)
        repositories.append(repository)
        return repository

    monkeypatch.setattr(memory_runtime, "SQLiteMemoryRepository", capture_repository)
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "root": str(tmp_path / "runtime"),
                "workspace": "w",
                "scopes": ["s"],
                "ontology_version": "v1",
                "predicates": [],
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text('{"scope":"s"}')

    assert main(["memory", "--config", str(config), "doctor", "--request", str(request)]) == 0

    assert len(repositories) == 1
    with pytest.raises(sqlite3.ProgrammingError, match="closed"):
        repositories[0]._db.execute("SELECT 1")


def test_opt_in_trace_keeps_memory_source_text_out_of_trace_store(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    import json

    from capicove.cli import main

    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "root": str(tmp_path / "runtime"),
                "workspace": "w",
                "scopes": ["s"],
                "ontology_version": "v1",
                "predicates": [],
            }
        )
    )
    request = tmp_path / "request.json"
    request.write_text(
        json.dumps(
            {
                "scope": "s",
                "content": "nevercopythissourceintotrace",
                "event_time": datetime.now(UTC).isoformat(),
                "idempotency_key": "r",
            }
        )
    )
    trace = tmp_path / "trace.db"
    assert (
        main(
            [
                "--trace-store",
                str(trace),
                "memory",
                "--config",
                str(config),
                "remember",
                "--request",
                str(request),
            ]
        )
        == 0
    )
    capsys.readouterr()
    request.write_text(json.dumps({"scope": "s", "text": "nevercopythissourceintotrace"}))
    assert (
        main(
            [
                "--trace-store",
                str(trace),
                "memory",
                "--config",
                str(config),
                "recall",
                "--request",
                str(request),
            ]
        )
        == 0
    )
    assert "nevercopythissourceintotrace" in capsys.readouterr().out
    assert b"nevercopythissourceintotrace" not in trace.read_bytes()


def test_authorizer_failure_is_closed_and_does_not_leak_details(tmp_path: Path) -> None:
    def denied(ctx: InvocationContext, scope: str) -> bool:
        raise RuntimeError("private credential backend detail")

    runtime = build_memory_runtime(
        tmp_path / "runtime",
        workspace="w",
        scopes=frozenset({"s"}),
        ontology_version="v1",
        predicates=(),
        authorize_scope=denied,
    )
    result = runtime["recall"](context(), {"scope": "s"})
    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert "private" not in result.error.message


def test_mcp_uses_host_context_and_exact_tool_schemas(tmp_path: Path) -> None:
    from collections.abc import Callable, Mapping

    from capicove.mcp.memory import register_memory_mcp
    from capicove.tools.memory import MEMORY_SPECS

    callbacks: dict[str, Callable[[Mapping[str, object]], Mapping[str, object]]] = {}
    schemas: dict[str, Mapping[str, object]] = {}

    class Server:
        def tool(
            self, *, name: str, description: str, input_schema: Mapping[str, object]
        ) -> Callable[
            [Callable[[Mapping[str, object]], Mapping[str, object]]],
            Callable[[Mapping[str, object]], Mapping[str, object]],
        ]:
            schemas[name] = input_schema

            def register(
                callback: Callable[[Mapping[str, object]], Mapping[str, object]],
            ) -> Callable[[Mapping[str, object]], Mapping[str, object]]:
                callbacks[name] = callback
                return callback

            return register

    runtime = build_memory_runtime(
        tmp_path / "runtime",
        workspace="w",
        scopes=frozenset({"s"}),
        ontology_version="v1",
        predicates=(),
    )
    register_memory_mcp(
        Server(), runtime, context_factory=lambda: context(permissions=frozenset({"memory:read"}))
    )
    assert len(callbacks) == 11
    assert "capicove_memory_work_extractions" in callbacks
    assert "capicove_memory_resolve_entity" in callbacks
    for name, spec in MEMORY_SPECS.items():
        assert schemas["capicove_memory_" + name] == spec.input_schema
    result = callbacks["capicove_memory_remember_episode"](
        {
            "scope": "s",
            "content": "not permitted",
            "event_time": datetime.now(UTC).isoformat(),
            "idempotency_key": "a",
        }
    )
    assert result["status"] == "failed"
    assert callbacks["capicove_memory_recall"]({"scope": "s"})["status"] == "succeeded"


def test_tools_assert_temporal_history_archive_and_scope_erasure(tmp_path: Path) -> None:
    from collections.abc import Mapping

    from capicove.contracts.knowledge import Predicate

    runtime = build_memory_runtime(
        tmp_path / "runtime",
        workspace="w",
        scopes=frozenset({"s"}),
        ontology_version="v1",
        predicates=(Predicate("city", ("person",), "literal", ("string",), "one", "interval"),),
    )
    ctx = context(permissions=frozenset({"memory:read", "memory:write", "memory:erase"}))
    receipts = []
    for key, name, city, start in (
        ("a", "Alice", "Paris", "2026-01-01"),
        ("b", "alice", "London", "2026-02-01"),
    ):
        result = runtime["assert_fact"](
            ctx,
            {
                "scope": "s",
                "subject_name": name,
                "subject_type": "person",
                "predicate": "city",
                "object": {"kind": "string", "value": city},
                "valid_from": start + "T00:00:00+00:00",
                "event_time": datetime.now(UTC).isoformat(),
                "evidence_text": name + " lives in " + city,
                "idempotency_key": key,
            },
        )
        assert result.status is ToolStatus.SUCCEEDED, result.error
        assert result.result is not None
        receipts.append(result.result)
    historic = runtime["recall"](
        ctx, {"scope": "s", "types": ["semantic"], "valid_at": "2026-01-15T00:00:00+00:00"}
    )
    assert historic.result is not None
    assert "Paris" in str(historic.result) and "London" not in str(historic.result)
    current = runtime["recall"](
        ctx,
        {"scope": "s", "types": ["semantic"], "min_revision": receipts[1]["committed_revision"]},
    )
    assert current.result is not None and "London" in str(current.result)
    history = runtime["get_fact_history"](ctx, {"scope": "s", "fact_id": receipts[0]["fact_id"]})
    assert history.result is not None
    facts = history.result["facts"]
    assert isinstance(facts, (tuple, list)) and len(facts) == 2
    assert all(isinstance(fact, Mapping) for fact in facts)
    archived = runtime["archive_memory"](
        ctx, {"scope": "s", "memory_id": receipts[1]["memory_id"], "idempotency_key": "archive"}
    )
    assert archived.status is ToolStatus.SUCCEEDED
    after = runtime["recall"](ctx, {"scope": "s", "types": ["semantic"]})
    assert after.result is not None and not after.result["hits"]
    assert (
        runtime["erase_scope"](ctx, {"scope": "s", "idempotency_key": "erase"}).status
        is ToolStatus.SUCCEEDED
    )
    assert runtime["recall"](ctx, {"scope": "s"}).status is ToolStatus.FAILED
