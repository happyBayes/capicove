"""Request contexts remain distinct through the shared Tool binding path."""

from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from typing import NoReturn

from capicove.contracts import InvocationContext, ToolResult, ToolSpec, ToolStatus
from capicove.mcp.bindings import McpToolBinding, default_context, invoke_binding


def test_explicit_contexts_do_not_call_or_mutate_startup_factory() -> None:
    spec = ToolSpec(
        "capicove.test.identity",
        1,
        "Echo authorized actor",
        {"type": "object", "additionalProperties": False},
        {"type": "object", "properties": {"actor": {"type": "string"}}, "required": ["actor"]},
    )

    def handler(context: InvocationContext, arguments: Mapping[str, object]) -> ToolResult:
        return ToolResult(
            context.request_id,
            context.trace_id,
            spec.tool_id,
            1,
            ToolStatus.SUCCEEDED,
            result={"actor": context.actor_id},
        )

    def startup_factory(spec: ToolSpec) -> NoReturn:
        raise AssertionError("Explicit invocation must not consult startup authority")

    binding = McpToolBinding("capicove_test_identity", spec, handler, startup_factory)
    contexts = [replace(default_context(spec), actor_id=actor) for actor in ("alice", "bob")]
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(lambda context: invoke_binding(binding, {}, context=context), contexts)
        )
    assert [result["result"] for result in results] == [{"actor": "alice"}, {"actor": "bob"}]
    assert binding.context_factory is startup_factory
