"""Explicit CLI composition for opt-in Tool operation tracing."""

from __future__ import annotations

from collections.abc import Mapping

from capicove.contracts import InvocationContext, ToolHandler, ToolResult, ToolSpec
from capicove.tools.tracing import invoke_one_step_tool
from capicove.traces import TraceRecorder


def invoke_cli_tool(
    *,
    recorder: TraceRecorder | None,
    spec: ToolSpec,
    handler: ToolHandler,
    context: InvocationContext,
    arguments: Mapping[str, object],
) -> ToolResult:
    """Invoke one CLI Tool directly or through the shared trace boundary."""
    return invoke_one_step_tool(
        recorder=recorder,
        spec=spec,
        handler=handler,
        context=context,
        arguments=arguments,
    )
