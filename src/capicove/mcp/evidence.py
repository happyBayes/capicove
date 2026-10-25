"""MCP translation for evidence with a startup-owned authenticated context."""

from collections.abc import Callable, Mapping

from capicove.contracts import InvocationContext, ToolHandler
from capicove.mcp.search import McpCallback, McpRegistrationServer
from capicove.tools.evidence import CAPICOVE_EVIDENCE_SPEC


def register_evidence_mcp(
    server: McpRegistrationServer,
    handler: ToolHandler,
    *,
    context_factory: Callable[[], InvocationContext],
) -> McpCallback:
    """Register without deriving authority from caller-controlled arguments."""

    @server.tool(
        name="capicove_get_evidence",
        description=CAPICOVE_EVIDENCE_SPEC.description,
        input_schema=CAPICOVE_EVIDENCE_SPEC.input_schema,
    )
    def invoke(arguments: Mapping[str, object]) -> Mapping[str, object]:
        result = handler(context_factory(), arguments)
        error = result.error
        return {
            "request_id": result.request_id,
            "trace_id": result.trace_id,
            "tool_id": result.tool_id,
            "tool_version": result.tool_version,
            "status": result.status.value,
            "result": None if result.result is None else dict(result.result),
            "evidence_refs": list(result.evidence_refs),
            "error": None
            if error is None
            else {"code": error.code.value, "message": error.message, "retryable": error.retryable},
        }

    return invoke
