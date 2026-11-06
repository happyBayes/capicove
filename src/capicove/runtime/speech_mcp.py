"""Workspace-owned speech bindings with no model load during catalog discovery."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from capicove.contracts import InvocationContext, ToolResult, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.mcp_config import WorkspaceHostConfiguration
from capicove.runtime.speech import SpeechRuntime
from capicove.tools.audio_synthesis import AUDIO_CLONE_SPEC, AUDIO_SYNTHESIZE_SPEC


@contextmanager
def compose_speech_mcp(
    config: WorkspaceHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    *,
    catalog_only: bool = False,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Keep speech independent of the existing POSIX-only ASR/clip host."""
    raw = config.capabilities["audio_synthesis"]
    if set(raw) != {"schema_version", "runtime_config"} or not isinstance(
        raw["runtime_config"], str
    ):
        raise ValueError("audio_synthesis requires schema_version and runtime_config")
    runtime = SpeechRuntime(Path(raw["runtime_config"]))
    try:
        if runtime.output_root == config.workspace_root or not runtime.output_root.is_relative_to(
            config.workspace_root
        ):
            raise ValueError("speech output_root must be inside workspace_root")
        handlers = {
            AUDIO_SYNTHESIZE_SPEC.tool_id: runtime.synthesize,
            AUDIO_CLONE_SPEC.tool_id: runtime.clone,
        }

        def catalog(context: InvocationContext, arguments: Mapping[str, object], /) -> ToolResult:
            raise RuntimeError("catalog-only binding cannot execute")

        bindings = []
        for tool_id in config.enabled_tools:
            if tool_id not in handlers:
                raise ValueError("unsupported speech Tool")
            handler = handlers[tool_id]
            if handler.spec.required_permissions - config.permissions:
                raise ValueError("speech permissions missing")
            bindings.append(
                McpToolBinding(
                    "capicove_audio_" + tool_id.rsplit(".", 1)[1],
                    handler.spec,
                    catalog if catalog_only else handler,
                    context_factory,
                )
            )
        yield tuple(bindings)
    finally:
        runtime.close()
