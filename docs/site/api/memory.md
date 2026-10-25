# Temporal Memory API

Temporal Memory separates public contracts, the knowledge service, Agent-owned
admission policy, runtime composition, canonical Tools, and MCP transport. Hosts
must construct authorization context; model input never selects workspace,
filesystem paths, providers, or permissions.

## Contracts

::: capicove.contracts.memory
    options:
      members: [MemoryRecord, MemoryQuery, MemoryHit, MemoryRecall, MemoryContext]
      show_if_no_docstring: true

::: capicove.contracts.knowledge
    options:
      members: [TimeInterval, Authority, TypedLiteral, EntityRef, Predicate]
      show_if_no_docstring: true

## Knowledge and Agent services

::: capicove.knowledge.TemporalKnowledgeService
    options:
      show_if_no_docstring: true

::: capicove.agent.AgentMemoryService
    options:
      show_if_no_docstring: true

## Runtime composition

The host supplies a private root, workspace, authorized scopes, ontology version,
and predicate tuple. The result maps operation names to `ToolHandler` objects.
No extractor or dense provider is enabled by default. Unsupported backends raise
`RuntimeError("backend_unavailable")`; an unsafe root raises `ValueError`.

Runnable composition example (creates storage only inside a temporary directory):

<!-- docs-smoke: memory -->
```python
from pathlib import Path
from tempfile import TemporaryDirectory
from capicove.runtime.memory import build_memory_runtime

with TemporaryDirectory() as directory:
    handlers = build_memory_runtime(
        Path(directory),
        workspace="demo",
        scopes=frozenset({"notes"}),
        ontology_version="empty-demo-v1",
        predicates=(),
    )
    assert "remember_episode" in handlers and "recall" in handlers
```

This example constructs handlers; it grants no caller authority and writes no
Episode. A trusted host must supply the invocation context when calling them.
For a complete remember/recall request, use the
[Memory guide](../guides/temporal-memory.md).

::: capicove.runtime.memory.build_memory_runtime
    options:
      show_if_no_docstring: true

## Tool and MCP boundaries

The Tool specs are the authoritative argument schemas. Handlers validate those
arguments and the host context. Inspect `status` and `error` in the encoded
result; a transport response alone does not mean the operation succeeded.

::: capicove.tools.memory.memory_spec
    options:
      show_if_no_docstring: true

::: capicove.tools.memory.MemoryToolHandler
    options:
      show_if_no_docstring: true

::: capicove.tools.memory.invoke_memory_tool
    options:
      show_if_no_docstring: true

::: capicove.tools.memory.encode_memory_tool_result
    options:
      show_if_no_docstring: true

::: capicove.mcp.memory.register_memory_mcp
    options:
      show_if_no_docstring: true
