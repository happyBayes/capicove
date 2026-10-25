"""Worker process for `capicove http serve` (ADR-0025).

It composes the same host bindings as MCP and answers one JSON line per request
on a protocol-only stdout. The HTTP process starts it in a new process group so
shutdown can stop in-flight work, including external child processes.
"""

from __future__ import annotations

import base64
import json
import os
import sys
from contextlib import ExitStack


def main(argv: list[str]) -> int:
    """Serve invocations until stdin closes."""
    from capicove.mcp.bindings import InvalidToolArguments, invoke_binding
    from capicove.mcp.stdio import protocol_stdout
    from capicove.runtime.mcp_config import parse_mcp_host_config
    from capicove.runtime.mcp_host import compose_mcp_host

    with protocol_stdout() as protocol, ExitStack() as stack:

        def reply(message: dict[str, object]) -> None:
            protocol.write(json.dumps(message, sort_keys=True) + "\n")
            protocol.flush()

        try:
            # The server sends the configuration bytes it froze at startup.
            startup = json.loads(sys.stdin.readline())
            config = parse_mcp_host_config(base64.b64decode(startup["config"], validate=True))
            bindings = stack.enter_context(compose_mcp_host(config))
            expected = startup.get("workspace_identity")
            if expected is not None:
                metadata = os.stat(config.workspace_root, follow_symlinks=False)  # type: ignore[union-attr]
                if [metadata.st_dev, metadata.st_ino] != expected:
                    raise ValueError("workspace_root changed since the server started")
        except Exception:
            reply({"failed": "startup"})
            return 2
        selected = {item.spec.tool_id: item for item in bindings}
        reply({"ready": True})
        for line in sys.stdin:
            try:
                request = json.loads(line)
                item = selected[request["tool_id"]]
                reply({"result": invoke_binding(item, request["arguments"])})
            except InvalidToolArguments:
                reply({"error": "invalid_arguments"})
            except Exception:
                reply({"error": "internal"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
