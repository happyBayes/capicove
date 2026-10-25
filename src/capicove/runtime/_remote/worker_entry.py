"""Isolated worker process entry: one request, one signed response, then exit.

Invoked only by ``capicove.runtime.remote_workers`` inside fresh PID, network, IPC,
UTS and mount namespaces as an unprivileged domain identity. The supervisor
selects the operation registry module; request payloads cannot name code.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import socket
import sys
from typing import Any


MAX_MESSAGE_BYTES = 1024 * 1024


def canonical(value: Any) -> bytes:
    """Serialize deterministically for MAC computation."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()


def sign(key: bytes, envelope: dict[str, Any]) -> str:
    """Return the HMAC-SHA256 over the envelope fields except the MAC itself."""
    return hmac.new(key, canonical(envelope), hashlib.sha256).hexdigest()


def main() -> int:
    """Read the request, run the registered operation and send one envelope."""
    channel = socket.socket(fileno=int(sys.argv[1]))
    registry = importlib.import_module(sys.argv[2])
    request = json.loads(channel.recv(MAX_MESSAGE_BYTES))
    key = bytes.fromhex(request["key"])
    operation = registry.OPERATIONS.get(request["operation"])
    envelope: dict[str, Any] = {"fence": request["fence"], "operation": request["operation"]}
    if operation is None:
        envelope.update(status="failed", output={"error": "unknown_operation"})
    else:
        try:
            output = operation(request["payload"], channel=channel, key=key, fence=request["fence"])
        except Exception as error:
            envelope.update(status="failed", output={"error": type(error).__name__})
        else:
            if output is None:
                return 0
            envelope.update(status="succeeded", output=output)
            if len(canonical(envelope)) > MAX_MESSAGE_BYTES // 2:
                envelope.update(status="failed", output={"error": "output_too_large"})
    envelope["mac"] = sign(key, envelope)
    channel.send(canonical(envelope))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
