"""Production operations available inside isolated remote workers.

``probe`` is the isolation self-check used by readiness and qualification: it
reports only the worker's own identity and observed sandbox properties.
"""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


def _connect(address: tuple[str, int]) -> str:
    try:
        with socket.create_connection(address, timeout=1):
            return "connected"
    except OSError:
        return "refused_or_unreachable"


def probe(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    """Report identity, capabilities, environment keys, egress and writability."""
    home = Path(os.environ.get("HOME", "/nonexistent"))
    marker = home / ".capicove-probe"
    try:
        marker.write_text("probe", encoding="utf-8")
        marker.unlink()
        writable = True
    except OSError:
        writable = False
    port = payload.get("port")
    status = Path("/proc/self/status").read_text(encoding="ascii").splitlines()
    security = {
        key: value.strip()
        for key, _, value in (line.partition(":") for line in status)
        if key in {"CapInh", "CapPrm", "CapEff", "CapBnd", "CapAmb", "NoNewPrivs"}
    }
    return {
        "security": security,
        "root_writable": os.access("/", os.W_OK),
        "uid": os.getuid(),
        "gid": os.getgid(),
        "pid": os.getpid(),
        "groups": os.getgroups(),
        "environment": sorted(os.environ),
        "home_writable": writable,
        "loopback_connect": _connect(("127.0.0.1", int(port)))
        if isinstance(port, int)
        else "not_run",
        "external_connect": _connect(("1.1.1.1", 53)),
    }


def noop(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    """Return an empty successful result; used for admission and latency checks."""
    return {}


OPERATIONS: Mapping[str, Callable[..., Any]] = {"probe": probe, "noop": noop}
