"""Adversarial and filesystem operations for isolated-worker qualification tests only."""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from capicove.runtime._remote.worker_entry import canonical, sign
from capicove.runtime._remote.worker_operations import probe


def write(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    path = Path(os.environ["HOME"]) / str(payload["name"])
    path.write_text(str(payload["text"]), encoding="utf-8")
    path.chmod(0o600)
    return {"written": True}


def read(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    try:
        return {"status": "ok", "text": Path(str(payload["path"])).read_text(encoding="utf-8")}
    except PermissionError:
        return {"status": "denied"}
    except FileNotFoundError:
        return {"status": "absent"}


def spawn_and_hang(payload: Mapping[str, Any], **_: Any) -> None:
    marker = str(payload["marker"])
    for _index in range(3):
        subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(1000)", marker],
            start_new_session=True,
        )
    time.sleep(1000)


def forge_envelope(payload: Mapping[str, Any], *, channel: Any, key: bytes, fence: str) -> None:
    envelope: dict[str, Any] = {
        "fence": "replayed-" + fence,
        "operation": "forge_envelope",
        "status": "succeeded",
        "output": {"forged": True},
    }
    envelope["mac"] = sign(key, envelope)
    channel.send(canonical(envelope))


def oversized_output(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    return {"blob": "x" * (2 * 1024 * 1024)}


def crash(payload: Mapping[str, Any], **_: Any) -> None:
    os._exit(3)


def ffmpeg(payload: Mapping[str, Any], **_: Any) -> dict[str, Any] | None:
    if payload.get("hang"):
        subprocess.Popen(
            [
                *("ffmpeg", "-hide_banner", "-re", "-f", "lavfi", "-i", "anullsrc"),
                *("-metadata", f"title={payload['marker']}", "-f", "null", "-"),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(1000)
        return None
    completed = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "sine=duration=1", "-f", "null", "-"],
        capture_output=True,
        check=False,
    )
    return {"returncode": completed.returncode}


def fork_many(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    spawned = 0
    failure = ""
    for _index in range(int(payload["count"])):
        try:
            subprocess.Popen([sys.executable, "-c", "import time; time.sleep(1000)"])
        except OSError as error:
            failure = type(error).__name__
            break
        spawned += 1
    return {"spawned": spawned, "failure": failure}


def allocate(payload: Mapping[str, Any], **_: Any) -> dict[str, Any]:
    block = bytearray(int(payload["bytes"]))
    block[::4096] = b"x" * len(block[::4096])
    return {"allocated": len(block)}


OPERATIONS: Mapping[str, Callable[..., Any]] = {
    "probe": probe,
    "write": write,
    "read": read,
    "spawn_and_hang": spawn_and_hang,
    "forge_envelope": forge_envelope,
    "oversized_output": oversized_output,
    "crash": crash,
    "ffmpeg": ffmpeg,
    "fork_many": fork_many,
    "allocate": allocate,
}
