"""Measure peak RSS of an offline retrieval build over many large Markdown sources.

Fixtures are generated in this parent process. A separate child process then
imports and builds them, so neither pytest nor fixture construction is measured.
The child resets and reports its own peak resident set size as JSON: Linux
``VmHWM`` after ``/proc/self/clear_refs`` (a fresh child's ``ru_maxrss`` would start
at its parent's size), or ``ru_maxrss`` where that is unavailable.

Usage::

    uv run python scripts/measure_retrieval_build_memory.py --sources 50 --mib 16
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import time
from pathlib import Path


_PEAK_HELPER = """
import resource, sys

def reset_peak():
    # A child's ru_maxrss starts at its parent's RSS on Linux, so reset the
    # high-water mark (VmHWM) instead when the kernel supports it.
    try:
        with open("/proc/self/clear_refs", "w") as stream:
            stream.write("5")
        return True
    except OSError:
        return False

def peak_kib(reset):
    if reset:
        with open("/proc/self/status") as status:
            for line in status:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1])
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value // 1024 if sys.platform == "darwin" else value

RESET = reset_peak()
"""


def _write_source(path: Path, seed: int, size: int) -> None:
    paragraph = f"Source {seed} paragraph about lineage, evidence, and immutable notes. " * 4
    with path.open("wb") as target:
        written, index = 0, 0
        while True:
            block = f"Note{seed:03d}-{index:07d} {paragraph}\n\n".encode()
            if written + len(block) > size:
                break
            target.write(block)
            written += len(block)
            index += 1


def main() -> int:
    """Generate fixtures, run the measured child, and print one JSON record."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sources", type=int, default=50)
    parser.add_argument("--mib", type=int, default=16)
    parser.add_argument("--limit-mib", type=int, default=200)
    arguments = parser.parse_args()
    size = arguments.mib * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="capicove-memory-") as temporary:
        root = Path(temporary)
        workspace, notes = root / "workspace", root / "notes"
        workspace.mkdir(mode=0o700)
        notes.mkdir()
        fixture_digest = hashlib.sha256()
        for index in range(arguments.sources):
            path = notes / f"source-{index:03d}.md"
            _write_source(path, index, size)
            fixture_digest.update(hashlib.sha256(path.read_bytes()).digest())
        config = workspace / "host.json"
        config.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "server_name": "capicove-local",
                    "workspace_id": "measurement",
                    "actor_id": "owner",
                    "environment_id": "local",
                    "workspace_root": str(workspace),
                    "profile": "write",
                    "permissions": ["evidence:read", "evidence:search"],
                    "enabled_tools": ["capicove.search"],
                    "trace_store": None,
                    "capabilities": {
                        "retrieval": {
                            "schema_version": 1,
                            "source_roots": {"notes": str(notes)},
                            "index_root": str(workspace / "index"),
                            "corpus_ref": "corpus:measurement",
                            "retrieval_profile_ref": "local-default",
                            "max_source_bytes": size,
                        }
                    },
                    "network": {"enabled": False, "source_hosts": []},
                }
            )
        )
        config.chmod(0o600)
        child = _PEAK_HELPER + textwrap.dedent(
            f"""
            import json, time
            from pathlib import Path
            from capicove.runtime.local_retrieval import build_index, import_source
            from capicove.runtime.mcp_config import load_mcp_host_config
            config = load_mcp_host_config(Path({str(config)!r}))
            for index in range({arguments.sources}):
                import_source(config, alias="notes", relative_path=f"source-{{index:03d}}.md")
            before = peak_kib(RESET)
            started = time.perf_counter()
            result = build_index(config)
            print(json.dumps({{
                "peak_rss_before_build_kib": before,
                "peak_rss_kib": peak_kib(RESET),
                "build_seconds": round(time.perf_counter() - started, 1),
                "text_units": result["text_units"],
                "generation_id": result["generation_id"],
            }}))
            """
        )
        completed = subprocess.run(
            [sys.executable, "-c", child], capture_output=True, text=True, check=False
        )
        if completed.returncode != 0:
            print(completed.stderr, file=sys.stderr)
            return 1
        measured = json.loads(completed.stdout)
    peak_mib = measured["peak_rss_kib"] / 1024
    record = {
        **measured,
        "peak_rss_mib": round(peak_mib, 1),
        "limit_mib": arguments.limit_mib,
        "passed": peak_mib < arguments.limit_mib,
        "sources": arguments.sources,
        "source_mib": arguments.mib,
        "fixture_set_sha256": fixture_digest.hexdigest(),
        "method": "child VmHWM after clear_refs (else ru_maxrss); fixtures built in the parent",
        "python": platform.python_version(),
        "sqlite": sqlite3.sqlite_version,
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    print(json.dumps(record, indent=2, sort_keys=True))
    return 0 if record["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
