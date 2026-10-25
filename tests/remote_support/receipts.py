"""Integrity checks for operator-owned test observations, not release approval.

A digest authenticates neither the observer nor semantic correctness. Independent
review and the real-runner identity are still required by the release gate.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class Observation:
    """Bind observed bytes to the exact test and provisioned profile."""

    relative_path: str
    sha256: str
    test_id: str
    profile: str


def _artifact(path: Path, root: Path) -> tuple[Path, str]:
    try:
        relative = path.absolute().relative_to(root.absolute())
        if ".." in relative.parts:
            raise ValueError("observation escapes receipt root")
        cursor = root.absolute()
        if cursor.is_symlink():
            raise ValueError("symlink receipt root")
        for part in relative.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError("symlink observation")
        if not cursor.is_file():
            raise ValueError("missing observation")
        return cursor, relative.as_posix()
    except OSError as error:
        raise ValueError("unreadable observation") from error


def observe(path: Path, *, root: Path, test_id: str, profile: str) -> Observation:
    """Hash an actual private artifact produced by an independently observed run."""
    if not test_id.strip() or not profile.strip():
        raise ValueError("test and profile are required")
    artifact, relative = _artifact(path, root)
    with artifact.open("rb") as stream:
        digest = hashlib.file_digest(stream, "sha256").hexdigest()
    return Observation(relative, digest, test_id, profile)


def verify(observation: Observation, *, root: Path, test_id: str, profile: str) -> None:
    """Reject substituted artifact bytes, tests or provider profiles."""
    if observation.test_id != test_id or observation.profile != profile:
        raise ValueError("test/profile mismatch")
    actual = observe(root / observation.relative_path, root=root, test_id=test_id, profile=profile)
    if actual.sha256 != observation.sha256:
        raise ValueError("observation digest mismatch")


def require_complete(states: Mapping[str, str], *, required: set[str]) -> None:
    """Reject absent, skipped or unsuccessful requirements without reducing scope."""
    if not required or any(states.get(key) != "passed" for key in required):
        raise ValueError("required observations are incomplete")


def validate_candidate(candidate: Mapping[str, object], *, required_contracts: set[str]) -> None:
    """Validate a candidate record; callers must independently verify its origins."""
    import re

    for field, length in (("source_revision", 40), ("wheel_sha256", 64), ("lock_sha256", 64)):
        value = candidate.get(field)
        if (
            not isinstance(value, str)
            or re.fullmatch(r"[0-9a-f]{" + str(length) + "}", value) is None
        ):
            raise ValueError("candidate identity is not frozen")
    if candidate.get("source_dirty") is not False:
        raise ValueError("candidate source is dirty or unknown")
    if (
        candidate.get("protocol_version") != "2025-11-25"
        or candidate.get("sdk_version") != "1.30.0"
    ):
        raise ValueError("candidate protocol or SDK is unqualified")
    contracts = candidate.get("contracts")
    if not isinstance(contracts, dict) or not required_contracts:
        raise ValueError("candidate contract decisions are missing")
    if any(contracts.get(contract) != "accepted" for contract in required_contracts):
        raise ValueError("candidate requires an accepted contract amendment")


class RunObserver:
    """Witness a synthetic worker through an independent local HTTP endpoint.

    The runner owns the sealing key and observes process identity/exit itself.
    This qualifies harness plumbing only, never a model or production provider.
    A reviewer must still trust the runner and bind its results to release hashes.
    """

    def __init__(self, test_id: str, profile: str) -> None:
        import secrets
        import threading

        if not test_id or not profile:
            raise ValueError("test and profile are required")
        self._test_id = test_id
        self._profile = profile
        self._key = secrets.token_bytes(32)
        self._challenge = secrets.token_hex(32)
        self._lock = threading.Lock()
        self._calls = 0
        self._pid: int | None = None
        self._exit: int | None = None
        self._active = False
        self._entered = False
        self._worker_reserved = False

    @property
    def test_id(self) -> str:
        """Return the immutable run identity."""
        return self._test_id

    @property
    def profile(self) -> str:
        """Return the immutable observation profile."""
        return self._profile

    def __enter__(self) -> RunObserver:
        import http.server
        import threading

        if self._entered:
            raise ValueError("an observation cannot be reopened")
        self._entered = True
        observer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                if self.path != "/observe" or self.headers.get("Content-Length") != "64":
                    self.send_error(400)
                    return
                self.connection.settimeout(2)
                if self.rfile.read(64).decode("ascii") != observer._challenge:
                    self.send_error(403)
                    return
                with observer._lock:
                    observer._calls += 1
                self.send_response(204)
                self.end_headers()

            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = http.server.HTTPServer(("127.0.0.1", 0), Handler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._active = True
        self._thread.start()
        return self

    def __exit__(self, *args: object) -> None:
        self._active = False
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=3)

    def run_worker(self, *, fail: bool = False) -> subprocess.CompletedProcess[str]:
        import subprocess
        import sys

        with self._lock:
            if not self._active or self._worker_reserved:
                raise ValueError("one worker per active observation")
            self._worker_reserved = True
        # The child cannot access the sealing key. Its stdout is not the oracle.
        program = (
            "import sys, urllib.request; "
            "request=urllib.request.Request(sys.argv[1], data=sys.argv[2].encode(), method='POST'); "
            "urllib.request.build_opener(urllib.request.ProxyHandler({})).open(request, timeout=2).close(); sys.exit(int(sys.argv[3]))"
        )
        arguments = [
            sys.executable,
            "-c",
            program,
            f"http://127.0.0.1:{self._server.server_port}/observe",
            self._challenge,
            "1" if fail else "0",
        ]
        with subprocess.Popen(
            arguments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        ) as worker:
            self._pid = worker.pid
            try:
                stdout, stderr = worker.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                worker.kill()
                worker.communicate()
                self._exit = worker.returncode
                raise ValueError("observed worker exceeded deadline") from None
            self._exit = worker.returncode
            return subprocess.CompletedProcess(arguments, worker.returncode, stdout, stderr)

    def _payload(self) -> dict[str, object]:
        with self._lock:
            return {
                "test_id": self.test_id,
                "profile": self.profile,
                "provider_calls": self._calls,
                "worker_pid": self._pid,
                "worker_exit": self._exit,
                "evidence_kind": "synthetic-harness-only",
            }

    def _seal(self, payload: dict[str, object]) -> str:
        import hmac
        import json

        return hmac.new(
            self._key, json.dumps(payload, sort_keys=True).encode(), "sha256"
        ).hexdigest()

    def finish(self, *, required_events: set[str]) -> dict[str, object]:
        if not self._active or not required_events or self._exit != 0:
            raise ValueError("live successful worker observation required")
        payload = self._payload()
        observed = {"worker-exit"}
        if self._calls:
            observed.add("provider-request")
        if not required_events <= observed:
            raise ValueError("required observations missing")
        return {**payload, "seal": self._seal(payload)}

    def verify(self, receipt: dict[str, object]) -> None:
        import hmac

        if not self._active or self._exit != 0:
            raise ValueError("no active successful observed run")
        payload = {key: value for key, value in receipt.items() if key != "seal"}
        seal = receipt.get("seal")
        if (
            payload != self._payload()
            or not isinstance(seal, str)
            or not hmac.compare_digest(seal, self._seal(payload))
        ):
            raise ValueError("receipt is not from this observed run")
