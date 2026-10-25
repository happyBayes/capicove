"""Exercise protected artifacts and durable receipts using local authored data."""

import errno
import hashlib
import os
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from threading import Barrier, Event

import pytest

from capicove.contracts.graph_build import GraphScope


SCOPE = GraphScope("workspace", "private")
PAYLOAD = b'{"authored_result": "protected provider bytes"}'


class Gate:
    """Represent current local policy, including evidence deletion/revocation."""

    def __init__(self) -> None:
        self.denied_scopes: set[GraphScope] = set()
        self.denied_sources: set[str] = set()

    def require_scope(self, scope: GraphScope) -> None:
        if scope in self.denied_scopes:
            raise PermissionError("scope denied")

    def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
        self.require_scope(scope)
        if not evidence_refs or set(evidence_refs) & self.denied_sources:
            raise PermissionError("source unavailable")


class Clock:
    """Supply deterministic Unix seconds without sleeps."""

    now = 1000.0

    def __call__(self) -> float:
        return self.now


def test_identical_request_does_not_share_scope_cache() -> None:
    from capicove.graph_build.receipts import receipt_key

    private = receipt_key(GraphScope("w", "private"), "extract", "a" * 64)
    work = receipt_key(GraphScope("w", "work"), "extract", "a" * 64)
    other_workspace = receipt_key(GraphScope("other", "private"), "extract", "a" * 64)
    assert len({private, work, other_workspace}) == 3


def test_artifacts_survive_reopen_and_enforce_persisted_lineage(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts

    gate = Gate()
    root = tmp_path / "artifacts"
    artifacts = FilesystemProtectedArtifacts(root, gate, evidence_refs=("source:a",))
    ref = artifacts.put(SCOPE, PAYLOAD)
    assert ref == artifacts.put(SCOPE, PAYLOAD)
    assert artifacts.read(SCOPE, ref) == PAYLOAD
    reopened = FilesystemProtectedArtifacts(root, gate, evidence_refs=("source:b",))
    assert reopened.read(SCOPE, ref) == PAYLOAD
    assert reopened.put(SCOPE, PAYLOAD) != ref
    gate.denied_sources.add("source:a")
    with pytest.raises(PermissionError):
        reopened.read(SCOPE, ref)
    with pytest.raises(PermissionError):
        artifacts.put(SCOPE, PAYLOAD)
    for path in (root, *root.rglob("*")):
        assert path.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("scope", [GraphScope("other", "private"), GraphScope("workspace", "work")])
def test_artifact_scope_isolation(tmp_path: Path, scope: GraphScope) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts

    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", Gate(), evidence_refs=("a",))
    ref = artifacts.put(SCOPE, PAYLOAD)
    with pytest.raises(PermissionError):
        artifacts.read(scope, ref)
    assert artifacts.put(scope, PAYLOAD) != ref


def test_scope_revocation_blocks_every_artifact_operation(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    ref = artifacts.put(SCOPE, PAYLOAD)
    gate.denied_scopes.add(SCOPE)
    with pytest.raises(PermissionError):
        artifacts.put(SCOPE, b"new")
    with pytest.raises(PermissionError):
        artifacts.read(SCOPE, ref)


@pytest.mark.parametrize(
    "ref", ["../outside", "/etc/passwd", "artifact:../x", "artifact:" + "A" * 64]
)
def test_artifact_refs_are_never_paths(tmp_path: Path, ref: str) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts

    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", Gate(), evidence_refs=("a",))
    with pytest.raises(ValueError):
        artifacts.read(SCOPE, ref)


def test_artifact_corruption_and_symlink_are_rejected(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )

    root = tmp_path / "artifacts"
    artifacts = FilesystemProtectedArtifacts(root, Gate(), evidence_refs=("a",))
    ref = artifacts.put(SCOPE, PAYLOAD)
    blob = root / (ref.removeprefix("artifact:") + ".json")
    original = blob.read_bytes()
    blob.write_bytes(b"damaged")
    with pytest.raises(ArtifactStoreError):
        artifacts.read(SCOPE, ref)
    with pytest.raises(ArtifactStoreError):
        artifacts.put(SCOPE, PAYLOAD)
    blob.unlink()
    outside = tmp_path / "outside"
    outside.write_bytes(original)
    blob.symlink_to(outside)
    with pytest.raises(ArtifactStoreError):
        artifacts.read(SCOPE, ref)
    with pytest.raises(ArtifactStoreError):
        artifacts.put(SCOPE, PAYLOAD)
    assert outside.read_bytes() == original


def test_artifact_root_symlink_and_insecure_permissions_are_rejected(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )

    real = tmp_path / "real"
    real.mkdir(mode=0o700)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    for root in (linked, linked / "nested"):
        with pytest.raises(ArtifactStoreError):
            FilesystemProtectedArtifacts(root, Gate(), evidence_refs=("a",))
    real.chmod(0o755)
    with pytest.raises(ArtifactStoreError):
        FilesystemProtectedArtifacts(real, Gate(), evidence_refs=("a",))


def test_receipt_recovery_expiry_stale_attempt_and_duplicate_completion(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, ReceiptConflictError, receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "receipts.sqlite3"
    store = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "a" * 64)
    receipt = InferenceReceipt(
        key,
        "succeeded",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        2,
        1,
        (("input_tokens", 5), ("output_tokens", 7)),
        1,
        ("invalid_record",),
    )
    assert store.get(SCOPE, key) is None
    assert store.claim(SCOPE, key, "first", 1010.0)
    assert not store.claim(SCOPE, key, "second", 1020.0)
    assert store.get(SCOPE, key) is None
    reopened = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    assert not reopened.claim(SCOPE, key, "second", 1020.0)
    clock.now = 1010.0
    with pytest.raises(ReceiptConflictError):
        reopened.complete(SCOPE, key, "first", receipt)
    assert reopened.claim(SCOPE, key, "second", 1020.0)
    with pytest.raises(ReceiptConflictError):
        store.complete(SCOPE, key, "first", receipt)
    reopened.complete(SCOPE, key, "second", receipt)
    clock.now = 1030.0
    reopened.complete(SCOPE, key, "second", receipt)
    with pytest.raises(ReceiptConflictError):
        reopened.complete(SCOPE, key, "second", replace(receipt, call_count=3))
    with pytest.raises(ReceiptConflictError):
        store.complete(SCOPE, key, "first", receipt)
    assert store.get(SCOPE, key) == receipt
    assert not store.claim(SCOPE, key, "third", 1040.0)
    with pytest.raises(FrozenInstanceError):
        receipt.status = "failed"  # type: ignore[misc]
    assert PAYLOAD not in database.read_bytes()
    assert b"protected provider bytes" not in database.read_bytes()
    assert database.stat().st_mode & 0o077 == 0


def test_compound_scope_and_concurrent_claims(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "receipts.sqlite3"
    key = receipt_key(SCOPE, "extract", "b" * 64)
    barrier = Barrier(8)

    def claim(index: int) -> bool:
        store = SqliteReceiptStore(database, gate, artifacts, clock=clock)
        barrier.wait(timeout=10)
        return store.claim(SCOPE, key, f"attempt:{index}", 1100.0)

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert sum(executor.map(claim, range(8))) == 1
    store = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    for scope in (GraphScope("other", "private"), GraphScope("workspace", "work")):
        assert store.claim(scope, key, "independent", 1100.0)
        assert store.get(scope, key) is None


def test_receipt_cannot_bypass_artifact_access_or_integrity(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, ReceiptStoreError, receipt_key

    gate, clock = Gate(), Clock()
    root = tmp_path / "artifacts"
    artifacts = FilesystemProtectedArtifacts(root, gate, evidence_refs=("a",))
    store = SqliteReceiptStore(tmp_path / "receipts" / "r.sqlite3", gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "c" * 64)
    ref = artifacts.put(SCOPE, PAYLOAD)
    receipt = InferenceReceipt(
        key, "succeeded", ref, hashlib.sha256(PAYLOAD).hexdigest(), 1, 0, (), 0, ()
    )
    assert store.claim(SCOPE, key, "attempt", 1100.0)
    with pytest.raises(ReceiptStoreError):
        store.complete(SCOPE, key, "attempt", replace(receipt, result_digest="0" * 64))
    gate.denied_sources.add("a")
    with pytest.raises(PermissionError):
        store.complete(SCOPE, key, "attempt", receipt)
    gate.denied_sources.clear()
    store.complete(SCOPE, key, "attempt", receipt)
    gate.denied_sources.add("a")
    for operation in (
        lambda: store.get(SCOPE, key),
        lambda: store.claim(SCOPE, key, "reuse", 1100.0),
        lambda: store.complete(SCOPE, key, "attempt", receipt),
    ):
        with pytest.raises(PermissionError):
            operation()
    gate.denied_sources.clear()
    (root / (ref.removeprefix("artifact:") + ".json")).write_bytes(b"corrupt")
    with pytest.raises(ArtifactStoreError):
        store.get(SCOPE, key)


def test_scope_denial_blocks_all_receipt_operations(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    store = SqliteReceiptStore(tmp_path / "receipts" / "r.sqlite3", gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "d" * 64)
    receipt = InferenceReceipt(
        key,
        "succeeded",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        1,
        0,
        (),
        0,
        (),
    )
    gate.denied_scopes.add(SCOPE)
    for operation in (
        lambda: store.get(SCOPE, key),
        lambda: store.claim(SCOPE, key, "attempt", 1100.0),
        lambda: store.complete(SCOPE, key, "attempt", receipt),
    ):
        with pytest.raises(PermissionError):
            operation()


def test_sqlite_schema_corruption_and_symlink_rejection(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    SqliteReceiptStore(database, gate, artifacts)
    with sqlite3.connect(database) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert tables == {"inference_units"}
        connection.execute("PRAGMA user_version=999")
    with pytest.raises(ReceiptStoreError):
        SqliteReceiptStore(database, gate, artifacts)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999
    database.write_bytes(b"invalid sqlite bytes")
    with pytest.raises(ReceiptStoreError):
        SqliteReceiptStore(database, gate, artifacts)
    database.unlink()
    external = tmp_path / "external"
    external.write_bytes(b"untouched")
    database.symlink_to(external)
    with pytest.raises(ReceiptStoreError):
        SqliteReceiptStore(database, gate, artifacts)
    assert external.read_bytes() == b"untouched"


def test_stored_receipt_corruption_fails_closed(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, ReceiptStoreError, receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    store = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "e" * 64)
    receipt = InferenceReceipt(
        key,
        "succeeded",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        1,
        0,
        (),
        0,
        (),
    )
    assert store.claim(SCOPE, key, "attempt", 1100.0)
    store.complete(SCOPE, key, "attempt", receipt)
    with sqlite3.connect(database) as connection:
        connection.execute(
            "UPDATE inference_units SET receipt_json = ?", ('{"status":"succeeded"}',)
        )
    with pytest.raises(ReceiptStoreError):
        store.get(SCOPE, key)
    with pytest.raises(ReceiptStoreError):
        store.claim(SCOPE, key, "new", 1100.0)


def test_failed_receipt_retry_never_reuses_fencing_token(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, ReceiptConflictError, receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    store = SqliteReceiptStore(tmp_path / "receipts" / "r.sqlite3", gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "f" * 64)
    receipt = InferenceReceipt(
        key,
        "failed",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        1,
        0,
        (),
        0,
        ("invalid_output",),
    )
    assert store.claim(SCOPE, key, "first", 1100.0)
    store.complete(SCOPE, key, "first", receipt)
    assert store.get(SCOPE, key) == receipt
    assert not store.claim(SCOPE, key, "first", 1100.0)
    assert store.claim(SCOPE, key, "second", 1100.0)
    clock.now = 1200.0
    assert not store.claim(SCOPE, key, "first", 1300.0)
    assert not store.claim(SCOPE, key, "second", 1300.0)
    assert store.claim(SCOPE, key, "third", 1300.0)
    with pytest.raises(ReceiptConflictError):
        store.complete(SCOPE, key, "first", receipt)


def test_claim_deadline_is_rechecked_after_reuse_authorization(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, receipt_key

    clock = Clock()

    class DelayedGate(Gate):
        delay = False

        def require_access(self, evidence_refs: tuple[str, ...], scope: GraphScope) -> None:
            super().require_access(evidence_refs, scope)
            if self.delay:
                clock.now = 1200.0

    gate = DelayedGate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    store = SqliteReceiptStore(tmp_path / "receipts" / "r.sqlite3", gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "1" * 64)
    receipt = InferenceReceipt(
        key,
        "failed",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        1,
        0,
        (),
        0,
        (),
    )
    assert store.claim(SCOPE, key, "first", 1100.0)
    store.complete(SCOPE, key, "first", receipt)
    gate.delay = True
    with pytest.raises(ValueError):
        store.claim(SCOPE, key, "second", 1100.0)
    assert store.get(SCOPE, key) == receipt


def test_concurrent_completion_accepts_only_one_distinct_result(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, ReceiptConflictError, receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    store = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "2" * 64)
    receipt = InferenceReceipt(
        key,
        "succeeded",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        1,
        0,
        (),
        0,
        (),
    )
    assert store.claim(SCOPE, key, "owner", 1100.0)
    barrier = Barrier(6)

    def complete(index: int) -> bool:
        competing = SqliteReceiptStore(database, gate, artifacts, clock=clock)
        barrier.wait(timeout=10)
        try:
            competing.complete(SCOPE, key, "owner", replace(receipt, call_count=index + 1))
        except ReceiptConflictError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=6) as executor:
        assert sum(executor.map(complete, range(6))) == 1
    assert store.get(SCOPE, key) is not None


def test_process_crash_rolls_back_uncommitted_claim_change(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    store = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "3" * 64)
    assert store.claim(SCOPE, key, "owner", 1100.0)
    crashed = subprocess.run(
        [
            sys.executable,
            "-c",
            "import os, sqlite3, sys; c=sqlite3.connect(sys.argv[1]); c.execute('BEGIN IMMEDIATE'); c.execute('UPDATE inference_units SET lease_until=1'); os._exit(23)",
            str(database),
        ],
        check=False,
        capture_output=True,
    )
    assert crashed.returncode == 23
    reopened = SqliteReceiptStore(database, gate, artifacts, clock=clock)
    assert not reopened.claim(SCOPE, key, "new", 1100.0)
    clock.now = 1200.0
    assert reopened.claim(SCOPE, key, "new", 1300.0)


def test_schema_creation_rolls_back_when_version_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    real_connect = sqlite3.connect

    def denied_version(
        action: int, name: str | None, value: str | None, db: str | None, trigger: str | None
    ) -> int:
        return (
            sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_PRAGMA and name == "user_version" and value is not None
            else sqlite3.SQLITE_OK
        )

    def connect(*args: object, **kwargs: object) -> sqlite3.Connection:
        connection: sqlite3.Connection = real_connect(*args, **kwargs)  # type: ignore[call-overload]
        connection.set_authorizer(denied_version)
        return connection

    with monkeypatch.context() as patch:
        patch.setattr(sqlite3, "connect", connect)
        with pytest.raises(ReceiptStoreError):
            SqliteReceiptStore(database, gate, artifacts)
    with real_connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert (
            connection.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall() == []
        )
    SqliteReceiptStore(database, gate, artifacts)


@pytest.mark.parametrize("lease", [1000.0, float("nan"), float("inf"), True])
def test_invalid_leases_do_not_claim(tmp_path: Path, lease: float) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import receipt_key

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    store = SqliteReceiptStore(tmp_path / "receipts" / "r.sqlite3", gate, artifacts, clock=Clock())
    key = receipt_key(SCOPE, "extract", "4" * 64)
    with pytest.raises(ValueError):
        store.claim(SCOPE, key, "attempt", lease)
    assert store.claim(SCOPE, key, "attempt", 1100.0)


def test_artifact_root_replacement_and_file_permissions_fail_closed(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )

    root = tmp_path / "artifacts"
    artifacts = FilesystemProtectedArtifacts(root, Gate(), evidence_refs=("a",))
    ref = artifacts.put(SCOPE, PAYLOAD)
    blob = root / (ref.removeprefix("artifact:") + ".json")
    blob.chmod(0o644)
    with pytest.raises(ArtifactStoreError):
        artifacts.read(SCOPE, ref)
    blob.chmod(0o600)
    moved = tmp_path / "moved"
    root.rename(moved)
    root.symlink_to(moved, target_is_directory=True)
    with pytest.raises(ArtifactStoreError):
        artifacts.read(SCOPE, ref)


def test_sqlite_sidecar_symlink_and_changed_schema_fail_closed(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError, receipt_key

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    store = SqliteReceiptStore(database, gate, artifacts)
    external = tmp_path / "external"
    external.write_bytes(b"unrelated")
    sidecar = database.with_name(database.name + "-journal")
    sidecar.symlink_to(external)
    key = receipt_key(SCOPE, "extract", "5" * 64)
    with pytest.raises(ReceiptStoreError):
        store.get(SCOPE, key)
    assert external.read_bytes() == b"unrelated"
    sidecar.unlink()
    with sqlite3.connect(database) as connection:
        connection.execute("ALTER TABLE inference_units RENAME COLUMN unit_key TO bad_key")
    with pytest.raises(ReceiptStoreError):
        store.get(SCOPE, key)


def test_concurrent_artifact_publication_has_no_partial_reads(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts

    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", Gate(), evidence_refs=("a",))
    barrier = Barrier(8)

    def publish(index: int) -> str:
        barrier.wait(timeout=10)
        ref = artifacts.put(SCOPE, PAYLOAD)
        assert artifacts.read(SCOPE, ref) == PAYLOAD
        return ref

    with ThreadPoolExecutor(max_workers=8) as executor:
        assert len(set(executor.map(publish, range(8)))) == 1


def test_completed_receipt_does_not_leak_across_compound_scope(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import InferenceReceipt, ReceiptConflictError, receipt_key

    gate, clock = Gate(), Clock()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    store = SqliteReceiptStore(tmp_path / "receipts" / "r.sqlite3", gate, artifacts, clock=clock)
    key = receipt_key(SCOPE, "extract", "6" * 64)
    receipt = InferenceReceipt(
        key,
        "succeeded",
        artifacts.put(SCOPE, PAYLOAD),
        hashlib.sha256(PAYLOAD).hexdigest(),
        1,
        0,
        (),
        0,
        (),
    )
    assert store.claim(SCOPE, key, "attempt", 1100.0)
    store.complete(SCOPE, key, "attempt", receipt)
    for scope in (GraphScope("other", "private"), GraphScope("workspace", "work")):
        assert store.get(scope, key) is None
        with pytest.raises(ReceiptConflictError):
            store.complete(scope, key, "attempt", receipt)
        assert store.claim(scope, key, "attempt", 1100.0)
        with pytest.raises(PermissionError):
            store.complete(scope, key, "attempt", receipt)
        assert store.get(scope, key) is None
    assert store.get(SCOPE, key) == receipt


@pytest.mark.parametrize(
    "changes",
    [
        {"call_count": -1},
        {"repair_count": 2},
        {"rejected_count": True},
        {"usage": (("input_tokens", -1),)},
        {"usage": (("input_tokens", 1), ("input_tokens", 2))},
        {"issue_codes": ("Raw source text is not a code",)},
        {"status": "unknown"},
        {"result_artifact_ref": "/path"},
    ],
)
def test_invalid_receipt_metadata_is_rejected(changes: dict[str, object]) -> None:
    from capicove.graph_build.receipts import InferenceReceipt

    receipt = InferenceReceipt(
        "receipt:" + "a" * 64, "succeeded", "artifact:" + "b" * 64, "c" * 64, 1, 0, (), 0, ()
    )
    with pytest.raises((TypeError, ValueError)):
        replace(receipt, **changes)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("original", "replacement"),
    [
        (
            "PRIMARY KEY (workspace_ref, scope_ref, unit_key)",
            "PRIMARY KEY (workspace_ref, scope_ref, unit_key, status)",
        ),
        ("workspace_ref TEXT NOT NULL", "workspace_ref INTEGER NOT NULL"),
        ("scope_ref TEXT NOT NULL", "scope_ref TEXT"),
        (
            "status TEXT NOT NULL CHECK (status IN ('claimed', 'succeeded', 'failed', 'cancelled'))",
            "status TEXT NOT NULL",
        ),
    ],
)
def test_receipt_schema_rejects_altered_table_semantics(
    tmp_path: Path, original: str, replacement: str
) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    SqliteReceiptStore(database, gate, artifacts)
    with sqlite3.connect(database) as connection:
        definition = connection.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='inference_units'"
        ).fetchone()[0]
        assert original in definition
        connection.execute("DROP TABLE inference_units")
        connection.execute(definition.replace(original, replacement))
    with pytest.raises(ReceiptStoreError):
        SqliteReceiptStore(database, gate, artifacts)


@pytest.mark.parametrize(
    "extra_object",
    [
        "CREATE TRIGGER erase_claim AFTER INSERT ON inference_units "
        "BEGIN DELETE FROM inference_units; END",
        "CREATE INDEX extra_index ON inference_units(status)",
        "CREATE UNIQUE INDEX extra_unique ON inference_units(scope_ref)",
        "CREATE VIEW extra_view AS SELECT * FROM inference_units",
    ],
)
def test_receipt_schema_rejects_unexpected_objects_before_claim(
    tmp_path: Path, extra_object: str
) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError, receipt_key

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    database = tmp_path / "receipts" / "r.sqlite3"
    store = SqliteReceiptStore(database, gate, artifacts, clock=Clock())
    with sqlite3.connect(database) as connection:
        connection.execute(extra_object)
    key = receipt_key(SCOPE, "extract", "7" * 64)
    with pytest.raises(ReceiptStoreError):
        store.claim(SCOPE, key, "attempt", 1100.0)
    with sqlite3.connect(database) as connection:
        assert connection.execute("SELECT * FROM inference_units").fetchall() == []
    with pytest.raises(ReceiptStoreError):
        SqliteReceiptStore(database, gate, artifacts)


def test_schema_creation_rejects_preexisting_view_without_mutation(tmp_path: Path) -> None:
    from capicove.graph_build._adapters.artifacts import FilesystemProtectedArtifacts
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    artifacts = FilesystemProtectedArtifacts(tmp_path / "artifacts", gate, evidence_refs=("a",))
    directory = tmp_path / "receipts"
    directory.mkdir(mode=0o700)
    database = directory / "r.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE VIEW existing_view AS SELECT 1")
    database.chmod(0o600)
    with pytest.raises(ReceiptStoreError):
        SqliteReceiptStore(database, gate, artifacts)
    with sqlite3.connect(database) as connection:
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT type, name FROM sqlite_master").fetchall() == [
            ("view", "existing_view")
        ]


@pytest.mark.parametrize("storage", ["artifact", "receipt"])
@pytest.mark.parametrize("failed_parent", ["base", "nested"])
def test_parent_directory_sync_failure_prevents_storage_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: str, failed_parent: str
) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    existing_artifacts = FilesystemProtectedArtifacts(
        tmp_path / "existing-artifacts", gate, evidence_refs=("a",)
    )
    nested = tmp_path / "new-parent"
    root = nested / "store"
    target = tmp_path if failed_parent == "base" else nested
    real_fsync = os.fsync

    def fail_target_directory_sync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if target.exists():
            expected = target.stat()
            if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
                raise OSError(errno.EIO, "injected directory sync failure")
        real_fsync(descriptor)

    # All I/O stays real; only a sync of the exact parent inode fails. Successful
    # construction/publication would wrongly acknowledge an undurable path.
    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_target_directory_sync)
        with pytest.raises((ArtifactStoreError, ReceiptStoreError)):
            if storage == "artifact":
                artifacts = FilesystemProtectedArtifacts(root, gate, evidence_refs=("a",))
                artifacts.put(SCOPE, PAYLOAD)
            else:
                SqliteReceiptStore(root / "r.sqlite3", gate, existing_artifacts)
    assert nested.is_dir()
    if failed_parent == "base":
        assert not root.exists()
    else:
        assert root.is_dir()
        assert list(root.iterdir()) == []


@pytest.mark.parametrize("storage", ["artifact", "receipt"])
@pytest.mark.parametrize("failed_parent", ["base", "nested"])
def test_retry_cannot_bypass_persisting_an_existing_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: str, failed_parent: str
) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    existing_artifacts = FilesystemProtectedArtifacts(
        tmp_path / "existing-artifacts", gate, evidence_refs=("a",)
    )
    nested = tmp_path / "new-parent"
    root = nested / "store"
    target = tmp_path if failed_parent == "base" else nested
    real_fsync = os.fsync

    def fail_target_directory_sync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if target.exists():
            expected = target.stat()
            if (info.st_dev, info.st_ino) == (expected.st_dev, expected.st_ino):
                raise OSError(errno.EIO, "persistent parent sync failure")
        real_fsync(descriptor)

    def construct() -> None:
        if storage == "artifact":
            FilesystemProtectedArtifacts(root, gate, evidence_refs=("a",)).put(SCOPE, PAYLOAD)
        else:
            SqliteReceiptStore(root / "r.sqlite3", gate, existing_artifacts)

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", fail_target_directory_sync)
        for _ in range(2):
            with pytest.raises((ArtifactStoreError, ReceiptStoreError)):
                construct()
    if failed_parent == "base":
        assert not root.exists()
    else:
        assert root.is_dir()
        assert list(root.iterdir()) == []
    # The failed mkdir's directory remains in place; recovery needs a successful
    # sync, not removal/recreation of that directory or any unrelated content.
    construct()
    assert list(root.iterdir())


@pytest.mark.parametrize("storage", ["artifact", "receipt"])
def test_concurrent_constructor_must_sync_an_observed_existing_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, storage: str
) -> None:
    from capicove.graph_build._adapters.artifacts import (
        ArtifactStoreError,
        FilesystemProtectedArtifacts,
    )
    from capicove.graph_build._adapters.sqlite import SqliteReceiptStore
    from capicove.graph_build.receipts import ReceiptStoreError

    gate = Gate()
    existing_artifacts = FilesystemProtectedArtifacts(
        tmp_path / "existing-artifacts", gate, evidence_refs=("a",)
    )
    root = tmp_path / "store"
    parent = tmp_path.stat()
    creator_waiting, release_creator = Event(), Event()
    real_fsync = os.fsync

    def pause_creator_and_fail_parent_sync(descriptor: int) -> None:
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (parent.st_dev, parent.st_ino):
            if not creator_waiting.is_set():
                creator_waiting.set()
                if not release_creator.wait(timeout=10):
                    raise TimeoutError("test did not release constructor")
            raise OSError(errno.EIO, "concurrent parent sync failure")
        real_fsync(descriptor)

    def construct() -> bool:
        try:
            if storage == "artifact":
                FilesystemProtectedArtifacts(root, gate, evidence_refs=("a",)).put(SCOPE, PAYLOAD)
            else:
                SqliteReceiptStore(root / "r.sqlite3", gate, existing_artifacts)
        except (ArtifactStoreError, ReceiptStoreError):
            return False
        return True

    with monkeypatch.context() as patch:
        patch.setattr(os, "fsync", pause_creator_and_fail_parent_sync)
        with ThreadPoolExecutor(max_workers=2) as executor:
            creator = executor.submit(construct)
            try:
                assert creator_waiting.wait(timeout=10)
                assert root.is_dir()
                # The first constructor has created the child but cannot yet
                # establish durability; the second sees exactly that window.
                observer_succeeded = executor.submit(construct).result(timeout=10)
            finally:
                release_creator.set()
            assert not creator.result(timeout=10)
    assert not observer_succeeded
    assert list(root.iterdir()) == []
    assert construct()
