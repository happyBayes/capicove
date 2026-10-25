"""RFC-0018 amendment 1: source retirement, ledger version 2, and bounded builds."""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
import textwrap
import threading
import time
from pathlib import Path

import pytest

from capicove.cli.app import main
from tests.unit.runtime.test_local_retrieval import (
    Workspace,
    build,
    evidence,
    first_hit,
    import_source,
    last_stderr,
    run_json,
    search,
)
from tests.unit.runtime.test_local_retrieval import (
    workspace as workspace,  # re-exported pytest fixture
)


pytestmark = pytest.mark.skipif(os.name != "posix", reason="retrieval is POSIX-only (ADR-0024)")


def forget(
    capsys: pytest.CaptureFixture[str], space: Workspace, relative: str, root: str = "notes"
) -> tuple[int, object]:
    return run_json(
        capsys,
        [
            "source",
            "forget",
            "--runtime-config",
            str(space.config),
            "--root",
            root,
            "--relative-path",
            relative,
        ],
    )


def ledger(space: Workspace) -> sqlite3.Connection:
    connection = sqlite3.connect(space.index / "imports.sqlite3")
    connection.row_factory = sqlite3.Row
    return connection


def ledger_events(space: Workspace) -> list[tuple[str, str]]:
    with ledger(space) as connection:
        return [
            (row["event"], row["source_ref"])
            for row in connection.execute(
                "SELECT event, source_ref FROM ledger_events ORDER BY sequence"
            )
        ]


def hits(payload: dict[str, object]) -> list[object]:
    result = payload["result"]
    assert isinstance(result, dict)
    found = result["hits"]
    assert isinstance(found, list)
    return found


def test_forget_then_build_removes_hits_and_stales_evidence(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace.notes / "lyon.md").write_text("Bob lives in Lyon.\n", encoding="utf-8")
    import_source(capsys, workspace, "paris.md")
    import_source(capsys, workspace, "lyon.md")
    _, first = build(capsys, workspace)
    _, found = search(capsys, workspace, "Paris")
    old_evidence = str(first_hit(found)["evidence_id"])
    before = (workspace.notes / "paris.md").read_bytes()
    generations_before = sorted(path.name for path in workspace.index.glob("generations/*"))

    code, forgotten = forget(capsys, workspace, "paris.md")
    assert code == 0
    assert forgotten == {"source_ref": "notes/paris.md", "forgotten": True}
    _, still_active = search(capsys, workspace, "Paris")
    assert hits(still_active), "retirement changes nothing until the next build"

    code, second = build(capsys, workspace)
    assert code == 0
    assert isinstance(first, dict) and isinstance(second, dict)
    assert second["generation_id"] != first["generation_id"]
    _, gone = search(capsys, workspace, "Paris")
    assert hits(gone) == []
    _, kept = search(capsys, workspace, "Lyon")
    assert hits(kept)
    code, stale = evidence(capsys, workspace, old_evidence)
    assert code == 1 and stale["status"] == "failed"
    assert (workspace.notes / "paris.md").read_bytes() == before
    generations_after = {path.name for path in workspace.index.glob("generations/*")}
    assert set(generations_before) <= generations_after


def test_repeated_forget_is_a_no_op(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    forget(capsys, workspace, "paris.md")
    events = ledger_events(workspace)

    code, again = forget(capsys, workspace, "paris.md")
    code_unknown, never = forget(capsys, workspace, "never-imported.md")

    assert code == 0 and again == {"source_ref": "notes/paris.md", "forgotten": False}
    assert code_unknown == 0 and isinstance(never, dict) and never["forgotten"] is False
    assert ledger_events(workspace) == events


def test_forgetting_final_source_builds_an_empty_generation(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    build(capsys, workspace)
    _, found = search(capsys, workspace, "Paris")
    old_evidence = str(first_hit(found)["evidence_id"])
    forget(capsys, workspace, "paris.md")

    code, empty = build(capsys, workspace)

    assert code == 0
    assert isinstance(empty, dict)
    assert empty["sources"] == 0 and empty["text_units"] == 0 and empty["activated"] is True
    code, payload = search(capsys, workspace, "Paris")
    assert code == 0 and hits(payload) == []
    code, _ = evidence(capsys, workspace, old_evidence)
    assert code == 1


def test_never_imported_workspace_keeps_its_build_error(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    code, _ = build(capsys, workspace)

    assert code == 2
    assert "no sources have been imported" in last_stderr()


def test_deleted_source_needs_forget_before_build(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace.notes / "lyon.md").write_text("Bob lives in Lyon.\n", encoding="utf-8")
    import_source(capsys, workspace, "paris.md")
    import_source(capsys, workspace, "lyon.md")
    build(capsys, workspace)
    (workspace.notes / "paris.md").unlink()

    code, _ = build(capsys, workspace)
    assert code == 2
    assert "changed since import" in last_stderr()

    code, forgotten = forget(capsys, workspace, "paris.md")
    assert code == 0 and isinstance(forgotten, dict) and forgotten["forgotten"] is True
    code, _ = build(capsys, workspace)
    assert code == 0
    _, kept = search(capsys, workspace, "Lyon")
    assert hits(kept)


def test_identical_reimport_after_forget_restores_eligibility(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    _, original = build(capsys, workspace)
    forget(capsys, workspace, "paris.md")
    build(capsys, workspace)

    code, reimported = import_source(capsys, workspace, "paris.md")
    assert code == 0 and isinstance(reimported, dict) and reimported["changed"] is True
    code, restored = build(capsys, workspace)

    assert code == 0
    assert isinstance(original, dict) and isinstance(restored, dict)
    assert restored["generation_id"] == original["generation_id"]
    _, found = search(capsys, workspace, "Paris")
    assert hits(found)
    assert [event for event, _ in ledger_events(workspace)] == ["import", "retire", "import"]


@pytest.mark.parametrize(
    ("relative", "root", "profile", "message"),
    [
        ("../escape.md", "notes", "write", "normalized"),
        ("/etc/passwd", "notes", "write", "normalized"),
        ("paris.md", "other", "write", "not configured"),
        ("paris.md", "notes", "read", "write profile"),
    ],
)
def test_forget_rejects_unsafe_requests_without_writes(
    workspace: Workspace,
    capsys: pytest.CaptureFixture[str],
    relative: str,
    root: str,
    profile: str,
    message: str,
) -> None:
    import_source(capsys, workspace, "paris.md")
    events = ledger_events(workspace)
    workspace.write_config(profile=profile)

    code, payload = forget(capsys, workspace, relative, root=root)

    assert code == 2 and payload is None
    assert message in last_stderr()
    assert ledger_events(workspace) == events


def _write_version_one_ledger(space: Workspace, digest: str, size: int) -> None:
    space.index.mkdir(mode=0o700)
    path = space.index / "imports.sqlite3"
    os.close(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600))
    with sqlite3.connect(path) as connection:
        connection.execute(
            "CREATE TABLE imports (sequence INTEGER PRIMARY KEY AUTOINCREMENT, "
            "source_ref TEXT NOT NULL, alias TEXT NOT NULL, relative_path TEXT NOT NULL, "
            "sha256 TEXT NOT NULL, size_bytes INTEGER NOT NULL, media_type TEXT NOT NULL, "
            "import_version TEXT NOT NULL, imported_at TEXT NOT NULL)"
        )
        connection.execute(
            "INSERT INTO imports (source_ref, alias, relative_path, sha256, size_bytes, "
            "media_type, import_version, imported_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "notes/paris.md",
                "notes",
                "paris.md",
                digest,
                size,
                "text/markdown",
                "capicove-local-import-v1",
                "2026-10-04T00:00:00+00:00",
            ),
        )


def test_version_one_ledger_migrates_on_next_write(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import hashlib

    payload = (workspace.notes / "paris.md").read_bytes()
    _write_version_one_ledger(workspace, hashlib.sha256(payload).hexdigest(), len(payload))

    code, forgotten = forget(capsys, workspace, "paris.md")

    assert code == 0 and isinstance(forgotten, dict) and forgotten["forgotten"] is True
    assert ledger_events(workspace) == [("import", "notes/paris.md"), ("retire", "notes/paris.md")]
    with ledger(workspace) as connection:
        version = connection.execute(
            "SELECT value FROM ledger_meta WHERE key = 'ledger_version'"
        ).fetchone()
        assert version[0] == "2"
        with pytest.raises(sqlite3.Error):
            connection.execute("SELECT sha256 FROM imports").fetchall()
        with pytest.raises(sqlite3.Error):
            connection.execute("INSERT INTO imports (source_ref) VALUES ('x')")


def test_interrupted_migration_keeps_version_one_ledger(
    workspace: Workspace, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    import hashlib

    import capicove.runtime.local_retrieval as runtime

    payload = (workspace.notes / "paris.md").read_bytes()
    _write_version_one_ledger(workspace, hashlib.sha256(payload).hexdigest(), len(payload))

    def interrupted(connection: sqlite3.Connection) -> None:
        raise sqlite3.OperationalError("simulated crash after copying events")

    monkeypatch.setattr(runtime, "_create_sentinel_view", interrupted)

    code, _ = forget(capsys, workspace, "paris.md")

    assert code == 2
    with ledger(workspace) as connection:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert "imports" in tables
        assert "ledger_events" not in tables
        assert connection.execute("SELECT count(*) FROM imports").fetchone()[0] == 1


def test_unknown_ledger_version_fails_closed(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    with ledger(workspace) as connection:
        connection.execute("UPDATE ledger_meta SET value = '3' WHERE key = 'ledger_version'")
    events = ledger_events(workspace)

    code, _ = forget(capsys, workspace, "paris.md")

    assert code == 2
    assert "ledger version" in last_stderr()
    assert ledger_events(workspace) == events


def test_ledger_mutations_wait_for_the_index_lock(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import fcntl

    import_source(capsys, workspace, "paris.md")
    lock = os.open(workspace.index / ".build.lock", os.O_WRONLY)
    fcntl.flock(lock, fcntl.LOCK_EX)
    finished = threading.Event()
    codes: list[int] = []

    def run() -> None:
        codes.append(
            main(
                [
                    "source",
                    "forget",
                    "--runtime-config",
                    str(workspace.config),
                    "--root",
                    "notes",
                    "--relative-path",
                    "paris.md",
                ]
            )
        )
        finished.set()

    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert not finished.wait(0.5), "forget must wait while another process holds the lock"
        assert ledger_events(workspace) == [("import", "notes/paris.md")]
    finally:
        os.close(lock)
    worker.join(10)
    capsys.readouterr()
    assert codes == [0]
    assert ledger_events(workspace)[-1] == ("retire", "notes/paris.md")


PEAK_RSS_HELPER = """
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


def _markdown_source(path: Path, seed: int, size: int) -> None:
    paragraph = f"Paragraph {seed} discusses lineage, evidence, and immutable sources. " * 4
    block = (paragraph + "\n\n").encode()
    with path.open("wb") as target:
        written = 0
        index = 0
        while written < size:
            chunk = block.replace(b"Paragraph", f"Note{index:07d}".encode(), 1)
            target.write(chunk)
            written += len(chunk)
            index += 1


@pytest.mark.slow
@pytest.mark.qualification
def test_build_memory_is_bounded_by_one_source(workspace: Workspace, tmp_path: Path) -> None:
    sixteen_mib = 16 * 1024 * 1024
    workspace.write_config(retrieval_max_source_bytes=sixteen_mib)
    for index in range(8):
        _markdown_source(workspace.notes / f"large-{index}.md", index, sixteen_mib - 65536)
    script = PEAK_RSS_HELPER + textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        from capicove.runtime.local_retrieval import build_index, import_source
        from capicove.runtime.mcp_config import load_mcp_host_config
        config = load_mcp_host_config(Path({str(workspace.config)!r}))
        for index in range(8):
            import_source(config, alias="notes", relative_path=f"large-{{index}}.md")
        baseline = peak_kib(RESET)
        result = build_index(config)
        peak = peak_kib(RESET)
        print(json.dumps({{"baseline_kib": baseline, "peak_kib": peak,
                          "units": result["text_units"]}}))
        """
    )
    started = time.perf_counter()
    # Measure only the build: pytest-cov would otherwise trace the child process too.
    environment = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("COV_CORE_", "COVERAGE_"))
    }
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
        env=environment,
    )
    assert completed.returncode == 0, completed.stderr
    measured = json.loads(completed.stdout)
    assert measured["units"] > 0
    assert measured["peak_kib"] < 200 * 1024, (measured, time.perf_counter() - started)


# Review findings (2026-10-06)

# Produced by the previous release (commit d8c8e30) for the fixture below.
PREVIOUS_RELEASE_GENERATION = {
    "generation_id": "local-1a57b1c3cf9906c93f2ea41a95c8f0dcd2cb38ecc25904bfe734942391f61bfa",
    "searchable_unit_manifest_sha256": (
        "abd35caacf7a81ded4efe9916d83044fee1c05a8829c781000b8c3aadbdf127e"  # pragma: allowlist secret
    ),
    "content_sha256": "b49e2eb2cb81fd0b4a0195cba200bcc4c5f599c5e9c6f8f0377abcaf09769ec9",  # pragma: allowlist secret
}


def test_generation_identity_matches_the_previous_release(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    for path in workspace.notes.iterdir():
        path.unlink()
    (workspace.notes / "a.md").write_text(
        "# Alpha\n\nFirst paragraph about lineage.\n\nSecond paragraph.\n", encoding="utf-8"
    )
    (workspace.notes / "b.md").write_text("Beta note.\n", encoding="utf-8")
    workspace.write_config(workspace_id="workspace", retrieval_corpus_ref="corpus:golden")
    for name in ("a.md", "b.md"):
        import_source(capsys, workspace, name)

    code, built = build(capsys, workspace)

    assert code == 0, last_stderr()
    assert isinstance(built, dict)
    assert built["generation_id"] == PREVIOUS_RELEASE_GENERATION["generation_id"]
    with sqlite3.connect(workspace.index / "catalog.sqlite3") as connection:
        (record,) = [
            json.loads(row[0])
            for row in connection.execute("SELECT record_json FROM text_index_generations")
        ]
    manifest = record["manifest"]
    for field in ("searchable_unit_manifest_sha256", "content_sha256"):
        assert manifest[field] == PREVIOUS_RELEASE_GENERATION[field], field


def test_deleted_empty_source_still_fails_the_build(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    (workspace.notes / "empty.md").write_bytes(b"")
    import_source(capsys, workspace, "paris.md")
    code, _ = import_source(capsys, workspace, "empty.md")
    assert code == 0
    (workspace.notes / "empty.md").unlink()

    code, _ = build(capsys, workspace)

    assert code == 2
    assert "changed since import" in last_stderr()


@pytest.mark.slow
@pytest.mark.qualification
def test_build_memory_does_not_grow_with_the_number_of_sources(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import tracemalloc

    from capicove.runtime.local_retrieval import build_index
    from capicove.runtime.mcp_config import WorkspaceHostConfiguration, load_mcp_host_config

    paragraph = "Short paragraph about immutable evidence and lineage. " * 40 + "\n\n"
    body = paragraph * (256 * 1024 // len(paragraph))

    def peak_for(count: int, space: Workspace) -> int:
        for index in range(count):
            (space.notes / f"n{count}-{index}.md").write_text(body, encoding="utf-8")
            import_source(capsys, space, f"n{count}-{index}.md")
        config = load_mcp_host_config(space.config)
        assert isinstance(config, WorkspaceHostConfiguration)
        tracemalloc.start()
        try:
            build_index(config)
            return tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()

    few = peak_for(10, workspace)
    many = peak_for(150, workspace)

    assert many < few * 1.5, (few, many)


def test_a_fifo_index_lock_fails_instead_of_blocking(
    workspace: Workspace, capsys: pytest.CaptureFixture[str]
) -> None:
    import_source(capsys, workspace, "paris.md")
    lock = workspace.index / ".build.lock"
    lock.unlink()
    os.mkfifo(lock, 0o600)
    script = (
        "import sys; from capicove.cli.app import main; "
        f"sys.exit(main(['index', 'build', '--runtime-config', {str(workspace.config)!r}]))"
    )

    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=30
    )

    assert completed.returncode == 2
    assert "lock" in completed.stderr
