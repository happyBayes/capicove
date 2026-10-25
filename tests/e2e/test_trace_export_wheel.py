"""Exercise Trace export validation from an isolated installed wheel."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import WheelCache


pytestmark = [pytest.mark.e2e, pytest.mark.slow]


def _run(
    arguments: list[str], *, environment: dict[str, str], cwd: Path | None = None
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        arguments,
        capture_output=True,
        text=True,
        timeout=180,
        env=environment,
        cwd=cwd,
        check=False,
    )


def test_installed_trace_export_validate_and_inspect(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    """A clean wheel can validate its own package and reject an unsafe alias."""
    environment = {**os.environ, "UV_OFFLINE": "1"}
    environment.pop("PYTHONPATH", None)
    installed = wheel_cache.install()
    python, command = installed.python, installed.executable

    store = tmp_path / "traces.sqlite3"
    seeded = _run(
        [
            python,
            "-c",
            """
from datetime import UTC, datetime
from pathlib import Path
import sys
from capicove.traces import EpisodeCompleted, EpisodeStarted, EpisodeStatus, TraceRecordDraft
from capicove.traces._adapters.sqlite import SqliteTraceStore

store = SqliteTraceStore(Path(sys.argv[1]))
now = datetime(2026, 9, 30, tzinfo=UTC)
store.append(TraceRecordDraft(
    episode_id="wheel-episode", trace_id="wheel-trace", occurred_at=now,
    payload=EpisodeStarted(
        goal="private wheel trace content", workspace_id="wheel-workspace",
        actor_id="wheel-actor", environment_id="local", environment_version=1,
        tool_catalog_sha256="a" * 64, active_asset_versions={},
    ),
), idempotency_key="start")
store.append(TraceRecordDraft(
    episode_id="wheel-episode", trace_id="wheel-trace", occurred_at=now,
    payload=EpisodeCompleted(EpisodeStatus.SUCCEEDED),
), idempotency_key="complete")
""",
            str(store),
        ],
        environment=environment,
        cwd=tmp_path,
    )
    assert seeded.returncode == 0, seeded.stderr

    output = tmp_path / "trace-export"
    exported = _run(
        [
            command,
            "traces",
            "export",
            "--store",
            str(store),
            "--output",
            str(output),
            "--format",
            "json",
        ],
        environment=environment,
        cwd=tmp_path,
    )
    assert exported.returncode == 0, exported.stderr
    manifest = json.loads(exported.stdout)
    assert manifest["records"] == 2
    assert manifest["episodes"] == 1
    assert "private wheel trace content" not in exported.stdout

    validated = _run(
        [command, "traces", "validate", str(output), "--output", "json"],
        environment=environment,
        cwd=tmp_path,
    )
    assert validated.returncode == 0, validated.stderr
    assert json.loads(validated.stdout)["valid"] is True
    inspected = _run(
        [command, "traces", "inspect", str(output), "--output", "json"],
        environment=environment,
        cwd=tmp_path,
    )
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["export_id"] == manifest["export_id"]

    if os.name == "posix":
        alias = tmp_path / "alias"
        alias.symlink_to(output, target_is_directory=True)
        denied = _run(
            [command, "traces", "validate", str(alias), "--output", "json"],
            environment=environment,
            cwd=tmp_path,
        )
        assert denied.returncode == 1, denied.stderr
        assert any(
            issue["code"] == "EXPORT_UNSAFE_PATH" for issue in json.loads(denied.stdout)["issues"]
        )
        denied_inspect = _run(
            [command, "traces", "inspect", str(alias), "--output", "json"],
            environment=environment,
            cwd=tmp_path,
        )
        assert denied_inspect.returncode == 2
        assert str(output) not in denied_inspect.stderr

    if os.name != "posix":
        return

    operator_root = tmp_path / "operator-exports"
    operator_case = _run(
        [
            python,
            "-c",
            """
import json
import hashlib
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
import capicove
from capicove.export import TraceExporter, TraceExportOptions
from capicove.traces import EpisodeStarted, TraceQuery, TraceRecordDraft
from capicove.traces._adapters.sqlite import SqliteTraceStore
from capicove.traces.operator import TraceOperatorExporter

store = SqliteTraceStore(Path(sys.argv[1]))
root = Path(sys.argv[2])
assert Path(capicove.__file__).resolve().is_relative_to(Path(sys.prefix).resolve())
root.mkdir(mode=0o700)
operator = TraceOperatorExporter(
    reader=store,
    workspace_id="wheel-workspace",
    output_roots={"private": root},
    permissions=frozenset({"trace:export"}),
)
first = operator.export(
    output_root_alias="private", query=TraceQuery(), idempotency_key="owned"
)
repeated = operator.export(
    output_root_alias="private", query=TraceQuery(), idempotency_key="owned"
)
store.append(TraceRecordDraft(
    episode_id="foreign-wheel-episode", trace_id="foreign-wheel-trace",
    occurred_at=datetime(2026, 9, 30, tzinfo=UTC),
    payload=EpisodeStarted(
        goal="private foreign wheel trace content", workspace_id="foreign-workspace",
        actor_id="wheel-actor", environment_id="local", environment_version=1,
        tool_catalog_sha256="a" * 64, active_asset_versions={},
    ),
), idempotency_key="foreign-start")
denied = False
try:
    operator.export(
        output_root_alias="private", query=TraceQuery(include_incomplete=True),
        idempotency_key="foreign",
    )
except ValueError:
    denied = True
foreign_key = "preexisting-foreign"
foreign_digest = hashlib.sha256(
    f"wheel-workspace\\0private\\0{foreign_key}".encode()
).hexdigest()[:32]
TraceExporter(
    store,
    clock=lambda: datetime(2026, 9, 30, tzinfo=UTC),
    export_id_factory=lambda: f"export:{foreign_digest}",
).export(
    root / f"trace-{foreign_digest}",
    options=TraceExportOptions(query=TraceQuery(include_incomplete=True)),
)
foreign_retry_denied = False
try:
    operator.export(
        output_root_alias="private", query=TraceQuery(include_incomplete=True),
        idempotency_key=foreign_key,
    )
except ValueError:
    foreign_retry_denied = True
print(json.dumps({
    "result_fields": sorted(asdict(first)),
    "owned_episodes": first.included_episode_count,
    "retry_matches": repeated == first,
    "foreign_denied": denied,
    "foreign_retry_denied": foreign_retry_denied,
    "package_count": len(tuple(root.iterdir())),
}))
""",
            str(store),
            str(operator_root),
        ],
        environment=environment,
        cwd=tmp_path,
    )
    assert operator_case.returncode == 0, operator_case.stderr
    operator_result = json.loads(operator_case.stdout)
    assert operator_result == {
        "result_fields": [
            "export_id",
            "included_episode_count",
            "incomplete_episode_count",
            "media_count",
            "media_mode",
            "missing_media_count",
            "omitted_restricted_episode_count",
            "record_count",
            "shard_count",
        ],
        "owned_episodes": 1,
        "retry_matches": True,
        "foreign_denied": True,
        "foreign_retry_denied": True,
        "package_count": 2,
    }
    assert "private foreign wheel trace content" not in operator_case.stdout
