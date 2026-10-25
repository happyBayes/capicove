"""RFC-0020 acceptance: loopback HTTP from a clean wheel with only the `api` extra."""

from __future__ import annotations

import json
import os
import signal
import subprocess
from pathlib import Path

import httpx
import pytest

from tests.e2e.test_retrieval_wheel import install_wheel, run
from tests.e2e.wheel_cache import WheelCache
from tests.unit.cli.test_http import reap_server_descendants, start
from tests.unit.runtime.test_local_retrieval import retrieval_config_body


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(os.name != "posix", reason="HTTP serve is POSIX-only (ADR-0025)"),
    # Kill the server and its separate-session worker even if an assertion fails.
    pytest.mark.usefixtures(reap_server_descendants.__name__),
]


def serve(
    capicove: str, config: Path, cwd: Path
) -> tuple[subprocess.Popen[str], dict[str, object]]:
    process, ready, output = start(config, executable=capicove, cwd=cwd)
    assert ready is not None, output
    return process, ready


@pytest.mark.slow
def test_http_retrieval_journey_from_installed_wheel(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    python, capicove = install_wheel(wheel_cache, extra="api")
    probe = subprocess.run(
        [python, "-c", "import importlib.util as u; print(u.find_spec('mcp') is None)"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert probe.stdout.strip() == "True"
    operator = tmp_path / "operator"
    operator.mkdir()
    workspace = tmp_path / "workspace"
    workspace.mkdir(mode=0o700)
    workspace.chmod(0o700)
    notes = tmp_path / "notes"
    notes.mkdir()
    (notes / "travel.md").write_text("# Travel\n\nAlice moved to Paris.\n", encoding="utf-8")
    config = workspace / "host.json"
    config.write_text(json.dumps(retrieval_config_body(workspace, notes, workspace / "index")))
    config.chmod(0o600)
    runtime = ("--runtime-config", str(config))
    run(
        capicove,
        operator,
        "source",
        "import",
        *runtime,
        "--root",
        "notes",
        "--relative-path",
        "travel.md",
    )
    run(capicove, operator, "index", "build", *runtime)

    evidence_ids = []
    for _ in range(2):  # The second pass proves a clean restart with a rotated token.
        process, ready = serve(capicove, config, operator)
        base = f"http://{ready['address']}:{ready['port']}"
        token = Path(str(ready["token_file"])).read_text().strip()
        headers = {"Authorization": f"Bearer {token}"}
        assert httpx.get(f"{base}/v1/tools").status_code == 401
        found = httpx.post(f"{base}/v1/search", headers=headers, json={"query": "Paris"}).json()
        evidence_id = found["evidence_refs"][0]
        opened = httpx.post(
            f"{base}/v1/evidence", headers=headers, json={"evidence_id": evidence_id}
        ).json()
        assert "Alice moved to Paris" in opened["result"]["metadata"]["text"]
        evidence_ids.append(evidence_id)
        process.send_signal(signal.SIGTERM)
        assert process.wait(15) == 0
        assert not Path(str(ready["token_file"])).exists()
    assert evidence_ids[0] == evidence_ids[1]
