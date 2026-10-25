"""Independent CLI client exercises the same host-bound maintenance handlers."""

import json
import subprocess
import sys
from pathlib import Path

from tests.integration.wiki.test_maintenance_publication import prepared
from tests.wiki_maintenance_support.oracle import snapshot


def test_cli_status_and_offline_ask(tmp_path: Path) -> None:
    service, proposal, review = prepared(tmp_path)
    service.apply(proposal.proposal_id, review.review_id)
    service.build_discovery()
    before = snapshot(service.wiki)
    cases: tuple[tuple[str, ...], ...] = (
        ("maintain", "status", str(tmp_path)),
        ("ask", str(tmp_path), "Plants"),
    )
    for arguments in cases:
        result = subprocess.run(
            [sys.executable, "-m", "capicove", "wiki", *arguments],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        value = json.loads(result.stdout)
        assert value["status"] == "succeeded"
        if arguments[0] == "ask":
            assert value["result"]["answer"]["status"] == "evidence_only"
    assert snapshot(service.wiki) == before
