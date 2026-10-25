"""The installed wheel runs capture -> task -> submit -> preview -> review -> apply -> ask.

Every step goes through the installed ``capicove wiki maintain`` command; only the
explicit v3 preparation and the submission bytes are produced with the installed
package's public Python modules. The reviewer is a synthetic protocol fixture, not
semantic acceptance.
"""

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from tests.e2e.wheel_cache import WheelCache
from tests.wiki_maintenance_support.oracle import snapshot


pytestmark = pytest.mark.e2e

_SUBMISSION = """
import json, sys
from capicove.contracts.wiki_maintenance import (
    ClaimDraft, CompilationSubmission, MaintenanceRecord, PageDraft, Section,
)
from capicove.wiki.maintenance.codec import encode_record
task = json.loads(sys.argv[1])
claim = ClaimDraft(
    draft_id="claim-one", text="Plants use light.", evidence_aliases=(task["evidence"][0]["alias"],),
    status="current", knowledge="asserted", premise_ids=(), derivation=None,
)
page = PageDraft(
    draft_id="plant", operation="create", page_id=None, title="Plants", kind="concept",
    aliases=(), sections=(Section(heading="Overview", claim_ids=("claim-one",)),),
    claims=(claim,), relations=(),
)
value = CompilationSubmission(
    task_id=task["task_id"], task_version=task["task_version"], idempotency_key="wheel-one",
    dispositions=("new",), rationale="New source-backed concept.", pages=(page,), impact=(),
    provenance=None,
)
record = MaintenanceRecord(schema_version=1, record_type="compilation_submission", payload=value)
sys.stdout.write(encode_record(record).decode())
"""


def _maintain(
    executable: str, workspace: Path, operation: str, request: dict[str, Any] | None = None
) -> dict[str, Any]:
    arguments = [executable, "wiki", "maintain", operation, str(workspace)]
    if request is not None:
        path = workspace.parent / f"request-{operation}.json"
        path.write_text(json.dumps(request))
        arguments += ["--request", str(path), "--source-root", str(workspace.parent / "sources")]
    completed = subprocess.run(arguments, capture_output=True, text=True, timeout=120, check=False)
    assert completed.returncode == 0, completed.stdout + completed.stderr
    value = json.loads(completed.stdout)
    assert value["status"] == "succeeded", value
    return dict(value["result"])


@pytest.mark.slow
def test_installed_wheel_runs_the_reviewed_maintenance_journey(
    tmp_path: Path, wheel_cache: WheelCache
) -> None:
    installed = wheel_cache.install(("wiki",))
    python, executable = installed.python, installed.executable
    workspace = tmp_path / "workspace"
    sources = tmp_path / "sources"
    sources.mkdir()
    source = sources / "plants.md"
    source.write_bytes(b"Plants use light.\n")
    subprocess.run(
        [executable, "wiki", "init", str(workspace), "--date", "2026-10-23"],
        check=True,
        capture_output=True,
        timeout=120,
    )
    subprocess.run(
        [
            python,
            "-c",
            "import sys; from pathlib import Path; "
            "from capicove.wiki.maintenance.migration import migrate_v3; "
            "migrate_v3(Path(sys.argv[1]), preview=False)",
            str(workspace / "wiki"),
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    pages = workspace / "wiki" / "pages"
    pages_before = snapshot(pages)
    captured = _maintain(executable, workspace, "capture", {"source": str(source)})
    task = _maintain(
        executable, workspace, "task", {"source_ids": [captured["source_id"]], "query": "Plants"}
    )
    wire = subprocess.run(
        [python, "-c", _SUBMISSION, json.dumps(task)],
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    ).stdout
    proposal = _maintain(executable, workspace, "submit", {"submission": wire})
    preview = _maintain(executable, workspace, "preview", {"proposal_id": proposal["proposal_id"]})
    assert "Plants use light." in preview["diff"]
    decisions = [
        {"subject_id": subject, "verdict": "supported", "reason": "Synthetic protocol check."}
        for subject in preview["review_subjects"]
    ]
    review = _maintain(
        executable,
        workspace,
        "review",
        {"proposal_id": proposal["proposal_id"], "decisions": decisions, "accept": True},
    )
    assert review["accepted"] is True
    assert snapshot(pages) == pages_before
    published = _maintain(
        executable,
        workspace,
        "apply",
        {"proposal_id": proposal["proposal_id"], "review_id": review["review_id"]},
    )
    assert published["status"] != "aborted"
    created = sorted(set(snapshot(pages)) - set(pages_before))
    assert len(created) == 1 and created[0].startswith("concepts/")
    status = _maintain(executable, workspace, "status")
    assert status["wiki_revision"] == published["wiki_revision"]
    answer = _maintain(
        executable, workspace, "ask", {"question": "Plants", "include_history": False}
    )
    assert answer["answer"]["status"] == "evidence_only"
    raw = next((workspace / "wiki" / "raw").glob("*.md"))
    assert raw.read_bytes() == source.read_bytes()
