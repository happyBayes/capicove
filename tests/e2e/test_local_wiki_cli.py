import json
import subprocess
import sys
from pathlib import Path


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "capicove", "wiki", *arguments],
        check=False,
        capture_output=True,
        text=True,
    )


def test_installed_cli_completes_the_local_wiki_journey(tmp_path: Path) -> None:
    source = tmp_path / "knowledge.md"
    source.write_text("# Knowledge\n\nOffline evidence remains searchable.\n", encoding="utf-8")

    initialized = _run("init", str(tmp_path), "--date", "2026-09-01")
    ingested = _run("ingest", str(tmp_path), str(source), "--date", "2026-09-01")
    built = _run("build", str(tmp_path), "--date", "2026-09-01")
    queried = _run("query", str(tmp_path), "offline evidence")

    assert initialized.returncode == 0, initialized.stderr
    assert ingested.returncode == 0, ingested.stderr
    assert built.returncode == 0, built.stderr
    assert queried.returncode == 0, queried.stderr
    payload = json.loads(queried.stdout)
    assert payload["hits"][0]["title"] == "Knowledge"
    assert payload["hits"][0]["source_ids"]
