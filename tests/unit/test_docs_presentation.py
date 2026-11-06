"""The visual-refresh gate must reject changes to published documentation."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("change", ["none", "text", "link", "navigation", "page", "search"])
def test_presentation_gate(tmp_path: Path, change: str) -> None:
    before = tmp_path / "before"
    after = tmp_path / "after"
    original = (
        '<header>Capicove</header><main><nav><a href="guide/">Guide</a></nav>'
        '<article><h1 id="title">Title</h1><p>Original text</p></article></main>'
        "<footer>Next</footer>"
    )
    for directory in (before, after):
        (directory / "search").mkdir(parents=True)
        (directory / "index.html").write_text(original)
        (directory / "search/search_index.json").write_text('{"docs": []}')
    changed = original.replace("<main>", '<div class="hero">Photo</div><main>')
    if change == "text":
        changed = changed.replace("Original text", "Rewritten text")
    elif change == "link":
        changed = changed.replace('href="guide/"', 'href="elsewhere/"')
    elif change == "navigation":
        changed = changed.replace("Guide</a>", "Renamed</a>")
    elif change == "page":
        (after / "extra.html").write_text(original)
    elif change == "search":
        (after / "search/search_index.json").write_text('{"docs": ["changed"]}')
    (after / "index.html").write_text(changed)
    script = Path(__file__).parents[2] / "scripts/check_docs_presentation.py"
    result = subprocess.run(
        [sys.executable, str(script), str(before), str(after)],
        capture_output=True,
        text=True,
        check=False,
        timeout=10,
    )
    assert result.returncode == (0 if change == "none" else 1), result.stderr
