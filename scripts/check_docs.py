"""Check maintained documentation examples and generated local HTML, offline."""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import re
import shlex
import subprocess
import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs/site"
SITE = ROOT / "site"


def run_cli(arguments: list[str]) -> subprocess.CompletedProcess[str]:
    """Run only explicitly selected CLI probes in the current environment."""
    return subprocess.run(
        [sys.executable, "-m", "capicove", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )


def marked_example(path: str, name: str, language: str) -> str:
    """Read one maintained example, never arbitrary website code."""
    content = (DOCS / path).read_text()
    match = re.search(rf"<!-- docs-smoke: {name} -->\s*```{language}\n(.*?)\n```", content, re.S)
    assert match, f"Missing maintained example: {name}"
    return match.group(1)


class Page(HTMLParser):
    """Collect rendered targets and links without a browser dependency."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.ids: set[str] = set()
        self.links: list[str] = []
        self.feed(path.read_text())

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = dict(attrs)
        if values.get("id"):
            self.ids.add(str(values["id"]))
        if tag == "a" and values.get("href"):
            self.links.append(str(values["href"]))


def check_examples() -> None:
    """Check actual example output and its immutable source identity."""
    fixture = DOCS / "assets/quickstart.ppm"
    before = hashlib.sha256(fixture.read_bytes()).hexdigest()
    command = shlex.split(
        marked_example("getting-started/quickstart.md", "image", "bash").replace("\\\n", "")
    )
    assert command[:4] == ["uv", "run", "--no-sync", "capicove"]
    result = run_cli(command[4:])
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "succeeded" and payload["error"] is None
    assert payload["result"]["keep"] is True
    assert payload["result"]["metrics"][0]["values"] == [1.0]
    assert payload["result"]["lineage"]["source_hashes"] == [before]
    assert hashlib.sha256(fixture.read_bytes()).hexdigest() == before
    missing = run_cli(
        [
            "image",
            "run",
            "--tool",
            "image_aspect_ratio_filter",
            "--input",
            str(fixture.parent / "missing-docs-fixture.ppm"),
            "--output",
            "json",
        ]
    )
    assert missing.returncode != 0, "Missing input must not report success"

    memory = subprocess.run(
        [sys.executable, "-c", marked_example("api/memory.md", "memory", "python")],
        cwd=ROOT,
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert memory.returncode == 0, memory.stderr

    # Bind the real documented call to the public signature without needing data.
    from capicove.processing import trace_export_snapshot

    guide = (DOCS / "guides/data-processing.md").read_text()
    blocks = re.findall(r"```python\n(.*?)\n```", guide, re.S)
    calls = [
        node
        for block in blocks
        for node in ast.walk(ast.parse(block))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "trace_export_snapshot"
    ]
    assert len(calls) == 1
    inspect.signature(trace_export_snapshot).bind(
        Path("exports/trace-package"),
        **{kw.arg: ast.literal_eval(kw.value) for kw in calls[0].keywords if kw.arg},
    )


def check_commands() -> None:
    """Ensure the overview's documented commands accept --help."""
    overview = (DOCS / "cli/index.md").read_text()
    commands = re.findall(r"^capicove (.+)$", overview, re.M)
    assert commands
    for command in commands:
        args = shlex.split(command)
        if "--help" not in args:
            args.append("--help")
        result = run_cli(args)
        assert result.returncode == 0, f"{command}: {result.stderr}"
    top = run_cli(["--help"])
    registered = set(re.search(r"\{([^}]+)\}", top.stdout).group(1).split(","))
    assert registered <= {shlex.split(command)[0] for command in commands}


def check_html() -> None:
    """Verify local links, fragments, public symbols and readable formulas."""
    pages = {path.resolve(): Page(path) for path in SITE.rglob("*.html")}
    assert pages, "Build the site first"
    for path, page in pages.items():
        for link in page.links:
            parts = urlsplit(link)
            if parts.scheme or parts.netloc:
                continue
            if parts.path.startswith("/"):
                target = (SITE / unquote(parts.path).lstrip("/")).resolve()
            else:
                target = (path.parent / unquote(parts.path)).resolve() if parts.path else path
            if target.is_dir():
                target /= "index.html"
            assert target.exists(), f"{path.relative_to(SITE)}: missing {link}"
            if parts.fragment and target in pages:
                assert unquote(parts.fragment) in pages[target].ids, f"{path}: broken {link}"
    memory = pages[(SITE / "api/memory/index.html").resolve()].ids
    for symbol in (
        "capicove.contracts.memory.MemoryQuery",
        "capicove.contracts.memory.MemoryHit",
        "capicove.runtime.memory.build_memory_runtime",
        "capicove.tools.memory.invoke_memory_tool",
        "capicove.mcp.memory.register_memory_mcp",
    ):
        assert symbol in memory, f"Public symbol did not render: {symbol}"
    retrieval = (SITE / "api/retrieval/index.html").read_text()
    assert ":math:" not in retrieval and ".. math::" not in retrieval
    print(f"PASS: examples, command coverage, symbols and links in {len(pages)} HTML pages")


if __name__ == "__main__":
    check_examples()
    check_commands()
    check_html()
