"""Public snapshots exclude development material and unsafe archive members."""

import importlib
import io
import tarfile
from pathlib import Path

import pytest


def archive(name: str = "README.md", *, symlink: bool = False) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as output:
        for path in dict.fromkeys(("LICENSE", "pyproject.toml", "README.md", name)):
            item = tarfile.TarInfo(path)
            item.size = 4
            if symlink and path == name:
                item.type = tarfile.SYMTYPE
                item.linkname = "/private"
            output.addfile(item, io.BytesIO(b"test"))
    return stream.getvalue()


def test_public_snapshot_contains_only_file_inventory() -> None:
    module = importlib.import_module("scripts.public_snapshot")
    report = module.inspect_snapshot(archive())
    assert set(report) == {"LICENSE", "pyproject.toml", "README.md"}
    assert len(report["README.md"]) == 64


@pytest.mark.parametrize(
    "name",
    [
        "../escape",
        ".git/config",
        "docs/reports/budget.xlsx",
        "weights/model.onnx",
        "credentials.json",
        "tests/datasets/user.md",
        "sources/private.md",
        ".internal/agent-policy.md",
        "docs/internal/release-notes.md",
        "docs/prd/product.md",
        "docs/superpowers/plans/task.md",
        "docs/development/private-handoff.md",
        "docs/development/new-internal-record.json",
        "assets/concepts/design.png",
        "assets/logo-2.png",
        "outputs/recording.mp3",
        "models/config.json",
        ".cache/huggingface/token",
        ".aws/credentials",
        ".ssh/id_ed25519",
        ".netrc",
        "a/../b",
        ".",
        "C:escape",
    ],
)
def test_public_snapshot_rejects_private_payloads(name: str) -> None:
    module = importlib.import_module("scripts.public_snapshot")
    with pytest.raises(ValueError):
        module.inspect_snapshot(archive(name))


def test_public_snapshot_rejects_links() -> None:
    module = importlib.import_module("scripts.public_snapshot")
    with pytest.raises(ValueError):
        module.inspect_snapshot(archive("link", symlink=True))


def test_public_documentation_reference_is_not_private_reference_tree() -> None:
    module = importlib.import_module("scripts.public_snapshot")
    assert "docs/site/reference/index.md" in module.inspect_snapshot(
        archive("docs/site/reference/index.md")
    )


@pytest.mark.parametrize(
    "name",
    [
        "docs/development/remote-mcp-operations.json",
        "docs/development/code-search-eval-v1.json",
        "docs/development/workflow.md",
        "docs/architecture/graph-pdf-adaptation.md",
        "docs/site/assets/quickstart.ppm",
        "assets/logo.png",
        "tests/fixtures/example.json",
    ],
)
def test_public_snapshot_keeps_contracts_and_authored_fixtures(name: str) -> None:
    module = importlib.import_module("scripts.public_snapshot")
    assert name in module.inspect_snapshot(archive(name))


def test_public_snapshot_rejects_missing_relative_document_links() -> None:
    module = importlib.import_module("scripts.public_snapshot")
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as output:
        for name, body in {
            "LICENSE": b"MIT",
            "pyproject.toml": b"[project]",
            "README.md": b"[Internal plan](docs/superpowers/plan.md)",
        }.items():
            member = tarfile.TarInfo(name)
            member.size = len(body)
            output.addfile(member, io.BytesIO(body))
    with pytest.raises(ValueError, match="document link"):
        module.inspect_snapshot(payload.getvalue())


def test_working_snapshot_does_not_follow_parent_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = importlib.import_module("scripts.public_snapshot")
    root, outside = tmp_path / "checkout", tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "README.md").write_text("private local note")
    (root / "docs").symlink_to(outside, target_is_directory=True)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda command, **_: b"docs/README.md\0" if command[1] == "ls-files" else b"",
    )
    with pytest.raises(ValueError, match="regular files"):
        module.working_snapshot(root)
