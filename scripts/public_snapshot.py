"""Inspect/export a source-only snapshot without Git metadata or private history.

The committed export attributes define reviewed exclusions. This does not create
Git repositories, change visibility, or establish copyright/redistribution rights.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import posixpath
import re
import subprocess
import tarfile
import tomllib
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


_PRIVATE_PARTS = {
    ".git",
    ".agents",
    ".codex",
    ".superpowers",
    ".venv",
    ".aws",
    ".ssh",
    ".huggingface",
    "__pycache__",
    "sources",
}
_ROOT = Path(__file__).resolve().parents[1]
_POLICY = tomllib.loads((_ROOT / ".public-export.toml").read_text(encoding="utf-8"))
_PRIVATE_SUFFIXES = {
    ".xlsx",
    ".xls",
    ".mp3",
    ".wav",
    ".mp4",
    ".zip",
    ".sqlite",
    ".sqlite3",
    ".gguf",
    ".onnx",
    ".safetensors",
    ".pt",
    ".pth",
    ".bin",
    ".pem",
    ".key",
}


def _private_path(path: PurePosixPath) -> bool:
    name = str(path)
    if any(
        name == directory or name.startswith(directory + "/")
        for directory in _POLICY["private_directories"]
    ):
        return True
    if name.startswith("docs/development/"):
        return name.removeprefix("docs/development/") not in _POLICY["public_development_files"]
    return name.startswith("assets/") and name not in _POLICY["public_assets"]


def _check_document_links(documents: dict[str, str], inventory: dict[str, str]) -> None:
    """Reject relative file links whose targets disappeared from the public tree."""
    for name, body in documents.items():
        # Examples can contain illustrative links; only prose links are checked.
        prose = re.sub(r"(?ms)^```.*?^```[^\n]*", "", body)
        links = re.findall(r"\[[^\]\n]*\]\(([^)\n]+)\)", prose)
        links += re.findall(r'(?:src|href)="([^"\n]+)"', prose)
        for link in links:
            target = link.strip().split(' "', 1)[0].strip("<>")
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or not parsed.path or target.startswith("/"):
                continue
            resolved = posixpath.normpath(
                posixpath.join(str(PurePosixPath(name).parent), unquote(parsed.path))
            )
            if resolved not in inventory and not any(
                item.startswith(resolved.rstrip("/") + "/") for item in inventory
            ):
                raise ValueError(f"unavailable public document link: {name} -> {target}")


def inspect_snapshot(payload: bytes) -> dict[str, str]:
    """Reject ambiguous paths, links, models, user data and internal artifacts."""
    inventory: dict[str, str] = {}
    documents: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
        for member in archive:
            path = PurePosixPath(member.name.rstrip("/"))
            if (
                not path.parts
                or path.is_absolute()
                or str(path) != member.name.rstrip("/")
                or ".." in path.parts
                or "\\" in member.name
                or ":" in member.name
                or set(path.parts) & _PRIVATE_PARTS
                or _private_path(path)
                or path.suffix.lower() in _PRIVATE_SUFFIXES
                or path.parts[0] == "reference"
                or str(path).startswith(("tests/datasets/", "docs/reports/"))
                or path.name
                in {
                    "credentials.json",
                    ".env",
                    "Untitled",
                    ".netrc",
                    ".pypirc",
                    ".npmrc",
                    ".git-credentials",
                }
                or (path.name.startswith(".env.") and path.name != ".env.example")
            ):
                raise ValueError(f"unapproved public snapshot path: {member.name}")
            if member.isdir():
                continue
            if not member.isfile() or member.name in inventory or member.size > 8 * 1024 * 1024:
                raise ValueError("snapshot requires unique bounded regular files")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("snapshot member unavailable")
            body = stream.read()
            inventory[member.name] = hashlib.sha256(body).hexdigest()
            if path.suffix == ".md":
                documents[member.name] = body.decode("utf-8")
    if not {"LICENSE", "README.md", "pyproject.toml"} <= inventory.keys():
        raise ValueError("snapshot is missing required project metadata")
    _check_document_links(documents, inventory)
    return inventory


def working_snapshot(root: Path) -> bytes:
    """Inspect edited and new non-ignored files without staging or creating commits.

    Parent attributes are checked too: git archive prunes export-ignored directories.
    This draft cannot be published through the command line's --output option.
    """
    root = root.resolve()
    listed = subprocess.check_output(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root
    )
    names = sorted({name.decode() for name in listed.split(b"\0") if name})
    candidates = sorted(
        {part for name in names for part in [name, *(str(p) for p in PurePosixPath(name).parents)]}
        - {"."}
    )
    attributes = subprocess.check_output(
        ["git", "check-attr", "-z", "--stdin", "export-ignore"],
        input=b"\0".join(name.encode() for name in candidates),
        cwd=root,
    ).split(b"\0")
    excluded = {
        attributes[index].decode()
        for index in range(0, len(attributes) - 1, 3)
        if attributes[index + 2] == b"set"
    }
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name in names:
            path = root / name
            if any(name == item or name.startswith(item + "/") for item in excluded):
                continue
            if path.is_symlink() or any(
                (root / parent).is_symlink() for parent in PurePosixPath(name).parents
            ):
                raise ValueError(f"snapshot requires regular files: {name}")
            if not path.exists():
                continue  # Reflect an unstaged deletion in this draft.
            if not path.is_file() or path.stat().st_size > 8 * 1024 * 1024:
                raise ValueError(f"snapshot requires bounded regular files: {name}")
            body = path.read_bytes()
            member = tarfile.TarInfo(name)
            member.mode = path.stat().st_mode & 0o777
            member.size = len(body)
            archive.addfile(member, io.BytesIO(body))
    return output.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path, help="New .tar path; omit for inspection only.")
    parser.add_argument(
        "--working-attributes",
        action="store_true",
        help="Inspect draft export exclusions; cannot publish an archive.",
    )
    parser.add_argument(
        "--working-tree", action="store_true", help="Inspect current file contents without staging."
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if args.working_tree and args.working_attributes:
        parser.error("choose either working-tree or working-attributes inspection")
    if args.output and (args.working_attributes or args.working_tree):
        parser.error("draft attributes cannot produce a publication artifact")
    if args.output and subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip():
        parser.error("publication requires a clean reviewed checkout")
    tree = subprocess.check_output(
        ["git", "rev-parse", "--verify", "--end-of-options", args.ref + "^{tree}"],
        cwd=root,
        text=True,
    ).strip()
    command = ["git", "archive", "--format=tar"]
    if args.working_attributes:
        command.append("--worktree-attributes")
    payload = (
        working_snapshot(root)
        if args.working_tree
        else subprocess.check_output([*command, tree], cwd=root)
    )
    inventory = inspect_snapshot(payload)
    if args.output:
        with args.output.open("xb") as stream:
            stream.write(payload)
        with args.output.with_suffix(".manifest.json").open("x") as stream:
            json.dump({"schema_version": 1, "files": inventory}, stream, indent=2)
            stream.write("\n")
    print(
        json.dumps(
            {
                "status": "passed",
                "files": len(inventory),
                "sha256": hashlib.sha256(payload).hexdigest(),
                "draft_attributes": args.working_attributes,
                "draft_working_tree": args.working_tree,
            }
        )
    )


if __name__ == "__main__":
    main()
