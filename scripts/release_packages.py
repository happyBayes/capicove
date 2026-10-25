"""Build and inspect the six release distributions; never upload anything."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Any

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name, parse_sdist_filename, parse_wheel_filename
from packaging.version import Version


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = {
    "capicove": ROOT,
    "capicove-execution-spark": ROOT / "plugins/execution-spark",
    "capicove-processor-paddleocr": ROOT / "plugins/processor-paddleocr",
    "capicove-retriever-elasticsearch": ROOT / "plugins/retriever-elasticsearch",
    "capicove-retriever-opensearch": ROOT / "plugins/retriever-opensearch",
    "capicove-retriever-faiss": ROOT / "plugins/retriever-faiss",
}
ALLOWED_SUFFIXES = {".py", ".json", ".toml", ".sql", ".yaml", ".yml", ".txt", ".md", ".typed"}


def _safe_member(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if (
        path.is_absolute()
        or ".." in path.parts
        or "\\" in name
        or ":" in name
        or str(path) != name.rstrip("/")
        or not path.parts
    ):
        raise ValueError(f"unsafe archive member: {name}")
    if any(
        (part.startswith(".") and not (part == ".gitignore" and len(path.parts) == 2))
        or part in {"__pycache__", "credentials.json"}
        for part in path.parts
    ):
        raise ValueError(f"unapproved archive member: {name}")
    return path


def _archive_files(path: Path) -> dict[str, bytes]:
    files: dict[str, bytes] = {}
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            for entry in archive.infolist():
                _safe_member(entry.filename)
                if entry.is_dir():
                    continue
                if entry.filename in files or (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("duplicate or linked archive member")
                if entry.file_size > 8 * 1024 * 1024:
                    raise ValueError("unexpected large distribution payload")
                files[entry.filename] = archive.read(entry)
    else:
        with tarfile.open(path, "r:gz") as archive:
            for entry in archive:
                _safe_member(entry.name)
                if entry.isdir():
                    continue
                if not entry.isfile() or entry.name in files or entry.size > 8 * 1024 * 1024:
                    raise ValueError("unexpected source distribution member")
                source = archive.extractfile(entry)
                if source is None:
                    raise ValueError("unreadable distribution member")
                files[entry.name] = source.read()
    return files


def inspect_archive(path: Path, *, version: str) -> dict[str, Any]:
    """Reject payloads outside the source-package and distribution allowlist."""
    files = _archive_files(path)
    wheel = path.suffix == ".whl"
    metadata_files = [name for name in files if name.endswith(".dist-info/METADATA") if wheel]
    if not wheel:
        metadata_files = [
            name
            for name in files
            if len(PurePosixPath(name).parts) == 2 and name.endswith("/PKG-INFO")
        ]
    if len(metadata_files) != 1:
        raise ValueError("distribution must contain exactly one root metadata file")
    metadata = BytesParser().parsebytes(files[metadata_files[0]])
    name = canonicalize_name(str(metadata["Name"]))
    if name not in PROJECTS:
        raise ValueError("unexpected distribution name")
    if str(metadata["Version"]) != version:
        raise ValueError("distribution version mismatch")
    parsed = parse_wheel_filename(path.name) if wheel else parse_sdist_filename(path.name)
    if parsed[0] != name or str(parsed[1]) != version:
        raise ValueError("distribution filename does not match metadata")
    if metadata.get("License-Expression") != "MIT":
        raise ValueError("distribution license metadata must be MIT")
    if not any(PurePosixPath(member).name == "LICENSE" for member in files):
        raise ValueError("distribution must include its license file")
    for requirement in metadata.get_all("Requires-Dist", []):
        if Requirement(requirement).url is not None:
            raise ValueError("direct URL or local dependency is not publishable")
    package = name.replace("-", "_")
    prefix = f"{package}-{version}"
    for member in files:
        parts = PurePosixPath(member).parts
        if wheel:
            if parts[0] == package:
                if PurePosixPath(member).suffix not in ALLOWED_SUFFIXES and parts[-1] != "py.typed":
                    raise ValueError(f"unapproved package payload: {member}")
            elif parts[0] == f"{prefix}.dist-info":
                if parts[1:] not in [
                    ("METADATA",),
                    ("WHEEL",),
                    ("RECORD",),
                    ("entry_points.txt",),
                    ("licenses", "LICENSE"),
                ]:
                    raise ValueError(f"unexpected wheel metadata: {member}")
            else:
                raise ValueError(f"unapproved wheel payload: {member}")
        else:
            if parts[0] != prefix:
                raise ValueError("source distribution root mismatch")
            relative = parts[1:]
            if relative[:2] == ("src", package):
                if PurePosixPath(member).suffix not in ALLOWED_SUFFIXES and parts[-1] != "py.typed":
                    raise ValueError(f"unapproved source payload: {member}")
            elif name == "capicove" and relative in {
                ("scripts", "vlm", filename)
                for filename in (
                    "README.md",
                    "client.py",
                    "deploy.py",
                    "models.toml",
                    "qwen35-9b.toml",
                )
            }:
                pass
            elif relative not in [
                ("pyproject.toml",),
                ("PKG-INFO",),
                ("README.md",),
                ("LICENSE",),
                (".gitignore",),
            ]:
                raise ValueError(f"unapproved source distribution payload: {member}")
    return {
        "name": name,
        "version": version,
        "filename": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "files": sorted(files),
    }


def inspect_release(directory: Path, *, version: str) -> list[dict[str, Any]]:
    """Require exactly one wheel and sdist for every release package."""
    paths = sorted([*directory.glob("*.whl"), *directory.glob("*.tar.gz")])
    if len(paths) != len(PROJECTS) * 2:
        raise ValueError("release inventory must contain six wheels and six source distributions")
    results = [inspect_archive(path, version=version) for path in paths]
    actual = {
        (result["name"], "wheel" if result["filename"].endswith(".whl") else "sdist")
        for result in results
    }
    expected = {(name, kind) for name in PROJECTS for kind in ("wheel", "sdist")}
    if actual != expected:
        raise ValueError("release inventory does not match the six expected packages")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    version = str(Version(args.version))
    if Version(version).is_devrelease or Version(version).local is not None:
        raise ValueError("release artifacts cannot use development or local versions")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.build:
        if any(directory.iterdir()):
            raise ValueError("build output must be empty to prevent stale release artifacts")
        for project in PROJECTS.values():
            subprocess.run(
                ["uv", "build", "--no-sources", "--out-dir", str(directory), str(project)],
                check=True,
            )
    results = inspect_release(directory, version=version)
    (directory / "release-manifest.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8"
    )
    (directory / "SHA256SUMS").write_text(
        "".join(f"{item['sha256']}  {item['filename']}\n" for item in results), encoding="utf-8"
    )
    print(f"Verified {len(results)} clean distributions for {version}")


if __name__ == "__main__":
    main()
