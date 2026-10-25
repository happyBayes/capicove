"""Reject unsafe or incomplete release distributions before publishing."""

from __future__ import annotations

import importlib
import io
import tarfile
import zipfile
from pathlib import Path

import pytest


release = importlib.import_module("scripts.release_packages")


@pytest.mark.parametrize(
    "name", ["README.md", "client.py", "deploy.py", "models.toml", "qwen35-9b.toml"]
)
def test_sdist_includes_reviewed_vlm_scripts(tmp_path: Path, name: str) -> None:
    path = tmp_path / "capicove-0.1.0.tar.gz"
    files = {
        "PKG-INFO": b"Metadata-Version: 2.4\nName: capicove\nVersion: 0.1.0\nLicense-Expression: MIT\n",
        "LICENSE": b"MIT License",
        f"scripts/vlm/{name}": b"reviewed source",
    }
    with tarfile.open(path, "w:gz") as archive:
        for member, content in files.items():
            info = tarfile.TarInfo(f"capicove-0.1.0/{member}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    assert (
        f"capicove-0.1.0/scripts/vlm/{name}"
        in release.inspect_archive(path, version="0.1.0")["files"]
    )


def test_sdist_rejects_unreviewed_vlm_script(tmp_path: Path) -> None:
    path = tmp_path / "capicove-0.1.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        for name, content in {
            "PKG-INFO": b"Metadata-Version: 2.4\nName: capicove\nVersion: 0.1.0\nLicense-Expression: MIT\n",
            "LICENSE": b"MIT License",
            "scripts/vlm/private.toml": b"private configuration",
        }.items():
            info = tarfile.TarInfo(f"capicove-0.1.0/{name}")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
    with pytest.raises(ValueError, match="unapproved"):
        release.inspect_archive(path, version="0.1.0")


def wheel(
    tmp_path: Path,
    *,
    version: str = "0.1.0",
    dependency: str = "numpy>=1.26",
    extra: str | None = None,
    license_file: bool = True,
) -> Path:
    path = tmp_path / f"capicove-{version}-py3-none-any.whl"
    metadata = f"Metadata-Version: 2.4\nName: capicove\nVersion: {version}\nLicense-Expression: MIT\nRequires-Dist: {dependency}\n"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("capicove/__init__.py", "")
        archive.writestr(f"capicove-{version}.dist-info/METADATA", metadata)
        archive.writestr(f"capicove-{version}.dist-info/WHEEL", "Wheel-Version: 1.0\n")
        archive.writestr(f"capicove-{version}.dist-info/RECORD", "")
        if license_file:
            archive.writestr(f"capicove-{version}.dist-info/licenses/LICENSE", "MIT License")
        if extra:
            archive.writestr(extra, "unapproved")
    return path


def test_valid_wheel_has_inventory_and_hash(tmp_path: Path) -> None:
    result = release.inspect_archive(wheel(tmp_path), version="0.1.0")
    assert result["name"] == "capicove"
    assert len(result["sha256"]) == 64
    assert result["version"] == "0.1.0"


@pytest.mark.parametrize(
    "name",
    [
        "../secret",
        "/etc/passwd",
        "capicove/model.safetensors",
        "capicove/.env",
        "tests/datasets/private.txt",
        "capicove/__pycache__/a.pyc",
        "capicove/credentials.json",
        "capicove/../../secret",
    ],
)
def test_unapproved_payloads_fail(tmp_path: Path, name: str) -> None:
    with pytest.raises(ValueError):
        release.inspect_archive(wheel(tmp_path, extra=name), version="0.1.0")


def test_wrong_version_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="version"):
        release.inspect_archive(wheel(tmp_path, version="0.1.0.dev0"), version="0.1.0")


def test_missing_license_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="license"):
        release.inspect_archive(wheel(tmp_path, license_file=False), version="0.1.0")


def test_local_dependency_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="dependency"):
        release.inspect_archive(
            wheel(tmp_path, dependency="capicove @ file:///private/source"), version="0.1.0"
        )


def test_sdist_symlink_fails(tmp_path: Path) -> None:
    path = tmp_path / "capicove-0.1.0.tar.gz"
    with tarfile.open(path, "w:gz") as archive:
        item = tarfile.TarInfo("capicove-0.1.0/src/capicove/link")
        item.type = tarfile.SYMTYPE
        item.linkname = "/private"
        archive.addfile(item)
    with pytest.raises(ValueError):
        release.inspect_archive(path, version="0.1.0")


def test_incomplete_release_cannot_be_accepted(tmp_path: Path) -> None:
    wheel(tmp_path)
    with pytest.raises(ValueError, match="inventory"):
        release.inspect_release(tmp_path, version="0.1.0")


def test_filename_must_match_metadata(tmp_path: Path) -> None:
    path = wheel(tmp_path)
    renamed = path.with_name("capicove-9.9.9-py3-none-any.whl")
    path.rename(renamed)
    with pytest.raises(ValueError, match="filename"):
        release.inspect_archive(renamed, version="0.1.0")


def test_normalized_duplicate_member_fails(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsafe"):
        release.inspect_archive(wheel(tmp_path, extra="capicove//__init__.py"), version="0.1.0")
