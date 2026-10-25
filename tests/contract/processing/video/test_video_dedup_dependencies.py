"""Dependency contracts for video file deduplication tests."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest


_ROOT = Path(__file__).parents[4]


@pytest.mark.contract
def test_default_test_group_and_lock_install_video_dedup_hash_dependency() -> None:
    with (_ROOT / "pyproject.toml").open("rb") as stream:
        project_configuration = tomllib.load(stream)
    with (_ROOT / "uv.lock").open("rb") as stream:
        lock_configuration = tomllib.load(stream)

    test_requirements = project_configuration["dependency-groups"]["test"]
    locked_project = next(
        package for package in lock_configuration["package"] if package["name"] == "capicove"
    )
    locked_test_requirements = locked_project["dev-dependencies"]["test"]

    assert any(str(requirement).startswith("xxhash") for requirement in test_requirements)
    assert any(requirement["name"] == "xxhash" for requirement in locked_test_requirements)
