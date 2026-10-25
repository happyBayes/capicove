"""Fixtures shared by the end-to-end journeys."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from tests.e2e.wheel_cache import ROOT_VARIABLE, WheelCache


@pytest.fixture(scope="session")
def wheel_cache(tmp_path_factory: pytest.TempPathFactory) -> WheelCache:
    """Share one wheel build and one environment per selection across the session.

    ``scripts/run_tests.py`` sets the root to a per-run directory on the uv cache's
    filesystem so installs can hardlink. Otherwise the root sits in the session's
    base temporary directory, which every xdist worker shares.
    """
    configured = os.environ.get(ROOT_VARIABLE)
    if configured:
        return WheelCache(Path(configured))
    base = tmp_path_factory.getbasetemp()
    if os.environ.get("PYTEST_XDIST_WORKER"):
        base = base.parent
    return WheelCache(base / "capicove-wheels")
