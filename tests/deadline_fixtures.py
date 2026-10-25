"""Reusable test-clock fixtures for long-running suites."""

from datetime import UTC, datetime, timedelta
from types import ModuleType, SimpleNamespace

import pytest


def refresh_module_deadlines(
    module: ModuleType | SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    *,
    refresh_now: bool,
) -> None:
    """Refresh relative test deadlines immediately before a test runs."""
    now = datetime.now(UTC)
    monkeypatch.setattr(module, "FUTURE", now + timedelta(hours=1))
    if refresh_now:
        monkeypatch.setattr(module, "NOW", now)
