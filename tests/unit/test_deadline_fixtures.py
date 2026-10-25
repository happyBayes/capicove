"""Test deadlines stay usable after a long collection-to-execution delay."""

from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest

from tests.deadline_fixtures import refresh_module_deadlines


def test_refresh_module_deadlines_replaces_expired_import_time_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imported_at = datetime.now(UTC) - timedelta(hours=2)
    module = SimpleNamespace(NOW=imported_at, FUTURE=imported_at + timedelta(hours=1))

    refresh_module_deadlines(module, monkeypatch, refresh_now=True)

    assert module.NOW > imported_at
    assert module.FUTURE > datetime.now(UTC) + timedelta(minutes=30)
    assert module.FUTURE - module.NOW == timedelta(hours=1)
