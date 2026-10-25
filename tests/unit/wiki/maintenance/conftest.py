"""Shared synthetic fixtures for maintenance unit tests."""

from pathlib import Path

import pytest

from tests.wiki_maintenance_support.faults import Effects, FaultBarriers
from tests.wiki_maintenance_support.fixtures import WikiCase, make_case


@pytest.fixture
def wiki_case(tmp_path: Path) -> WikiCase:
    return make_case(tmp_path)


@pytest.fixture
def effects() -> Effects:
    return Effects()


@pytest.fixture
def fault_barriers(tmp_path: Path) -> FaultBarriers:
    return FaultBarriers(tmp_path)
