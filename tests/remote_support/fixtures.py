"""Shared fixture composition; providers must opt into their real profiles."""

from collections.abc import Iterator

import pytest

from tests.remote_support.barriers import FaultBarrier
from tests.remote_support.identities import FixtureSubject, principal_matrix
from tests.remote_support.receipts import RunObserver


@pytest.fixture(name="principal_matrix")
def principal_matrix_fixture() -> tuple[FixtureSubject, ...]:
    return principal_matrix()


@pytest.fixture
def fault_barrier() -> Iterator[FaultBarrier]:
    barrier = FaultBarrier("before-release")
    try:
        yield barrier
    finally:
        barrier.release()


@pytest.fixture
def run_observer(request: pytest.FixtureRequest) -> Iterator[RunObserver]:
    with RunObserver(request.node.nodeid, "synthetic-cpu") as observer:
        yield observer
