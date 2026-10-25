"""Shared acceptance fixtures for authenticated remote MCP components."""

from collections.abc import Iterator

import pytest

from tests.remote_support.fixtures import fault_barrier, principal_matrix_fixture, run_observer


__all__ = ["fault_barrier", "principal_matrix_fixture", "run_observer"]


@pytest.fixture(autouse=True)
def _independent_sse_shutdown_state() -> Iterator[None]:
    """Keep one in-process uvicorn shutdown from closing later SSE responses.

    sse-starlette latches the process-global ``AppStatus.should_exit`` once it
    observes any uvicorn server exiting. Live tests stop their own servers, so a
    later SSE response in the same worker would otherwise end with zero bytes and
    leave its client waiting forever.
    """
    try:
        from sse_starlette.sse import AppStatus
    except ImportError:
        yield
        return
    AppStatus.should_exit = False
    yield
    AppStatus.should_exit = False
