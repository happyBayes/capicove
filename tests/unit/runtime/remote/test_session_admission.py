"""HTTP request capacity is bounded across applications and sessions."""

import importlib
from dataclasses import replace
from typing import Any

import pytest

from capicove.contracts.remote import PrincipalKey


def limiter(**limits: int) -> Any:
    module = importlib.import_module("capicove.runtime.remote_sessions")
    return module.RequestAdmission(**limits)


def principal(subject: str = "alice") -> PrincipalKey:
    return PrincipalKey("https://issuer.test", "tenant", "application", "human", subject)


def test_app_capacity_includes_different_subjects_and_releases_after_failure() -> None:
    admission = limiter(global_limit=4, application_limit=1, stream_limit=2)
    with admission.reserve(principal(), session_id="session", streaming=False):
        with pytest.raises(PermissionError):
            with admission.reserve(principal("bob"), session_id="other", streaming=False):
                pytest.fail("One app bypassed its request quota through another user")
        with admission.reserve(
            replace(principal(), application_id="other-app"),
            session_id="foreign",
            streaming=False,
        ):
            pass
    with pytest.raises(RuntimeError):
        with admission.reserve(principal(), session_id="session", streaming=False):
            raise RuntimeError("consumer disconnected")
    with admission.reserve(principal(), session_id="session", streaming=False):
        pass


def test_stream_capacity_is_session_bound_and_json_requests_are_separate() -> None:
    admission = limiter(global_limit=8, application_limit=8, stream_limit=2)
    with admission.reserve(principal(), session_id="session", streaming=True):
        with admission.reserve(principal(), session_id="session", streaming=True):
            with pytest.raises(PermissionError):
                with admission.reserve(principal(), session_id="session", streaming=True):
                    pytest.fail("Third stream was admitted")
            with admission.reserve(principal(), session_id="session", streaming=False):
                pass
            with admission.reserve(principal(), session_id="other", streaming=True):
                pass
    with admission.reserve(principal(), session_id="session", streaming=True):
        pass


def test_global_capacity_and_no_unbounded_retained_application_keys() -> None:
    admission = limiter(global_limit=1, application_limit=1, stream_limit=1)
    with admission.reserve(principal(), session_id=None, streaming=False):
        with pytest.raises(PermissionError):
            with admission.reserve(
                replace(principal(), issuer="https://other.test"),
                session_id=None,
                streaming=False,
            ):
                pytest.fail("Global quota exceeded")
    assert admission.active_requests == 0
    assert admission.tracked_applications == 0
    assert admission.tracked_sessions == 0


@pytest.mark.parametrize("value", [0, -1, True, 1.5, 10001])
def test_invalid_capacity_fails_before_use(value: int) -> None:
    with pytest.raises(ValueError):
        limiter(global_limit=value)


def test_session_creation_reserves_pending_capacity_and_cleanup_is_owned() -> None:
    module = importlib.import_module("capicove.runtime.remote_sessions")
    capacity = module.SessionCapacity(global_limit=2, application_limit=1)
    with capacity.reserve(principal()) as register:
        with pytest.raises(PermissionError):
            with capacity.reserve(principal("bob")):
                pytest.fail("Concurrent initialization bypassed the application session cap")
        register("session")
        with pytest.raises(ValueError):
            register("second-session")
    with pytest.raises(PermissionError):
        with capacity.reserve(principal("bob")):
            pytest.fail("Registered session lost its reservation")
    with capacity.reserve(replace(principal(), application_id="other")) as register:
        register("other-session")
    capacity.remove("missing-session")
    with pytest.raises(PermissionError):
        with capacity.reserve(replace(principal(), application_id="third")):
            pytest.fail("Global session capacity exceeded")
    capacity.remove("session")
    capacity.remove("session")
    with capacity.reserve(principal()) as register:
        register("replacement-session")


def test_failed_session_initialization_releases_reservation() -> None:
    module = importlib.import_module("capicove.runtime.remote_sessions")
    capacity = module.SessionCapacity(global_limit=1, application_limit=1)
    with pytest.raises(RuntimeError):
        with capacity.reserve(principal()):
            raise RuntimeError("initialization failed")
    with capacity.reserve(principal()) as register:
        register("session")
