"""Bounded request admission for a single remote gateway process.

These counters limit active HTTP requests and streams, not durable operations or
isolated workers. A deployment with multiple gateways needs shared admission.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager

from capicove.contracts.remote import PrincipalKey


class RequestAdmission:
    """Reserve global, enrolled-application and session-stream capacity together."""

    def __init__(
        self, *, global_limit: int = 128, application_limit: int = 32, stream_limit: int = 2
    ) -> None:
        for value in (global_limit, application_limit, stream_limit):
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError("Request capacity must be an integer in [1, 10000]")
        self._global_limit = global_limit
        self._application_limit = application_limit
        self._stream_limit = stream_limit
        self._lock = threading.Lock()
        self._active = 0
        self._applications: dict[tuple[str, str, str], int] = {}
        self._streams: dict[tuple[PrincipalKey, str], int] = {}

    @property
    def active_requests(self) -> int:
        """Return the current process-wide reservation count."""
        with self._lock:
            return self._active

    @property
    def tracked_applications(self) -> int:
        """Return the number of applications with active reservations."""
        with self._lock:
            return len(self._applications)

    @property
    def tracked_sessions(self) -> int:
        """Return the number of sessions with active stream reservations."""
        with self._lock:
            return len(self._streams)

    @contextmanager
    def reserve(
        self, principal: PrincipalKey, *, session_id: str | None, streaming: bool
    ) -> Iterator[None]:
        """Reject overload without waiting or retaining rejected identity keys."""
        if (
            not isinstance(principal, PrincipalKey)
            or type(streaming) is not bool
            or (session_id is not None and (not session_id or len(session_id) > 256))
        ):
            raise ValueError("A bounded authenticated request identity is required")
        application = (principal.issuer, principal.tenant_id, principal.application_id)
        stream = (principal, session_id) if streaming and session_id is not None else None
        with self._lock:
            if (
                self._active >= self._global_limit
                or self._applications.get(application, 0) >= self._application_limit
                or (stream is not None and self._streams.get(stream, 0) >= self._stream_limit)
            ):
                raise PermissionError("Remote request capacity exceeded")
            self._active += 1
            self._applications[application] = self._applications.get(application, 0) + 1
            if stream is not None:
                self._streams[stream] = self._streams.get(stream, 0) + 1
        try:
            yield
        finally:
            with self._lock:
                self._active -= 1
                self._applications[application] -= 1
                if not self._applications[application]:
                    del self._applications[application]
                if stream is not None:
                    self._streams[stream] -= 1
                    if not self._streams[stream]:
                        del self._streams[stream]


class SessionCapacity:
    """Bound pending and established sessions across delegated application users."""

    def __init__(self, *, global_limit: int = 1000, application_limit: int = 64) -> None:
        for value in (global_limit, application_limit):
            if type(value) is not int or not 1 <= value <= 10000:
                raise ValueError("Session capacity must be an integer in [1, 10000]")
        self._global_limit = global_limit
        self._application_limit = application_limit
        self._lock = threading.Lock()
        self._active = 0
        self._applications: dict[tuple[str, str, str], int] = {}
        self._sessions: dict[str, tuple[str, str, str]] = {}

    def _release(self, application: tuple[str, str, str]) -> None:
        self._active -= 1
        self._applications[application] -= 1
        if not self._applications[application]:
            del self._applications[application]

    @contextmanager
    def reserve(self, principal: PrincipalKey) -> Iterator[Callable[[str], None]]:
        """Reserve before SDK initialization; retain only a registered session."""
        if not isinstance(principal, PrincipalKey):
            raise ValueError("Session admission requires an authenticated principal")
        application = principal.issuer, principal.tenant_id, principal.application_id
        with self._lock:
            if (
                self._active >= self._global_limit
                or self._applications.get(application, 0) >= self._application_limit
            ):
                raise PermissionError("Remote session capacity exceeded")
            self._active += 1
            self._applications[application] = self._applications.get(application, 0) + 1
        registered = False
        open_reservation = True

        def register(session_id: str) -> None:
            nonlocal registered
            with self._lock:
                if (
                    not open_reservation
                    or registered
                    or not isinstance(session_id, str)
                    or not 1 <= len(session_id) <= 256
                    or session_id in self._sessions
                ):
                    raise ValueError("Invalid or reused session registration")
                self._sessions[session_id] = application
                registered = True

        try:
            yield register
        finally:
            with self._lock:
                open_reservation = False
                if not registered:
                    self._release(application)

    def remove(self, session_id: str) -> None:
        """Release a retired session once; unknown IDs cannot decrement capacity."""
        with self._lock:
            application = self._sessions.pop(session_id, None)
            if application is not None:
                self._release(application)
