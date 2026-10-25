"""Opt-in single-flight scheduling for cooperative candidate review callbacks."""

from __future__ import annotations

import threading
from collections.abc import Callable
from types import TracebackType


class BackgroundReview:
    """Run one bounded-wait review at a time with no registry approval interface.

    Callbacks must check cancellation before persisting candidates. Python threads
    cannot forcibly stop a host callback; a cancelled live worker retains the slot.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._closed = False
        self._failed = False
        self._running = False
        self._pending: Callable[[threading.Event], None] | None = None

    @property
    def failed(self) -> bool:
        """Expose failure without leaking callback exception content."""
        with self._lock:
            return self._failed

    def submit(self, review: Callable[[threading.Event], None], *, coalesce: bool = False) -> bool:
        """Start one worker, optionally retaining only the latest follow-up hook."""
        with self._lock:
            if self._closed:
                return False
            if self._running:
                if coalesce:
                    self._pending = review
                    return True
                return False
            self._cancel = threading.Event()
            self._failed = False
            self._running = True
            self._thread = threading.Thread(target=self._run, args=(review,), daemon=True)
            self._thread.start()
            return True

    def _run(self, review: Callable[[threading.Event], None]) -> None:
        while True:
            try:
                review(self._cancel)
            except Exception:
                with self._lock:
                    self._failed = True
            with self._lock:
                if self._pending is None or self._closed:
                    self._running = False
                    return
                review = self._pending
                self._pending = None
                self._cancel = threading.Event()

    def cancel(self, *, timeout: float = 0.1) -> bool:
        """Signal cancellation and wait at most five seconds; return worker exit."""
        if not 0 <= timeout <= 5:
            raise ValueError("timeout must be between zero and five seconds")
        with self._lock:
            self._pending = None
            self._cancel.set()
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def wait(self, *, timeout: float = 0.1) -> bool:
        """Wait a bounded interval for completion without requesting cancellation."""
        if not 0 <= timeout <= 5:
            raise ValueError("timeout must be between zero and five seconds")
        with self._lock:
            thread = self._thread
        if thread is None:
            return True
        thread.join(timeout)
        return not thread.is_alive()

    def close(self) -> None:
        """Permanently reject new work and request bounded cooperative shutdown."""
        with self._lock:
            self._closed = True
        self.cancel()

    def __enter__(self) -> BackgroundReview:
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()
