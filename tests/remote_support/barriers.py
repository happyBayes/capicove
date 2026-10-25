"""Deterministic pause points for authority changes and publication races."""

import threading


class FaultBarrier:
    """Pause a worker at an observed point without depending on arbitrary sleep."""

    def __init__(self, phase: str, timeout: float = 5.0) -> None:
        if not phase or timeout <= 0:
            raise ValueError("a phase and positive timeout are required")
        self.phase = phase
        self.timeout = timeout
        self._arrived = threading.Event()
        self._released = threading.Event()

    def arrive(self) -> None:
        """Signal that the worker reached the barrier, then wait for release."""
        self._arrived.set()
        if not self._released.wait(self.timeout):
            raise TimeoutError(f"barrier {self.phase} was not released")

    def wait_until_arrived(self) -> None:
        """Wait until a real worker reaches the selected boundary."""
        if not self._arrived.wait(self.timeout):
            raise TimeoutError(f"barrier {self.phase} was not reached")

    def release(self) -> None:
        """Permit the paused worker to resume; safe during fixture cleanup."""
        self._released.set()
