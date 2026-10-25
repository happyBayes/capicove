"""Ownership and cancellation-safe execution for a single local OCR service."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from functools import partial
from threading import Lock
from typing import ParamSpec, Protocol, TypeVar

import anyio

from capicove.processing.image.ocr_configuration import ConfiguredImageOcrProvider


_P = ParamSpec("_P")
_T = TypeVar("_T")


class OcrProvider(ConfiguredImageOcrProvider, Protocol):
    """A configured provider whose release is owned by OcrRuntime."""

    def close(self) -> None: ...


class OcrServiceBusyError(RuntimeError):
    """Another upload or native operation owns the admission slot."""


class OcrServiceClosedError(RuntimeError):
    """The service has stopped accepting work."""


class OcrRuntime:
    """Own one provider and retain admission until native work has actually finished."""

    def __init__(self, factory: Callable[[], OcrProvider]) -> None:
        self._factory = factory
        self._provider: OcrProvider | None = None
        self._owner_lock = Lock()
        self._admission = Lock()
        self._closing = False
        self.fingerprint = ""

    @property
    def provider(self) -> OcrProvider:
        if self._provider is None:
            raise RuntimeError("OCR runtime is not ready")
        return self._provider

    def __call__(self) -> OcrProvider:
        """Also satisfy the existing provider factory composition contract."""
        self.start()
        return self.provider

    def start(self) -> None:
        """Allocate once; CLI may call this before Uvicorn installs signal handlers."""
        with self._owner_lock:
            if self._closing:
                raise RuntimeError("OCR runtime has been closed")
            if self._provider is None:
                provider = self._factory()
                try:
                    self.fingerprint = provider.fingerprint.sha256
                except BaseException:
                    provider.close()
                    raise
                self._provider = provider

    def close(self) -> None:
        """Stop admission, drain active work and release even a non-idempotent provider once."""
        self._closing = True
        with self._admission, self._owner_lock:
            provider, self._provider = self._provider, None
            if provider is not None:
                provider.close()

    @asynccontextmanager
    async def admit(self) -> AsyncIterator[None]:
        if self._closing:
            raise OcrServiceClosedError("the service is shutting down")
        if not self._admission.acquire(blocking=False):
            raise OcrServiceBusyError("another request is running")
        try:
            if self._closing:
                raise OcrServiceClosedError("the service is shutting down")
            yield
        finally:
            self._admission.release()

    async def run(self, function: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs) -> _T:
        """Drain a worker before propagating cancellation, preserving its files and slot."""
        with anyio.CancelScope(shield=True):
            worker = asyncio.create_task(
                anyio.to_thread.run_sync(partial(function, *args, **kwargs))
            )
            cancelled = False
            while True:
                try:
                    result = await asyncio.shield(worker)
                except asyncio.CancelledError:
                    if worker.done():
                        if not worker.cancelled():
                            worker.exception()
                        raise
                    cancelled = True
                except Exception:
                    if cancelled:
                        raise asyncio.CancelledError from None
                    raise
                else:
                    if cancelled:
                        raise asyncio.CancelledError
                    return result

        raise asyncio.CancelledError
