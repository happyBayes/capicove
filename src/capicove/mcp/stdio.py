"""Shared optional MCP stdio lifecycle with a dedicated protocol stream."""

from __future__ import annotations

import asyncio
import importlib
import io
import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager, redirect_stdout
from typing import TYPE_CHECKING, Any, TextIO


if TYPE_CHECKING:
    from mcp.server import Server


def _windows_stdout_handles() -> tuple[int, int] | None:
    """Capture the Win32 and CRT stdout handles before changing either one."""
    if os.name != "nt":
        return None
    import ctypes

    try:
        msvcrt = importlib.import_module("msvcrt")
        crt_handle = int(msvcrt.get_osfhandle(1))
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        get_std_handle = kernel32.GetStdHandle
    except (ImportError, AttributeError, OSError, TypeError, ValueError) as error:
        raise RuntimeError("Windows standard output redirection is unavailable") from error
    get_std_handle.argtypes = (ctypes.c_uint32,)
    get_std_handle.restype = ctypes.c_void_p
    native_handle = get_std_handle(0xFFFFFFF5)
    if crt_handle == -1 or native_handle in (None, ctypes.c_void_p(-1).value):
        raise RuntimeError("Windows standard output handle is unavailable")
    return int(native_handle), crt_handle


def _set_windows_stdout_handle(handle: int) -> None:
    """Set the Win32 stdout handle used by native writers and child processes."""
    import ctypes

    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)  # type: ignore[attr-defined]
        set_std_handle = kernel32.SetStdHandle
    except (AttributeError, TypeError, ValueError) as error:
        raise RuntimeError("Windows standard output redirection is unavailable") from error
    set_std_handle.argtypes = (ctypes.c_uint32, ctypes.c_void_p)
    set_std_handle.restype = ctypes.c_int
    if not set_std_handle(0xFFFFFFF5, ctypes.c_void_p(handle)):
        raise OSError(
            int(ctypes.get_last_error()),  # type: ignore[attr-defined]
            "Windows standard output handle could not be redirected",
        )


def _sync_windows_stdout_handle(descriptor: int) -> None:
    """Match the Win32 standard handle to the current CRT descriptor."""
    if os.name == "nt":
        msvcrt = importlib.import_module("msvcrt")
        handle = int(msvcrt.get_osfhandle(descriptor))
        if handle == -1:
            raise RuntimeError("Windows standard output descriptor is unavailable")
        _set_windows_stdout_handle(handle)


@contextmanager
def protocol_stdout(stdout: TextIO | None = None) -> Iterator[TextIO]:
    """Reserve stdout before composing providers in a dedicated host process.

    Duplicate fd 1 for the protocol, then route inherited/native writes to
    stderr. Explicit non-fd-1 streams (including embedded/test streams) retain
    Python-only redirection. This process-global context must not be overlapped
    by independent hosts in the same process.
    """
    protocol = stdout if stdout is not None else sys.stdout
    if protocol is None:
        raise RuntimeError("MCP protocol stdout is unavailable")
    try:
        descriptor = protocol.fileno()
    except (AttributeError, io.UnsupportedOperation):
        descriptor = None
    if descriptor != 1:
        with redirect_stdout(sys.stderr):
            yield protocol
        return
    try:
        error_descriptor = sys.stderr.fileno()
    except (AttributeError, io.UnsupportedOperation, ValueError) as error:
        raise RuntimeError("MCP protocol requires a separate stderr descriptor") from error
    if error_descriptor == 1:
        raise RuntimeError("MCP protocol requires a separate stderr descriptor")
    original_windows_handles = _windows_stdout_handles()
    protocol.flush()
    saved = os.dup(1)
    pending_error: BaseException | None = None
    try:
        # Duplicating a standard descriptor is inheritable on Windows.
        os.set_inheritable(saved, False)
        with os.fdopen(
            os.dup(saved), "w", encoding="utf-8", buffering=1, newline="\n"
        ) as dedicated:
            os.set_inheritable(dedicated.fileno(), False)
            os.dup2(error_descriptor, 1)
            active_error: BaseException | None = None
            try:
                _sync_windows_stdout_handle(1)
                with redirect_stdout(sys.stderr):
                    yield dedicated
            except BaseException as error:
                active_error = error
                raise
            finally:
                restore_error: Exception | None = None
                try:
                    os.dup2(saved, 1)
                except Exception as error:
                    restore_error = error
                try:
                    if (
                        original_windows_handles is not None
                        and original_windows_handles[0] != original_windows_handles[1]
                    ):
                        _set_windows_stdout_handle(original_windows_handles[0])
                    elif restore_error is None:
                        _sync_windows_stdout_handle(1)
                except Exception as error:
                    if restore_error is None:
                        restore_error = error
                    else:
                        restore_error.add_note(f"MCP native stdout restoration failed: {error!r}")
                if restore_error is not None:
                    if active_error is None:
                        raise restore_error
                    active_error.add_note(f"MCP stdout restoration failed: {restore_error!r}")
    except BaseException as error:
        pending_error = error
        raise
    finally:
        try:
            os.close(saved)
        except OSError as error:
            if pending_error is None:
                raise
            pending_error.add_note(f"MCP stdout descriptor cleanup failed: {error!r}")


def run_stdio(server: Server[Any], *, stdout: TextIO | None = None) -> None:
    """Drain synchronous work before exit and isolate process diagnostics."""
    stdio = importlib.import_module("mcp.server.stdio")
    anyio = importlib.import_module("anyio")

    async def serve() -> None:
        async with stdio.stdio_server(stdout=anyio.wrap_file(protocol)) as (reader, writer):
            await server.run(reader, writer, server.create_initialization_options())

    with protocol_stdout(stdout) as protocol:
        asyncio.run(serve())
