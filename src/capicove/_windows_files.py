"""Native Windows handle checks shared by private MCP files and public image reads."""

from __future__ import annotations

import ctypes
import ntpath
import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from ctypes import wintypes
from pathlib import Path
from typing import Any, cast


class WindowsPathPolicyError(ValueError):
    """A Windows path, handle identity, ownership, or ACL violates local policy."""


_WINDOWS_WRITE_RIGHTS = (
    0x0002  # FILE_WRITE_DATA / FILE_ADD_FILE
    | 0x0004  # FILE_APPEND_DATA / FILE_ADD_SUBDIRECTORY
    | 0x0010  # FILE_WRITE_EA
    | 0x0040  # FILE_DELETE_CHILD
    | 0x0100  # FILE_WRITE_ATTRIBUTES
    | 0x00010000  # DELETE
    | 0x00040000  # WRITE_DAC
    | 0x00080000  # WRITE_OWNER
    | 0x01000000  # ACCESS_SYSTEM_SECURITY
    | 0x02000000  # MAXIMUM_ALLOWED
    | 0x10000000  # GENERIC_ALL
    | 0x40000000  # GENERIC_WRITE
)
_WINDOWS_PRIVATE_RIGHTS = (
    _WINDOWS_WRITE_RIGHTS
    | 0x0001  # FILE_READ_DATA / FILE_LIST_DIRECTORY
    | 0x0008  # FILE_READ_EA
    | 0x0020  # FILE_EXECUTE / FILE_TRAVERSE
    | 0x0080  # FILE_READ_ATTRIBUTES
    | 0x00020000  # READ_CONTROL
    | 0x80000000  # GENERIC_READ
    | 0x20000000  # GENERIC_EXECUTE
)
_WINDOWS_INHERITED_FILE_RIGHTS = 0x0001 | 0x0002 | 0x0004 | 0x00010000
_WINDOWS_REPARSE_POINT = 0x0400
_WINDOWS_DIRECTORY = 0x0010


class _WindowsFileAttributeTagInfo(ctypes.Structure):
    _fields_ = [("attributes", wintypes.DWORD), ("reparse_tag", wintypes.DWORD)]


class _WindowsAclHeader(ctypes.Structure):
    _fields_ = [
        ("revision", wintypes.BYTE),
        ("reserved", wintypes.BYTE),
        ("size", wintypes.WORD),
        ("ace_count", wintypes.WORD),
        ("reserved2", wintypes.WORD),
    ]


class _WindowsAceHeader(ctypes.Structure):
    _fields_ = [
        ("ace_type", wintypes.BYTE),
        ("flags", wintypes.BYTE),
        ("size", wintypes.WORD),
    ]


def _validate_windows_acl(
    owner_sid: str,
    current_sid: str,
    aces: list[tuple[int, int, int, str]],
    *,
    token_owner_sid: str | None = None,
    owner_required: bool = True,
    private_acl: bool = False,
    require_private_inheritance: bool = False,
) -> None:
    """Apply owner and ACL policy to the native DACL entries of one handle."""
    trusted_token_owner = token_owner_sid in {"S-1-5-18", "S-1-5-32-544"}
    if (
        owner_required
        and owner_sid != current_sid
        and not (trusted_token_owner and owner_sid == token_owner_sid)
    ):
        raise WindowsPathPolicyError(
            "MCP Windows file ACL owner must be the process user or trusted token owner"
        )
    if require_private_inheritance and not private_acl:
        raise ValueError("private inheritance requires a private ACL")
    trusted_writers = {current_sid, "S-1-5-18", "S-1-5-32-544"}
    inherited_current_rights = 0
    for ace_type, ace_flags, mask, sid in aces:
        if ace_type not in (0, 1):
            raise WindowsPathPolicyError("MCP Windows file ACL has an unsupported ACE")
        if ace_type == 1:
            continue
        # Object-inherit rights reach direct child files; container-inherit rights do not.
        if require_private_inheritance and sid == current_sid and ace_flags & 0x01:
            inherited_current_rights |= mask
        creator_owner_child_only = sid == "S-1-3-0" and bool(ace_flags & 0x08)
        if sid not in trusted_writers and not creator_owner_child_only:
            forbidden = _WINDOWS_PRIVATE_RIGHTS if private_acl else _WINDOWS_WRITE_RIGHTS
            if mask & forbidden:
                raise WindowsPathPolicyError(
                    "MCP Windows file ACL grants forbidden access to another user"
                )
    if require_private_inheritance:
        if inherited_current_rights & 0x10000000:  # GENERIC_ALL
            inherited_current_rights |= _WINDOWS_INHERITED_FILE_RIGHTS
        if inherited_current_rights & 0x80000000:  # GENERIC_READ
            inherited_current_rights |= 0x0001
        if inherited_current_rights & 0x40000000:  # GENERIC_WRITE
            inherited_current_rights |= 0x0002 | 0x0004
        if (
            inherited_current_rights & _WINDOWS_INHERITED_FILE_RIGHTS
            != _WINDOWS_INHERITED_FILE_RIGHTS
        ):
            raise WindowsPathPolicyError(
                "MCP Windows trace parent must inherit private read, write, and delete rights"
            )


def _windows_apis() -> tuple[Any, Any]:
    """Bind native handle and ACL calls only on Windows."""
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise ValueError("MCP Windows file security APIs are unavailable")
    kernel32 = loader("kernel32", use_last_error=True)
    advapi32 = loader("advapi32", use_last_error=True)
    kernel32.CreateFileW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL
    kernel32.GetDriveTypeW.argtypes = [wintypes.LPCWSTR]
    kernel32.GetDriveTypeW.restype = wintypes.UINT
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.GetCurrentThread.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandleEx.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    ]
    kernel32.GetFileInformationByHandleEx.restype = wintypes.BOOL
    kernel32.GetFinalPathNameByHandleW.argtypes = [
        wintypes.HANDLE,
        wintypes.LPWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
    ]
    kernel32.GetFinalPathNameByHandleW.restype = wintypes.DWORD
    kernel32.GetFileSizeEx.argtypes = [wintypes.HANDLE, ctypes.POINTER(ctypes.c_longlong)]
    kernel32.GetFileSizeEx.restype = wintypes.BOOL
    kernel32.ReadFile.argtypes = [
        wintypes.HANDLE,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
        ctypes.c_void_p,
    ]
    kernel32.ReadFile.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    advapi32.OpenProcessToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.OpenThreadToken.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.BOOL,
        ctypes.POINTER(wintypes.HANDLE),
    ]
    advapi32.OpenThreadToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    ]
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.GetSecurityInfo.argtypes = [
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.IsValidSid.argtypes = [ctypes.c_void_p]
    advapi32.IsValidSid.restype = wintypes.BOOL
    advapi32.IsValidAcl.argtypes = [ctypes.c_void_p]
    advapi32.IsValidAcl.restype = wintypes.BOOL
    advapi32.GetLengthSid.argtypes = [ctypes.c_void_p]
    advapi32.GetLengthSid.restype = wintypes.DWORD
    advapi32.ConvertSidToStringSidW.argtypes = [
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    advapi32.GetAce.restype = wintypes.BOOL
    return kernel32, advapi32


def _windows_sid_string(kernel32: Any, advapi32: Any, pointer: int) -> str:
    if not pointer or not advapi32.IsValidSid(pointer):
        raise WindowsPathPolicyError("MCP Windows file ACL has an invalid SID")
    text_pointer = ctypes.c_void_p()
    success = advapi32.ConvertSidToStringSidW(pointer, ctypes.byref(text_pointer))
    try:
        if not success or text_pointer.value is None:
            raise WindowsPathPolicyError("MCP Windows file ACL SID could not be read")
        return ctypes.wstring_at(text_pointer.value)
    finally:
        if text_pointer.value:
            kernel32.LocalFree(text_pointer)


def _windows_current_sids(kernel32: Any, advapi32: Any) -> tuple[str, str]:
    token = wintypes.HANDLE()
    if not advapi32.OpenThreadToken(kernel32.GetCurrentThread(), 0x0008, True, ctypes.byref(token)):
        if ctypes.get_last_error() != 1008:  # type: ignore[attr-defined]  # ERROR_NO_TOKEN
            raise WindowsPathPolicyError("MCP Windows process token could not be read")
        if not advapi32.OpenProcessToken(kernel32.GetCurrentProcess(), 0x0008, ctypes.byref(token)):
            raise WindowsPathPolicyError("MCP Windows process token could not be read")
    try:

        def sid_for(info_class: int) -> str:
            needed = wintypes.DWORD()
            advapi32.GetTokenInformation(token, info_class, None, 0, ctypes.byref(needed))
            if ctypes.get_last_error() != 122 or not 8 <= needed.value <= 4096:  # type: ignore[attr-defined]
                raise WindowsPathPolicyError(
                    "MCP Windows process token has invalid SID information"
                )
            buffer = ctypes.create_string_buffer(needed.value)
            if not advapi32.GetTokenInformation(
                token, info_class, buffer, needed.value, ctypes.byref(needed)
            ):
                raise WindowsPathPolicyError("MCP Windows process token could not be read")
            pointer = ctypes.cast(buffer, ctypes.POINTER(ctypes.c_void_p)).contents.value
            if pointer is None:
                raise WindowsPathPolicyError("MCP Windows process token has no SID")
            return _windows_sid_string(kernel32, advapi32, pointer)

        return sid_for(1), sid_for(4)  # TokenUser, TokenOwner
    finally:
        kernel32.CloseHandle(token)


def _windows_current_sid(kernel32: Any, advapi32: Any) -> str:
    return _windows_current_sids(kernel32, advapi32)[0]


def _windows_validate_handle_acl(
    kernel32: Any,
    advapi32: Any,
    handle: int,
    *,
    owner_required: bool,
    private_acl: bool,
    require_private_inheritance: bool,
) -> None:
    owner = ctypes.c_void_p()
    dacl = ctypes.c_void_p()
    descriptor = ctypes.c_void_p()
    error = advapi32.GetSecurityInfo(
        handle,
        1,  # SE_FILE_OBJECT
        0x0001 | 0x0004,  # OWNER_SECURITY_INFORMATION | DACL_SECURITY_INFORMATION
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    try:
        if error:
            raise WindowsPathPolicyError("MCP Windows file owner and ACL could not be read")
        if not owner.value or not dacl.value:
            raise WindowsPathPolicyError("MCP Windows file requires an owner and a non-null ACL")
        current_sid, token_owner_sid = _windows_current_sids(kernel32, advapi32)
        owner_sid = _windows_sid_string(kernel32, advapi32, owner.value)
        if not advapi32.IsValidAcl(dacl):
            raise WindowsPathPolicyError("MCP Windows file ACL is invalid")
        header = ctypes.cast(dacl, ctypes.POINTER(_WindowsAclHeader)).contents
        if header.size < ctypes.sizeof(_WindowsAclHeader):
            raise WindowsPathPolicyError("MCP Windows file ACL is invalid")
        aces: list[tuple[int, int, int, str]] = []
        for index in range(header.ace_count):
            ace = ctypes.c_void_p()
            if not advapi32.GetAce(dacl, index, ctypes.byref(ace)) or not ace.value:
                raise WindowsPathPolicyError("MCP Windows file ACL could not be read")
            ace_header = ctypes.cast(ace, ctypes.POINTER(_WindowsAceHeader)).contents
            if ace_header.size < 16:
                raise WindowsPathPolicyError("MCP Windows file ACL contains an invalid ACE")
            if ace_header.ace_type not in (0, 1):
                raise WindowsPathPolicyError("MCP Windows file ACL contains an unsupported ACE")
            mask = ctypes.c_uint32.from_address(ace.value + 4).value
            sid_pointer = ace.value + 8
            if not advapi32.IsValidSid(sid_pointer):
                raise WindowsPathPolicyError("MCP Windows file ACL contains an invalid SID")
            if advapi32.GetLengthSid(sid_pointer) > ace_header.size - 8:
                raise WindowsPathPolicyError("MCP Windows file ACL contains an invalid SID")
            sid = _windows_sid_string(kernel32, advapi32, sid_pointer)
            aces.append((ace_header.ace_type, ace_header.flags, mask, sid))
        _validate_windows_acl(
            owner_sid,
            current_sid,
            aces,
            token_owner_sid=token_owner_sid,
            owner_required=owner_required,
            private_acl=private_acl,
            require_private_inheritance=require_private_inheritance,
        )
    finally:
        if descriptor.value:
            kernel32.LocalFree(descriptor)


def _windows_drive_root(path: Path) -> str:
    drive, tail = ntpath.splitdrive(str(path))
    if not path.is_absolute() or re.fullmatch(r"[A-Za-z]:", drive) is None:
        raise WindowsPathPolicyError("MCP Windows paths must be absolute local-drive paths")
    if ":" in tail:
        raise WindowsPathPolicyError("MCP Windows paths must not contain alternate data streams")
    if ".." in path.parts:
        raise WindowsPathPolicyError("MCP Windows paths must not contain parent traversal")
    return drive + "\\"


def _require_local_windows_drive(path: Path) -> None:
    root = _windows_drive_root(path)
    try:
        kernel32, _ = _windows_apis()
        drive_type = kernel32.GetDriveTypeW(root)
    except (AttributeError, OSError, TypeError, ValueError) as error:
        raise WindowsPathPolicyError("MCP Windows drive type could not be verified") from error
    if drive_type != 3:  # DRIVE_FIXED; reject remote, unknown, and removable volumes.
        raise WindowsPathPolicyError("MCP Windows path must be on a fixed local drive")


def _windows_check_reparse_components(path: Path, *, include_self: bool = True) -> None:
    _require_local_windows_drive(path)
    parts = (path, *path.parents) if include_self else tuple(path.parents)
    for part in reversed(parts):
        metadata = os.lstat(part)
        attributes = getattr(metadata, "st_file_attributes", None)
        if attributes is None or attributes & _WINDOWS_REPARSE_POINT:
            raise WindowsPathPolicyError("MCP Windows path must not contain reparse points")


@contextmanager
def _windows_checked_handle(
    path: Path,
    *,
    directory: bool,
    owner_required: bool = True,
    private_acl: bool = False,
    require_private_inheritance: bool = False,
) -> Iterator[tuple[Any, int]]:
    """Open the named local-drive object and validate its actual handle."""
    if require_private_inheritance and (not directory or not private_acl):
        raise ValueError("private inheritance requires a private directory")
    _windows_check_reparse_components(path, include_self=False)
    kernel32, advapi32 = _windows_apis()
    handle = kernel32.CreateFileW(
        str(path),
        0x0001 | 0x0080 | (0x00020000 if owner_required or private_acl else 0),
        0x0001 | 0x0002 | 0x0004,  # FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE
        None,
        3,  # OPEN_EXISTING
        0x00200000 | (0x02000000 if directory else 0),
        None,
    )
    if handle == ctypes.c_void_p(-1).value or handle is None:
        raise ValueError("MCP Windows path could not be securely opened")
    try:
        info = _WindowsFileAttributeTagInfo()
        if not kernel32.GetFileInformationByHandleEx(
            handle, 9, ctypes.byref(info), ctypes.sizeof(info)
        ):
            raise WindowsPathPolicyError("MCP Windows path attributes could not be read")
        if (
            info.attributes & (_WINDOWS_REPARSE_POINT | 0x0040)
            or bool(info.attributes & _WINDOWS_DIRECTORY) != directory
        ):
            raise WindowsPathPolicyError("MCP Windows path is a reparse point or wrong file type")
        final = ctypes.create_unicode_buffer(32768)
        length = kernel32.GetFinalPathNameByHandleW(handle, final, len(final), 0)
        if not length or length >= len(final):
            raise WindowsPathPolicyError("MCP Windows final path could not be verified")
        resolved = final.value
        if not resolved.startswith("\\\\?\\"):
            raise WindowsPathPolicyError("MCP Windows final path could not be verified")
        resolved = resolved[4:]
        if ntpath.normcase(ntpath.normpath(resolved)) != ntpath.normcase(
            ntpath.normpath(str(path))
        ):
            raise WindowsPathPolicyError("MCP Windows path resolved through a link or alias")
        if owner_required or private_acl:
            _windows_validate_handle_acl(
                kernel32,
                advapi32,
                handle,
                owner_required=owner_required,
                private_acl=private_acl,
                require_private_inheritance=require_private_inheritance,
            )
        yield kernel32, cast(int, handle)
    finally:
        kernel32.CloseHandle(handle)


def require_windows_file_apis() -> None:
    """Fail before host startup if native Windows security APIs are unavailable."""
    if os.name != "nt":
        raise ValueError("MCP host requires native Windows file security APIs")
    try:
        _windows_apis()
    except (AttributeError, OSError, ValueError) as error:
        raise ValueError("MCP host requires native Windows file security APIs") from error


def reject_windows_reparse_components(path: Path, *, include_self: bool = True) -> None:
    """Reject links and junctions in every existing component of a path."""
    _windows_check_reparse_components(path, include_self=include_self)


def validate_windows_path(
    path: Path,
    *,
    directory: bool,
    owner_required: bool = True,
    private_acl: bool = False,
    require_private_inheritance: bool = False,
) -> None:
    """Validate the actual Windows file or directory behind a local-drive path."""
    with _windows_checked_handle(
        path,
        directory=directory,
        owner_required=owner_required,
        private_acl=private_acl,
        require_private_inheritance=require_private_inheritance,
    ):
        pass


def validate_windows_file_target(path: Path) -> None:
    """Check a private file path before creating or opening a trace store."""
    validate_windows_path(
        path.parent, directory=True, private_acl=True, require_private_inheritance=True
    )
    reject_windows_reparse_components(path, include_self=False)
    try:
        os.lstat(path)
    except FileNotFoundError:
        pass
    else:
        validate_windows_path(path, directory=False, private_acl=True)


def read_windows_secure_file(
    path: Path, *, max_bytes: int, owner_required: bool = True, private_acl: bool = False
) -> bytes:
    """Read a bounded regular file from one validated no-follow Windows handle."""
    if max_bytes < 0:
        raise ValueError("MCP Windows file size limit must be nonnegative")
    with _windows_checked_handle(
        path, directory=False, owner_required=owner_required, private_acl=private_acl
    ) as (kernel32, handle):
        size = ctypes.c_longlong()
        if (
            not kernel32.GetFileSizeEx(handle, ctypes.byref(size))
            or not 0 <= size.value <= max_bytes
        ):
            raise ValueError("MCP Windows file exceeds its size limit or has no bounded file size")
        if size.value == 0:
            return b""
        buffer = ctypes.create_string_buffer(size.value)
        offset = 0
        while offset < size.value:
            read = wintypes.DWORD()
            if (
                not kernel32.ReadFile(
                    handle,
                    ctypes.byref(buffer, offset),
                    size.value - offset,
                    ctypes.byref(read),
                    None,
                )
                or read.value == 0
            ):
                raise ValueError("MCP Windows file could not be read completely")
            offset += read.value
        return buffer.raw
