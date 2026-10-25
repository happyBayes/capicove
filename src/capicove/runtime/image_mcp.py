"""Compose deterministic image Tools using explicit startup filesystem roots."""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from pathlib import Path

from capicove.contracts import InvocationContext, ToolSpec
from capicove.mcp.bindings import McpToolBinding
from capicove.runtime.mcp_config import McpHostConfiguration
from capicove.traces import TraceRecorder


def _open_posix_image_root(root: Path) -> int:
    """Pin a configured image root without following any ancestor link."""
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    try:
        descriptor = os.open(root.anchor, flags)
    except OSError:
        raise ValueError("image read_roots must be existing directories without symlinks") from None
    try:
        for component in root.parts[1:]:
            next_descriptor = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = next_descriptor
        return descriptor
    except OSError:
        os.close(descriptor)
        raise ValueError("image read_roots must be existing directories without symlinks") from None


def _require_current_posix_image_root(root: Path, pinned: int) -> None:
    """Reject a configured image root replaced during provider composition."""
    try:
        current = _open_posix_image_root(root)
    except ValueError:
        raise ValueError("image read_roots changed during startup") from None
    try:
        before = os.fstat(pinned)
        after = os.fstat(current)
        if (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino):
            raise ValueError("image read_roots changed during startup")
    finally:
        os.close(current)


@contextmanager
def compose_image_mcp(
    host: McpHostConfiguration,
    context_factory: Callable[[ToolSpec], InvocationContext],
    recorder: TraceRecorder | None,
) -> Iterator[tuple[McpToolBinding, ...]]:
    """Build only selected local providers; do not acquire models or access a network."""
    from capicove.processing.image import (
        ImageAspectRatioFilter,
        ImageLshSearch,
        ImagePhash,
        ImageShapeFilter,
        ImageSizeFilter,
        ImageSubplotFilter,
    )
    from capicove.tools.image import IMAGE_TOOL_SPECS_BY_NAME, build_image_tool_handler

    if set(host.capability_config) != {"read_roots"}:
        raise ValueError("image capability_config requires only read_roots")
    raw_roots = host.capability_config["read_roots"]
    if not isinstance(raw_roots, tuple) or not raw_roots or len(raw_roots) > 32:
        raise ValueError("image read_roots must contain 1 to 32 absolute directories")
    with ExitStack() as stack:
        roots: list[Path] = []
        pinned_roots: list[tuple[Path, int]] = []
        pinned_identities: dict[Path, tuple[int, int]] = {}
        for raw_root in raw_roots:
            if not isinstance(raw_root, str) or not Path(raw_root).is_absolute():
                raise ValueError("image read_roots must contain absolute directories")
            root = Path(raw_root)
            if os.name == "nt":
                from capicove._windows_files import WindowsPathPolicyError, validate_windows_path

                try:
                    validate_windows_path(root, directory=True, owner_required=False)
                except (OSError, WindowsPathPolicyError):
                    raise ValueError("image read_roots violate Windows path policy") from None
                roots.append(root)
            else:
                descriptor = _open_posix_image_root(root)
                stack.callback(os.close, descriptor)
                pinned_roots.append((root, descriptor))
                try:
                    resolved = root.resolve(strict=True)
                except OSError:
                    raise ValueError("image read_roots changed during startup") from None
                roots.append(resolved)
                metadata = os.fstat(descriptor)
                pinned_identities[resolved] = (metadata.st_dev, metadata.st_ino)
        if len(set(roots)) != len(roots):
            raise ValueError("image read_roots must not contain duplicates")
        factories: dict[str, Callable[[], object]] = {
            "image_aspect_ratio_filter": ImageAspectRatioFilter,
            "image_shape_filter": ImageShapeFilter,
            "image_size_filter": ImageSizeFilter,
            "image_subplot_filter": ImageSubplotFilter,
            "image_phash": ImagePhash,
            "image_lsh_search": ImageLshSearch,
        }
        supported = {IMAGE_TOOL_SPECS_BY_NAME[name].tool_id: name for name in factories}
        if any(tool_id not in supported for tool_id in host.enabled_tools):
            raise ValueError(
                "image profile includes an unavailable Tool; select a supported local Tool"
            )
        bindings: list[McpToolBinding] = []
        for tool_id in sorted(host.enabled_tools):
            name = supported[tool_id]
            spec = IMAGE_TOOL_SPECS_BY_NAME[name]
            if not spec.required_permissions.issubset(host.permissions):
                raise ValueError("configured permissions do not authorize every enabled Tool")
            bindings.append(
                McpToolBinding(
                    "capicove_"
                    + tool_id.removeprefix("capicove.").replace(".", "_").replace("-", "_"),
                    spec,
                    build_image_tool_handler(
                        name,
                        factories[name](),
                        read_roots=tuple(roots),
                        read_root_identities=pinned_identities if os.name == "posix" else None,
                    ),
                    context_factory,
                    recorder=recorder,
                )
            )
        for root, descriptor in pinned_roots:
            _require_current_posix_image_root(root, descriptor)
        yield tuple(bindings)
