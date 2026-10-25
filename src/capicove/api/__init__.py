"""Optional HTTP application boundaries, loaded only when explicitly requested."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    from capicove.api.search import create_search_app
    from capicove.api.siglip2 import build_service, create_app

_EXPORTS = {
    "build_service": "capicove.api.siglip2",
    "create_app": "capicove.api.siglip2",
    "create_search_app": "capicove.api.search",
}


def __getattr__(name: str) -> Any:
    """Import one optional boundary on first use; model SDKs stay unloaded otherwise."""
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(name)
    value = getattr(import_module(module), name)
    globals()[name] = value
    return value


__all__ = ["build_service", "create_app", "create_search_app"]
