"""Capability-neutral explicit model installer discovery (RFC-0023)."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Protocol, runtime_checkable

from capicove.model._entry_points import select_entry_points


@runtime_checkable
class ModelInstaller(Protocol):
    """Plan pinned files and install only following explicit authorization.

    Plans retain RFC-0016's inventory shape. Converted assets require their own
    verified receipts; unknown output hashes cannot be treated as pinned sources.
    """

    def model_plan(self, profile_id: str, destination: Path) -> object:
        """Return the existing immutable installation plan without networking."""

    def install_models(self, profile_id: str, destination: Path, *, confirmed: bool) -> object:
        """Install only the planned inventory after operator confirmation."""


def load_model_installer(
    name: str, *, entry_points: Iterable[object] | None = None
) -> ModelInstaller:
    """Load a selected installer or an existing parser installer compatibly."""
    if name == "bge-reranker-v2-m3":
        from capicove.model.reranker_assets import BgeRerankerInstaller

        return BgeRerankerInstaller()
    matches = select_entry_points("capicove.model_installers", name, entry_points)
    if len(matches) > 1:
        raise ValueError("duplicate model installer entry point")
    if matches:
        factory = matches[0].load()
        if not callable(factory):
            raise TypeError("model installer entry point must load a callable")
        installer = factory()
    else:
        from capicove.plugins import load_document_parser_plugin

        installer = load_document_parser_plugin(name)
    if not isinstance(installer, ModelInstaller):
        raise TypeError("incompatible model installer")
    return installer
