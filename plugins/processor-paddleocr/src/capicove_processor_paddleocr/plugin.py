"""Dependency-light PaddleOCR plugin descriptor."""

from __future__ import annotations

import importlib.util
from collections.abc import Mapping
from pathlib import Path

from capicove.documents import DocumentParserBackend
from capicove.plugins import PluginAvailability, PluginMetadata
from capicove_processor_paddleocr.backend import PaddleOcrDocumentParserBackend
from capicove_processor_paddleocr.models import (
    ModelInstallationPlan,
    install_models,
    load_book_balanced_manifest,
    plan_model_installation,
    verify_installed_models,
)
from capicove_processor_paddleocr.runtime import resolve_runtime_identity


class PaddleOcrDocumentParserPlugin:
    """Expose PP-StructureV3 without importing its runtime during discovery."""

    @property
    def metadata(self) -> PluginMetadata:
        return PluginMetadata(
            plugin_id="capicove.processor.paddleocr",
            plugin_version="0.1.0",
            capability="document_parser",
            contract_min=1,
            contract_max=1,
            optional_packages=(
                "paddleocr",
                "paddlepaddle",
                "paddlex",
                "pypdf",
                "pypdfium2",
            ),
            external_binaries=("hf",),
            network_access=False,
        )

    def availability(self) -> PluginAvailability:
        required_modules = (
            ("paddleocr", "install capicove-processor-paddleocr with paddleocr[doc-parser]"),
            ("paddle", "install a compatible PaddlePaddle runtime"),
            ("paddlex", "install capicove-processor-paddleocr with PaddleX"),
            ("pypdf", "install capicove-processor-paddleocr with pypdf"),
            ("pypdfium2", "install capicove-processor-paddleocr with pypdfium2"),
        )
        for module, detail in required_modules:
            if importlib.util.find_spec(module) is None:
                return PluginAvailability(False, detail)
        try:
            resolve_runtime_identity(load_book_balanced_manifest().inference_engine)
        except (RuntimeError, ValueError) as error:
            return PluginAvailability(False, str(error))
        return PluginAvailability(True)

    def create(self, configuration: Mapping[str, object]) -> DocumentParserBackend:
        model_root = configuration.get("model_root")
        device = configuration.get("device", "cpu")
        if not isinstance(model_root, str) or not model_root.strip():
            raise ValueError("model_root is required")
        if not isinstance(device, str) or not device.strip():
            raise ValueError("device must be a nonblank string")
        enable_mkldnn = configuration.get("enable_mkldnn", False)
        if type(enable_mkldnn) is not bool:
            raise ValueError("enable_mkldnn must be a boolean")
        return PaddleOcrDocumentParserBackend(
            Path(model_root), device=device, enable_mkldnn=enable_mkldnn
        )

    def model_plan(self, profile_id: str, destination: Path) -> ModelInstallationPlan:
        manifest = load_book_balanced_manifest()
        if profile_id != manifest.profile_id:
            raise ValueError(f"unknown PaddleOCR profile: {profile_id}")
        return plan_model_installation(manifest, destination)

    def install_models(
        self,
        profile_id: str,
        destination: Path,
        *,
        confirmed: bool,
    ) -> ModelInstallationPlan:
        manifest = load_book_balanced_manifest()
        if profile_id != manifest.profile_id:
            raise ValueError(f"unknown PaddleOCR profile: {profile_id}")
        return install_models(manifest, destination, confirmed=confirmed)

    def verify_models(self, profile_id: str, destination: Path) -> None:
        """Verify Paddle's locked format without weakening other asset validators."""
        manifest = load_book_balanced_manifest()
        if profile_id != manifest.profile_id:
            raise ValueError(f"unknown PaddleOCR profile: {profile_id}")
        verify_installed_models(manifest, destination)


def paddleocr_document_parser_plugin() -> PaddleOcrDocumentParserPlugin:
    """Return the dependency-light plugin descriptor."""
    return PaddleOcrDocumentParserPlugin()


__all__ = ["PaddleOcrDocumentParserPlugin", "paddleocr_document_parser_plugin"]
