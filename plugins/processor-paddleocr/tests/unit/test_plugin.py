from __future__ import annotations

import importlib.machinery
import importlib.util
import sys
from pathlib import Path

import pytest
from capicove_processor_paddleocr import paddleocr_document_parser_plugin
from capicove_processor_paddleocr.backend import pipeline_configuration, select_requested_pages
from pytest import MonkeyPatch

from capicove.plugins import DocumentParserPlugin


def test_plugin_import_is_lazy_and_metadata_is_compatible() -> None:
    plugin = paddleocr_document_parser_plugin()

    assert isinstance(plugin, DocumentParserPlugin)
    assert plugin.metadata.capability == "document_parser"
    assert plugin.metadata.optional_packages == (
        "paddleocr",
        "paddlepaddle",
        "paddlex",
        "pypdf",
        "pypdfium2",
    )
    assert plugin.metadata.network_access is False
    assert "paddleocr" not in sys.modules
    assert "paddle" not in sys.modules


@pytest.mark.parametrize(
    ("available", "expected"),
    (
        (frozenset(), "paddleocr[doc-parser]"),
        (frozenset({"paddleocr"}), "PaddlePaddle runtime"),
        (frozenset({"paddleocr", "paddle"}), "PaddleX"),
        (frozenset({"paddleocr", "paddle", "paddlex"}), "pypdf"),
        (frozenset({"paddleocr", "paddle", "paddlex", "pypdf"}), "pypdfium2"),
    ),
)
def test_plugin_reports_each_installation_step(
    monkeypatch: MonkeyPatch,
    available: frozenset[str],
    expected: str,
) -> None:
    def find_spec(name: str) -> importlib.machinery.ModuleSpec | None:
        if name in available:
            return importlib.machinery.ModuleSpec(name, loader=None)
        return None

    monkeypatch.setattr(importlib.util, "find_spec", find_spec)

    result = paddleocr_document_parser_plugin().availability()

    assert result.available is False
    assert expected in (result.detail or "")


def test_plugin_availability_rejects_an_incompatible_runtime(
    monkeypatch: MonkeyPatch,
) -> None:
    def reject_runtime(inference_engine: object) -> tuple[tuple[str, str], ...]:
        del inference_engine
        raise RuntimeError("incompatible paddleocr runtime 4.0.0")

    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None),
    )
    monkeypatch.setattr(
        "capicove_processor_paddleocr.plugin.resolve_runtime_identity",
        reject_runtime,
    )

    result = paddleocr_document_parser_plugin().availability()

    assert result.available is False
    assert result.detail == "incompatible paddleocr runtime 4.0.0"


def test_plugin_availability_accepts_the_validated_runtime(
    monkeypatch: MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: importlib.machinery.ModuleSpec(name, loader=None),
    )
    monkeypatch.setattr(
        "capicove_processor_paddleocr.plugin.resolve_runtime_identity",
        lambda inference_engine: (("paddleocr.version", "3.7.0"),),
    )

    result = paddleocr_document_parser_plugin().availability()

    assert result.available is True


def test_pipeline_configuration_binds_local_models_and_disables_optional_modules(
    tmp_path: Path,
) -> None:
    for name in ("pp-doclayout-l", "pp-ocrv5-server-det", "pp-ocrv5-server-rec"):
        (tmp_path / name).mkdir()

    configuration = pipeline_configuration(tmp_path, device="cpu")

    assert configuration == {
        "layout_detection_model_name": "PP-DocLayout-L",
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_recognition_model_name": "PP-OCRv5_server_rec",
        "layout_detection_model_dir": str((tmp_path / "pp-doclayout-l").resolve()),
        "text_detection_model_dir": str((tmp_path / "pp-ocrv5-server-det").resolve()),
        "text_recognition_model_dir": str((tmp_path / "pp-ocrv5-server-rec").resolve()),
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "use_table_recognition": False,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
        "use_seal_recognition": False,
        "use_region_detection": False,
        "device": "cpu",
        "text_det_limit_side_len": 736,
        "text_det_limit_type": "max",
        "text_recognition_batch_size": 1,
        "cpu_threads": 2,
    }


def test_pipeline_configuration_rejects_missing_model_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="missing model"):
        pipeline_configuration(tmp_path, device="cpu")


def test_requested_page_range_preserves_original_pdf_indices() -> None:
    raw = tuple({"res": {"page_index": index}} for index in range(4))

    selected = select_requested_pages(raw, start=1, end=3)

    assert [item["res"]["page_index"] for item in selected] == [1, 2]  # type: ignore[index]
