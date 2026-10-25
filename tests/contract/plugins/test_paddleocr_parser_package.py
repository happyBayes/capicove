"""Distribution contract for the optional PaddleOCR document parser."""

from __future__ import annotations

import tomllib
from pathlib import Path


PLUGIN_ROOT = Path("plugins/processor-paddleocr")


def test_paddleocr_parser_declares_plugin_dependencies_and_external_runtime() -> None:
    project = tomllib.loads((PLUGIN_ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    dependencies = set(project["dependencies"])

    assert project["entry-points"]["capicove.document_parsers"] == {
        "paddleocr": ("capicove_processor_paddleocr:paddleocr_document_parser_plugin")
    }
    assert {
        "huggingface-hub>=0.35,<2",
        "packaging>=24,<27",
        "paddleocr[doc-parser]>=3.7,<3.8",
        "pypdf>=5,<7",
    } <= dependencies
    assert not any(dependency.startswith("paddlepaddle") for dependency in dependencies)
