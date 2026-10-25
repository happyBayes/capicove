from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import capicove_processor_paddleocr.pdf_metadata as metadata
import pytest
from capicove_processor_paddleocr.pdf_metadata import translate_outline
from pytest import MonkeyPatch


@dataclass(frozen=True)
class _Destination:
    title: str
    page: int


@pytest.fixture(autouse=True)
def _stub_verified_pypdf_import(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        metadata,
        "import_verified_runtime_module",
        lambda name: sys.modules[name],
    )


def _page_number(value: object) -> int:
    if not isinstance(value, _Destination):
        raise TypeError("value is not a test destination")
    return value.page


def test_nested_pdf_outline_translates_to_canonical_levels() -> None:
    introduction = _Destination("Introduction", 2)
    scope = _Destination("1.1 Scope", 4)
    details = _Destination("1.2 Details", 8)
    next_chapter = _Destination("Chapter 2", 12)

    outline = translate_outline(
        [introduction, [scope, details], next_chapter],
        page_number=_page_number,
    )

    assert [(item.title, item.level, item.page_index) for item in outline] == [
        ("Introduction", 1, 2),
        ("Chapter 2", 1, 12),
    ]
    assert [(item.title, item.level) for item in outline[0].children] == [
        ("1.1 Scope", 2),
        ("1.2 Details", 2),
    ]


def test_invalid_outline_destinations_are_skipped() -> None:
    outline = translate_outline(
        [_Destination("Valid", 1), object(), _Destination("Later", 3)],
        page_number=_page_number,
    )

    assert [item.title for item in outline] == ["Valid", "Later"]


def test_fallback_pdf_numbers_are_not_printed_labels(monkeypatch: MonkeyPatch) -> None:
    reader = SimpleNamespace(
        outline=[],
        page_labels=["1", "2"],
        trailer={"/Root": {}},
        get_destination_page_number=_page_number,
    )
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    _, labels = metadata.extract_pdf_metadata(Path("authorized.pdf"))

    assert labels == {}


def test_explicit_pdf_labels_are_preserved(monkeypatch: MonkeyPatch) -> None:
    reader = SimpleNamespace(
        outline=[],
        page_labels=["i", "1"],
        trailer={"/Root": {"/PageLabels": {}}},
        get_destination_page_number=_page_number,
    )
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    _, labels = metadata.extract_pdf_metadata(Path("authorized.pdf"))

    assert labels == {0: "i", 1: "1"}


def test_encrypted_pdf_is_rejected_before_accessing_outline(monkeypatch: MonkeyPatch) -> None:
    class Reader:
        is_encrypted = True

        @property
        def outline(self) -> object:
            raise AssertionError("encrypted outline must not be read")

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: Reader()))
    with pytest.raises(ValueError, match="encrypted"):
        metadata.extract_pdf_metadata(Path("encrypted.pdf"))


def test_unverified_pypdf_is_rejected_before_reader_construction(
    monkeypatch: MonkeyPatch,
) -> None:
    called = False

    def reader(path: Path) -> object:
        nonlocal called
        called = True
        return object()

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=reader))

    def reject(name: str) -> object:
        raise RuntimeError(f"unverified runtime module: {name}")

    monkeypatch.setattr(metadata, "import_verified_runtime_module", reject)

    with pytest.raises(RuntimeError, match="unverified runtime module"):
        metadata.extract_pdf_metadata(Path("authorized.pdf"))
    assert called is False
