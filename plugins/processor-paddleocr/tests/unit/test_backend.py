from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from capicove_processor_paddleocr.backend import (
    PaddleOcrDocumentParserBackend,
    _dominant_unsupported_script,
    _embedded_text,
    _predict_pages,
    _restrict_outline,
)
from pytest import MonkeyPatch

from capicove.contracts import (
    DocumentOutlineEntry,
    DocumentPageRange,
    DocumentParseRequest,
    LocalPathSource,
)
from capicove.documents import DocumentBackendUnavailableError


def _backend(tmp_path: Path) -> PaddleOcrDocumentParserBackend:
    for name in ("pp-doclayout-l", "pp-ocrv5-server-det", "pp-ocrv5-server-rec"):
        (tmp_path / name).mkdir()
    return PaddleOcrDocumentParserBackend(tmp_path)


def test_cpu_kernel_selection_is_explicit_configuration(tmp_path: Path) -> None:
    for name in ("pp-doclayout-l", "pp-ocrv5-server-det", "pp-ocrv5-server-rec"):
        (tmp_path / name).mkdir()
    backend = PaddleOcrDocumentParserBackend(tmp_path, enable_mkldnn=False)
    assert backend._configuration["enable_mkldnn"] is False


def _usable_raw_page(*, page_index: int = 0) -> dict[str, object]:
    return {
        "page_index": page_index,
        "width": 100,
        "height": 200,
        "parsing_res_list": [
            {
                "block_id": 0,
                "block_label": "text",
                "block_content": "English source text",
                "block_bbox": [10, 10, 90, 30],
            }
        ],
    }


@pytest.fixture(autouse=True)
def _stub_verified_model_installation(monkeypatch: MonkeyPatch) -> None:
    import capicove_processor_paddleocr.backend as backend
    import capicove_processor_paddleocr.pdf_metadata as pdf_metadata

    monkeypatch.setattr(
        "capicove_processor_paddleocr.backend.verify_installed_models",
        lambda manifest, root: None,
    )
    monkeypatch.setattr(
        "capicove_processor_paddleocr.backend.resolve_runtime_identity",
        lambda inference_engine: (("paddleocr.version", "3.7.0"),),
    )
    monkeypatch.setattr(backend, "verify_loaded_runtime_modules", lambda: None)
    monkeypatch.setattr(backend, "enforce_offline_runtime_policy", lambda: None)
    monkeypatch.setattr(
        backend,
        "import_verified_runtime_module",
        lambda name: sys.modules[name],
    )
    monkeypatch.setattr(
        pdf_metadata,
        "import_verified_runtime_module",
        lambda name: sys.modules[name],
    )


def test_inference_receives_only_requested_pages_one_at_a_time(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    source = tmp_path / "source.pdf"
    source.write_bytes(b"authorized PDF")
    seen: list[list[int]] = []

    class Page:
        def __init__(self, index: int) -> None:
            self.index = index
            self.images: tuple[object, ...] = ()

        def extract_text(self) -> str:
            return "English source text"

    class Writer:
        def __init__(self) -> None:
            self.pages: list[int] = []

        def add_page(self, page: Page) -> None:
            self.pages.append(page.index)

        def write(self, path: object) -> None:
            Path(str(path)).write_text(json.dumps(self.pages))

    reader = SimpleNamespace(
        pages=[Page(0), Page(1), Page(2), Page(3)],
        outline=[],
        trailer={"/Root": {}},
        is_encrypted=False,
        get_destination_page_number=lambda value: 0,
    )
    monkeypatch.setitem(
        sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader, PdfWriter=Writer)
    )

    class Pipeline:
        def __init__(self, **configuration: object) -> None:
            pass

        def predict(self, *, input: str) -> list[dict[str, object]]:
            assert Path(input) != source, "the whole PDF must never enter inference"
            pages = json.loads(Path(input).read_text())
            seen.append(pages)
            assert len(pages) == 1
            return [_usable_raw_page()]

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "book-balanced",
        2,
        page_range=DocumentPageRange(1, 3),
    )

    result = _backend(tmp_path).parse(request)

    assert seen == [[1], [2]]
    assert [page.page_index for page in result.pages] == [1, 2]
    assert source.read_bytes() == b"authorized PDF"


def test_source_hash_is_checked_before_importing_runtime(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.backend as backend

    def forbidden_metadata(path: Path) -> object:
        raise AssertionError("mismatched source reached the PDF dependency")

    monkeypatch.setattr(backend, "extract_pdf_metadata", forbidden_metadata)
    source = tmp_path / "source.pdf"
    source.write_bytes(b"changed PDF")
    request = DocumentParseRequest(
        LocalPathSource(str(source)), "0" * 64, "application/pdf", "book-balanced", 2
    )

    with pytest.raises(ValueError, match=r"source.*hash"):
        _backend(tmp_path).parse(request)


def test_downstream_processing_uses_stable_authorized_source_snapshot(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.backend as backend

    source = tmp_path / "source.pdf"
    source.write_bytes(b"authorized PDF")
    snapshots: list[Path] = []

    def metadata(path: Path) -> tuple[tuple[object, ...], dict[int, str]]:
        snapshots.append(path)
        assert path != source
        assert path.read_bytes() == b"authorized PDF"
        source.write_bytes(b"changed after authorization")
        return (), {}

    def predict(
        path: Path, request: DocumentParseRequest, configuration: dict[str, object]
    ) -> tuple[dict[str, object], ...]:
        del request, configuration
        snapshots.append(path)
        assert path.read_bytes() == b"authorized PDF"
        return (_usable_raw_page(),)

    monkeypatch.setattr(backend, "extract_pdf_metadata", metadata)
    monkeypatch.setattr(backend, "_predict_pages", predict)
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "book-balanced",
        2,
    )

    result = _backend(tmp_path).parse(request)

    assert len(result.pages) == 1
    assert snapshots[0] == snapshots[1]
    assert not snapshots[0].exists()
    assert source.read_bytes() == b"changed after authorization"


def test_model_identity_is_verified_before_source_processing(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.backend as backend

    source = tmp_path / "source.pdf"
    source.write_bytes(b"authorized PDF")

    def reject_models(manifest: object, root: Path) -> None:
        del manifest, root
        raise ValueError("model identity is invalid")

    def forbidden_metadata(path: Path) -> object:
        raise AssertionError("invalid models reached the PDF dependency")

    monkeypatch.setattr(backend, "verify_installed_models", reject_models)
    monkeypatch.setattr(backend, "extract_pdf_metadata", forbidden_metadata)
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "book-balanced",
        2,
    )

    with pytest.raises(DocumentBackendUnavailableError) as caught:
        _backend(tmp_path).parse(request)
    assert isinstance(caught.value.__cause__, ValueError)
    assert str(caught.value.__cause__) == "model identity is invalid"


def test_runtime_identity_is_verified_before_models_or_source(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.backend as backend

    source = tmp_path / "source.pdf"
    source.write_bytes(b"authorized PDF")

    def reject_runtime(inference_engine: object) -> tuple[tuple[str, str], ...]:
        del inference_engine
        raise RuntimeError("incompatible paddleocr runtime")

    def forbidden_models(manifest: object, root: Path) -> None:
        raise AssertionError("invalid runtime reached model verification")

    def forbidden_snapshot(source: Path, destination: Path, expected: str) -> None:
        raise AssertionError("invalid runtime reached source access")

    monkeypatch.setattr(backend, "resolve_runtime_identity", reject_runtime)
    monkeypatch.setattr(backend, "verify_installed_models", forbidden_models)
    monkeypatch.setattr(backend, "_snapshot_authorized_source", forbidden_snapshot)
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "book-balanced",
        2,
    )

    with pytest.raises(DocumentBackendUnavailableError) as caught:
        _backend(tmp_path).parse(request)
    assert str(caught.value.__cause__) == "incompatible paddleocr runtime"


def test_relative_model_root_is_bound_when_backend_is_constructed(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    import capicove_processor_paddleocr.backend as backend

    first = tmp_path / "first"
    second = tmp_path / "second"
    model_root = first / "models"
    second.mkdir()
    for name in ("pp-doclayout-l", "pp-ocrv5-server-det", "pp-ocrv5-server-rec"):
        (model_root / name).mkdir(parents=True)
    monkeypatch.chdir(first)
    parser = PaddleOcrDocumentParserBackend(Path("models"))
    monkeypatch.chdir(second)
    seen: list[Path] = []

    def stop_after_verification(manifest: object, root: Path) -> None:
        del manifest
        seen.append(root)
        raise RuntimeError("verification reached")

    monkeypatch.setattr(backend, "verify_installed_models", stop_after_verification)
    source = second / "source.pdf"
    source.write_bytes(b"authorized PDF")
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "book-balanced",
        2,
    )

    with pytest.raises(DocumentBackendUnavailableError) as caught:
        parser.parse(request)
    assert str(caught.value.__cause__) == "verification reached"
    assert seen == [model_root.resolve()]


@pytest.mark.parametrize(
    "runtime_key",
    ("paddleocr.version", "paddlex.version", "pypdfium2.version"),
)
def test_exact_runtime_versions_change_receipt_fingerprints(
    tmp_path: Path, monkeypatch: MonkeyPatch, runtime_key: str
) -> None:
    import capicove_processor_paddleocr.backend as backend

    source = tmp_path / "source.pdf"
    source.write_bytes(b"authorized PDF")
    identities = iter(
        (
            ((runtime_key, "3.7.0"),),
            ((runtime_key, "3.7.1"),),
        )
    )
    monkeypatch.setattr(
        backend,
        "resolve_runtime_identity",
        lambda inference_engine: next(identities),
    )
    monkeypatch.setattr(backend, "extract_pdf_metadata", lambda path: ((), {}))
    monkeypatch.setattr(
        backend,
        "_predict_pages",
        lambda path, request, configuration: (_usable_raw_page(),),
    )
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "book-balanced",
        2,
    )
    parser = _backend(tmp_path)

    first = parser.parse(request)
    second = parser.parse(request)

    assert first.receipt.configuration_sha256 != second.receipt.configuration_sha256
    assert first.receipt.processing_sha256 != second.receipt.processing_sha256


def test_range_keeps_descendant_bookmarks_when_parent_starts_before_range() -> None:
    outline = (DocumentOutlineEntry("Chapter 1", 1, 0, (DocumentOutlineEntry("1.1 Scope", 2, 5),)),)
    restricted = _restrict_outline(outline, frozenset({5}))
    assert [(entry.title, entry.level, entry.page_index) for entry in restricted] == [
        ("1.1 Scope", 2, 5),
    ]


@pytest.mark.parametrize("encrypted", [False, True])
def test_pdf_guards_run_before_model_initialization(
    tmp_path: Path, monkeypatch: MonkeyPatch, encrypted: bool
) -> None:
    reader = SimpleNamespace(pages=[0], is_encrypted=encrypted)
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    def forbidden_pipeline(**configuration: object) -> None:
        raise AssertionError("invalid PDF reached model initialization")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=forbidden_pipeline))
    path = tmp_path / "source.pdf"
    request = DocumentParseRequest(
        LocalPathSource(str(path)),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        1,
        page_range=DocumentPageRange(1, 3),
    )
    with pytest.raises(ValueError, match="encrypted" if encrypted else "page bounds"):
        _predict_pages(path, request, {})


@pytest.mark.parametrize("count", [0, 2])
def test_single_page_result_count_is_validated_and_temporary_file_removed(
    tmp_path: Path, monkeypatch: MonkeyPatch, count: int
) -> None:
    page = SimpleNamespace(extract_text=lambda: "English source text", images=())
    reader = SimpleNamespace(pages=[page], is_encrypted=False)
    temporary_paths: list[Path] = []

    class Writer:
        def add_page(self, page: int) -> None:
            pass

        def write(self, path: Path) -> None:
            path.write_bytes(b"single page")
            temporary_paths.append(path)

    class Pipeline:
        def __init__(self, **configuration: object) -> None:
            pass

        def predict(self, *, input: str) -> list[dict[str, object]]:
            return [{"page_index": 0}] * count

    monkeypatch.setitem(
        sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader, PdfWriter=Writer)
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    path = tmp_path / "source.pdf"
    request = DocumentParseRequest(
        LocalPathSource(str(path)),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        1,
    )
    with pytest.raises(ValueError, match="no result" if count == 0 else "multiple results"):
        _predict_pages(path, request, {})
    assert temporary_paths
    assert all(not temporary.exists() for temporary in temporary_paths)


@pytest.mark.parametrize(
    "native_text",
    (
        "Ανεργία 4,0% και 21.289 άνεργοι",
        "Сохранить ключевую ставку на уровне 21,00% годовых.",
        "Ανεργία",
        "Отчёт",
    ),
)
def test_unsupported_script_native_text_is_rejected(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    native_text: str,
) -> None:
    page = SimpleNamespace(extract_text=lambda: native_text, images=())
    reader = SimpleNamespace(pages=[page], is_encrypted=False)

    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    def forbidden_pipeline(**configuration: object) -> None:
        raise AssertionError("reliable native text reached OCR")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=forbidden_pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
    )

    with pytest.raises(ValueError, match="does not support the dominant PDF text script"):
        _predict_pages(tmp_path / "source.pdf", request, {})


def test_unsupported_language_hint_is_rejected_for_native_text_page(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    page = SimpleNamespace(
        extract_text=lambda **options: "Сохранить ключевую ставку",
        images=(),
    )
    reader = SimpleNamespace(pages=[page], is_encrypted=False)
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
        language_hints=("ru",),
    )

    with pytest.raises(ValueError, match="unsupported OCR language hint: ru"):
        _predict_pages(tmp_path / "source.pdf", request, {})


def test_supported_script_native_text_keeps_layout_ocr(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    page = SimpleNamespace(
        extract_text=lambda: "pdf.js",
        mediabox=SimpleNamespace(width=612, height=792),
        images=(),
    )
    reader = SimpleNamespace(pages=[page], is_encrypted=False)
    seen: list[str] = []

    class Writer:
        def add_page(self, value: object) -> None:
            assert value is page

        def write(self, path: Path) -> None:
            path.write_bytes(b"single page")

    class Pipeline:
        def __init__(self, **configuration: object) -> None:
            pass

        def predict(self, *, input: str) -> list[dict[str, object]]:
            seen.append(input)
            return [
                {
                    "page_index": 0,
                    "width": 100,
                    "height": 200,
                    "parsing_res_list": [{"block_content": "第一章 Chapter 1"}],
                    "overall_ocr_res": {"rec_texts": ["第一章", "Chapter 1"]},
                }
            ]

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda path: reader, PdfWriter=Writer),
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
    )

    result = _predict_pages(tmp_path / "source.pdf", request, {})

    assert len(seen) == 1
    assert result[0]["page_index"] == 0


def test_raster_image_page_is_rejected_despite_incidental_supported_text(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    page = SimpleNamespace(extract_text=lambda: "pdf.js", images=(object(),))
    reader = SimpleNamespace(pages=[page], is_encrypted=False)
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    def forbidden_pipeline(**configuration: object) -> None:
        raise AssertionError("unverified image script reached OCR")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=forbidden_pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
        language_hints=("en",),
    )

    with pytest.raises(ValueError, match="containing raster images"):
        _predict_pages(tmp_path / "source.pdf", request, {})


@pytest.mark.parametrize(
    "ocr_payload",
    (
        {"parsing_res_list": [{"block_content": "Сохранить ключевую ставку"}]},
        {
            "parsing_res_list": [],
            "overall_ocr_res": {"rec_texts": ["Анализ", "ключевой ставки"]},
        },
    ),
)
def test_unsupported_script_ocr_output_is_rejected_with_a_supported_text_layer(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    ocr_payload: dict[str, object],
) -> None:
    page = SimpleNamespace(extract_text=lambda: "English source text", images=())
    reader = SimpleNamespace(pages=[page], is_encrypted=False)

    class Writer:
        def add_page(self, value: object) -> None:
            assert value is page

        def write(self, path: Path) -> None:
            path.write_bytes(b"single page")

    class Pipeline:
        def __init__(self, **configuration: object) -> None:
            pass

        def predict(self, *, input: str) -> list[dict[str, object]]:
            return [{"page_index": 0, "width": 100, "height": 200, **ocr_payload}]

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda path: reader, PdfWriter=Writer),
    )
    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=Pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
    )

    with pytest.raises(ValueError, match="OCR output contains a dominant unsupported script"):
        _predict_pages(tmp_path / "source.pdf", request, {})


def test_native_text_fallback_rejects_incidental_script_and_extraction_errors() -> None:
    assert _dominant_unsupported_script("Chapter \u03b1") is None
    assert _dominant_unsupported_script("A" * 144 + "Ж" * 16) is None
    assert _dominant_unsupported_script("ﾊﾝｶｸｶﾀｶﾅ" * 3) is None
    assert _dominant_unsupported_script("\u039d\u03b1\u03b9") == "\u039d\u03b1\u03b9"
    assert _dominant_unsupported_script("\u0414\u0430") == "\u0414\u0430"
    assert _dominant_unsupported_script("Ανεργία") == "Ανεργία"
    assert _dominant_unsupported_script("Отчёт") == "Отчёт"

    def fail_extraction() -> str:
        raise ValueError("broken font map")

    assert _embedded_text(SimpleNamespace(extract_text=fail_extraction)) is None

    pypdf_error = type("PyPdfError", (Exception,), {"__module__": "pypdf.errors"})
    limit_reached_error = type(
        "LimitReachedError",
        (pypdf_error,),
        {"__module__": "pypdf.errors"},
    )

    def fail_pypdf_extraction() -> str:
        raise limit_reached_error("broken stream")

    assert _embedded_text(SimpleNamespace(extract_text=fail_pypdf_extraction)) is None


@pytest.mark.parametrize("native_text", ("", "text\ufffd", "text\x01"))
def test_unreliable_native_text_is_rejected_even_with_supported_hint(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
    native_text: str,
) -> None:
    page = SimpleNamespace(
        extract_text=lambda: native_text,
        mediabox=SimpleNamespace(width=612, height=792),
        images=(),
    )
    reader = SimpleNamespace(pages=[page], is_encrypted=False)
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    def forbidden_pipeline(**configuration: object) -> None:
        raise AssertionError("unverifiable source script reached OCR")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=forbidden_pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
        language_hints=("en",),
    )

    with pytest.raises(ValueError, match="without a reliable nonblank text layer"):
        _predict_pages(tmp_path / "source.pdf", request, {})


def test_profile_mismatch_fails_before_pdf_or_runtime_access(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    import capicove_processor_paddleocr.backend as backend

    source = tmp_path / "source.pdf"
    source.write_bytes(b"authorized PDF")

    def forbidden_metadata(path: Path) -> object:
        raise AssertionError("profile mismatch reached the PDF dependency")

    monkeypatch.setattr(backend, "extract_pdf_metadata", forbidden_metadata)
    request = DocumentParseRequest(
        LocalPathSource(str(source)),
        hashlib.sha256(source.read_bytes()).hexdigest(),
        "application/pdf",
        "wrong-profile",
        2,
    )

    with pytest.raises(ValueError, match="profile"):
        _backend(tmp_path).parse(request)


def test_unsupported_ocr_language_fails_before_runtime_initialization(
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    page = SimpleNamespace(
        extract_text=lambda: "",
        mediabox=SimpleNamespace(width=612, height=792),
        images=(),
    )
    reader = SimpleNamespace(pages=[page], is_encrypted=False)
    monkeypatch.setitem(sys.modules, "pypdf", SimpleNamespace(PdfReader=lambda path: reader))

    def forbidden_pipeline(**configuration: object) -> None:
        raise AssertionError("unsupported OCR language reached runtime initialization")

    monkeypatch.setitem(sys.modules, "paddleocr", SimpleNamespace(PPStructureV3=forbidden_pipeline))
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
        language_hints=("ru",),
    )

    with pytest.raises(ValueError, match="language hint"):
        _predict_pages(tmp_path / "source.pdf", request, {})


def test_model_receipt_failure_is_a_backend_availability_error(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    def fail(*args: object) -> None:
        raise ValueError("invalid model receipt")

    monkeypatch.setattr("capicove_processor_paddleocr.backend.verify_installed_models", fail)
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
    )
    with pytest.raises(DocumentBackendUnavailableError):
        _backend(tmp_path).parse(request)


def test_encrypted_document_is_an_explicit_input_rejection(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    from capicove.documents import DocumentRejectedError

    monkeypatch.setitem(
        sys.modules,
        "pypdf",
        SimpleNamespace(PdfReader=lambda path: SimpleNamespace(is_encrypted=True)),
    )
    request = DocumentParseRequest(
        LocalPathSource(str(tmp_path / "source.pdf")),
        "a" * 64,
        "application/pdf",
        "book-balanced",
        2,
    )
    with pytest.raises(DocumentRejectedError):
        _predict_pages(tmp_path / "source.pdf", request, {})
