"""Boundary regressions for the experimental canonical adapter."""

import sys
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar


ROOT = Path(__file__).resolve().parent
repository = ROOT.parents[3]
sys.path[:0] = [str(repository / "src"), str(repository / "plugins/processor-paddleocr/src")]
_fixture = tempfile.TemporaryDirectory()
_assets = Path(_fixture.name) / "assets.json"
_assets.write_text('{"models": {}}')
configuration = {"assets": str(_assets), "binaries": {}}
from backend import (  # noqa: E402 - explicit isolated worker bootstrap
    ExperimentalHybridBackend,
    can_extract_raster,
    native_normalize,
)

from capicove.contracts import (  # noqa: E402 - explicit isolated worker bootstrap
    DocumentPageRange,
    DocumentParseRequest,
    LocalPathSource,
)


class Tests(unittest.TestCase):
    def test_original_profile_is_not_silently_replaced(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = ExperimentalHybridBackend(configuration, Path(temp) / "out")
            request = DocumentParseRequest(
                LocalPathSource("/missing.pdf"),
                "a" * 64,
                "application/pdf",
                "book-balanced",
                2,
                (),
                DocumentPageRange(0, 1),
            )
            with self.assertRaisesRegex(ValueError, "profile"):
                backend.parse(request)
            self.assertFalse((Path(temp) / "out").exists())

    def test_explicit_bounded_page_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            backend = ExperimentalHybridBackend(configuration, Path(temp) / "out")
            request = DocumentParseRequest(
                LocalPathSource("/missing.pdf"),
                "a" * 64,
                "application/pdf",
                "book-hybrid-local",
                1,
                (),
                DocumentPageRange(0, 2),
            )
            with self.assertRaisesRegex(ValueError, "exactly one"):
                backend.parse(request)

    def test_source_hash_mismatch_before_provider_use(self):
        with tempfile.TemporaryDirectory() as temp:
            p = Path(temp) / "source.pdf"
            p.write_bytes(b"not authorized bytes")
            backend = ExperimentalHybridBackend(configuration, Path(temp) / "out")
            request = DocumentParseRequest(
                LocalPathSource(str(p)),
                "a" * 64,
                "application/pdf",
                "book-hybrid-local",
                1,
                (),
                DocumentPageRange(0, 1),
            )
            with self.assertRaisesRegex(ValueError, "source hash"):
                backend.parse(request)

    def test_annotation_forces_composited_render(self):
        page = {"/Annots": ["visible annotation"]}
        self.assertFalse(can_extract_raster(page))

    def test_rotation_forces_composited_render(self):
        self.assertFalse(can_extract_raster({"/Rotate": 90}))

    def test_numeric_conflicts_survive_comparison(self):
        self.assertNotEqual(native_normalize("100 USD"), native_normalize("900 USD"))
        self.assertNotEqual(native_normalize('3½"'), native_normalize('312"'))
        self.assertEqual(native_normalize("\uff08iii\uff09"), native_normalize("(iii)"))
        self.assertNotEqual(native_normalize("(iii)"), native_normalize("(ii)"))


class RasterExtractionTests(unittest.TestCase):
    def test_nonempty_operations_in_falsey_pdf_stream(self):
        from types import SimpleNamespace

        class Box(list):
            left = 0
            bottom = 0
            width = 612
            height = 792

        class Contents(dict):
            operations: ClassVar[list] = [
                ([], b"q"),
                ([612, 0, 0, 792, 0, 0], b"cm"),
                (["/Image"], b"Do"),
                ([], b"Q"),
            ]

        class Page(dict):
            mediabox = Box([0, 0, 612, 792])
            cropbox = Box([0, 0, 612, 792])
            images: ClassVar[list] = [
                SimpleNamespace(
                    indirect_reference=SimpleNamespace(
                        get_object=lambda: {
                            "/ColorSpace": "/DeviceGray",
                            "/BitsPerComponent": 1,
                        }
                    )
                )
            ]

            def get_contents(self):
                return Contents()

        self.assertTrue(can_extract_raster(Page()))


class ModelScopeTests(unittest.TestCase):
    def test_english_fragment_keeps_chinese_document_context(self):
        from backend import model_for_region

        self.assertEqual(model_for_region("Self Training)", "技術願景", ("zh", "en")), "server")
        self.assertEqual(model_for_region("-2-", "季度報告", ("zh",)), "server")
        self.assertEqual(model_for_region("Revenue 16.30", "Annual report", ("en",)), "en")


class LifecycleTests(unittest.TestCase):
    def test_identity_rechecked_after_provider_error(self):
        import hashlib
        from unittest.mock import patch

        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.pdf"
            source.write_bytes(b"bounded fixture")
            backend = ExperimentalHybridBackend(configuration, Path(temp) / "out")
            request = DocumentParseRequest(
                LocalPathSource(str(source)),
                hashlib.sha256(source.read_bytes()).hexdigest(),
                "application/pdf",
                "book-hybrid-local",
                1,
                (),
                DocumentPageRange(0, 1),
            )
            with (
                patch.object(backend, "_verify") as verify,
                patch.object(
                    backend, "_parse_page", side_effect=ValueError("unsupported test page")
                ),
            ):
                with self.assertRaisesRegex(ValueError, "unsupported test page"):
                    backend.parse(request)
                self.assertEqual(verify.call_count, 2)

    def test_configuration_is_detached(self):
        local = {**configuration, "binaries": {"vision": {"path": "original"}}}
        backend = ExperimentalHybridBackend(local, Path("unused"))
        local["binaries"]["vision"]["path"] = "changed"
        self.assertEqual(backend.configuration["binaries"]["vision"]["path"], "original")
