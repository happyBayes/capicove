"""Tests for the versioned image Tool catalog."""

from __future__ import annotations

import base64
import hashlib
import os
import time
from datetime import UTC, datetime, timedelta
from io import BytesIO
from pathlib import Path, PureWindowsPath
from typing import cast

import pytest
from jsonschema import Draft202012Validator  # type: ignore[import-untyped]

from capicove import tools as public_tools
from capicove.contracts import InvocationContext, ToolErrorCode, ToolStatus
from capicove.processing.image.deduplication import (
    ImageLshSearch,
    ImagePerceptualDeduplicator,
    ImagePhash,
)
from capicove.processing.image.filters import ImageAspectRatioFilter
from capicove.processing.image.mappers import (
    ImageRemoveBackgroundMapper,
    ImgDiffDifferenceAreaGeneratorMapper,
    ImgDiffDifferenceCaptionGeneratorMapper,
)
from capicove.processing.image.models import ImageSource
from capicove.processing.image.ocr import (
    ImageOcr,
    OcrProviderOutput,
    OcrProviderUnavailableError,
)
from capicove.tools.image import (
    IMAGE_TOOL_SPECS_BY_NAME,
    build_image_tool_handler,
    register_image_tools,
)
from capicove.tools.registry import ToolRegistry
from tests.unit.processing.image.fakes import FixedImageTextScorer


EXPECTED_NAMES = {
    "image_remove_background_mapper",
    "imgdiff_difference_area_generator_mapper",
    "imgdiff_difference_caption_generator_mapper",
    "image_aesthetics_filter",
    "image_aspect_ratio_filter",
    "image_nsfw_filter",
    "image_shape_filter",
    "image_size_filter",
    "image_subplot_filter",
    "image_text_matching_filter",
    "image_text_similarity_filter",
    "image_watermark_filter",
    "image_pair_similarity_filter",
    "image_phash",
    "image_lsh_search",
    "image_perceptual_deduplicator",
    "ray_image_perceptual_deduplicator",
    "image_ocr",
}


class _CatalogCapability:
    """Expose every operation name needed to validate catalog composition."""

    def map(self, *args: object) -> object:
        del args
        return ()

    def evaluate(self, *args: object) -> object:
        del args
        raise NotImplementedError

    def group(self, *args: object) -> object:
        del args
        raise NotImplementedError

    def recognize(self, *args: object) -> object:
        del args
        return ()

    def process(self, *args: object) -> object:
        del args
        raise NotImplementedError

    def search(self, *args: object, **kwargs: object) -> object:
        del args, kwargs
        raise NotImplementedError


def test_image_tool_handler_builder_is_public() -> None:
    assert public_tools.build_image_tool_handler is build_image_tool_handler


def _context(
    *,
    permissions: frozenset[str],
    expired: bool = False,
    deadline_seconds: float = 60.0,
) -> InvocationContext:
    now = datetime.now(UTC)
    return InvocationContext(
        request_id="request-1",
        trace_id="trace-1",
        workspace_id="workspace-1",
        actor_id="actor-1",
        environment_id="environment-1",
        deadline=(
            now - timedelta(seconds=1) if expired else now + timedelta(seconds=deadline_seconds)
        ),
        permissions=permissions,
    )


def test_image_tool_catalog_contains_every_requested_capability() -> None:
    assert set(IMAGE_TOOL_SPECS_BY_NAME) == EXPECTED_NAMES
    assert len({spec.tool_id for spec in IMAGE_TOOL_SPECS_BY_NAME.values()}) == 18
    assert all(spec.network_access is False for spec in IMAGE_TOOL_SPECS_BY_NAME.values())


def test_image_tool_catalog_schemas_are_valid_draft_2020_12() -> None:
    for spec in IMAGE_TOOL_SPECS_BY_NAME.values():
        Draft202012Validator.check_schema(dict(spec.input_schema))
        Draft202012Validator.check_schema(dict(spec.output_schema))


def test_image_tool_catalog_uses_capability_specific_input_schemas() -> None:
    dedup = IMAGE_TOOL_SPECS_BY_NAME["image_perceptual_deduplicator"].input_schema
    caption = IMAGE_TOOL_SPECS_BY_NAME["imgdiff_difference_caption_generator_mapper"].input_schema
    pair = IMAGE_TOOL_SPECS_BY_NAME["image_pair_similarity_filter"].input_schema

    assert dedup["required"] == ["records"]
    assert caption["required"] == ["areas", "sources"]
    assert pair["properties"]["sources"]["minItems"] == 2  # type: ignore[index]
    assert pair["properties"]["sources"]["maxItems"] == 2  # type: ignore[index]
    assert dedup["properties"]["records"]["maxItems"] == 1024  # type: ignore[index]
    assert caption["properties"]["areas"]["properties"]["regions"]["maxItems"] == 64  # type: ignore[index]


def test_phash_and_lsh_tool_catalogs_are_closed_and_bounded() -> None:
    phash = IMAGE_TOOL_SPECS_BY_NAME["image_phash"]
    search = IMAGE_TOOL_SPECS_BY_NAME["image_lsh_search"]

    assert phash.tool_id == "capicove.image.phash"
    assert search.tool_id == "capicove.image.lsh-search"
    assert search.resources.filesystem_read is False
    assert phash.input_schema["additionalProperties"] is False
    assert phash.input_schema["properties"]["enable_lsh"]["type"] == "boolean"  # type: ignore[index]
    assert phash.input_schema["properties"]["max_hamming_distance"]["maximum"] == 64  # type: ignore[index]
    assert search.input_schema["properties"]["candidates"]["maxItems"] == 100000  # type: ignore[index]
    assert search.input_schema["properties"]["top_k"]["maximum"] == 1000  # type: ignore[index]
    hash_schema = search.input_schema["properties"]["query"]["properties"]["hashes"][  # type: ignore[index]
        "items"
    ]
    assert hash_schema["pattern"] == "^[0-9A-Fa-f]{16}$"
    assert search.output_schema["properties"]["matches"]["items"]["additionalProperties"] is False  # type: ignore[index]


def test_image_tool_catalog_redacts_all_confidential_text_shapes() -> None:
    spec = IMAGE_TOOL_SPECS_BY_NAME["image_perceptual_deduplicator"]

    assert "records.*.text" in spec.redacted_fields


def test_image_tool_catalog_closes_nested_output_objects() -> None:
    filter_output = IMAGE_TOOL_SPECS_BY_NAME["image_aesthetics_filter"].output_schema
    artifact_output = IMAGE_TOOL_SPECS_BY_NAME["image_remove_background_mapper"].output_schema
    region_output = IMAGE_TOOL_SPECS_BY_NAME[
        "imgdiff_difference_caption_generator_mapper"
    ].output_schema
    dedup_output = IMAGE_TOOL_SPECS_BY_NAME["image_perceptual_deduplicator"].output_schema

    assert filter_output["properties"]["metrics"]["items"]["additionalProperties"] is False  # type: ignore[index]
    assert filter_output["properties"]["lineage"]["additionalProperties"] is False  # type: ignore[index]
    assert artifact_output["properties"]["artifacts"]["items"]["additionalProperties"] is False  # type: ignore[index]
    assert region_output["properties"]["regions"]["items"]["additionalProperties"] is False  # type: ignore[index]
    assert dedup_output["properties"]["duplicate_groups"]["items"]["additionalProperties"] is False  # type: ignore[index]


def test_image_tool_catalog_uses_perceptual_deduplication_names() -> None:
    assert "image_deduplicator" not in IMAGE_TOOL_SPECS_BY_NAME
    assert IMAGE_TOOL_SPECS_BY_NAME["image_perceptual_deduplicator"].tool_id == (
        "capicove.image.perceptual-deduplicate"
    )


def test_image_phash_handler_serializes_hashes_groups_and_lineage() -> None:
    class FixedHasher:
        provider_id = "fixed-phash"
        provider_version = "1"
        hash_version = "fixed-phash-64-v1"

        def phash(self, image: object) -> int:
            del image
            return 15

    encoded = base64.b64encode(_png_bytes("white")).decode()
    handler = build_image_tool_handler(
        "image_phash",
        ImagePhash(hasher=FixedHasher(), enable_lsh=True),
        read_roots=(),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {
            "records": [
                {"record_id": "b", "images": [{"base64": encoded}]},
                {"record_id": "a", "images": [{"base64": encoded}]},
            ],
            "enable_lsh": True,
            "max_hamming_distance": 0,
        },
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["records"][0]["hashes"] == ["000000000000000f"]  # type: ignore[index]
    assert result.result["duplicate_groups"][0]["kept_record_id"] == "a"  # type: ignore[index]
    assert result.result["lineage"]["candidate_strategy"] == "lsh"  # type: ignore[index]
    assert result.result["lineage"]["provider_id"] == "fixed-phash"  # type: ignore[index]
    assert len(result.evidence_refs) == 1


def test_image_lsh_search_handler_normalizes_hashes_and_stable_order() -> None:
    handler = build_image_tool_handler(
        "image_lsh_search",
        ImageLshSearch(),
        read_roots=(),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {
            "query": {"hashes": ["0000000000000000"]},
            "candidates": [
                {"record_id": "z", "hashes": ["0000000000000001"]},
                {"record_id": "a", "hashes": ["0000000000000002"]},
            ],
            "hash_version": "phash-64-v1",
            "max_hamming_distance": 1,
            "top_k": 2,
        },
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    matches = cast(list[dict[str, object]], result.result["matches"])
    assert [item["record_id"] for item in matches] == ["a", "z"]
    assert result.result["matches"][0]["hashes"] == ["0000000000000002"]  # type: ignore[index]
    assert result.result["lineage"]["comparison_count"] == 2  # type: ignore[index]
    assert result.evidence_refs == ()


def test_image_ocr_tool_catalog_declares_closed_local_process_contract() -> None:
    spec = IMAGE_TOOL_SPECS_BY_NAME["image_ocr"]

    assert spec.tool_id == "capicove.image.ocr"
    assert spec.required_permissions == frozenset({"image:read", "model:execute"})
    assert spec.resources.subprocesses == ("tesseract",)
    assert spec.network_access is False
    assert spec.input_schema["properties"]["languages"]["maxItems"] == 8  # type: ignore[index]
    assert spec.output_schema["properties"]["results"]["items"]["additionalProperties"] is False  # type: ignore[index]


def test_register_image_tools_binds_all_injected_handlers() -> None:
    registry = ToolRegistry()
    capabilities = {name: _CatalogCapability() for name in IMAGE_TOOL_SPECS_BY_NAME}

    register_image_tools(registry, capabilities, read_roots=(Path.cwd(),))

    assert registry.list_specs() == tuple(
        sorted(IMAGE_TOOL_SPECS_BY_NAME.values(), key=lambda spec: spec.registry_key)
    )


def test_image_ocr_handler_serializes_structured_words_and_lineage() -> None:
    from capicove.processing.image.models import OcrWord

    class Provider:
        provider_id = "fixed-ocr"
        provider_version = "1"

        def recognize(self, image: object, languages: tuple[str, ...]) -> OcrProviderOutput:
            del image
            assert languages == ("chi_sim", "eng")
            return OcrProviderOutput((OcrWord("hello", 0, 0, 10, 10, 0.9, 1, 1, 1, 1, 1),))

    handler = build_image_tool_handler(
        "image_ocr",
        ImageOcr(Provider()),
        read_roots=(),
    )
    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": [{"base64": base64.b64encode(_png_bytes("white")).decode()}]},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["results"][0]["full_text"] == "hello"  # type: ignore[index]
    assert result.result["results"][0]["words"][0]["confidence"] == 0.9  # type: ignore[index]
    assert result.result["lineage"]["parameters"]["languages"] == "chi_sim+eng"  # type: ignore[index]
    assert result.evidence_refs == (result.result["results"][0]["source_hash"],)  # type: ignore[index]


def test_image_ocr_handler_accepts_safe_language_override() -> None:
    class Provider:
        provider_id = "fixed-ocr"
        provider_version = "1"

        def recognize(self, image: object, languages: tuple[str, ...]) -> OcrProviderOutput:
            del image
            assert languages == ("eng",)
            return OcrProviderOutput(())

    handler = build_image_tool_handler(
        "image_ocr",
        ImageOcr(Provider()),
        read_roots=(),
    )
    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": [], "languages": ["eng"]},
    )

    assert result.status is ToolStatus.SUCCEEDED


def test_image_ocr_handler_rejects_duplicate_languages() -> None:
    handler = build_image_tool_handler("image_ocr", ImageOcr(_UnavailableOcr()), read_roots=())

    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": [], "languages": ["eng", "eng"]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT


def test_image_ocr_handler_maps_provider_unavailable() -> None:
    handler = build_image_tool_handler("image_ocr", ImageOcr(_UnavailableOcr()), read_roots=())

    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": [{"base64": base64.b64encode(_png_bytes("white")).decode()}]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.UNAVAILABLE


def test_image_ocr_handler_maps_invalid_provider_output_to_internal() -> None:
    from capicove.processing.image.models import OcrWord

    class InvalidOcr:
        provider_id = "invalid-ocr"
        provider_version = "1"

        def recognize(self, image: object, languages: tuple[str, ...]) -> OcrProviderOutput:
            del image, languages
            return OcrProviderOutput((OcrWord("outside", 100, 0, 10, 10, 0.9, 1, 1, 1, 1, 1),))

    handler = build_image_tool_handler("image_ocr", ImageOcr(InvalidOcr()), read_roots=())
    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": [{"base64": base64.b64encode(_png_bytes("white")).decode()}]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INTERNAL


class _UnavailableOcr:
    provider_id = "unavailable-ocr"
    provider_version = None

    def recognize(self, image: object, languages: tuple[str, ...]) -> OcrProviderOutput:
        del image, languages
        raise OcrProviderUnavailableError("unavailable")


def test_image_tool_builder_rejects_incompatible_capability(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="evaluate"):
        build_image_tool_handler(
            "image_aspect_ratio_filter",
            object(),
            read_roots=(tmp_path,),
        )


def test_image_tool_handler_enforces_permissions_and_deadline(tmp_path: Path) -> None:
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ImageAspectRatioFilter(),
        read_roots=(tmp_path,),
    )

    denied = handler(_context(permissions=frozenset()), {"sources": []})
    expired = handler(
        _context(permissions=frozenset({"image:read"}), expired=True),
        {"sources": []},
    )

    assert denied.status is ToolStatus.REJECTED
    assert denied.error is not None and denied.error.code is ToolErrorCode.PERMISSION_DENIED
    assert expired.status is ToolStatus.FAILED
    assert expired.error is not None and expired.error.code is ToolErrorCode.TIMEOUT


def test_image_tool_handler_enforces_cooperative_running_deadline(
    tmp_path: Path,
) -> None:
    from capicove.processing.image.invocation import check_image_deadline

    class BlockingFilter:
        def evaluate(self, sources: object) -> object:
            del sources
            while True:
                check_image_deadline()
                time.sleep(0.002)

    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        BlockingFilter(),
        read_roots=(tmp_path,),
    )
    started = time.monotonic()
    result = handler(
        _context(
            permissions=frozenset({"image:read"}),
            deadline_seconds=0.02,
        ),
        {"sources": []},
    )
    elapsed = time.monotonic() - started
    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.TIMEOUT
    assert elapsed < 0.25


def test_image_tool_handler_rejects_wrong_capability_result_type() -> None:
    from capicove.processing.image.mappers import ImgDiffDifferenceAreaGeneratorMapper

    class WrongFilter:
        def evaluate(self, sources: object) -> object:
            del sources
            return ImgDiffDifferenceAreaGeneratorMapper().map(
                (ImageSource(data=_png_bytes("white")), ImageSource(data=_png_bytes("black")))
            )

    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        WrongFilter(),
        read_roots=(),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": []},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INTERNAL


def test_image_tool_handler_rejects_text_beyond_schema_limit() -> None:
    handler = build_image_tool_handler(
        "image_text_matching_filter",
        _CatalogCapability(),
        read_roots=(),
    )

    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": [], "text": "x" * 8193},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT


def test_image_tool_handler_rejects_encoded_media_before_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import capicove.tools.image as image_tools

    decoded = False

    def decode(*args: object, **kwargs: object) -> bytes:
        nonlocal decoded
        del args, kwargs
        decoded = True
        return b"image"

    monkeypatch.setattr(image_tools, "_MAX_BASE64_LENGTH", 8)
    monkeypatch.setattr("capicove.tools.image.base64.b64decode", decode)
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ImageAspectRatioFilter(),
        read_roots=(),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"base64": "a" * 9}]},
    )

    assert result.status is ToolStatus.FAILED
    assert decoded is False


def test_image_tool_handler_authorizes_paths_and_serializes_lineage(tmp_path: Path) -> None:
    from PIL import Image

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    image_path = allowed / "image.png"
    Image.new("RGB", (20, 10), "white").save(image_path)
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ImageAspectRatioFilter(min_ratio=1.5, max_ratio=2.5),
        read_roots=(allowed,),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(image_path)}]},
    )
    rejected = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(tmp_path / "outside.png")}]},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None and result.result["keep"] is True
    assert result.result["lineage"]["parameters"]["min_ratio"] == 1.5  # type: ignore[index]
    assert rejected.status is ToolStatus.REJECTED
    assert rejected.error is not None
    assert rejected.error.code is ToolErrorCode.PERMISSION_DENIED


def test_image_tool_materializes_authorized_bytes_before_capability_can_swap_path(
    tmp_path: Path,
) -> None:
    from PIL import Image

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source_path = allowed / "source.png"
    outside_path = tmp_path / "outside.png"
    Image.new("RGB", (20, 10), "white").save(source_path)
    Image.new("RGB", (10, 20), "black").save(outside_path)

    class SwappingFilter:
        def evaluate(self, sources: tuple[ImageSource, ...]) -> object:
            source_path.unlink()
            source_path.symlink_to(outside_path)
            return ImageAspectRatioFilter(min_ratio=1.5, max_ratio=2.5).evaluate(sources)

    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        SwappingFilter(),
        read_roots=(allowed,),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(source_path)}]},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None and result.result["keep"] is True


def test_image_tool_materializes_authorized_bytes_before_file_growth(tmp_path: Path) -> None:
    from PIL import Image

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source_path = allowed / "source.png"
    Image.new("RGB", (20, 10), "white").save(source_path)

    class GrowingFilter:
        def evaluate(self, sources: tuple[ImageSource, ...]) -> object:
            Image.new("RGB", (10, 20), "black").save(source_path, format="BMP")
            return ImageAspectRatioFilter(min_ratio=1.5, max_ratio=2.5).evaluate(sources)

    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        GrowingFilter(),
        read_roots=(allowed,),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(source_path)}]},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None and result.result["keep"] is True


def test_image_tool_fails_closed_without_descriptor_relative_path_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from PIL import Image

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source_path = allowed / "source.png"
    Image.new("RGB", (20, 10), "white").save(source_path)
    invoked = False

    class ObservingFilter:
        def evaluate(self, sources: tuple[ImageSource, ...]) -> object:
            nonlocal invoked
            invoked = True
            return ImageAspectRatioFilter().evaluate(sources)

    monkeypatch.setattr("capicove.tools.image.os.supports_dir_fd", set())
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ObservingFilter(),
        read_roots=(allowed,),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(source_path)}]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.UNAVAILABLE
    assert invoked is False


def test_windows_image_local_path_syntax_is_checked_without_filesystem_access() -> None:
    from capicove.tools.image import _require_windows_local_path

    accepted = PureWindowsPath(r"C:\allowed\sample.png")
    assert _require_windows_local_path(accepted) == accepted
    for text in (
        r"\\remote-server\share\sample.png",
        r"\\?\C:\allowed\sample.png",
        r"C:\allowed\..\outside.png",
        r"C:\allowed\sample.png:stream",
        r"C:relative.png",
    ):
        with pytest.raises(ValueError):
            _require_windows_local_path(PureWindowsPath(text))


@pytest.mark.skipif(os.name != "nt", reason="Windows native file handle behavior")
def test_windows_image_read_root_is_not_resolved_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def forbidden_resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("Windows image read roots must not be resolved")

    with monkeypatch.context() as patch:
        patch.setattr(type(tmp_path), "resolve", forbidden_resolve)
        handler = build_image_tool_handler(
            "image_aspect_ratio_filter",
            ImageAspectRatioFilter(),
            read_roots=(tmp_path,),
            write_roots=(tmp_path,),
        )

    assert handler._read_roots == (tmp_path,)
    assert handler._write_roots == (tmp_path,)


@pytest.mark.skipif(os.name != "nt", reason="Windows native file handle behavior")
def test_windows_image_rejects_unc_before_path_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter", ImageAspectRatioFilter(), read_roots=(tmp_path,)
    )

    def forbidden_resolve(*args: object, **kwargs: object) -> Path:
        raise AssertionError("Windows image arguments must not be resolved")

    with monkeypatch.context() as patch:
        patch.setattr(type(tmp_path), "resolve", forbidden_resolve)
        result = handler(
            _context(permissions=frozenset({"image:read"})),
            {"sources": [{"path": r"\\remote-server\share\sample.png"}]},
        )

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED


@pytest.mark.skipif(os.name != "nt", reason="Windows native file handle behavior")
def test_windows_native_image_reader_rejects_reparse_source(tmp_path: Path) -> None:
    from capicove._windows_files import read_windows_secure_file

    source = tmp_path / "source.png"
    source.write_bytes(b"source-bytes")
    alias = tmp_path / "alias.png"
    alias.symlink_to(source)

    assert read_windows_secure_file(source, max_bytes=1024, owner_required=False) == b"source-bytes"
    with pytest.raises(ValueError):
        read_windows_secure_file(alias, max_bytes=1024, owner_required=False)


@pytest.mark.skipif(os.name != "nt", reason="Windows native file handle behavior")
def test_windows_image_handler_classifies_native_path_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PIL import Image

    from capicove._windows_files import WindowsPathPolicyError

    source = tmp_path / "sample.png"
    Image.new("RGB", (2, 3), "red").save(source)
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter", ImageAspectRatioFilter(), read_roots=(tmp_path,)
    )

    def reject_path(*args: object, **kwargs: object) -> bytes:
        raise WindowsPathPolicyError("reparse path")

    monkeypatch.setattr("capicove._windows_files.read_windows_secure_file", reject_path)
    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(source)}]},
    )
    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED


@pytest.mark.skipif(not hasattr(os, "mkfifo"), reason="FIFO files require POSIX")
def test_image_tool_rejects_fifo_without_invoking_capability(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    fifo = allowed / "image.pipe"
    os.mkfifo(fifo)
    invoked = False

    class ObservingFilter:
        def evaluate(self, sources: tuple[ImageSource, ...]) -> object:
            nonlocal invoked
            invoked = True
            return ImageAspectRatioFilter().evaluate(())

    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ObservingFilter(),
        read_roots=(allowed,),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"path": str(fifo)}]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT
    assert invoked is False


def test_image_tool_enforces_cumulative_decoded_pixel_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "capicove.processing.image.invocation._MAX_INVOCATION_DECODED_PIXELS",
        300,
    )
    handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ImageAspectRatioFilter(),
        read_roots=(),
    )

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [_inline_png("white"), _inline_png("black")]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT


class _CaptionProvider:
    provider_id = "test-caption"
    provider_version = "1"
    model_id = "test-caption-model"
    model_revision = "1"
    model_manifest_sha256 = "b" * 64
    preprocess_revision = "test-caption-v1"
    device = "cpu"

    def caption(self, image: object) -> str:
        from PIL import Image

        assert isinstance(image, Image.Image)
        return "dark" if image.convert("L").getbbox() is None else "light"

    def describe_difference(self, *args: object) -> str:
        del args
        return "the region became dark"


class _CountingCaptionProvider(_CaptionProvider):
    def __init__(self) -> None:
        self.caption_calls = 0
        self.difference_calls = 0

    def caption(self, image: object) -> str:
        self.caption_calls += 1
        return super().caption(image)

    def describe_difference(self, *args: object) -> str:
        self.difference_calls += 1
        return super().describe_difference(*args)


def _inline_png(color: str) -> dict[str, str]:
    from PIL import Image

    buffer = BytesIO()
    Image.new("RGB", (40, 40), color).save(buffer, format="PNG")
    return {
        "base64": base64.b64encode(buffer.getvalue()).decode("ascii"),
        "media_type": "image/png",
    }


def _png_bytes(color: str) -> bytes:
    return base64.b64decode(_inline_png(color)["base64"])


def test_difference_area_tool_output_feeds_caption_tool_input() -> None:
    permissions = frozenset({"image:read", "model:execute"})
    sources = [_inline_png("white"), _inline_png("black")]
    area_handler = build_image_tool_handler(
        "imgdiff_difference_area_generator_mapper",
        ImgDiffDifferenceAreaGeneratorMapper(pixel_threshold=10),
        read_roots=(),
    )
    caption_handler = build_image_tool_handler(
        "imgdiff_difference_caption_generator_mapper",
        ImgDiffDifferenceCaptionGeneratorMapper(
            _CaptionProvider(),
            validator=FixedImageTextScorer((1.0,)),
        ),
        read_roots=(),
    )

    area_result = area_handler(_context(permissions=permissions), {"sources": sources})
    assert area_result.result is not None
    caption_result = caption_handler(
        _context(permissions=permissions),
        {"sources": sources, "areas": area_result.result},
    )

    assert caption_result.status is ToolStatus.SUCCEEDED
    assert caption_result.result is not None
    assert len(caption_result.result["regions"]) == 1  # type: ignore[arg-type]


def test_difference_caption_tool_deduplicates_identical_regions() -> None:
    permissions = frozenset({"image:read", "model:execute"})
    sources = [_inline_png("white"), _inline_png("black")]
    area_handler = build_image_tool_handler(
        "imgdiff_difference_area_generator_mapper",
        ImgDiffDifferenceAreaGeneratorMapper(pixel_threshold=10),
        read_roots=(),
    )
    provider = _CountingCaptionProvider()
    caption_handler = build_image_tool_handler(
        "imgdiff_difference_caption_generator_mapper",
        ImgDiffDifferenceCaptionGeneratorMapper(
            provider,
            validator=FixedImageTextScorer((1.0,)),
        ),
        read_roots=(),
    )
    area_result = area_handler(_context(permissions=permissions), {"sources": sources})
    assert area_result.result is not None
    regions = area_result.result["regions"]
    assert isinstance(regions, list)
    duplicated = dict(area_result.result)
    duplicated["regions"] = [*regions, *regions]

    result = caption_handler(
        _context(permissions=permissions),
        {"sources": sources, "areas": duplicated},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None and len(result.result["regions"]) == 1  # type: ignore[arg-type]
    assert provider.caption_calls == 2
    assert provider.difference_calls == 1


def test_difference_caption_tool_rejects_excessive_region_count() -> None:
    handler = build_image_tool_handler(
        "imgdiff_difference_caption_generator_mapper",
        _CatalogCapability(),
        read_roots=(),
    )
    regions = [{"x": index, "y": 0, "width": 1, "height": 1} for index in range(65)]

    result = handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {
            "sources": [_inline_png("white"), _inline_png("black")],
            "areas": {
                "regions": regions,
                "lineage": {
                    "operation_version": 1,
                    "parameters": {},
                    "provider_id": "test",
                    "source_hashes": ["a" * 64, "b" * 64],
                },
            },
        },
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT


def test_difference_caption_tool_enforces_model_call_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "capicove.processing.image.invocation._MAX_INVOCATION_MODEL_CALLS",
        2,
    )
    permissions = frozenset({"image:read", "model:execute"})
    sources = [_inline_png("white"), _inline_png("black")]
    area_handler = build_image_tool_handler(
        "imgdiff_difference_area_generator_mapper",
        ImgDiffDifferenceAreaGeneratorMapper(pixel_threshold=10),
        read_roots=(),
    )
    provider = _CountingCaptionProvider()
    caption_handler = build_image_tool_handler(
        "imgdiff_difference_caption_generator_mapper",
        ImgDiffDifferenceCaptionGeneratorMapper(
            provider,
            validator=FixedImageTextScorer((1.0,)),
        ),
        read_roots=(),
    )
    area_result = area_handler(_context(permissions=permissions), {"sources": sources})
    assert area_result.result is not None

    result = caption_handler(
        _context(permissions=permissions),
        {"sources": sources, "areas": area_result.result},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INVALID_INPUT
    assert provider.caption_calls == 2
    assert provider.difference_calls == 0


class _AlphaProvider:
    provider_id = "test-alpha"
    provider_version = "1"
    model_id = "test-alpha-model"
    model_revision = "1"
    model_manifest_sha256 = "b" * 64
    preprocess_revision = "test-alpha-v1"
    device = "cpu"

    def remove(self, image: object, options: object) -> object:
        from PIL import Image

        del options
        assert isinstance(image, Image.Image)
        return image.convert("RGBA")


def test_background_tool_parses_options_authorizes_output_and_serializes_artifact(
    tmp_path: Path,
) -> None:
    output = tmp_path / "output"
    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(_AlphaProvider(), output),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(output,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {
            "sources": [_inline_png("white")],
            "options": {
                "alpha_matting": True,
                "background_color": [0, 0, 0, 0],
            },
        },
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    artifacts = result.result["artifacts"]
    assert isinstance(artifacts, list)
    assert Path(artifacts[0]["path"]).is_file()
    assert result.result["lineage"]["parameters"]["alpha_matting"] is True  # type: ignore[index]
    assert result.result["lineage"]["model_manifest_sha256"] == "b" * 64  # type: ignore[index]


def test_background_tool_fails_closed_without_descriptor_relative_output_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    provider = _AlphaProvider()
    monkeypatch.setattr("capicove.tools.image.os.supports_dir_fd", set())
    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(provider, output),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(output,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {"sources": [_inline_png("white")]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.UNAVAILABLE
    assert not output.exists()


def test_background_tool_accepts_authorized_symlink_output_root(tmp_path: Path) -> None:
    authorized = tmp_path / "authorized"
    real_output = authorized / "real-output"
    output_link = authorized / "output-link"
    real_output.mkdir(parents=True)
    output_link.symlink_to(real_output, target_is_directory=True)
    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(_AlphaProvider(), output_link),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(authorized,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {"sources": [_inline_png("white")]},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    artifacts = result.result["artifacts"]
    assert isinstance(artifacts, list)
    assert Path(artifacts[0]["path"]).resolve().parent == real_output


def test_background_tool_keeps_artifact_bound_when_authorized_symlink_is_retargeted(
    tmp_path: Path,
) -> None:
    authorized = tmp_path / "authorized"
    original_output = authorized / "original-output"
    replacement_output = authorized / "replacement-output"
    output_link = authorized / "output-link"
    original_output.mkdir(parents=True)
    replacement_output.mkdir()
    output_link.symlink_to(original_output, target_is_directory=True)

    class RetargetingProvider(_AlphaProvider):
        def remove(self, image: object, options: object) -> object:
            result = super().remove(image, options)
            output_link.unlink()
            output_link.symlink_to(replacement_output, target_is_directory=True)
            return result

    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(RetargetingProvider(), output_link),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(authorized,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {"sources": [_inline_png("white")]},
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    artifacts = result.result["artifacts"]
    assert isinstance(artifacts, list)
    artifact = artifacts[0]
    artifact_path = Path(artifact["path"])
    assert artifact_path.parent == original_output
    assert hashlib.sha256(artifact_path.read_bytes()).hexdigest() == artifact["sha256"]
    assert list(replacement_output.iterdir()) == []


def test_background_tool_rolls_back_if_authorized_directory_changes_during_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    authorized = tmp_path / "authorized"
    output = authorized / "output"
    parked = authorized / "parked"
    real_link = os.link

    def publish_then_replace_directory(
        source: str,
        destination: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        real_link(
            source,
            destination,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )
        output.rename(parked)
        output.mkdir()

    monkeypatch.setattr(
        "capicove.processing.image.mappers.remove_background.os.link",
        publish_then_replace_directory,
    )
    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(_AlphaProvider(), output),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(authorized,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {"sources": [_inline_png("white")]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INTERNAL
    assert list(parked.iterdir()) == []
    assert list(output.iterdir()) == []


def test_background_tool_rejects_unauthorized_output_before_writing(tmp_path: Path) -> None:
    authorized = tmp_path / "authorized"
    outside = tmp_path / "outside"
    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(_AlphaProvider(), outside),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(authorized,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {"sources": [_inline_png("white")]},
    )

    assert result.status is ToolStatus.REJECTED
    assert result.error is not None and result.error.code is ToolErrorCode.PERMISSION_DENIED
    assert not outside.exists()


def test_background_tool_does_not_escape_when_output_directory_is_swapped(
    tmp_path: Path,
) -> None:
    authorized = tmp_path / "authorized"
    output = authorized / "output"
    outside = tmp_path / "outside"
    parked = authorized / "parked"
    outside.mkdir()

    class SwappingProvider(_AlphaProvider):
        def remove(self, image: object, options: object) -> object:
            result = super().remove(image, options)
            output.rename(parked)
            output.symlink_to(outside, target_is_directory=True)
            return result

    handler = build_image_tool_handler(
        "image_remove_background_mapper",
        ImageRemoveBackgroundMapper(SwappingProvider(), output),  # type: ignore[arg-type]
        read_roots=(),
        write_roots=(authorized,),
    )
    permissions = frozenset({"artifact:write", "image:read", "image:write", "model:execute"})

    result = handler(
        _context(permissions=permissions),
        {"sources": [_inline_png("white")]},
    )

    assert result.status is ToolStatus.FAILED
    assert result.error is not None and result.error.code is ToolErrorCode.INTERNAL
    assert list(outside.iterdir()) == []
    assert list(parked.iterdir()) == []


def test_perceptual_deduplication_tool_parses_records_and_serializes_groups() -> None:
    handler = build_image_tool_handler(
        "image_perceptual_deduplicator",
        ImagePerceptualDeduplicator(),
        read_roots=(),
    )
    source = _inline_png("white")

    result = handler(
        _context(permissions=frozenset({"image:read"})),
        {
            "records": [
                {"record_id": "b", "images": [source], "text": "same"},
                {"record_id": "a", "images": [source], "text": "same"},
            ]
        },
    )

    assert result.status is ToolStatus.SUCCEEDED
    assert result.result is not None
    assert result.result["retained_record_ids"] == ["a"]
    assert len(result.result["duplicate_groups"]) == 1  # type: ignore[arg-type]


def test_image_tool_maps_invalid_inline_media_and_unavailable_provider() -> None:
    invalid_handler = build_image_tool_handler(
        "image_aspect_ratio_filter",
        ImageAspectRatioFilter(),
        read_roots=(),
    )

    class UnavailableFilter:
        def evaluate(self, sources: object) -> object:
            del sources
            raise RuntimeError("provider unavailable; install optional dependency")

    unavailable_handler = build_image_tool_handler(
        "image_aesthetics_filter",
        UnavailableFilter(),
        read_roots=(),
    )
    invalid = invalid_handler(
        _context(permissions=frozenset({"image:read"})),
        {"sources": [{"base64": "not-base64"}]},
    )
    unavailable = unavailable_handler(
        _context(permissions=frozenset({"image:read", "model:execute"})),
        {"sources": []},
    )

    assert invalid.error is not None and invalid.error.code is ToolErrorCode.INVALID_INPUT
    assert unavailable.error is not None
    assert unavailable.error.code is ToolErrorCode.UNAVAILABLE
