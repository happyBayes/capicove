"""Resource-specific authorization and bounded evidence projections."""

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from unittest.mock import MagicMock

import pytest

from capicove.contracts import InvocationContext
from capicove.contracts.retrieval import ProcessingFingerprint, SearchableUnit, TargetType
from capicove.retrieval.video_evidence import VideoEvidenceReader, video_evidence_bindings
from capicove.retrieval.video_ports import VideoUnitCatalogReader


PREVIEW = b"\x89PNG\r\n\x1a\n" + b"fixture"
REF = "resource:preview:" + sha256(PREVIEW).hexdigest()
RECEIPT = b"{}"
EVIDENCE = "resource:sha256:" + sha256(RECEIPT).hexdigest()


def unit() -> SearchableUnit:
    return SearchableUnit(
        "keyframe:1",
        TargetType.KEYFRAME,
        "a" * 64,
        "/private/source.mp4",
        EVIDENCE,
        (REF,),
        "run:1",
        "v1",
        ProcessingFingerprint("b" * 64),
        timestamp_ms=1000,
    )


def context() -> InvocationContext:
    return InvocationContext(
        "r",
        "t",
        "workspace",
        "actor",
        "environment",
        datetime.now(UTC) + timedelta(seconds=10),
        permissions=frozenset({"evidence:read"}),
    )


def setup(*, allow: bool = True) -> tuple[VideoEvidenceReader, MagicMock, MagicMock]:
    catalog = MagicMock(spec=VideoUnitCatalogReader)
    catalog.get.return_value = unit()
    read = MagicMock(side_effect={REF: PREVIEW, EVIDENCE: RECEIPT}.__getitem__)
    authorize = MagicMock(return_value=allow)
    reader = VideoEvidenceReader(
        catalog=catalog,
        evidence_units=video_evidence_bindings((unit(),)),
        read_resource=read,
        authorize_resource=authorize,
    )
    return reader, read, authorize


def test_authorization_is_per_request_and_every_resource() -> None:
    reader, read, authorize = setup()
    for _ in range(2):
        result = reader.read(
            context=context(), evidence_id=EVIDENCE, sections=("metadata", "preview", "lineage")
        )
        assert result["preview"] == {"resource_ref": REF, "media_type": "image/png"}
        assert "/private" not in str(result)
    refs = [call.args[2] for call in authorize.call_args_list]
    assert refs.count(EVIDENCE) == 2
    assert refs.count("keyframe:1") == 2
    assert refs.count(REF) == 2
    assert read.call_count == 4
    authorize.return_value = False
    with pytest.raises(PermissionError):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("preview",))
    assert read.call_count == 4


def test_absent_authorizer_denies_before_catalog_or_content() -> None:
    catalog = MagicMock(spec=VideoUnitCatalogReader)
    read = MagicMock()
    reader = VideoEvidenceReader(
        catalog=catalog, evidence_units=video_evidence_bindings((unit(),)), read_resource=read
    )
    with pytest.raises(PermissionError):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("metadata",))
    catalog.get.assert_not_called()
    read.assert_not_called()


@pytest.mark.parametrize("section", ["scores", "provider_diagnostics"])
def test_unavailable_sections_are_never_fabricated(section: str) -> None:
    reader, read, _ = setup()
    with pytest.raises(LookupError, match="unsupported"):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=(section,))
    read.assert_not_called()


@pytest.mark.parametrize("section,origin", [("transcript", "asr"), ("ocr", "ocr")])
def test_temporal_text_expansion_authorizes_representation(section: str, origin: str) -> None:
    from capicove.contracts.retrieval import (
        CharacterRange,
        TemporalRange,
        TextChunkLocator,
        TextOrigin,
    )

    payload = "你好 world".encode()
    ref = "resource:sha256:" + sha256(payload).hexdigest()
    text_unit = replace(
        unit(),
        unit_id="text:1",
        unit_type=TargetType.TEXT_CHUNK,
        timestamp_ms=None,
        representation_refs=(ref,),
        text_locator=TextChunkLocator(
            origin=TextOrigin(origin),
            character_range=CharacterRange(0, 2),
            time_range=TemporalRange(0, 1000),
        ),
    )
    catalog = MagicMock(spec=VideoUnitCatalogReader)
    catalog.get.return_value = text_unit
    read = MagicMock(side_effect={ref: payload, EVIDENCE: RECEIPT}.__getitem__)
    authorize = MagicMock(return_value=True)
    reader = VideoEvidenceReader(
        catalog=catalog,
        evidence_units=video_evidence_bindings((text_unit,)),
        read_resource=read,
        authorize_resource=authorize,
    )
    result = reader.read(context=context(), evidence_id=EVIDENCE, sections=(section,))
    assert result[section] == [
        {
            "unit_id": "text:1",
            "text": "你好",
            "representation_ref": ref,
            "time_range": {"start_ms": 0, "end_ms": 1000},
        }
    ]
    authorize.side_effect = lambda context, evidence_id, reference: reference != ref
    with pytest.raises(PermissionError):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=(section,))
    authorize.side_effect = None
    read.side_effect = lambda reference: RECEIPT if reference == EVIDENCE else b"changed"
    with pytest.raises(ValueError, match="hash"):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=(section,))


def test_erasure_hash_failure_and_unknown_evidence_never_succeed() -> None:
    reader, read, _ = setup()
    read.side_effect = FileNotFoundError
    with pytest.raises(LookupError):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("preview",))
    read.side_effect = lambda ref: RECEIPT if ref == EVIDENCE else b"changed"
    with pytest.raises(ValueError, match="hash"):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("preview",))
    with pytest.raises(LookupError):
        reader.read(context=context(), evidence_id="evidence:unknown", sections=("metadata",))


def test_only_requested_metadata_and_deadline() -> None:
    reader, read, _ = setup()
    assert set(reader.read(context=context(), evidence_id=EVIDENCE, sections=("metadata",))) == {
        "metadata"
    }
    assert [call.args[0] for call in read.call_args_list] == [EVIDENCE]
    with pytest.raises(TimeoutError):
        reader.read(
            context=replace(context(), deadline=datetime.now(UTC) - timedelta(seconds=1)),
            evidence_id=EVIDENCE,
            sections=("metadata",),
        )


def test_preview_specific_denial_prevents_preview_read() -> None:
    reader, read, authorize = setup()
    authorize.side_effect = lambda context, evidence_id, reference: reference != REF
    with pytest.raises(PermissionError):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("preview",))
    assert [call.args[0] for call in read.call_args_list] == [EVIDENCE]


@pytest.mark.parametrize("count", [32, 33])
def test_unit_and_four_kib_response_limits(count: int) -> None:
    units = tuple(replace(unit(), unit_id=f"keyframe:{i}") for i in range(count))
    catalog = MagicMock(spec=VideoUnitCatalogReader)
    catalog.get.side_effect = {u.unit_id: u for u in units}.get
    reader = VideoEvidenceReader(
        catalog=catalog,
        evidence_units=video_evidence_bindings(units),
        read_resource=lambda reference: RECEIPT,
        authorize_resource=lambda context, evidence_id, reference: True,
    )
    with pytest.raises(LookupError, match="capacity"):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("metadata", "lineage"))
    if count == 33:
        catalog.get.assert_not_called()


def test_catalog_binding_and_ambiguous_preview_are_rejected() -> None:
    first = unit()
    second = replace(
        first, unit_id="keyframe:2", representation_refs=("resource:preview:" + ("c" * 64),)
    )
    catalog = MagicMock(spec=VideoUnitCatalogReader)
    catalog.get.side_effect = {u.unit_id: u for u in (first, second)}.get
    reader = VideoEvidenceReader(
        catalog=catalog,
        evidence_units=video_evidence_bindings((first, second)),
        read_resource=lambda ref: RECEIPT,
        authorize_resource=lambda context, evidence_id, reference: True,
    )
    with pytest.raises(LookupError, match="unique"):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("preview",))
    catalog.get.side_effect = lambda unit_id: replace(first, evidence_ref="evidence:other")
    with pytest.raises(ValueError, match="binding mismatch"):
        reader.read(context=context(), evidence_id=EVIDENCE, sections=("metadata",))
