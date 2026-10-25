"""Canonical temporal evidence and character-offset behavior."""

from dataclasses import replace

import pytest

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange, TextOrigin
from capicove.processing.audio.models import (
    AudioBackendLineage,
    AudioTranscript,
    AudioTranscriptionSegment,
)
from capicove.processing.video.retrieval.models import TemporalOcrText
from capicove.processing.video.retrieval.text_units import (
    asr_searchable_units,
    ocr_searchable_units,
)


def transcript() -> AudioTranscript:
    return AudioTranscript(
        "helloworld",
        "en",
        (
            AudioTranscriptionSegment(0, 0.0, 1.25, "hello"),
            AudioTranscriptionSegment(1, 1.25, 2.0, "world"),
        ),
        (),
        AudioBackendLineage("local", "1", "model", "a" * 64, "cpu", "float32"),
    )


def test_asr_offsets_bounds_and_lineage() -> None:
    units = asr_searchable_units(
        source_hash="a" * 64,
        artifact_ref="artifact:v",
        evidence_ref="evidence:v",
        transcript=transcript(),
        representation_ref="representation:asr",
        processing_run_ref="run:1",
        processing_fingerprint=ProcessingFingerprint("b" * 64),
    )
    assert [u.text_locator.time_range for u in units if u.text_locator] == [
        TemporalRange(0, 1250),
        TemporalRange(1250, 2000),
    ]
    assert [u.text_locator.character_range.start_char for u in units if u.text_locator] == [0, 5]
    assert all(u.evidence_ref == "evidence:v" and u.processing_run_ref == "run:1" for u in units)
    empty = replace(transcript(), text="", segments=())
    assert (
        asr_searchable_units(
            source_hash="a" * 64,
            artifact_ref="artifact:v",
            evidence_ref="evidence:v",
            transcript=empty,
            representation_ref="representation:asr",
            processing_run_ref="run:1",
            processing_fingerprint=ProcessingFingerprint("b" * 64),
        )
        == ()
    )


def test_ocr_temporal_and_each_representation_offsets() -> None:
    items = (
        TemporalOcrText("first", TemporalRange(0, 1), "evidence:1", "representation:1"),
        TemporalOcrText("second", TemporalRange(1000, 1001), "evidence:2", "representation:2"),
    )
    units = ocr_searchable_units(
        source_hash="a" * 64,
        artifact_ref="artifact:v",
        ocr=items,
        processing_run_ref="run:1",
        processing_fingerprint=ProcessingFingerprint("b" * 64),
    )
    assert all(
        u.text_locator
        and u.text_locator.origin == TextOrigin.OCR
        and u.text_locator.page_number is None
        and u.text_locator.character_range.start_char == 0
        for u in units
    )
    with pytest.raises(ValueError, match="ordered"):
        ocr_searchable_units(
            source_hash="a" * 64,
            artifact_ref="artifact:v",
            ocr=tuple(reversed(items)),
            processing_run_ref="run:1",
            processing_fingerprint=ProcessingFingerprint("b" * 64),
        )
