"""Deterministic canonical ASR/OCR projections without inline text."""

from capicove.contracts.retrieval import (
    CharacterRange,
    ProcessingFingerprint,
    SearchableUnit,
    TargetType,
    TemporalRange,
    TextChunkLocator,
    TextOrigin,
)
from capicove.processing.audio.models import AudioTranscript
from capicove.processing.video.retrieval.models import TemporalOcrText


def _project(
    *,
    source_hash: str,
    artifact_ref: str,
    entries: tuple[TemporalOcrText, ...],
    origin: TextOrigin,
    processing_run_ref: str,
    processing_fingerprint: ProcessingFingerprint,
) -> tuple[SearchableUnit, ...]:
    units = []
    offset = 0
    previous_end = 0
    for entry in entries:
        if entry.time_range.start_ms < previous_end:
            raise ValueError("text ranges must be ordered and non-overlapping")
        previous_end = entry.time_range.end_ms
        identity = ProcessingFingerprint.from_manifest(
            {
                "source": source_hash,
                "origin": origin.value,
                "start": entry.time_range.start_ms,
                "end": entry.time_range.end_ms,
                "text": entry.text,
                "representation": entry.representation_ref,
                "fingerprint": processing_fingerprint.sha256,
            }
        )
        units.append(
            SearchableUnit(
                unit_id=f"{origin.value}:{identity.sha256}",
                unit_type=TargetType.TEXT_CHUNK,
                source_hash=source_hash,
                artifact_ref=artifact_ref,
                evidence_ref=entry.evidence_ref,
                representation_refs=(entry.representation_ref,),
                processing_run_ref=processing_run_ref,
                processing_version="temporal-text-v1",
                processing_fingerprint=processing_fingerprint,
                text_locator=TextChunkLocator(
                    origin=origin,
                    character_range=CharacterRange(offset, offset + len(entry.text)),
                    time_range=entry.time_range,
                ),
            )
        )
        offset += len(entry.text)
    return tuple(units)


def asr_searchable_units(
    *,
    source_hash: str,
    artifact_ref: str,
    evidence_ref: str,
    transcript: AudioTranscript,
    representation_ref: str,
    processing_run_ref: str,
    processing_fingerprint: ProcessingFingerprint,
) -> tuple[SearchableUnit, ...]:
    """Project transcript character offsets and rounded millisecond ranges."""
    entries = tuple(
        TemporalOcrText(
            s.text,
            TemporalRange(round(s.start_seconds * 1000), round(s.end_seconds * 1000)),
            evidence_ref,
            representation_ref,
        )
        for s in transcript.segments
    )
    return _project(
        source_hash=source_hash,
        artifact_ref=artifact_ref,
        entries=entries,
        origin=TextOrigin.ASR,
        processing_run_ref=processing_run_ref,
        processing_fingerprint=processing_fingerprint,
    )


def ocr_searchable_units(
    *,
    source_hash: str,
    artifact_ref: str,
    ocr: tuple[TemporalOcrText, ...],
    processing_run_ref: str,
    processing_fingerprint: ProcessingFingerprint,
) -> tuple[SearchableUnit, ...]:
    """Project OCR offsets independently in each addressed representation."""
    # Each OCR entry addresses its own preview-derived text representation.
    result: list[SearchableUnit] = []
    previous_end = 0
    for entry in ocr:
        if entry.time_range.start_ms < previous_end:
            raise ValueError("OCR ranges must be ordered and non-overlapping")
        previous_end = entry.time_range.end_ms
        result.extend(
            _project(
                source_hash=source_hash,
                artifact_ref=artifact_ref,
                entries=(entry,),
                origin=TextOrigin.OCR,
                processing_run_ref=processing_run_ref,
                processing_fingerprint=processing_fingerprint,
            )
        )
    return tuple(result)
