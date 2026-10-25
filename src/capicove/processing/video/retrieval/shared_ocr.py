"""Adapt public image OCR to temporal video evidence without owning an engine."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime
from typing import TYPE_CHECKING

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange
from capicove.processing.video.ports import BackendUnavailableError
from capicove.processing.video.retrieval.models import RetrievalKeyframe, TemporalOcrText
from capicove.processing.video.retrieval.pipeline import check_deadline
from capicove.processing.video.retrieval.ports import PreviewStore, ProcessingManifestStore


if TYPE_CHECKING:
    from capicove.processing.image import ImageOcr


class SharedImageOCRProcessor:
    """Reuse an injected image operation over verified, persisted keyframes.

    The caller provisions the public ImageOcr service and compatible processing
    fingerprint. No PDF internals, recognition SDK, model loading or download is
    used here. Frame observations do not establish text persistence across a scene.
    """

    def __init__(
        self,
        service: ImageOcr,
        previews: PreviewStore,
        representations: ProcessingManifestStore,
        *,
        languages: tuple[str, ...] | None = None,
        provider_fingerprint: ProcessingFingerprint | None = None,
    ) -> None:
        from capicove.processing.image.ocr import DEFAULT_OCR_LANGUAGES, validate_ocr_languages

        if not isinstance(provider_fingerprint, ProcessingFingerprint):
            raise ValueError("provider_fingerprint is required for video OCR")
        self._service = service
        self._previews = previews
        self._representations = representations
        self._languages = validate_ocr_languages(
            DEFAULT_OCR_LANGUAGES if languages is None else languages
        )
        self._fingerprint = ProcessingFingerprint.from_manifest(
            {
                "provider_fingerprint": provider_fingerprint.sha256,
                "languages": list(self._languages),
                "temporal_policy": "keyframe-point-1ms-v1",
                "representation_policy": "exact-image-full-text-utf8-v1",
            }
        )

    @property
    def fingerprint(self) -> ProcessingFingerprint:
        """Bind provisioned model identity, effective languages and time policy."""
        return self._fingerprint

    def recognize(
        self, keyframes: tuple[RetrievalKeyframe, ...], *, deadline: datetime
    ) -> tuple[TemporalOcrText, ...]:
        """Keep exact text, structured recognition lineage and a 1 ms observation."""
        from capicove.processing.image import ImageSource
        from capicove.processing.image.invocation import image_invocation_deadline
        from capicove.processing.image.ocr import OcrProviderUnavailableError

        check_deadline(deadline)
        times = tuple(frame.timestamp_ms for frame in keyframes)
        if times != tuple(sorted(set(times))):
            raise ValueError("OCR keyframes must be unique and ordered")
        output: list[TemporalOcrText] = []
        for frame in keyframes:
            check_deadline(deadline)
            payload = self._previews.read(frame.preview_resource_ref)
            if hashlib.sha256(payload).hexdigest() != frame.preview_sha256:
                raise ValueError("OCR preview hash mismatch")
            try:
                # A bounded image invocation per frame avoids retaining a batch of
                # decoded images; the absolute video deadline is never extended.
                with image_invocation_deadline(deadline):
                    results = self._service.recognize(
                        (ImageSource(data=payload),), languages=self._languages
                    )
            except OcrProviderUnavailableError as error:
                raise BackendUnavailableError("shared image OCR unavailable") from error
            if len(results) != 1 or results[0].source_hash != frame.preview_sha256:
                raise ValueError("OCR result source hash or count mismatch")
            result = results[0]
            if not result.full_text.strip():
                continue
            check_deadline(deadline)
            text_ref = self._representations.put(result.full_text.encode("utf-8"))
            receipt = {
                "schema_version": 1,
                "processing_fingerprint": self.fingerprint.sha256,
                "keyframe_unit_id": frame.unit_id,
                "timestamp_ms": frame.timestamp_ms,
                "preview_resource_ref": frame.preview_resource_ref,
                "recognition": asdict(result),
                "text_representation_ref": text_ref,
            }
            evidence_ref = self._representations.put(
                json.dumps(receipt, ensure_ascii=False, sort_keys=True).encode("utf-8")
            )
            output.append(
                TemporalOcrText(
                    result.full_text,
                    TemporalRange(frame.timestamp_ms, frame.timestamp_ms + 1),
                    evidence_ref,
                    text_ref,
                )
            )
        check_deadline(deadline)
        return tuple(output)
