"""Video OCR delegates recognition to the public image operation."""

import json
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from hashlib import sha256
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image

from capicove.contracts.retrieval import ProcessingFingerprint, TemporalRange
from capicove.processing.image import ImageOcr, OcrWord
from capicove.processing.image.ocr import OcrProviderOutput, OcrProviderUnavailableError
from capicove.processing.video.ports import BackendUnavailableError
from capicove.processing.video.retrieval._adapters.local import LocalContentStore
from capicove.processing.video.retrieval.models import RetrievalKeyframe
from capicove.processing.video.retrieval.shared_ocr import SharedImageOCRProcessor


class Provider:
    provider_id = "fixture-ocr"
    provider_version = "1"

    def __init__(self) -> None:
        self.calls = 0
        self.empty = False
        self.unavailable = False

    def recognize(self, image: Image.Image, languages: tuple[str, ...]) -> OcrProviderOutput:
        self.calls += 1
        if self.unavailable:
            raise OcrProviderUnavailableError("missing local model")
        return OcrProviderOutput(
            () if self.empty else (OcrWord("100.01", 0, 0, 8, 8, 0.9, 1, 1, 1, 1, 1),)
        )


def setup(
    tmp_path: Path,
) -> tuple[SharedImageOCRProcessor, Provider, LocalContentStore, RetrievalKeyframe]:
    store = LocalContentStore(tmp_path)
    output = BytesIO()
    Image.new("RGB", (16, 16)).save(output, format="PNG")
    payload = output.getvalue()
    ref = store.put(payload)
    frame = RetrievalKeyframe(
        "kf:1",
        1200,
        TemporalRange(0, 4000),
        ref,
        sha256(payload).hexdigest(),
        0.8,
        ProcessingFingerprint("b" * 64),
    )
    provider = Provider()
    processor = SharedImageOCRProcessor(
        ImageOcr(provider), store, store, provider_fingerprint=ProcessingFingerprint("c" * 64)
    )
    return processor, provider, store, frame


def test_public_image_ocr_preserves_text_and_point_evidence(tmp_path: Path) -> None:
    processor, provider, store, frame = setup(tmp_path)
    (result,) = processor.recognize((frame,), deadline=datetime.now(UTC) + timedelta(seconds=5))
    assert provider.calls == 1
    assert result.text == "100.01"
    assert result.time_range == TemporalRange(1200, 1201)
    assert store.read(result.representation_ref) == b"100.01"
    receipt = json.loads(store.read(result.evidence_ref))
    assert receipt["recognition"]["provider_id"] == "fixture-ocr"
    assert receipt["recognition"]["source_hash"] == frame.preview_sha256
    assert receipt["preview_resource_ref"] == frame.preview_resource_ref


def test_hash_mismatch_prevents_recognition(tmp_path: Path) -> None:
    processor, provider, _, frame = setup(tmp_path)
    with pytest.raises(ValueError, match="hash"):
        processor.recognize(
            (replace(frame, preview_sha256="a" * 64),),
            deadline=datetime.now(UTC) + timedelta(seconds=5),
        )
    assert provider.calls == 0


def test_expired_deadline_prevents_recognition(tmp_path: Path) -> None:
    processor, provider, _, frame = setup(tmp_path)
    with pytest.raises(TimeoutError):
        processor.recognize((frame,), deadline=datetime.now(UTC) - timedelta(seconds=1))
    assert provider.calls == 0


def test_empty_text_is_not_a_searchable_unit(tmp_path: Path) -> None:
    processor, provider, _, frame = setup(tmp_path)
    provider.empty = True
    assert processor.recognize((frame,), deadline=datetime.now(UTC) + timedelta(seconds=5)) == ()


def test_unavailable_provider_uses_pipeline_optional_failure(tmp_path: Path) -> None:
    processor, provider, _, frame = setup(tmp_path)
    provider.unavailable = True
    with pytest.raises(BackendUnavailableError):
        processor.recognize((frame,), deadline=datetime.now(UTC) + timedelta(seconds=5))


def test_identity_is_required_and_language_changes_fingerprint(tmp_path: Path) -> None:
    store = LocalContentStore(tmp_path)
    service = ImageOcr(Provider())
    with pytest.raises(ValueError, match="provider_fingerprint"):
        SharedImageOCRProcessor(service, store, store)
    first = SharedImageOCRProcessor(
        service, store, store, provider_fingerprint=ProcessingFingerprint("c" * 64)
    )
    changed = SharedImageOCRProcessor(
        service,
        store,
        store,
        provider_fingerprint=ProcessingFingerprint("c" * 64),
        languages=("eng",),
    )
    upgraded = SharedImageOCRProcessor(
        service, store, store, provider_fingerprint=ProcessingFingerprint("d" * 64)
    )
    assert len({p.fingerprint for p in (first, changed, upgraded)}) == 3
    assert (
        first.fingerprint
        == SharedImageOCRProcessor(
            service,
            store,
            store,
            provider_fingerprint=ProcessingFingerprint("c" * 64),
            languages=("chi_sim", "eng"),
        ).fingerprint
    )
