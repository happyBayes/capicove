from __future__ import annotations

from dataclasses import FrozenInstanceError
from pathlib import Path, PurePosixPath

import pytest

from capicove.batch.models import (
    BatchItem,
    BatchModality,
    BatchState,
    OperationFingerprint,
    SourceSnapshot,
)


_SHA_A = "a" * 64
_SHA_B = "b" * 64


def _snapshot() -> SourceSnapshot:
    return SourceSnapshot(
        device=1,
        inode=2,
        size_bytes=3,
        modified_ns=4,
        changed_ns=5,
    )


def test_batch_state_exposes_every_persisted_lifecycle_value() -> None:
    assert {state.value for state in BatchState} == {
        "pending",
        "running",
        "succeeded",
        "failed",
        "skipped",
        "source_changed",
    }


def test_batch_item_preserves_validated_immutable_source_identity() -> None:
    item = BatchItem(
        item_id=f"item:{_SHA_A}",
        modality=BatchModality.AUDIO,
        source_path=Path("/data/meeting.wav"),
        relative_path=PurePosixPath("calls/meeting.wav"),
        media_type="audio/wav",
        sha256=_SHA_B,
        snapshot=_snapshot(),
    )

    assert item.size_bytes == 3
    with pytest.raises(FrozenInstanceError):
        item.media_type = "audio/mpeg"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("item_id", "item:not-a-digest", "item_id"),
        ("source_path", Path("relative.wav"), "source_path"),
        ("relative_path", PurePosixPath("../escape.wav"), "relative_path"),
        ("media_type", " ", "media_type"),
        ("sha256", "ABC", "sha256"),
    ],
)
def test_batch_item_rejects_ambiguous_or_unsafe_identity(
    field: str,
    value: object,
    message: str,
) -> None:
    values: dict[str, object] = {
        "item_id": f"item:{_SHA_A}",
        "modality": BatchModality.AUDIO,
        "source_path": Path("/data/meeting.wav"),
        "relative_path": PurePosixPath("meeting.wav"),
        "media_type": "audio/wav",
        "sha256": _SHA_B,
        "snapshot": _snapshot(),
    }
    values[field] = value

    with pytest.raises(ValueError, match=message):
        BatchItem(**values)  # type: ignore[arg-type]


def test_source_snapshot_rejects_negative_stat_values() -> None:
    with pytest.raises(ValueError, match="size_bytes"):
        SourceSnapshot(device=1, inode=2, size_bytes=-1, modified_ns=4, changed_ns=5)


def test_operation_fingerprint_requires_complete_content_digests() -> None:
    fingerprint = OperationFingerprint(
        operation_id="capicove.audio.transcribe",
        operation_version="1",
        modalities=frozenset({BatchModality.AUDIO}),
        parameters_sha256=_SHA_A,
        provider_fingerprint=_SHA_B,
    )

    assert fingerprint.modalities == frozenset({BatchModality.AUDIO})

    with pytest.raises(ValueError, match="modalities"):
        OperationFingerprint(
            operation_id="capicove.audio.transcribe",
            operation_version="1",
            modalities=frozenset(),
            parameters_sha256=_SHA_A,
            provider_fingerprint=_SHA_B,
        )
    with pytest.raises(ValueError, match="provider_fingerprint"):
        OperationFingerprint(
            operation_id="capicove.audio.transcribe",
            operation_version="1",
            modalities=frozenset({BatchModality.AUDIO}),
            parameters_sha256=_SHA_A,
            provider_fingerprint="whisper.cpp",
        )
