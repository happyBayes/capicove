from __future__ import annotations

from dataclasses import replace
from pathlib import Path, PurePosixPath

import pytest

from capicove.batch.models import (
    BatchItem,
    BatchModality,
    OperationFingerprint,
    SourceSnapshot,
)
from capicove.batch.planner import BatchPlanner


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64


def _item(
    relative_path: str,
    *,
    item_sha256: str,
    source_sha256: str = _SHA_C,
    modality: BatchModality = BatchModality.AUDIO,
) -> BatchItem:
    return BatchItem(
        item_id=f"item:{item_sha256}",
        modality=modality,
        source_path=Path("/data") / relative_path,
        relative_path=PurePosixPath(relative_path),
        media_type="audio/wav",
        sha256=source_sha256,
        snapshot=SourceSnapshot(1, 2, 3, 4, 5),
    )


def _operation() -> OperationFingerprint:
    return OperationFingerprint(
        operation_id="capicove.audio.transcribe",
        operation_version="1",
        modalities=frozenset({BatchModality.AUDIO}),
        parameters_sha256=_SHA_A,
        provider_fingerprint=_SHA_B,
    )


def test_planner_is_order_independent_and_reuses_identical_content_keys() -> None:
    first = _item("z.wav", item_sha256=_SHA_A)
    second = _item("nested/a.wav", item_sha256=_SHA_B)
    planner = BatchPlanner()

    forward = planner.plan((first, second), _operation())
    reverse = planner.plan((second, first), _operation())

    assert [planned.item.relative_path for planned in forward.items] == [
        PurePosixPath("nested/a.wav"),
        PurePosixPath("z.wav"),
    ]
    assert forward.items[0].cache_key == forward.items[1].cache_key
    assert forward.plan_id == reverse.plan_id
    assert forward.operation == _operation()


def test_cache_key_changes_for_every_cache_affecting_input() -> None:
    item = _item("meeting.wav", item_sha256=_SHA_A)
    operation = _operation()
    planner = BatchPlanner()
    baseline = planner.plan((item,), operation).items[0].cache_key

    variants = (
        planner.plan((replace(item, sha256=_SHA_D),), operation).items[0].cache_key,
        planner.plan(
            (item,),
            replace(operation, operation_id="capicove.audio.transcribe.v2"),
        )
        .items[0]
        .cache_key,
        planner.plan((item,), replace(operation, operation_version="2")).items[0].cache_key,
        planner.plan(
            (item,),
            replace(operation, parameters_sha256=_SHA_D),
        )
        .items[0]
        .cache_key,
        planner.plan(
            (item,),
            replace(operation, provider_fingerprint=_SHA_D),
        )
        .items[0]
        .cache_key,
    )

    assert len({baseline, *variants}) == 6
    assert baseline.startswith("cache:")


def test_planner_rejects_duplicate_item_identity() -> None:
    item = _item("meeting.wav", item_sha256=_SHA_A)
    duplicate = replace(item, relative_path=PurePosixPath("copy.wav"))

    with pytest.raises(ValueError, match="duplicate item_id"):
        BatchPlanner().plan((item, duplicate), _operation())


def test_planner_rejects_a_modality_the_operation_does_not_support() -> None:
    image = _item(
        "frame.png",
        item_sha256=_SHA_A,
        modality=BatchModality.IMAGE,
    )

    with pytest.raises(ValueError, match="does not support image"):
        BatchPlanner().plan((image,), _operation())
