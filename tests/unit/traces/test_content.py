"""Tests for immutable trace content and media values."""

from __future__ import annotations

import math
from dataclasses import FrozenInstanceError
from typing import cast

import pytest

from capicove.traces._json import FrozenJson, freeze_json, thaw_json
from capicove.traces.content import (
    AgentLineage,
    LineageAvailability,
    MediaContent,
    MediaLocator,
    MediaLocatorKind,
    MediaReference,
    TextContent,
    TraceError,
    TraceLineage,
    TraceModality,
    TraceWarning,
)


SHA256 = "a" * 64


def _media(**changes: object) -> MediaReference:
    values: dict[str, object] = {
        "media_id": "media-1",
        "modality": TraceModality.IMAGE,
        "media_type": "image/png",
        "sha256": SHA256,
        "size_bytes": 12,
        "locator": MediaLocator(MediaLocatorKind.WORKSPACE_RELATIVE, "inputs/page.png"),
    }
    values.update(changes)
    return MediaReference(**values)  # type: ignore[arg-type]


def test_content_values_support_all_modalities_and_are_frozen() -> None:
    references = (
        _media(),
        _media(
            media_id="media-2",
            modality=TraceModality.VIDEO,
            media_type="video/mp4",
            start_seconds=1.0,
            end_seconds=2.0,
            frame_indices=(3, 9),
        ),
        _media(
            media_id="media-3",
            modality=TraceModality.AUDIO,
            media_type="audio/wav",
        ),
    )

    assert tuple(reference.modality.value for reference in references) == (
        "image",
        "video",
        "audio",
    )
    assert TextContent("visible").text == "visible"
    assert MediaContent("media-2").media_id == "media-2"
    with pytest.raises(FrozenInstanceError):
        references[0].media_id = "changed"  # type: ignore[misc]


def test_lineage_deeply_freezes_caller_owned_values() -> None:
    source: dict[str, object] = {"nested": {"items": [1, "two"]}}
    lineage = TraceLineage(source)

    nested = source["nested"]
    assert isinstance(nested, dict)
    items = nested["items"]
    assert isinstance(items, list)
    items.append("changed")
    source["later"] = True

    assert thaw_json(cast(FrozenJson, lineage.values)) == {"nested": {"items": [1, "two"]}}
    with pytest.raises(TypeError):
        lineage.values["changed"] = True  # type: ignore[index]


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"sha256": "bad"}, "sha256"),
        ({"size_bytes": -1}, "size_bytes"),
        ({"start_seconds": 2.0, "end_seconds": 1.0}, "time"),
        ({"frame_indices": (2, 2)}, "frame_indices"),
    ],
)
def test_media_reference_rejects_invalid_identity_and_selection(
    change: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _media(**change)


@pytest.mark.parametrize("value", ["/absolute/file.wav", "../escape.wav"])
def test_media_locator_rejects_absolute_and_escaping_paths(value: str) -> None:
    with pytest.raises(ValueError, match="locator"):
        MediaLocator(MediaLocatorKind.BUNDLE, value)


@pytest.mark.parametrize(
    "value",
    [
        b"bytes",
        {"set"},
        {1: "non-string key"},
        math.nan,
        math.inf,
    ],
)
def test_freeze_json_rejects_non_json_values(value: object) -> None:
    with pytest.raises(ValueError, match="value"):
        freeze_json(value, field_name="value")


def test_freeze_json_rejects_cycles_and_excessive_depth() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(ValueError, match="cycle"):
        freeze_json(cyclic, field_name="value")

    nested: object = 0
    for _ in range(34):
        nested = [nested]
    with pytest.raises(ValueError, match="depth"):
        freeze_json(nested, field_name="value")


def test_common_values_enforce_identity_and_boolean_contracts() -> None:
    with pytest.raises(ValueError, match="captured"):
        AgentLineage(LineageAvailability.CAPTURED)
    lineage = AgentLineage(LineageAvailability.CAPTURED, model_id="model-1")
    assert lineage.model_id == "model-1"

    with pytest.raises(ValueError, match="retryable"):
        TraceError("internal", "failed", retryable=1)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="message"):
        TraceWarning("warning", " ")
