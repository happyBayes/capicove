"""Tests for image-model selection and managed destination policy."""

from __future__ import annotations

from pathlib import Path

import pytest

from capicove.model.manifest import load_packaged_manifest
from capicove.model.selection import resolve_model_home, select_models


def test_default_profile_excludes_restricted_model() -> None:
    selected = select_models(
        load_packaged_manifest(),
        default_profile=True,
        keys=(),
        include_restricted=False,
    )

    assert [model.key for model in selected] == [
        "nsfw-falconsai",
        "watermark-siglip2",
        "clip-vit-b32",
        "blip-itm-coco",
        "blip-caption-base",
        "smolvlm-256m-instruct",
    ]


def test_explicit_restricted_model_requires_authorization() -> None:
    with pytest.raises(ValueError, match="--include-restricted"):
        select_models(
            load_packaged_manifest(),
            default_profile=False,
            keys=("aesthetics-cafe",),
            include_restricted=False,
        )


def test_explicit_restricted_model_can_be_selected() -> None:
    selected = select_models(
        load_packaged_manifest(),
        default_profile=False,
        keys=("aesthetics-cafe",),
        include_restricted=True,
    )

    assert tuple(model.key for model in selected) == ("aesthetics-cafe",)


def test_selection_deduplicates_keys_in_manifest_order() -> None:
    selected = select_models(
        load_packaged_manifest(),
        default_profile=False,
        keys=("clip-vit-b32", "nsfw-falconsai", "clip-vit-b32"),
        include_restricted=False,
    )

    assert tuple(model.key for model in selected) == (
        "nsfw-falconsai",
        "clip-vit-b32",
    )


def test_selection_rejects_empty_and_unknown_requests() -> None:
    manifest = load_packaged_manifest()

    with pytest.raises(ValueError, match="select --profile default or at least one --model"):
        select_models(
            manifest,
            default_profile=False,
            keys=(),
            include_restricted=False,
        )
    with pytest.raises(ValueError, match="unknown image model: missing"):
        select_models(
            manifest,
            default_profile=False,
            keys=("missing",),
            include_restricted=False,
        )


def test_explicit_destination_has_highest_precedence(tmp_path: Path) -> None:
    destination = tmp_path / "models"

    resolved = resolve_model_home(
        destination,
        {"CAPICOVE_MODEL_HOME": "/ignored"},
        "Darwin",
        Path("/Users/example"),
    )

    assert resolved == destination.resolve()
    assert not destination.exists()


@pytest.mark.parametrize(
    ("environment", "system", "home", "expected"),
    (
        (
            {"CAPICOVE_MODEL_HOME": "/managed/models"},
            "Darwin",
            Path("/Users/example"),
            Path("/managed/models"),
        ),
        (
            {},
            "Darwin",
            Path("/Users/example"),
            Path("/Users/example/Library/Caches/capicove/models"),
        ),
        (
            {"LOCALAPPDATA": "C:/Users/example/AppData/Local"},
            "Windows",
            Path("C:/Users/example"),
            Path("C:/Users/example/AppData/Local/capicove/models"),
        ),
        (
            {"XDG_CACHE_HOME": "/cache"},
            "Linux",
            Path("/home/example"),
            Path("/cache/capicove/models"),
        ),
        ({}, "Linux", Path("/home/example"), Path("/home/example/.cache/capicove/models")),
    ),
)
def test_model_home_uses_environment_and_platform_defaults(
    environment: dict[str, str],
    system: str,
    home: Path,
    expected: Path,
) -> None:
    assert resolve_model_home(None, environment, system, home) == expected
