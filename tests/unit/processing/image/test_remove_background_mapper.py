"""Tests for background-removal mapping and derived-image lineage."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest
from PIL import Image

from capicove.processing.image.mappers import (
    BackgroundRemovalOptions,
    ImageRemoveBackgroundMapper,
)
from capicove.processing.image.models import ImageSource
from tests.unit.processing.image.authored_images import write_authored_image


class AlphaProvider:
    """Create deterministic alpha output without a model dependency."""

    provider_id = "test-alpha"
    provider_version = "1"
    model_id = "test-alpha-model"
    model_revision = "sha256:test-alpha-model"
    model_manifest_sha256 = "b" * 64
    preprocess_revision = "test-alpha-v1"
    device = "cpu"

    def __init__(self) -> None:
        self.calls = 0
        self.options: list[BackgroundRemovalOptions] = []

    def remove(self, image: Image.Image, options: BackgroundRemovalOptions) -> Image.Image:
        self.calls += 1
        self.options.append(options)
        rgba = image.convert("RGBA")
        rgba.putalpha(128)
        return rgba


def test_remove_background_mapper_writes_png_without_mutating_source(tmp_path: Path) -> None:
    source_path = write_authored_image(tmp_path / "source.png")
    before = source_path.read_bytes()
    provider = AlphaProvider()
    mapper = ImageRemoveBackgroundMapper(provider, tmp_path / "output")

    result = mapper.map(
        (ImageSource(path=str(source_path)),),
        BackgroundRemovalOptions(alpha_matting=True),
    )

    assert source_path.read_bytes() == before
    assert provider.options[0].alpha_matting is True
    assert result[0].path.endswith(".png")
    assert result[0].source_hashes == (hashlib.sha256(before).hexdigest(),)
    assert Path(result[0].path).is_file()
    assert result[0].operation_version == 1
    assert dict(result[0].parameters)["alpha_matting"] is True
    assert result[0].model_id == "test-alpha-model"
    assert result[0].model_revision == "sha256:test-alpha-model"
    with Image.open(result[0].path) as output:
        assert output.mode == "RGBA"


def test_remove_background_mapper_reuses_duplicate_source(tmp_path: Path) -> None:
    provider = AlphaProvider()
    mapper = ImageRemoveBackgroundMapper(provider, tmp_path / "output")
    source = ImageSource(path=str(write_authored_image(tmp_path / "source.png")))

    result = mapper.map((source, source))

    assert provider.calls == 1
    assert result[0] == result[1]


def test_remove_background_mapper_rejects_corrupt_existing_artifact(tmp_path: Path) -> None:
    """Never return lineage whose digest disagrees with the persisted bytes."""
    mapper = ImageRemoveBackgroundMapper(AlphaProvider(), tmp_path / "output")
    source = ImageSource(path=str(write_authored_image(tmp_path / "source.png")))
    artifact = mapper.map((source,))[0]
    Path(artifact.path).write_bytes(b"corrupt")

    with pytest.raises(RuntimeError, match="content-addressed artifact"):
        mapper.map((source,))


def test_remove_background_mapper_never_overwrites_concurrent_digest_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "output"
    mapper = ImageRemoveBackgroundMapper(AlphaProvider(), output)
    source = ImageSource(path=str(write_authored_image(tmp_path / "source.png")))
    real_link = os.link
    competitor = b"concurrent-writer"

    def publish_competitor_first(
        source_name: str,
        destination_name: str,
        *,
        src_dir_fd: int | None = None,
        dst_dir_fd: int | None = None,
        follow_symlinks: bool = True,
    ) -> None:
        descriptor = os.open(
            destination_name,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
            dir_fd=dst_dir_fd,
        )
        try:
            os.write(descriptor, competitor)
        finally:
            os.close(descriptor)
        real_link(
            source_name,
            destination_name,
            src_dir_fd=src_dir_fd,
            dst_dir_fd=dst_dir_fd,
            follow_symlinks=follow_symlinks,
        )

    monkeypatch.setattr(
        "capicove.processing.image.mappers.remove_background.os.link",
        publish_competitor_first,
    )

    with pytest.raises(RuntimeError, match="content-addressed artifact"):
        mapper.map((source,))

    assert [path.read_bytes() for path in output.glob("*.png")] == [competitor]
