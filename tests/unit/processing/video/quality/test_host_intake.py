"""Trusted host snapshots must be validated without a second unmanaged copy."""

import hashlib
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.video.quality import QualityError, QualityLimits
from capicove.processing.video.quality._adapters.ffmpeg_quality import _executable
from capicove.processing.video.quality._adapters.local_media import VerifiedQualityIntake
from capicove.processing.video.quality.video_quality_ports import QualitySource


class Snapshot:
    def __init__(self, source: QualitySource) -> None:
        self.source = source

    @contextmanager
    def stage(self, path: Path, deadline: datetime) -> Iterator[QualitySource]:
        assert path == self.source.original_path
        yield self.source


def source(path: Path, *, digest: str | None = None) -> QualitySource:
    path.write_bytes(b"\x1aE\xdf\xa3fixture")
    return QualitySource(
        path,
        Path("/authorized/original.mkv"),
        digest or hashlib.sha256(path.read_bytes()).hexdigest(),
        path.stat().st_size,
        "",
    )


def test_verified_port_keeps_descriptor_locator_and_original_identity(tmp_path: Path) -> None:
    snapshot = source(tmp_path / "bound.mkv")
    intake = VerifiedQualityIntake(Snapshot(snapshot), QualityLimits())
    with intake.stage(snapshot.original_path, datetime.now(UTC) + timedelta(seconds=5)) as accepted:
        assert accepted.staged_path.stat().st_ino == snapshot.staged_path.stat().st_ino
        assert accepted.original_path == snapshot.original_path
        assert accepted.sha256 == snapshot.sha256
        assert accepted.demuxer == "matroska"


def test_forged_host_digest_is_rejected_before_provider(tmp_path: Path) -> None:
    snapshot = source(tmp_path / "bound.mkv", digest="0" * 64)
    with pytest.raises(QualityError, match="verified"):
        with VerifiedQualityIntake(Snapshot(snapshot), QualityLimits()).stage(
            snapshot.original_path, datetime.now(UTC) + timedelta(seconds=5)
        ):
            raise AssertionError("forged source must not reach a provider")


def test_changed_verified_copy_is_rejected_on_return(tmp_path: Path) -> None:
    snapshot = source(tmp_path / "bound.mkv")
    with pytest.raises(QualityError, match="verified"):
        with VerifiedQualityIntake(Snapshot(snapshot), QualityLimits()).stage(
            snapshot.original_path, datetime.now(UTC) + timedelta(seconds=5)
        ):
            snapshot.staged_path.write_bytes(b"\x1aE\xdf\xa3changed")


def test_configured_executable_locator_is_not_resolved(tmp_path: Path) -> None:
    executable = tmp_path / "real-program"
    executable.write_text("#!/bin/sh\nexit 0\n")
    executable.chmod(0o700)
    locator = tmp_path / "descriptor-locator"
    locator.symlink_to(executable)
    assert _executable(str(locator)) == locator
