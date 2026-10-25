import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from capicove.processing.video import AcceptedVideoSource, VideoSource
from capicove.processing.video.ports import ArtifactResolver, PathAuthorizer
from capicove.processing.video.source import VideoSourceError, accept_source


def _deadline() -> datetime:
    return datetime.now(UTC) + timedelta(minutes=1)


class AllowPaths(PathAuthorizer):
    def authorize_read(self, policy_ref: str, path: Path) -> None:
        assert policy_ref == "policy-1"

    def authorize_write(self, policy_ref: str, path: Path) -> None:
        assert policy_ref == "policy-1"


class FakeResolver(ArtifactResolver):
    def __init__(self, source: AcceptedVideoSource) -> None:
        self.source = source

    def resolve(
        self,
        ref: str,
        expected_sha256: str,
        deadline: datetime,
    ) -> AcceptedVideoSource:
        assert ref == "artifact:video:1"
        assert expected_sha256
        assert deadline.tzinfo is not None
        return self.source


def test_accept_source_hashes_bytes_without_mutating_them() -> None:
    payload = b"synthetic-video"

    accepted = accept_source(VideoSource(data=payload), _deadline())

    assert accepted.data == payload
    assert accepted.sha256 == hashlib.sha256(payload).hexdigest()
    assert accepted.source_id == f"sha256:{accepted.sha256}"


def test_accept_source_hashes_an_authorized_existing_path(tmp_path: Path) -> None:
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"synthetic-video")

    accepted = accept_source(
        VideoSource(path=source_path),
        _deadline(),
        path_authorizer=AllowPaths(),
        policy_ref="policy-1",
    )

    assert accepted.path == source_path.resolve()
    assert accepted.data is None
    assert accepted.mime_type == "video/mp4"


def test_accept_source_rejects_a_path_without_an_authorizer(tmp_path: Path) -> None:
    source_path = tmp_path / "clip.mp4"
    source_path.write_bytes(b"video")

    with pytest.raises(VideoSourceError, match="authorizer"):
        accept_source(VideoSource(path=source_path), _deadline())


def test_accept_source_rejects_a_missing_path(tmp_path: Path) -> None:
    with pytest.raises(VideoSourceError, match="does not exist"):
        accept_source(
            VideoSource(path=tmp_path / "missing.mp4"),
            _deadline(),
            path_authorizer=AllowPaths(),
            policy_ref="policy-1",
        )


def test_accept_source_rejects_artifact_hash_mismatch() -> None:
    payload = b"resolved-video"
    actual_hash = hashlib.sha256(payload).hexdigest()
    resolved = AcceptedVideoSource(
        data=payload,
        source_id=f"sha256:{actual_hash}",
        sha256=actual_hash,
        mime_type="video/mp4",
        artifact_ref="artifact:video:1",
    )

    with pytest.raises(VideoSourceError, match="hash"):
        accept_source(
            VideoSource(
                artifact_ref="artifact:video:1",
                artifact_sha256="0" * 64,
            ),
            _deadline(),
            artifact_resolver=FakeResolver(resolved),
        )
