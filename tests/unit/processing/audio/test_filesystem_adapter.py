from pathlib import Path

import pytest

from capicove.processing.audio._adapters.filesystem import ExplicitAudioPathAuthorizer
from capicove.processing.audio.ports import AudioPathPolicyError


def test_audio_path_authorizer_accepts_only_configured_read_roots(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    source = allowed / "meeting.wav"
    source.touch()
    authorizer = ExplicitAudioPathAuthorizer({"policy-1": (allowed,)})

    assert authorizer.authorize_read("policy-1", source) == source.resolve()

    link = allowed / "link.wav"
    link.symlink_to(source)
    assert authorizer.authorize_read("policy-1", link) == source.resolve()

    with pytest.raises(AudioPathPolicyError, match="outside configured read roots"):
        authorizer.authorize_read("policy-1", tmp_path / "private.wav")


def test_audio_path_authorizer_rejects_missing_policy_and_relative_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="absolute"):
        ExplicitAudioPathAuthorizer({"policy-1": (Path("relative"),)})

    authorizer = ExplicitAudioPathAuthorizer({"policy-1": (tmp_path,)})
    with pytest.raises(AudioPathPolicyError, match="not configured"):
        authorizer.authorize_read("missing", tmp_path / "meeting.wav")
