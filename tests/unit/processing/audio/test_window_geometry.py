"""Sample-exact native audio window geometry and receipts (RFC-0032)."""

from __future__ import annotations

import pytest

from capicove.contracts.retrieval import TemporalRange
from capicove.processing.audio.retrieval import (
    AudioWindowPolicy,
    AudioWindowReceipt,
    plan_windows,
)


POLICY = AudioWindowPolicy()


def _frames(windows: tuple[object, ...]) -> list[tuple[int, int]]:
    return [(w.start_frame, w.end_frame) for w in windows]  # type: ignore[attr-defined]


def test_reference_profile_is_16khz_mono_30s_windows_25s_hop() -> None:
    assert (POLICY.sample_rate_hz, POLICY.window_frames, POLICY.hop_frames) == (
        16_000,
        480_000,
        400_000,
    )


def test_thirty_one_seconds_yields_two_windows_including_short_tail() -> None:
    windows = plan_windows(31 * 16_000, POLICY)
    assert _frames(windows) == [(0, 480_000), (400_000, 496_000)]
    assert [w.time_range for w in windows] == [
        TemporalRange(0, 30_000),
        TemporalRange(25_000, 31_000),
    ]


def test_exactly_one_window_does_not_emit_contained_trailing_window() -> None:
    assert _frames(plan_windows(480_000, POLICY)) == [(0, 480_000)]
    assert _frames(plan_windows(1, POLICY)) == [(0, 1)]


def test_window_stops_when_an_emitted_window_reaches_the_end() -> None:
    assert _frames(plan_windows(880_000, POLICY)) == [(0, 480_000), (400_000, 880_000)]
    assert _frames(plan_windows(880_001, POLICY)) == [
        (0, 480_000),
        (400_000, 880_000),
        (800_000, 880_001),
    ]


def test_display_milliseconds_floor_start_and_ceil_end() -> None:
    policy = AudioWindowPolicy(sample_rate_hz=16_000, window_seconds=1, hop_seconds=1)
    windows = plan_windows(16_001, policy)
    assert windows[-1].start_frame == 16_000 and windows[-1].end_frame == 16_001
    assert windows[-1].time_range == TemporalRange(1_000, 1_001)
    assert windows[0].index == 0 and windows[-1].index == 1


@pytest.mark.parametrize("frames", [0, -1, True, 1.5])
def test_zero_or_invalid_duration_is_rejected(frames: object) -> None:
    with pytest.raises(ValueError):
        plan_windows(frames, POLICY)  # type: ignore[arg-type]


def test_limits_reject_overlong_audio_and_window_count_without_truncation() -> None:
    with pytest.raises(ValueError, match="duration"):
        plan_windows(4 * 3600 * 16_000 + 1, POLICY)
    tight = AudioWindowPolicy(max_windows=2)
    with pytest.raises(ValueError, match="window"):
        plan_windows(880_001, tight)


@pytest.mark.parametrize(
    "overrides",
    [
        {"sample_rate_hz": 7_999},
        {"window_seconds": 0},
        {"hop_seconds": 31},
        {"hop_seconds": 0},
        {"max_windows": 0},
        {"max_duration_seconds": 4 * 3600 + 1},
        {"window_seconds": True},
    ],
)
def test_policy_rejects_invalid_geometry(overrides: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        AudioWindowPolicy(**overrides)  # type: ignore[arg-type]


def test_every_policy_parameter_changes_the_fingerprint() -> None:
    base = POLICY.fingerprint.sha256
    for changed in (
        AudioWindowPolicy(sample_rate_hz=24_000),
        AudioWindowPolicy(window_seconds=40),
        AudioWindowPolicy(hop_seconds=20),
    ):
        assert changed.fingerprint.sha256 != base
    assert AudioWindowPolicy().fingerprint.sha256 == base


def test_receipt_binds_source_stream_samples_decoder_and_pcm() -> None:
    receipt = AudioWindowReceipt(
        source_sha256="a" * 64,
        audio_stream_index=0,
        selection_method="sole_stream_implicit",
        start_frame=400_000,
        end_frame=496_000,
        sample_rate_hz=16_000,
        channels=1,
        decoder="ffmpeg 6.1.1",
        pcm_sha256="b" * 64,
        preprocessing_sha256=POLICY.fingerprint.sha256,
    )
    assert receipt.time_range == TemporalRange(25_000, 31_000)
    payload = receipt.to_payload()
    assert payload["schema"] == "capicove.audio.window-receipt.v1"
    assert AudioWindowReceipt.from_payload(payload) == receipt
    other = AudioWindowReceipt.from_payload({**payload, "pcm_sha256": "c" * 64})
    assert other.digest != receipt.digest
    with pytest.raises(ValueError):
        AudioWindowReceipt.from_payload({**payload, "unexpected": 1})
    with pytest.raises(ValueError):
        AudioWindowReceipt.from_payload({**payload, "end_frame": 400_000})
