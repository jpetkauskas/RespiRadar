import pytest

from respiradar.sources import replay_frames


def test_replay_yields_every_frame_in_the_recording(sitting_recording):
    frames = list(replay_frames(sitting_recording))

    assert len(frames) == 773
    assert frames[0].iq.shape == (16, 21)


def test_replay_reports_the_distance_of_each_range_point(sitting_recording):
    frame = next(replay_frames(sitting_recording))

    assert len(frame.distances_m) == 21
    assert frame.distances_m[0] == pytest.approx(0.2975, abs=1e-3)
    assert frame.distances_m[-1] == pytest.approx(1.4975, abs=1e-3)


def test_replay_timestamps_advance_at_the_recorded_frame_rate(sitting_recording):
    frames = list(replay_frames(sitting_recording))

    assert frames[0].t == pytest.approx(0.0)
    assert frames[20].t == pytest.approx(1.0)


def test_config_round_trips_through_a_recording_without_gaining_a_point(sitting_recording):
    """The recording has 21 range points; floating point must not turn that into 22."""
    from respiradar.sources import recorded_config

    config = recorded_config(sitting_recording)

    assert config.num_points == 21


def test_default_config_covers_the_requested_range():
    from respiradar.sources import RadarConfig

    config = RadarConfig()

    assert config.num_points == 21
    assert config.distances_m[-1] <= config.end_m
