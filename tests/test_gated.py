"""The detector must not alarm about an empty room.

Every recording has a person in it throughout, so this case is untested by the bake-off
and was found only by simulating it. Ungated, the change-point detector alarms on about
half the frames of an empty room - nobody there produces no chest motion, which is exactly
what apnea looks like.
"""

import itertools

import numpy as np

from respiradar.bakeoff import Clip, folds
from respiradar.dataset import FeatureExtractor
from respiradar.detectors.changepoint import build_conservative
from respiradar.detectors.gated import build
from respiradar.sources import RadarConfig, simulated_frames


def _empty_room_clip(seconds: int = 180) -> Clip:
    config = RadarConfig(sweeps_per_frame=8)
    extractor = FeatureExtractor(config)
    times, rows = [], []
    frames = simulated_frames(config, breaths_per_min=None, realtime=False)
    for frame in itertools.islice(frames, seconds * int(config.frame_rate)):
        times.append(frame.t)
        rows.append(extractor.process(frame))
    t = np.asarray(times)
    return Clip("empty-room", t, np.asarray(rows), np.zeros(len(t), bool), [])


def test_gated_detector_stays_quiet_in_an_empty_room():
    clip = _empty_room_clip()
    detector = build()
    detector.fit(folds()[0][0])

    alarms = detector.predict(clip)

    assert not alarms[clip.t >= 25].any()


def test_the_gate_is_what_makes_the_difference():
    """Guards against the gate silently becoming a no-op."""
    clip = _empty_room_clip()
    ungated = build_conservative()
    ungated.fit(folds()[0][0])

    assert ungated.predict(clip)[clip.t >= 25].any()


def test_the_gate_costs_nothing_on_real_recordings():
    from respiradar.bakeoff import score

    gated = score(build())
    plain = score(build_conservative())

    assert gated.detected == plain.detected
    assert gated.false_alarms == plain.false_alarms


def test_the_best_detector_catches_every_hold_without_false_alarms():
    """The demo candidate: 4/4 holds, zero false alarms, leave-one-subject-out."""
    from respiradar.bakeoff import score
    from respiradar.detectors.gated import build_best

    result = score(build_best())

    assert result.detected == 4
    assert result.missed == 0
    assert result.false_alarms == 0


def test_the_best_detector_also_stays_quiet_in_an_empty_room():
    clip = _empty_room_clip()
    detector = build_best_detector()
    detector.fit(folds()[0][0])

    assert not detector.predict(clip)[clip.t >= 25].any()


def build_best_detector():
    from respiradar.detectors.gated import build_best

    return build_best()
