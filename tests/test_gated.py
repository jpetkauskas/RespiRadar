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


def test_the_gate_costs_no_detections_and_removes_false_alarms():
    """The gate must be free on holds and strictly better on empty scenes.

    It used to be free in both directions, because the only empty room available was
    simulated and nothing alarmed at it anyway. With a real wall recording scored, the
    ungated detector alarms and the gated one does not - so the gate now earns its place
    rather than merely costing nothing.
    """
    from respiradar.bakeoff import score

    gated = score(build())
    plain = score(build_conservative())

    assert gated.detected == plain.detected
    assert gated.false_alarms <= plain.false_alarms


def test_the_shipped_detector_is_not_obviously_broken():
    """A floor, not a target.

    This deliberately does NOT pin an exact hold count. An earlier version asserted 4/4
    with zero false alarms, which was true when there were four labelled holds and became
    false the moment nine more arrived -- the detector scored 9/13. Pinning a result rather
    than a property turns every new recording into a failing test, which trains you to edit
    the test instead of reading it. The bake-off table is where performance is compared;
    this only catches a detector that has stopped working altogether.
    """
    from respiradar.bakeoff import score
    from respiradar.detectors.gated import build_best

    result = score(build_best())

    assert result.detected > result.total_holds / 2
    assert result.false_alarms <= 3


def test_the_best_detector_also_stays_quiet_in_an_empty_room():
    clip = _empty_room_clip()
    detector = build_best_detector()
    detector.fit(folds()[0][0])

    assert not detector.predict(clip)[clip.t >= 25].any()


def build_best_detector():
    from respiradar.detectors.gated import build_best

    return build_best()


def test_the_live_detector_finds_holds_in_a_recording():
    """The live path: frames in one at a time, same answer as the batch evaluation.

    The detectors are written against a whole clip at once, which is right for scoring and
    wrong for a sensor. This runs the streaming adapter over a recording and checks it still
    catches the holds and stays silent on a subject with none.
    """
    import numpy as np

    from respiradar.dataset import session_by_name
    from respiradar.detectors.gated import build_best
    from respiradar.live import LiveDetector
    from respiradar.sources import recorded_config, replay_frames

    session = session_by_name("nishant-holds-2401")
    live = LiveDetector(recorded_config(session.path), build_best())
    times = []
    for frame in replay_frames(session.path):
        live.process(frame)
        times.append(frame.t)

    t = np.asarray(times)
    alarms = np.asarray(live.alarms)
    caught = [h for h in session.holds
              if alarms[(t >= h.start_s) & (t < h.end_s)].any()]

    assert len(caught) == len(session.holds)


def test_the_live_detector_stays_silent_on_a_subject_with_no_holds():
    import numpy as np

    from respiradar.dataset import session_by_name
    from respiradar.detectors.gated import build_best
    from respiradar.live import LiveDetector
    from respiradar.sources import recorded_config, replay_frames

    session = session_by_name("vishnu-sleeping")
    live = LiveDetector(recorded_config(session.path), build_best())
    times = []
    for frame in replay_frames(session.path):
        live.process(frame)
        times.append(frame.t)

    settled = np.asarray(times) >= 25
    assert not np.asarray(live.alarms)[settled].any()


def test_the_shipped_detector_is_silent_at_a_wall():
    """A real empty scene, not a simulated one.

    Pointed at a wall the scope reported a confident 16.1 bpm and "breathing". Two causes:
    the readout took the largest peak in a band of noise, and presence was decided per
    frame. A wall spikes to 15.2 on the slow-motion score while a person holding their
    breath drops to 2.7, so no per-frame threshold separates them - only time does.
    """
    import numpy as np

    from respiradar.bakeoff import Clip, folds
    from respiradar.dataset import FeatureExtractor, session_by_name
    from respiradar.detectors.gated import build_best
    from respiradar.sources import recorded_config, replay_frames

    session = session_by_name("wall")
    extractor = FeatureExtractor(recorded_config(session.path))
    times, rows = [], []
    for frame in replay_frames(session.path):
        times.append(frame.t)
        rows.append(extractor.process(frame))

    t = np.asarray(times)
    clip = Clip("wall", t, np.asarray(rows), np.zeros(len(t), bool), [])
    detector = build_best()
    detector.fit(folds()[0][0])

    assert not detector.predict(clip)[t >= 25].any()
