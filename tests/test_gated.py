"""The detector must not alarm about an empty room.

Every recording has a person in it throughout, so this case is untested by the bake-off
and was found only by simulating it. Ungated, the change-point detector alarms on about
half the frames of an empty room - nobody there produces no chest motion, which is exactly
what apnea looks like.

The 180 s simulated empty room is built once per test session by the `empty_room_clip`
fixture in conftest, and `bakeoff_score` memoises `bakeoff.score` per detector name. Both
are caches only - the clip and the scores are exactly what the tests computed before.
"""

from respiradar.bakeoff import folds
from respiradar.detectors.changepoint import build_conservative
from respiradar.detectors.gated import build

# The live-path tests replay a recording through `LiveDetector`, which re-runs the detector
# over a growing buffer twice a second - quadratic in the length of the recording. 90 s of
# nishant-holds-2401 still contains one whole labelled hold (30-60 s; the next runs
# 90-120 s), so the first 90 s tests the same property at a quarter of the cost. See
# LIVE_SECONDS below; raise it to replay more.
LIVE_SECONDS = 90.0


def _holds_within(session, seconds):
    return [h for h in session.holds if h.end_s <= seconds]


def test_gated_detector_stays_quiet_in_an_empty_room(empty_room_clip):
    clip = empty_room_clip
    detector = build()
    detector.fit(folds()[0][0])

    alarms = detector.predict(clip)

    assert not alarms[clip.t >= 25].any()


def test_the_gate_is_what_makes_the_difference(empty_room_clip):
    """Guards against the gate silently becoming a no-op."""
    clip = empty_room_clip
    ungated = build_conservative()
    ungated.fit(folds()[0][0])

    assert ungated.predict(clip)[clip.t >= 25].any()


def test_the_gate_costs_no_detections_and_removes_false_alarms(bakeoff_score):
    """The gate must be free on holds and strictly better on empty scenes.

    It used to be free in both directions, because the only empty room available was
    simulated and nothing alarmed at it anyway. With a real wall recording scored, the
    ungated detector alarms and the gated one does not - so the gate now earns its place
    rather than merely costing nothing.
    """
    gated = bakeoff_score(build())
    plain = bakeoff_score(build_conservative())

    assert gated.detected == plain.detected
    assert gated.false_alarms <= plain.false_alarms


def test_the_shipped_detector_is_not_obviously_broken(bakeoff_score):
    """A floor, not a target.

    This deliberately does NOT pin an exact hold count. An earlier version asserted 4/4
    with zero false alarms, which was true when there were four labelled holds and became
    false the moment nine more arrived -- the detector scored 9/13. Pinning a result rather
    than a property turns every new recording into a failing test, which trains you to edit
    the test instead of reading it. The bake-off table is where performance is compared;
    this only catches a detector that has stopped working altogether.
    """
    from respiradar.detectors.gated import build_best

    result = bakeoff_score(build_best())

    assert result.detected > result.total_holds / 2
    assert result.false_alarms <= 3


def test_the_best_detector_also_stays_quiet_in_an_empty_room(empty_room_clip):
    clip = empty_room_clip
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
        if frame.t > LIVE_SECONDS:
            break
        live.process(frame)
        times.append(frame.t)

    t = np.asarray(times)
    alarms = np.asarray(live.alarms)
    holds = _holds_within(session, LIVE_SECONDS)
    assert holds, "replayed too little of the session to contain a whole hold"
    caught = [h for h in holds
              if alarms[(t >= h.start_s) & (t < h.end_s)].any()]

    assert len(caught) == len(holds)


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
        if frame.t > LIVE_SECONDS:
            break
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
    from respiradar.dataset import load_cached
    from respiradar.detectors.gated import build_best

    # `load_cached` is the same extraction this test used to redo frame by frame - it is
    # what `build_cache` wrote by running FeatureExtractor over `replay_frames`. The wall
    # recording is registered like any other session, so the features are identical.
    t, X, _ = load_cached("wall")
    clip = Clip("wall", t, X, np.zeros(len(t), bool), [])
    detector = build_best()
    detector.fit(folds()[0][0])

    assert not detector.predict(clip)[t >= 25].any()
