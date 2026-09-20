"""Smoke test for the scope: it must extract, build and draw without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

import visualize  # noqa: E402
from respiradar.dataset import session_by_name  # noqa: E402


@pytest.fixture(scope="module")
def scope_data():
    """`visualize.extract` for one session.

    It keeps per-frame intermediates (range profiles, the chest IQ, per-bin motion) that the
    feature cache does not hold, so it has to replay the raw frames - but both tests want
    the same replay, so it is done once.
    """
    return visualize.extract(session_by_name("nishant-holds-2401"))


def test_scope_extracts_and_draws_a_session(scope_data):
    data = scope_data

    assert len(data["t"]) == len(data["wave"]) == len(data["alarms"])
    assert data["profile"].shape[1] == len(data["distances"])
    assert np.isfinite(data["wave"]).all()

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    scope = visualize.Scope(data, "test")
    for i in (0, len(data["t"]) // 2, len(data["t"]) - 1):
        scope.i = i
        scope.draw()

    assert scope.readouts["chest"].text().endswith("m")
    app.processEvents()


def test_the_breathing_band_isolates_a_plausible_rate(scope_data):
    """The wide 0.10 Hz band lets drift dominate; the scope uses 0.18-0.55 Hz instead."""
    data = scope_data

    breathing = data["wave"][data["t"] >= 30]
    spectrum = np.abs(np.fft.rfft(breathing - breathing.mean())) ** 2
    freqs = np.fft.rfftfreq(len(breathing), 1 / data["fs"]) * 60
    band = (freqs >= 6) & (freqs <= 40)
    peak = freqs[band][np.argmax(spectrum[band])]

    assert 8 <= peak <= 25, f"dominant period {peak:.1f} bpm is not breathing"


def test_panel_ten_replays_the_charts_over_the_detectors_window():
    """The chart panel must compute what the alarm computes, not a shorter replay of it.

    The scope used to replay the CUSUM charts over its 30 s *display* window. Twenty of
    those seconds are the charts' own warmup, leaving 10 s in which a running total can
    accumulate - the `energy` chart needs 15 s at its per-frame cap to reach threshold, so
    that curve could not reach 1.0 whatever the sensor saw - and the charts' 75-120 s
    reference windows collapsed onto the apnea they are supposed to measure against.

    Measured here: 9 s into a labelled hold the panel read 0.57 while the detector's own
    window read 1.45. A diagnostic panel that goes quiet exactly when the detector is most
    certain is worse than no panel.
    """
    import numpy as np

    from respiradar.bakeoff import Clip, folds
    from respiradar.dataset import load_cached
    from respiradar.detectors import changepoint

    session = session_by_name("nishant-holds-2401")
    t, X, _ = load_cached(session.name)
    fs = round(1 / float(np.median(np.diff(t))), 3)
    detector = changepoint.build()
    detector.fit(folds()[0][0])

    def replay(seconds, at_s):
        i = int(np.searchsorted(t, at_s))
        j0 = max(0, i - int(seconds * fs))
        clip = Clip("live", t[j0 : i + 1], X[j0 : i + 1],
                    np.zeros(i + 1 - j0, dtype=bool), [])
        return {k: v[-1] for k, v in detector.progress(clip).items()}

    hold = session.holds[0]
    at = hold.end_s - 1
    panel = replay(visualize.CHART_REPLAY_S, at)
    short = replay(visualize.WINDOW_S, at)

    assert max(panel.values()) >= 1.0, (
        f"no chart reaches its threshold {at:.0f}s in, inside a labelled hold: {panel}"
    )
    assert max(panel.values()) > max(short.values()), (
        "the display window is as good as the detector's - this test has stopped biting"
    )
