"""Smoke test for the scope: it must extract, build and draw without a display."""

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
from PySide6 import QtWidgets  # noqa: E402

import visualize  # noqa: E402
from respiradar.dataset import session_by_name  # noqa: E402


def test_scope_extracts_and_draws_a_session():
    session = session_by_name("nishant-holds-2401")
    data = visualize.extract(session)

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


def test_the_breathing_band_isolates_a_plausible_rate():
    """The wide 0.10 Hz band lets drift dominate; the scope uses 0.18-0.55 Hz instead."""
    session = session_by_name("nishant-holds-2401")
    data = visualize.extract(session)

    breathing = data["wave"][data["t"] >= 30]
    spectrum = np.abs(np.fft.rfft(breathing - breathing.mean())) ** 2
    freqs = np.fft.rfftfreq(len(breathing), 1 / data["fs"]) * 60
    band = (freqs >= 6) & (freqs <= 40)
    peak = freqs[band][np.argmax(spectrum[band])]

    assert 8 <= peak <= 25, f"dominant period {peak:.1f} bpm is not breathing"
