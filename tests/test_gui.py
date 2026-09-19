"""Headless smoke test: the window builds and the plots fill from real recorded data."""

import os
from functools import partial

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6 import QtWidgets  # noqa: E402

from respiradar.gui import Acquisition, Dashboard  # noqa: E402
from respiradar.sources import recorded_config, replay_frames  # noqa: E402


def test_dashboard_fills_its_plots_from_a_recording(sitting_recording):
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    config = recorded_config(sitting_recording)
    acquisition = Acquisition(partial(replay_frames, sitting_recording), config)
    acquisition.run()  # synchronous: replay the whole file, then inspect

    window = Dashboard(acquisition, config, "test")
    window.refresh()

    assert acquisition.error is None
    assert acquisition.total_frames == 773
    assert len(window.slow_curve.getData()[0]) == config.num_points
    assert len(window.displacement_curve.getData()[0]) > 0
    assert window.rate_label.text().endswith("bpm")
    app.processEvents()


def test_web_dashboard_payload_has_every_key_the_page_reads(sitting_recording):
    """respiradar/static/index.html reads these by name; a rename here blanks the page."""
    from respiradar.breathing import BreathingPipeline
    from respiradar.server import to_payload
    from respiradar.sources import recorded_config, replay_frames

    config = recorded_config(sitting_recording)
    pipeline = BreathingPipeline(config)
    result = [pipeline.process(f) for f in replay_frames(sitting_recording)][-1]

    payload = to_payload(result)

    for key in ("distances_m", "range_profile", "target_m", "times",
                "displacement_mm", "rate_bpm", "events"):
        assert key in payload
    assert payload["rate_bpm"] is not None
