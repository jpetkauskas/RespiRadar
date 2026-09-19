"""Standalone live dashboard: four stacked plots, same layout as Acconeer's reference app.

The radar is read on a worker thread; Qt polls the newest result on a timer. Nothing
blocks the UI, and a slow frame shows up as a stale plot rather than a frozen window.
"""

from __future__ import annotations

import threading
import traceback
from typing import Callable, Iterator

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtWidgets

from respiradar.breathing import AppState, BreathingPipeline, BreathingResult
from respiradar.sources import Frame, RadarConfig

REFRESH_MS = 50

STATE_COLOUR = {
    AppState.NO_PRESENCE: "#888888",
    AppState.DETERMINE_DISTANCE: "#d08b00",
    AppState.ESTIMATE_BREATHING_RATE: "#1f9d55",
    AppState.APNEA: "#d62728",
}


class Acquisition(threading.Thread):
    """Runs the source and the pipeline, keeping only the newest result."""

    def __init__(
        self,
        frames: Callable[[], Iterator[Frame]],
        config: RadarConfig,
        apnea_s: float = 10.0,
    ) -> None:
        super().__init__(daemon=True)
        self.frames = frames
        self.pipeline = BreathingPipeline(config, apnea_s=apnea_s)
        self.latest: BreathingResult | None = None
        self.error: str | None = None
        self.delayed_frames = 0
        self.total_frames = 0
        self._stop = threading.Event()

    def run(self) -> None:
        try:
            for frame in self.frames():
                if self._stop.is_set():
                    return
                self.total_frames += 1
                if frame.delayed:
                    self.delayed_frames += 1
                self.latest = self.pipeline.process(frame)
        except Exception as exc:  # surfaced in the status bar rather than killing the UI
            traceback.print_exc()
            self.error = str(exc)

    def stop(self) -> None:
        self._stop.set()


class Dashboard(QtWidgets.QMainWindow):
    def __init__(self, acquisition: Acquisition, config: RadarConfig, source_name: str) -> None:
        super().__init__()
        self.acquisition = acquisition
        self.config = config
        self.setWindowTitle(f"RespiRadar - {source_name}")
        self.resize(900, 1000)

        central = QtWidgets.QWidget()
        self.central = central
        layout = QtWidgets.QVBoxLayout(central)
        self.setCentralWidget(central)

        self.rate_label = QtWidgets.QLabel("--")
        self.rate_label.setAlignment(QtCore.Qt.AlignCenter)
        self.rate_label.setStyleSheet("font-size: 54px; font-weight: 600;")
        layout.addWidget(self.rate_label)

        self.state_label = QtWidgets.QLabel("Starting")
        self.state_label.setAlignment(QtCore.Qt.AlignCenter)
        self.state_label.setStyleSheet("font-size: 16px; color: #888;")
        layout.addWidget(self.state_label)

        win = pg.GraphicsLayoutWidget()
        layout.addWidget(win, stretch=1)

        # 1. Presence score against distance, with the analysed region shaded.
        self.presence_plot = win.addPlot(row=0, col=0)
        self.presence_plot.setTitle("Presence")
        self.presence_plot.setLabel("left", "Score")
        self.presence_plot.setLabel("bottom", "Distance (m)")
        self.presence_plot.addLegend()
        self.presence_plot.showGrid(x=True, y=True, alpha=0.2)
        self.slow_curve = self.presence_plot.plot(pen=pg.mkPen("#1f77b4", width=2), name="Slow motion")
        self.fast_curve = self.presence_plot.plot(pen=pg.mkPen("#d62728", width=2), name="Fast motion")
        self.analysed_region = pg.LinearRegionItem(movable=False, brush=pg.mkBrush(31, 157, 85, 45))
        self.analysed_region.setZValue(-10)
        self.presence_plot.addItem(self.analysed_region)
        self.analysed_region.hide()

        # 2. The chest displacement trace.
        self.displacement_plot = win.addPlot(row=1, col=0)
        self.displacement_plot.setTitle("Chest displacement")
        self.displacement_plot.setLabel("left", "Displacement (mm)")
        self.displacement_plot.setLabel("bottom", "Time (s)")
        self.displacement_plot.showGrid(x=True, y=True, alpha=0.2)
        self.displacement_curve = self.displacement_plot.plot(pen=pg.mkPen("#1f9d55", width=2))

        # 3. Where the energy sits across the breathing band.
        self.psd_plot = win.addPlot(row=2, col=0)
        self.psd_plot.setTitle("Spectrum")
        self.psd_plot.setLabel("left", "PSD")
        self.psd_plot.setLabel("bottom", "Breathing rate (bpm)")
        self.psd_plot.showGrid(x=True, y=True, alpha=0.2)
        self.psd_curve = self.psd_plot.plot(pen=pg.mkPen("#9467bd", width=2))
        self.psd_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen("#d62728", width=2, style=QtCore.Qt.DashLine))
        self.psd_plot.addItem(self.psd_marker)
        self.psd_marker.hide()

        # 4. Rate over time.
        self.rate_plot = win.addPlot(row=3, col=0)
        self.rate_plot.setTitle("Breathing rate")
        self.rate_plot.setLabel("left", "Breaths per minute")
        self.rate_plot.setLabel("bottom", "Time (s)")
        self.rate_plot.showGrid(x=True, y=True, alpha=0.2)
        self.rate_curve = self.rate_plot.plot(pen=pg.mkPen("#1f9d55", width=2))

        # 5. The anomaly score: breathing strength relative to this person's own baseline.
        self.ratio_plot = win.addPlot(row=4, col=0)
        self.ratio_plot.setTitle("Breathing vs. personal baseline")
        self.ratio_plot.setLabel("left", "Ratio")
        self.ratio_plot.setLabel("bottom", "Time (s)")
        self.ratio_plot.showGrid(x=True, y=True, alpha=0.2)
        self.ratio_plot.setYRange(0, 2)
        self.ratio_curve = self.ratio_plot.plot(pen=pg.mkPen("#1f77b4", width=2))
        self.ratio_plot.addItem(
            pg.InfiniteLine(
                pos=acquisition.pipeline.apnea_ratio,
                angle=0,
                pen=pg.mkPen("#d62728", width=1, style=QtCore.Qt.DashLine),
            )
        )

        self.status = self.statusBar()

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.refresh)
        self.timer.start(REFRESH_MS)

    def refresh(self) -> None:
        if self.acquisition.error:
            self.state_label.setText("Error")
            self.status.showMessage(self.acquisition.error)
            return

        result = self.acquisition.latest
        if result is None:
            return

        presence = result.presence
        self.slow_curve.setData(presence.distances_m, presence.inter)
        self.fast_curve.setData(presence.distances_m, presence.intra)

        if result.distances_being_analyzed is not None:
            low, high = result.distances_being_analyzed
            self.analysed_region.setRegion(
                (presence.distances_m[low], presence.distances_m[high])
            )
            self.analysed_region.show()
        else:
            self.analysed_region.hide()

        if len(result.displacement_mm):
            self.displacement_curve.setData(result.times, result.displacement_mm)
        if len(result.psd):
            self.psd_curve.setData(result.psd_freqs_hz * 60, result.psd)
        if len(result.rate_history):
            self.rate_curve.setData(result.rate_times, result.rate_history)
        self.ratio_curve.setData(result.ratio_times, result.ratio_history)

        state_text = result.app_state.value
        if result.breathing_ratio is not None:
            state_text += f"  -  breathing at {100 * result.breathing_ratio:.0f}% of baseline"
        if result.quiet_s > 0:
            state_text += f"  -  no breathing for {result.quiet_s:.0f} s"
        self.state_label.setText(state_text)
        self.state_label.setStyleSheet(
            f"font-size: 16px; color: {STATE_COLOUR[result.app_state]};"
        )

        apnea = result.app_state == AppState.APNEA
        self.central.setStyleSheet("background: #5c1010;" if apnea else "")
        if apnea:
            self.rate_label.setText(f"APNEA  {result.quiet_s:.0f} s")
            self.rate_label.setStyleSheet("font-size: 54px; font-weight: 600; color: #ff5252;")
            self.psd_marker.hide()
        elif result.rate_bpm is None:
            self.rate_label.setStyleSheet("font-size: 54px; font-weight: 600;")
            self.rate_label.setText("--")
            self.psd_marker.hide()
        else:
            self.rate_label.setStyleSheet("font-size: 54px; font-weight: 600;")
            self.rate_label.setText(f"{result.rate_bpm:.1f} bpm")
            self.psd_marker.setPos(result.rate_bpm)
            self.psd_marker.show()

        total = self.acquisition.total_frames
        message = f"{total} frames"
        if self.acquisition.delayed_frames:
            share = 100 * self.acquisition.delayed_frames / max(total, 1)
            message += f"  -  {share:.1f}% delayed: lower the frame rate or raise the baud rate"
        self.status.showMessage(message)

    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        self.acquisition.stop()
        super().closeEvent(event)


def run(
    frames: Callable[[], Iterator[Frame]],
    config: RadarConfig,
    source_name: str,
    apnea_s: float = 10.0,
) -> int:
    pg.setConfigOptions(antialias=True)
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    acquisition = Acquisition(frames, config, apnea_s=apnea_s)
    acquisition.start()
    window = Dashboard(acquisition, config, source_name)
    window.show()
    return app.exec()
