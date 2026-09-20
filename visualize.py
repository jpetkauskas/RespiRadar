#!/usr/bin/env python3
"""RespiRadar scope: watch a breathing signal being pulled out of radar noise.

    python visualize.py                          # the clearest session, replayed
    python visualize.py --session nishant-holds-3008
    python visualize.py --list

Eight panels, arranged as the pipeline runs. Each one is the input to the next, so reading
left to right and top to bottom is watching noise turn into a decision.

WHAT YOU ARE LOOKING AT, panel by panel:

1. RANGE PROFILE - how much energy comes back from each distance. This is the rawest view.
   The tall spike at 0.30 m is not a person, it is the sensor's own near field, and it is
   about six times brighter than the chest. An earlier version of this pipeline tracked that
   spike for 50-99% of frames and measured clutter instead of breathing.

2. IQ CONSTELLATION - the complex return from the chest over the last few seconds. Breathing
   shows up as the dot sweeping an arc: the chest moves a few millimetres, the round trip
   changes by a fraction of a wavelength, the phase rotates. During a hold the arc collapses
   to a blob. This is the raw physical measurement everything else is derived from.

3. RANGE-TIME MOTION - breathing-band motion at every distance, over time. A person shows as
   a bright horizontal band at their range. This is the panel that shows the sensor is
   spatially selective: it can ignore a moving curtain at 1.4 m while watching a chest at
   0.7 m.

4. MOTION PER RANGE BIN - how the chest bin is chosen. Not the brightest bin, the one that
   MOVES most, smoothed across neighbours because a chest is wider than one 6 cm bin, and
   averaged over 90 s so that a breath hold cannot make the tracker wander off looking for
   something else that is moving.

5. PHASE -> DISPLACEMENT - the unwrapped phase converted to millimetres. This is the signal
   before cleaning: real breathing is in there, buried under slow drift of far larger
   amplitude than the breathing itself.

6. THE BREATHING WAVE - the same trace band-passed to 0.18-0.55 Hz (11-33 breaths/min). This
   is the "seeing through noise" panel: individual breaths become countable, and a hold is a
   visibly flat stretch. Note the band matters - at the wider 0.10 Hz the drift in panel 5
   dominates and the dominant period reads 6-10 bpm, which is not breathing.

7. SPECTRUM - where the energy sits across the breathing band, with the peak marked. A
   breathing person has a sharp line. A held breath has no line at all.

8. DETECTOR - the alarm, against the labelled ground truth. Green shading is a real breath
   hold; red is where the detector alarmed. The gap between the start of green and the start
   of red is the detection latency, which is the number the whole project optimises.
"""

from __future__ import annotations

import argparse
import sys
import threading
import traceback

import numpy as np
import pyqtgraph as pg
from PySide6 import QtCore, QtGui, QtWidgets
from scipy import signal as sig

from respiradar.dataset import (
    FEATURE_NAMES,
    SESSIONS,
    FeatureExtractor,
    session_by_name,
)
from respiradar.detectors.gated import PRESENCE_THRESHOLD, PRESENCE_WINDOW_S
from respiradar.sources import BASE_STEP_M, recorded_config, replay_frames

WINDOW_S = 30.0          # how much history the time plots show
CONSTELLATION_S = 6.0
BAND = (0.18, 0.55)      # the band where breaths actually live

BG = "#0d1117"
FG = "#e6edf3"
ACCENT = "#58a6ff"
GOOD = "#3fb950"
WARN = "#d29922"
BAD = "#f85149"
DIM = "#30363d"


def extract(session, detector_name: str = "spectral"):
    """Run the pipeline once, keeping every intermediate the panels need."""
    config = recorded_config(session.path)
    ex = FeatureExtractor(config)

    times, profiles, chest_iq, bins, raw_mm, feats, bin_motion = [], [], [], [], [], [], []
    for frame in replay_frames(session.path):
        row = ex.process(frame)
        mean_sweep = frame.iq.mean(axis=0)
        times.append(frame.t)
        profiles.append(np.abs(mean_sweep))
        bins.append(ex._last_bin)
        chest_iq.append(mean_sweep[ex._last_bin])
        raw_mm.append(ex.raw[-1])
        feats.append(row)
        bin_motion.append(
            ex.bin_slow.copy() if ex.bin_slow is not None else np.zeros(len(mean_sweep))
        )

    t = np.asarray(times)
    out = dict(
        t=t,
        distances=config.distances_m,
        profile=np.asarray(profiles),
        chest_iq=np.asarray(chest_iq),
        bin_index=np.asarray(bins),
        raw_mm=np.asarray(raw_mm),
        features=np.asarray(feats),
        bin_motion=np.asarray(bin_motion),
        fs=config.frame_rate,
        holds=session.holds,
    )

    # The breathing wave: causal band-pass, the same one a live detector would use.
    sos = sig.butter(
        2, [BAND[0] / (out["fs"] / 2), BAND[1] / (out["fs"] / 2)], btype="bandpass", output="sos"
    )
    out["wave"] = sig.sosfilt(sos, out["raw_mm"] - out["raw_mm"][0])

    out["alarms"] = _run_detector(session, out, detector_name)
    return out


def _run_detector(session, data, name):
    """Best-effort: show the detector's alarms, but never let it break the display."""
    import importlib

    from respiradar.bakeoff import Clip

    y = np.zeros(len(data["t"]), dtype=bool)
    for hold in session.holds:
        y |= (data["t"] >= hold.start_s) & (data["t"] < hold.end_s)
    clip = Clip(session.name, data["t"], data["features"], y, list(session.holds))

    for module in (name, "spectral", "changepoint"):
        try:
            mod = importlib.import_module(f"respiradar.detectors.{module}")
            det = mod.build()
            if hasattr(det, "fit"):
                from respiradar.bakeoff import folds

                train = next(tr for tr, te in folds() if all(c.name != session.name for c in te))
                det.fit(train)
            alarms = np.asarray(det.predict(clip), dtype=bool)
            print(f"detector: {getattr(det, 'name', module)}")
            return alarms
        except Exception as exc:
            print(f"  ({module} unavailable here: {type(exc).__name__}: {exc})")
    return np.zeros(len(data["t"]), dtype=bool)


class LiveSource:
    """A growing buffer of the same arrays `extract` produces, filled by a worker thread.

    The scope does not care which it is given: replay hands it finished arrays, this hands it
    arrays that get longer. Everything downstream indexes the same way.
    """

    def __init__(self, frames, config, detector_name="gated"):
        import importlib

        from respiradar.live import LiveDetector

        mod = importlib.import_module(f"respiradar.detectors.{detector_name}")
        build = getattr(mod, "build_best", mod.build)
        self.live = LiveDetector(config, build())
        self.frames = frames
        self.config = config
        self.fs = config.frame_rate
        self.distances = config.distances_m
        self.holds = []
        self.error = None

        self._t, self._profile, self._iq = [], [], []
        self._bins, self._raw, self._feat, self._motion, self._alarm = [], [], [], [], []

        nyq = self.fs / 2
        self._sos = sig.butter(
            2, [BAND[0] / nyq, BAND[1] / nyq], btype="bandpass", output="sos"
        )
        self._zi = sig.sosfilt_zi(self._sos) * 0.0
        self._wave = []

        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        try:
            for frame in self.frames():
                row = self.live.process(frame)
                mean_sweep = frame.iq.mean(axis=0)
                ex = self.live.extractor
                self._t.append(frame.t)
                self._profile.append(np.abs(mean_sweep))
                self._bins.append(ex._last_bin)
                self._iq.append(mean_sweep[ex._last_bin])
                self._raw.append(ex.raw[-1])
                self._feat.append(row)
                self._motion.append(
                    ex.bin_slow.copy() if ex.bin_slow is not None
                    else np.zeros(len(mean_sweep))
                )
                value, self._zi = sig.sosfilt(
                    self._sos, [self._raw[-1] - self._raw[0]], zi=self._zi
                )
                self._wave.append(float(value[0]))
                self._alarm.append(self.live.alarm)
        except Exception as exc:
            traceback.print_exc()
            self.error = str(exc)

    def __len__(self):
        return len(self._t)

    def __getitem__(self, key):
        arrays = {
            "t": self._t, "profile": self._profile, "chest_iq": self._iq,
            "bin_index": self._bins, "raw_mm": self._raw, "features": self._feat,
            "bin_motion": self._motion, "wave": self._wave, "alarms": self._alarm,
        }
        if key in arrays:
            return np.asarray(arrays[key])
        return {"distances": self.distances, "fs": self.fs, "holds": self.holds}[key]


class Scope(QtWidgets.QMainWindow):
    def __init__(self, data, title: str, speed: float = 1.0, live: bool = False):
        super().__init__()
        self.d = data
        self.i = 0
        self.speed = speed
        self.playing = True
        self.live = live

        self.setWindowTitle(f"RespiRadar scope - {title}")
        self.resize(1680, 980)
        self.setStyleSheet(f"background:{BG}; color:{FG};")

        root = QtWidgets.QWidget()
        outer = QtWidgets.QVBoxLayout(root)
        outer.setContentsMargins(12, 10, 12, 10)
        self.setCentralWidget(root)
        outer.addLayout(self._header())

        pg.setConfigOptions(antialias=True, background=BG, foreground=FG)
        self.win = pg.GraphicsLayoutWidget()
        outer.addWidget(self.win, stretch=1)
        self._build_panels()

        outer.addLayout(self._controls())

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.step)
        self.timer.start(33)

    # ---------------------------------------------------------------- header
    def _header(self):
        bar = QtWidgets.QHBoxLayout()
        self.readouts = {}
        for key, label in [
            ("bpm", "BREATHING RATE"),
            ("chest", "CHEST AT"),
            ("amp", "CHEST MOTION"),
            ("presence", "PRESENCE"),
            ("state", "STATUS"),
        ]:
            box = QtWidgets.QVBoxLayout()
            cap = QtWidgets.QLabel(label)
            cap.setStyleSheet(f"color:#8b949e; font-size:11px; letter-spacing:1px;")
            val = QtWidgets.QLabel("--")
            val.setStyleSheet("font-size:30px; font-weight:600;")
            box.addWidget(cap)
            box.addWidget(val)
            self.readouts[key] = val
            bar.addLayout(box)
            bar.addSpacing(38)
        bar.addStretch(1)
        return bar

    # ---------------------------------------------------------------- panels
    def _panel(self, row, col, title, left, bottom, rowspan=1, colspan=1):
        p = self.win.addPlot(row=row, col=col, rowspan=rowspan, colspan=colspan)
        p.setTitle(title, color=FG, size="10pt")
        p.setLabel("left", left)
        p.setLabel("bottom", bottom)
        p.showGrid(x=True, y=True, alpha=0.12)
        p.getAxis("left").setTextPen(FG)
        p.getAxis("bottom").setTextPen(FG)
        return p

    def _build_panels(self):
        d = self.d

        # 1. range profile ------------------------------------------------
        self.p_profile = self._panel(0, 0, "1 - RANGE PROFILE: raw energy vs distance",
                                     "amplitude", "distance (m)")
        self.c_profile = self.p_profile.plot(pen=pg.mkPen(ACCENT, width=2),
                                             fillLevel=0, brush=pg.mkBrush(88, 166, 255, 45))
        self.chest_marker = pg.InfiniteLine(angle=90, pen=pg.mkPen(GOOD, width=2))
        self.p_profile.addItem(self.chest_marker)
        clutter = pg.LinearRegionItem([d["distances"][0] - 0.03, d["distances"][0] + 0.03],
                                      movable=False, brush=pg.mkBrush(248, 81, 73, 40))
        self.p_profile.addItem(clutter)

        # 2. IQ constellation ---------------------------------------------
        self.p_iq = self._panel(0, 1, "2 - IQ CONSTELLATION: breathing rotates the phase",
                                "Q", "I")
        self.p_iq.setAspectLocked(True)
        self.c_iq = self.p_iq.plot(pen=None, symbol="o", symbolSize=4,
                                   symbolPen=None, symbolBrush=pg.mkBrush(88, 166, 255, 110))
        self.c_iq_now = self.p_iq.plot(pen=None, symbol="o", symbolSize=11,
                                       symbolPen=pg.mkPen(FG, width=1),
                                       symbolBrush=pg.mkBrush(GOOD))

        # 3. range-time waterfall -----------------------------------------
        self.p_water = self._panel(0, 2, "3 - RANGE vs TIME: the body shows as a bright band",
                                   "distance (m)", "time (s)")
        self.img = pg.ImageItem()
        self.img.setColorMap(pg.colormap.get("magma"))
        self.p_water.addItem(self.img)

        # 4. per-bin motion -----------------------------------------------
        self.p_bins = self._panel(1, 0, "4 - MOTION PER RANGE BIN: how the chest is chosen",
                                  "breathing motion (mm)", "distance (m)")
        self.c_bins = pg.BarGraphItem(x=d["distances"], height=np.zeros(len(d["distances"])),
                                      width=0.045, brush=pg.mkBrush(88, 166, 255, 170))
        self.p_bins.addItem(self.c_bins)
        self.bin_pick = pg.InfiniteLine(angle=90, pen=pg.mkPen(GOOD, width=2))
        self.p_bins.addItem(self.bin_pick)

        # 5. raw displacement ---------------------------------------------
        self.p_raw = self._panel(1, 1, "5 - PHASE -> DISPLACEMENT: breathing buried in drift",
                                 "mm", "time (s)")
        self.c_raw = self.p_raw.plot(pen=pg.mkPen("#8b949e", width=1))

        # 6. the breathing wave -------------------------------------------
        self.p_wave = self._panel(1, 2, "6 - THE BREATHING WAVE: band-passed, breaths countable",
                                  "mm", "time (s)")
        self.c_wave = self.p_wave.plot(pen=pg.mkPen(GOOD, width=2))

        # 7. spectrum ------------------------------------------------------
        self.p_spec = self._panel(2, 0, "7 - SPECTRUM: a sharp line means breathing",
                                  "power", "breaths per minute")
        self.c_spec = self.p_spec.plot(pen=pg.mkPen("#bc8cff", width=2),
                                       fillLevel=0, brush=pg.mkBrush(188, 140, 255, 55))
        self.spec_peak = pg.InfiniteLine(angle=90, pen=pg.mkPen(WARN, width=2,
                                                               style=QtCore.Qt.DashLine))
        self.p_spec.addItem(self.spec_peak)

        # 8. detector ------------------------------------------------------
        self.p_det = self._panel(2, 1, "8 - DETECTOR: green = real hold, red = alarm",
                                 "", "time (s)", colspan=2)
        self.p_det.setYRange(0, 1)
        self.p_det.getAxis("left").setTicks([])
        self.c_det = self.p_det.plot(pen=pg.mkPen(BAD, width=2))
        self.truth_regions, self.alarm_regions = [], []
        for hold in d["holds"]:
            r = pg.LinearRegionItem([hold.start_s, hold.end_s], movable=False,
                                    brush=pg.mkBrush(63, 185, 80, 55))
            r.setZValue(-20)
            self.p_det.addItem(r)
        self.now_line = pg.InfiniteLine(angle=90, pen=pg.mkPen(FG, width=1))
        self.p_det.addItem(self.now_line)

        for c in range(3):
            self.win.ci.layout.setColumnStretchFactor(c, 1)

    # -------------------------------------------------------------- controls
    def _controls(self):
        bar = QtWidgets.QHBoxLayout()
        self.play_btn = QtWidgets.QPushButton("Pause")
        self.play_btn.clicked.connect(self.toggle)
        self.play_btn.setStyleSheet(
            f"background:{DIM}; color:{FG}; padding:6px 18px; border-radius:4px;")
        bar.addWidget(self.play_btn)

        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.setRange(0, max(0, len(self.d["t"]) - 1))
        self.slider.sliderMoved.connect(self.seek)
        bar.addWidget(self.slider, stretch=1)

        self.clock = QtWidgets.QLabel("0.0 s")
        self.clock.setStyleSheet("color:#8b949e; min-width:70px;")
        bar.addWidget(self.clock)
        return bar

    def toggle(self):
        self.playing = not self.playing
        self.play_btn.setText("Pause" if self.playing else "Play")

    def seek(self, value):
        self.i = int(value)
        self.draw()

    def step(self):
        if self.live:
            n = len(self.d["t"])
            if n < 2:
                return
            self.i = n - 1  # live always shows now
            self.slider.setRange(0, n - 1)
            self.slider.setValue(self.i)
        elif self.playing:
            self.i = (self.i + max(1, int(self.speed))) % len(self.d["t"])
            self.slider.setValue(self.i)
        self.draw()

    # ------------------------------------------------------------------ draw
    def draw(self):
        d, i = self.d, self.i
        t = d["t"]
        fs = d["fs"]
        lo = max(0, i - int(WINDOW_S * fs))
        bin_i = int(d["bin_index"][i])
        chest_m = d["distances"][bin_i]

        self.c_profile.setData(d["distances"], d["profile"][i])
        self.chest_marker.setPos(chest_m)

        c_lo = max(0, i - int(CONSTELLATION_S * fs))
        iq = d["chest_iq"][c_lo : i + 1]
        self.c_iq.setData(iq.real, iq.imag)
        self.c_iq_now.setData([iq.real[-1]], [iq.imag[-1]])

        motion = d["bin_motion"][lo : i + 1]
        if len(motion) > 2:
            self.img.setImage(motion, autoLevels=True)
            self.img.setRect(QtCore.QRectF(t[lo], d["distances"][0],
                                           t[i] - t[lo] or 1e-3,
                                           d["distances"][-1] - d["distances"][0]))
        self.c_bins.setOpts(height=d["bin_motion"][i])
        self.bin_pick.setPos(chest_m)

        raw = d["raw_mm"][lo : i + 1]
        self.c_raw.setData(t[lo : i + 1], raw - raw.mean() if len(raw) else raw)
        self.c_wave.setData(t[lo : i + 1], d["wave"][lo : i + 1])

        seg = d["wave"][max(0, i - int(20 * fs)) : i + 1]
        bpm = None
        if len(seg) > 64:
            spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=8 * len(seg))) ** 2
            fr = np.fft.rfftfreq(8 * len(seg), 1 / fs) * 60
            band = (fr >= BAND[0] * 60) & (fr <= BAND[1] * 60)
            self.c_spec.setData(fr[band], spec[band])
            in_band = spec[band]
            # Noise always has a largest peak, so reporting the argmax unconditionally
            # invents a breathing rate from an empty room. Require the peak to stand clear
            # of the rest of the band before believing it is a chest.
            if band.any() and in_band.max() > 6 * np.median(in_band):
                bpm = float(fr[band][int(np.argmax(in_band))])
                self.spec_peak.setPos(bpm)
                self.spec_peak.show()
            else:
                self.spec_peak.hide()

        self.c_det.setData(t[lo : i + 1], d["alarms"][lo : i + 1].astype(float))
        self.now_line.setPos(t[i])

        # Is anyone actually there? The same slow statistic the detector's gate uses: a
        # 60 s trailing median of the slow-motion score. Per-frame values cannot tell a
        # still person from a wall, and without this check the panels happily report a
        # breathing rate for an empty room - the largest peak in a band of noise is still
        # a peak.
        inter = d["features"][:, FEATURE_NAMES.index("inter")]
        window = int(PRESENCE_WINDOW_S * fs)
        present = float(np.median(inter[max(0, i - window + 1) : i + 1])) >= PRESENCE_THRESHOLD

        in_hold = any(h.start_s <= t[i] < h.end_s for h in d["holds"])
        alarming = bool(d["alarms"][i]) and present
        amp = float(np.std(d["wave"][max(0, i - int(4 * fs)) : i + 1])) if i > 10 else 0.0

        if not present:
            bpm = None
        self.readouts["bpm"].setText("--" if bpm is None else f"{bpm:.1f}")
        # A SIGNAL quality readout used to live here, scoring the autocorrelation of the
        # chest trace. It was removed because it did not work: measured across every
        # recording its median ran 0.24-0.44, and the WALL recording - nobody in the room -
        # scored 0.28, indistinguishable from a breathing person. It was measuring correlated
        # noise, and its "good" threshold of 0.45 was above the median of every session ever
        # recorded, so it could never be reached. A readout that cannot tell a person from a
        # wall is worse than no readout, because people tune the hardware against it.
        #
        # What does separate them is the presence statistic below: wall 2.4, people 26-30.
        self.readouts["presence"].setText("nobody" if not present else "person")
        self.readouts["presence"].setStyleSheet(
            f"font-size:30px; font-weight:600; color:{'#8b949e' if not present else GOOD};")
        self.readouts["bpm"].setStyleSheet(
            "font-size:30px; font-weight:600;" if bpm is not None
            else "font-size:30px; font-weight:600; color:#8b949e;")
        self.readouts["chest"].setText(f"{chest_m:.2f} m")
        self.readouts["amp"].setText(f"{amp:.2f} mm")
        if not present:
            self.readouts["state"].setText("no presence")
            colour = "#8b949e"
        elif bpm is None and not alarming:
            self.readouts["state"].setText("no breathing signal")
            colour = "#8b949e"
        elif alarming:
            self.readouts["state"].setText("APNEA")
            colour = BAD
        elif in_hold:
            self.readouts["state"].setText("holding")
            colour = WARN
        else:
            self.readouts["state"].setText("breathing")
            colour = GOOD
        self.readouts["state"].setStyleSheet(
            f"font-size:30px; font-weight:600; color:{colour};")
        self.clock.setText(f"{t[i]:.1f} s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--session", default="nishant-holds-2401",
                        help="recorded session to replay")
    parser.add_argument("--detector", default="spectral")
    parser.add_argument("--speed", type=float, default=2.0, help="replay frames per tick")
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--port", help="live: serial port of the XM125")
    parser.add_argument("--baudrate", type=int, default=230400)
    parser.add_argument("--no-flow-control", action="store_true")
    parser.add_argument("--simulate", action="store_true",
                        help="live pipeline, synthetic sensor - no hardware needed")
    parser.add_argument("--bpm", type=float, default=14.0)
    args = parser.parse_args()

    if args.list:
        for s in SESSIONS:
            print(f"  {s.name:<24} {s.subject:<9} {len(s.holds)} holds  {s.description}")
        return 0

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)

    # Go live when asked to, and also when a sensor is simply plugged in and no particular
    # recording was requested - which is what someone running `visualize.py` at a demo means.
    from respiradar.sources import find_serial_port

    detected = None if args.simulate else find_serial_port()
    session_requested = "--session" in sys.argv
    go_live = bool(args.port or args.simulate or (detected and not session_requested))

    if go_live:
        from functools import partial

        from respiradar.sources import (
            RadarConfig, radar_frames, simulated_frames,
        )

        config = RadarConfig(sweeps_per_frame=8)
        port = None if args.simulate else (args.port or detected)
        if port:
            from main import fit_config

            config = fit_config(config, args.baudrate)
            frames = partial(radar_frames, port, config=config, baudrate=args.baudrate,
                             flow_control=not args.no_flow_control)
            title = f"LIVE - {port}"
            print(f"connecting to {port} at {args.baudrate} baud ...")
        else:
            if not args.simulate:
                print("no radar found - running the simulator instead")
            frames = partial(simulated_frames, config, breaths_per_min=args.bpm,
                             realtime=True)
            title = f"LIVE - simulator @ {args.bpm:.0f} bpm"
        # "gated" - the change-point bank behind the presence gate. This is the one verified
        # frame-by-frame through the live path on real recordings.
        #
        # The rhythm veto was briefly the default here and is NOT, deliberately. It blocks an
        # alarm whenever it believes a breathing rhythm is still present, and on a weak
        # signal the residual after breathing stops can be MORE periodic than real breathing
        # (measured: autocorrelation 0.40-0.58 during holds against 0.16-0.37 while
        # breathing). On such a setup the veto suppresses exactly the detections it is meant
        # to protect. It scores well offline on recordings with a strong signal; it is not
        # safe as a live default until that is understood. --detector rhythm still selects it.
        data = LiveSource(frames, config, "gated")
        scope = Scope(data, title, live=True)
    else:
        session = session_by_name(args.session)
        if not session_requested:
            print("no radar detected - replaying a recording instead "
                  "(pass --simulate for the live pipeline without hardware)")
        print(f"extracting {session.name} ...")
        data = extract(session, args.detector)
        scope = Scope(data, f"{session.name} ({session.subject})", args.speed)

    scope.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
