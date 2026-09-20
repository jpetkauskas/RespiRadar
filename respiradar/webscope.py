"""The scope, over the network: every panel `visualize.py` draws, pushed to a browser.

    python -m respiradar.webscope                        # sensor if plugged in, else simulator
    python -m respiradar.webscope --session breath-hold  # replay a recording, in real time
    python -m respiradar.webscope --simulate --hold 35,30
    python -m respiradar.webscope --http-port 8000 --host 0.0.0.0

Then open the URL it prints from any device on the same network.

WHY NOT JUST IMPORT visualize.py
--------------------------------
`visualize.py` imports pyqtgraph and PySide6 at module scope, which do not belong on a
headless box and are a long build on aarch64. This module re-derives the same arrays from the
same primitives - `dataset.FeatureExtractor`, `live.LiveDetector`, the same band-pass - so the
two agree by construction rather than by one importing the other.

WHAT GOES OVER THE WIRE
-----------------------
Not the raw feed. At 20 Hz a full history every tick is tens of megabytes a minute, most of it
redrawing pixels that did not change. Instead:

- the time series are decimated to `SERIES_POINTS` over the last `HISTORY_S`, which is more
  resolution than a browser canvas can show anyway;
- the range-time heatmap sends only its newest column each tick, and the client scrolls its
  own canvas - the panel is 21 x 256 values, far too big to resend;
- floats are rounded before serialising, because JSON's 17 significant digits are noise.

Measured on a live session, that lands at ~13 KB a tick, so ~67 KB/s at the default 5 Hz.

THE ACQUISITION THREAD NEVER WAITS ON ANYTHING
----------------------------------------------
Same rule as everywhere else in this project: the loop pulling frames off the sensor does
feature extraction and nothing slower. The detector runs on `live.AlarmWorker`, and the
websocket only ever reads a snapshot. A browser that stalls, a client on bad wifi, or a
detector taking two seconds cannot back up the serial link. `respiradar/live.py` has the
measurements that make this non-negotiable.
"""

from __future__ import annotations

import argparse
import asyncio
import socket
import threading
import traceback
from collections import deque
from pathlib import Path

import numpy as np
from scipy import signal as sig

from respiradar.dataset import FEATURE_NAMES, FeatureExtractor
from respiradar.detectors.gated import PRESENCE_THRESHOLD, PRESENCE_WINDOW_S
from respiradar.evaluation import DEFAULT_WARMUP_S
from respiradar.sources import Frame, RadarConfig

STATIC = Path(__file__).parent / "static"

BAND = (0.18, 0.55)  # where breaths live; see visualize.py's panel 6
HISTORY_S = 30.0  # how much of the past the time panels show
IQ_S = 6.0  # how much of the past the constellation shows
SERIES_POINTS = 256  # after decimation, per time series
IQ_POINTS = 90
PUSH_INTERVAL_S = 0.2  # 5 Hz; the wave moves far slower than this
BUFFER_S = 120.0  # see unoq.BUFFER_S: evaluation cost is linear in it

SPECTRUM_S = 30.0
SPECTRUM_BAND = (0.1, 0.7)
PEAK_OVER_MEDIAN = 4.0  # a peak below this is noise, not a breathing rate

RATIO = FEATURE_NAMES.index("ratio_8s_q")
INTER = FEATURE_NAMES.index("inter")
HISTORY_COL = FEATURE_NAMES.index("seconds_of_history")


def _round(values, digits: int = 4):
    """Trim float noise before it becomes bandwidth, and never emit NaN or infinity.

    `json.dumps` writes those as bare `NaN` / `Infinity`, which is not JSON: the browser's
    `JSON.parse` throws and the panel goes blank with nothing in the log to explain it. An
    empty window or a zero-power spectrum is a normal thing to have during start-up, so this
    has to be handled here rather than hoped against.
    """
    array = np.round(np.asarray(values, dtype=float), digits)
    return [float(v) if np.isfinite(v) else None for v in np.atleast_1d(array)]


def _sample_indices(envelope: np.ndarray, points: int) -> np.ndarray:
    """Pick `points` source indices, one per bucket, at each bucket's largest excursion.

    ONE index set for every series, because the client draws them on top of each other: the
    alarm shading sits behind the wave, and the hold shading behind the alarm. Decimating each
    series independently - a plain stride here, an envelope there - leaves them different
    lengths and silently misaligned, so an alarm appears to start somewhere it did not.

    Choosing by largest excursion rather than striding also stops the wave aliasing away: a
    0.2 Hz breath sampled every 12th frame can vanish entirely, and these panels exist to
    show exactly that wave.
    """
    n = len(envelope)
    if n <= points:
        return np.arange(n)
    edges = np.linspace(0, n, points + 1).astype(int)
    out = np.empty(points, dtype=int)
    for i in range(points):
        lo, hi = edges[i], max(edges[i] + 1, edges[i + 1])
        out[i] = lo + int(np.argmax(np.abs(envelope[lo:hi])))
    return out


class ScopeFeed:
    """Runs the pipeline on a background thread and keeps everything the panels need."""

    def __init__(self, frames, config: RadarConfig, detector=None, holds=()) -> None:
        from respiradar.detectors.gated import build_best
        from respiradar.live import AlarmWorker, LiveDetector

        self.config = config
        self.fs = config.frame_rate
        self.distances = config.distances_m
        self.holds = [[float(h.start_s), float(h.end_s)] for h in holds]
        self.frames = frames
        self.status = "starting"
        self.source = "?"

        self.live = LiveDetector(
            config, detector or build_best(), buffer_s=BUFFER_S, evaluate_every_s=None
        )
        self.worker = AlarmWorker(self.live)

        nyq = self.fs / 2
        self.sos = sig.butter(2, [BAND[0] / nyq, BAND[1] / nyq], btype="bandpass", output="sos")
        self.zi = sig.sosfilt_zi(self.sos) * 0.0
        self._first_raw: float | None = None

        keep = int(HISTORY_S * self.fs)
        self.t: deque[float] = deque(maxlen=keep)
        self.wave: deque[float] = deque(maxlen=keep)
        self.raw: deque[float] = deque(maxlen=keep)
        self.ratio: deque[float] = deque(maxlen=keep)
        self.alarms: deque[bool] = deque(maxlen=keep)
        self.iq: deque[tuple[float, float]] = deque(maxlen=int(IQ_S * self.fs))
        self.presence_window: deque[float] = deque(maxlen=int(PRESENCE_WINDOW_S * self.fs))

        self.profile = np.zeros(len(self.distances))
        self.bin_motion = np.zeros(len(self.distances))
        self.chest_bin = 0
        self.activity = 0.0
        self.settling = True
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    # -- acquisition ----------------------------------------------------
    def _run(self) -> None:
        try:
            self.status = "running"
            for frame in self.frames:
                self._ingest(frame)
        except Exception as exc:
            traceback.print_exc()
            self.status = f"error: {exc}"
        else:
            self.status = "source ended"

    def _ingest(self, frame: Frame) -> None:
        with self.worker.lock:
            row = self.live.process(frame)
        extractor: FeatureExtractor = self.live.extractor
        raw = extractor.raw[-1]
        if self._first_raw is None:
            self._first_raw = raw
        value, self.zi = sig.sosfilt(self.sos, [raw - self._first_raw], zi=self.zi)

        mean_sweep = frame.iq.mean(axis=0)
        chest_bin = int(extractor._last_bin)

        with self.lock:
            self.t.append(float(frame.t))
            self.wave.append(float(value[0]))
            self.raw.append(float(raw))
            self.ratio.append(float(row[RATIO]))
            self.alarms.append(bool(self.worker.alarm))
            self.iq.append((float(mean_sweep[chest_bin].real), float(mean_sweep[chest_bin].imag)))
            self.presence_window.append(float(row[INTER]))
            self.profile = np.abs(mean_sweep)
            self.bin_motion = (
                extractor.bin_slow.copy()
                if extractor.bin_slow is not None
                else np.zeros(len(self.distances))
            )
            self.chest_bin = chest_bin
            self.activity = float(np.median(self.presence_window))
            self.settling = float(row[HISTORY_COL]) < DEFAULT_WARMUP_S

    # -- what the browser gets ------------------------------------------
    def _spectrum(self, wave: np.ndarray) -> tuple[list, list, float | None]:
        n = int(SPECTRUM_S * self.fs)
        x = wave[-n:]
        if len(x) < int(6 * self.fs):
            return [], [], None
        x = sig.detrend(x) * np.hanning(len(x))
        # Zero-padded so the peak can be read off finely; it adds no information, only a
        # smoother interpolation of the same spectrum.
        spectrum = np.abs(np.fft.rfft(x, n=8 * len(x))) ** 2
        freqs = np.fft.rfftfreq(8 * len(x), 1 / self.fs)
        band = (freqs >= SPECTRUM_BAND[0]) & (freqs <= SPECTRUM_BAND[1])
        if not band.any():
            return [], [], None
        f, p = freqs[band], spectrum[band]
        peak = float(f[int(np.argmax(p))])
        # The readout must not invent a breathing rate out of a noise floor: the largest bin
        # in a band of noise is still the largest bin.
        if p.max() <= PEAK_OVER_MEDIAN * np.median(p):
            peak = None
        scale = p.max() or 1.0
        return _round(f, 4), _round(p / scale, 4), peak

    def snapshot(self) -> dict:
        with self.lock:
            t = np.asarray(self.t)
            if len(t) < 2:
                return {"status": self.status, "source": self.source}
            wave = np.asarray(self.wave)
            raw = np.asarray(self.raw)
            ratio = np.asarray(self.ratio)
            alarms = np.asarray(self.alarms, dtype=float)
            iq = list(self.iq)
            profile = self.profile.copy()
            motion = self.bin_motion.copy()
            chest_bin, activity, settling = self.chest_bin, self.activity, self.settling

        present = activity >= PRESENCE_THRESHOLD
        freqs, psd, peak_hz = self._spectrum(wave)
        bpm = None if (peak_hz is None or not present) else peak_hz * 60
        amp = float(np.std(wave[-int(4 * self.fs):])) if len(wave) > int(4 * self.fs) else 0.0

        if not present:
            state = "no presence"
        elif bool(alarms[-1]):
            state = "APNEA"
        elif settling:
            state = "settling"
        elif bpm is None:
            state = "no breathing signal"
        else:
            state = "breathing"

        # The wave is the envelope worth preserving, so it chooses the sample points and
        # every other series follows it. See `_sample_indices`.
        keep = _sample_indices(wave, SERIES_POINTS)
        return {
            "status": self.status,
            "source": self.source,
            "t": round(float(t[-1]), 2),
            "fs": self.fs,
            "distances_m": _round(self.distances, 3),
            "profile": _round(profile, 1),
            "bin_motion": _round(motion, 4),
            "chest_bin": chest_bin,
            "chest_m": round(float(self.distances[chest_bin]), 2),
            "iq": [[round(a, 1), round(b, 1)] for a, b in iq[::max(1, len(iq) // IQ_POINTS)]],
            "series": {
                "t": _round(t[keep], 2),
                "wave": _round(wave[keep], 4),
                "raw": _round(raw[keep], 3),
                "ratio": _round(ratio[keep], 3),
                "alarm": [int(v) for v in alarms[keep]],
            },
            "spectrum": {"f": freqs, "p": psd, "peak_hz": peak_hz},
            "bpm": None if bpm is None else round(bpm, 1),
            "present": bool(present),
            "presence": round(activity / PRESENCE_THRESHOLD, 2),
            "amp_mm": round(amp, 3),
            "state": state,
            "holds": self.holds,
            "window_s": HISTORY_S,
        }


# -- the app ------------------------------------------------------------


def create_app(feed: ScopeFeed):
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import FileResponse

    app = FastAPI(title="RespiRadar scope")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC / "scope.html")

    @app.get("/snapshot")
    def snapshot() -> dict:
        """One frame of everything, for curl and for debugging without a browser."""
        return feed.snapshot()

    @app.websocket("/ws")
    async def ws(websocket: WebSocket) -> None:
        await websocket.accept()
        try:
            while True:
                # `snapshot` touches numpy under a lock. Handing it to a thread keeps the
                # event loop free, so one slow client cannot stall the others.
                payload = await asyncio.to_thread(feed.snapshot)
                await websocket.send_json(payload)
                await asyncio.sleep(PUSH_INTERVAL_S)
        except WebSocketDisconnect:
            pass
        except Exception:
            # A dropped client is routine - a phone locking its screen does it. It must never
            # reach the acquisition thread.
            pass

    return app


def _lan_address() -> str:
    """This machine's address on the LAN, for the URL to print. No traffic is sent."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sock.connect(("10.255.255.255", 1))
        return sock.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        sock.close()


def build_source(args) -> tuple[object, RadarConfig, str, tuple]:
    from respiradar.sources import find_serial_port, replay_frames, simulated_frames

    if args.session:
        from respiradar.dataset import session_by_name
        from respiradar.sources import recorded_config

        session = session_by_name(args.session)
        config = recorded_config(session.path)
        return (replay_frames(session.path, realtime=True), config,
                f"replay: {session.name}", tuple(session.holds))

    port = None if args.simulate else (args.port or find_serial_port())
    if port:
        from main import fit_config
        from respiradar.sources import radar_frames

        config = fit_config(RadarConfig(), args.baudrate)
        return (radar_frames(port, config=config, baudrate=args.baudrate), config,
                f"XM125 on {port}", ())

    config = RadarConfig()
    holds = tuple(tuple(float(v) for v in h.split(",")) for h in args.hold)
    label = "simulator: empty room" if args.empty else f"simulator: {args.bpm:.0f} bpm"
    return (simulated_frames(config, breaths_per_min=None if args.empty else args.bpm,
                             holds=holds), config, label, ())


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", help="serial port of the XM125")
    parser.add_argument("--baudrate", type=int, default=230400)
    parser.add_argument("--simulate", action="store_true", help="synthetic sensor, no hardware")
    parser.add_argument("--empty", action="store_true", help="--simulate an empty room")
    parser.add_argument("--bpm", type=float, default=14.0, help="--simulate breathing rate")
    parser.add_argument("--hold", action="append", default=[], metavar="START,DURATION",
                        help="--simulate a breath hold, in seconds; repeatable")
    parser.add_argument("--session", help="replay a recorded session by name")
    parser.add_argument("--host", default="0.0.0.0",
                        help="0.0.0.0 to serve the network, 127.0.0.1 for this machine only")
    parser.add_argument("--http-port", type=int, default=8000)
    args = parser.parse_args(argv)

    frames, config, label, holds = build_source(args)
    feed = ScopeFeed(frames, config, holds=holds)
    feed.source = label

    where = _lan_address() if args.host == "0.0.0.0" else args.host
    print(f"source: {label}")
    print(f"scope:  http://{where}:{args.http_port}")
    if args.host == "0.0.0.0":
        print("        (reachable from any device on this network)")
    uvicorn.run(create_app(feed), host=args.host, port=args.http_port, log_level="warning")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
