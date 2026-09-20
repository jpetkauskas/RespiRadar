"""Drive the Arduino UNO Q's LED matrix from the radar.

    python -m respiradar.unoq                          # sensor if plugged in, else simulator
    python -m respiradar.unoq --session breath-hold    # replay a recording, in real time
    python -m respiradar.unoq --simulate --hold 20,30  # fake a person who stops breathing
    python -m respiradar.unoq --sink terminal          # draw the matrix as text, no board

On the UNO Q the Linux side (the Dragonwing QRB2210) runs this, and the matrix hangs off the
STM32, so every frame goes over the Router Bridge as 104 bytes. Everywhere else `--sink
terminal` prints the same 8x13 grid as characters, which is how the layout gets developed and
reviewed without the hardware.

The pipeline is the live one, unchanged: `dataset.FeatureExtractor` produces a feature row per
frame, `gated.build_best()` decides whether to alarm, and `ledmatrix.MatrixRenderer` turns the
result into pixels. Nothing here re-decides anything the detector already decided.
"""

from __future__ import annotations

import argparse
import sys
from collections import deque

import numpy as np
from scipy import signal as sig

from respiradar.dataset import FEATURE_NAMES
from respiradar.detectors.gated import PRESENCE_THRESHOLD, PRESENCE_WINDOW_S
from respiradar.evaluation import DEFAULT_WARMUP_S
from respiradar.ledmatrix import MatrixRenderer, MatrixState, to_text
from respiradar.live import AlarmWorker, LiveDetector
from respiradar.sources import Frame, RadarConfig

# The band breaths actually live in. `visualize.py` explains the choice at length: at the
# wider 0.10 Hz the slow drift in the raw displacement dominates and the dominant period
# reads 6-10 bpm, which is not breathing. The constant is repeated rather than imported
# because `visualize.py` pulls in PySide6, and the UNO Q runs headless.
BAND = (0.18, 0.55)

INTER = FEATURE_NAMES.index("inter")
HISTORY = FEATURE_NAMES.index("seconds_of_history")

REFRESH_S = 0.1  # how often the matrix is redrawn; the blink and sweep animate at this rate

# Shorter than `live.BUFFER_S` (300 s) on purpose: the detector is re-run over the whole
# buffer every evaluation, so its cost is linear in this. 120 s leaves 100 s of usable
# output past `changepoint`'s 20 s warm-up and comfortably covers the gate's 60 s presence
# window, at roughly a sixth of the work.
BUFFER_S = 120.0


class PresenceTracker:
    """Is anyone there? The same slow statistic the detector's gate uses.

    A 60 s trailing median of the slow-motion score, not a per-frame test: instantaneously a
    wall and a person holding still are not separable (`detectors/gated.py` has the measured
    numbers). Evaluated every half second and held in between, exactly as the gate does, so
    the lamp and the alarm can never disagree about whether the room is occupied.
    """

    def __init__(self, fs: float) -> None:
        self.window = deque(maxlen=max(1, int(PRESENCE_WINDOW_S * fs)))
        self.every = max(1, int(0.5 * fs))
        self.activity = 0.0
        self._n = 0

    def update(self, row: np.ndarray) -> float:
        self.window.append(float(row[INTER]))
        if self._n % self.every == 0:
            self.activity = float(np.median(self.window))
        self._n += 1
        return self.activity

    @property
    def present(self) -> bool:
        return self.activity >= PRESENCE_THRESHOLD

    @property
    def ratio(self) -> float:
        """Activity as a multiple of the threshold. The status lamp's height comes from this."""
        return self.activity / PRESENCE_THRESHOLD


class MatrixFeed:
    """Frames in, `MatrixState` out. Holds the detector, the presence tracker and the filter."""

    def __init__(self, config: RadarConfig, detector=None, buffer_s: float = BUFFER_S) -> None:
        if detector is None:
            from respiradar.detectors.gated import build_best

            detector = build_best()
        # `evaluate_every_s=None`: the worker owns evaluation, `process` only extracts.
        self.live = LiveDetector(config, detector, buffer_s=buffer_s, evaluate_every_s=None)
        self.worker = AlarmWorker(self.live)
        self.presence = PresenceTracker(config.frame_rate)

        nyq = config.frame_rate / 2
        self.sos = sig.butter(2, [BAND[0] / nyq, BAND[1] / nyq], btype="bandpass", output="sos")
        self.zi = sig.sosfilt_zi(self.sos) * 0.0
        self._first_raw: float | None = None

    def process(self, frame: Frame) -> MatrixState:
        # The lock covers only the buffer mutation, which is the extractor's ~0.6 ms. The
        # worker holds it just long enough to copy the buffer, never while predicting.
        with self.worker.lock:
            row = self.live.process(frame)
        raw = self.live.extractor.raw[-1]
        if self._first_raw is None:
            self._first_raw = raw
        # Subtract the first sample rather than starting the filter from rest: otherwise the
        # displacement arrives as a step and the band-pass rings its way out of it for several
        # seconds, which on this display is a large fake breath at start-up.
        value, self.zi = sig.sosfilt(self.sos, [raw - self._first_raw], zi=self.zi)

        activity = self.presence.update(row)
        return MatrixState(
            t=frame.t,
            wave_mm=float(value[0]),
            present=self.presence.present,
            alarm=self.worker.alarm,
            settling=float(row[HISTORY]) < DEFAULT_WARMUP_S,
            presence=activity / PRESENCE_THRESHOLD,
        )


# -- where the pixels go ------------------------------------------------


class TerminalSink:
    """Redraw the matrix in place, as text. The development and review path."""

    def __init__(self) -> None:
        self._drawn = False

    def draw(self, frame: np.ndarray, state: MatrixState | None) -> None:
        if self._drawn:
            sys.stdout.write(f"\033[{frame.shape[0] + 2}A")
        self._drawn = True
        # `None` means a fixed pattern from the self-test, which is not a reading of
        # anything - labelling it "breathing, presence 1.00" would be inventing data.
        label = "" if state is None else (
            f"{state.display.value:<14} t={state.t:6.1f}s  presence={state.presence:4.2f}"
        )
        sys.stdout.write(f"\033[2K{label}\n\033[2K+{'-' * frame.shape[1]}+\n")
        for line in to_text(frame).splitlines():
            sys.stdout.write(f"\033[2K|{line}|\n")
        sys.stdout.flush()

    def close(self) -> None:
        sys.stdout.write("\n")


class BridgeSink:
    """Send the frame to the STM32 over the Router Bridge, which drives the real matrix.

    `arduino.app_utils` only exists on the UNO Q, so the import is deferred to construction:
    `respiradar.unoq` has to stay importable on the machine the code is written on.

    The `Frame` helper rescales from its declared `brightness_levels` to 0..255 on the way out.
    The sketch calls `setGrayscaleBits(3)`, so it expects 0..7 - which means the array must be
    declared with the default 256 levels and carry 0..7 values, so that the rescale is a no-op.
    Declaring `brightness_levels=8` would helpfully stretch 7 to 255 and the matrix would show
    a solid block.
    """

    def __init__(self) -> None:
        try:
            from arduino.app_utils import Bridge, Frame
        except ImportError as error:
            raise ValueError(
                "arduino.app_utils is only available on the UNO Q. Run this inside an "
                "Arduino App Lab app (see unoq/), or pass --sink terminal."
            ) from error
        self._Bridge, self._Frame = Bridge, Frame

    def draw(self, frame: np.ndarray, state: MatrixState) -> None:
        board = self._Frame(np.asarray(frame, dtype=np.uint8))
        self._Bridge.call("draw", board.to_board_bytes())

    def note(self, text: str) -> None:
        print(text, flush=True)

    def close(self) -> None:
        """Leave the matrix dark rather than frozen on the last frame."""
        try:
            self.draw(np.zeros((8, 13), dtype=np.uint8), None)
        except Exception:
            pass


# -- the loop -----------------------------------------------------------


def run(frames, config: RadarConfig, sink, detector=None, refresh_s: float = REFRESH_S) -> None:
    """Pump frames through the pipeline and onto the matrix until the source runs out."""
    feed = MatrixFeed(config, detector)
    renderer = MatrixRenderer()
    next_draw = 0.0
    try:
        for frame in frames:
            state = feed.process(frame)
            # The renderer sees every frame - it needs them to find each column's peak - but
            # the matrix is only redrawn at `refresh_s`. Pushing 20 frames a second over the
            # Bridge would spend the link on pixels nobody can see change.
            picture = renderer.update(state)
            if state.t >= next_draw:
                next_draw = state.t + refresh_s
                sink.draw(picture, state)
    except KeyboardInterrupt:
        pass
    finally:
        feed.worker.stop()
        sink.close()


def selftest(sink, seconds_each: float = 3.0, loop: bool = False) -> None:
    """Walk the known patterns, so a wiring fault can be told from a rendering one.

    Nothing here touches the radar. If this works and the radar does not, the problem is the
    sensor; if this does not work, nothing else is worth debugging yet.
    """
    import time

    from respiradar.ledmatrix import test_patterns

    patterns = test_patterns()
    try:
        while True:
            for n, (name, expect, frame) in enumerate(patterns, 1):
                print(f"\n[{n}/{len(patterns)}] {name}: you should see {expect}", flush=True)
                deadline = time.monotonic() + seconds_each
                while time.monotonic() < deadline:
                    # No state: these are fixed patterns, not a reading of anything.
                    sink.draw(frame, None)
                    time.sleep(0.1)
            if not loop:
                return
    except KeyboardInterrupt:
        pass
    finally:
        sink.close()


def main(argv: list[str] | None = None) -> int:
    from respiradar.sources import find_serial_port, replay_frames, simulated_frames

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sink", choices=("bridge", "terminal"), default=None,
                        help="where to draw (default: bridge on the UNO Q, terminal elsewhere)")
    parser.add_argument("--port", help="serial port of the XM125")
    parser.add_argument("--baudrate", type=int, default=230400)
    parser.add_argument("--simulate", action="store_true", help="synthetic sensor, no hardware")
    parser.add_argument("--session", help="replay a recorded session by name")
    parser.add_argument("--list", action="store_true", help="list recorded sessions and exit")
    parser.add_argument("--bpm", type=float, default=14.0, help="--simulate breathing rate")
    parser.add_argument("--empty", action="store_true", help="--simulate an empty room")
    parser.add_argument("--hold", action="append", default=[], metavar="START,DURATION",
                        help="--simulate a breath hold, in seconds; repeatable")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="--session playback speed; 0 for as fast as possible")
    parser.add_argument("--selftest", action="store_true",
                        help="walk known patterns to check the matrix; touches no radar")
    parser.add_argument("--loop", action="store_true", help="--selftest: repeat forever")
    parser.add_argument("--dwell", type=float, default=3.0,
                        help="--selftest: seconds to hold each pattern")
    args = parser.parse_args(argv)

    if args.list:
        from respiradar.dataset import SESSIONS

        for session in SESSIONS:
            print(f"  {session.name:24s} {session.subject:9s} {len(session.holds)} holds")
        return 0

    sink_name = args.sink or ("bridge" if _on_uno_q() else "terminal")
    make_sink = BridgeSink if sink_name == "bridge" else TerminalSink

    # Before anything that might open a serial port: the point of the self-test is to prove
    # the display alone, so it must not be able to fail for a radar reason.
    if args.selftest:
        selftest(make_sink(), seconds_each=args.dwell, loop=args.loop)
        return 0

    if args.session:
        from respiradar.dataset import session_by_name
        from respiradar.sources import recorded_config

        session = session_by_name(args.session)
        config = recorded_config(session.path)
        frames = replay_frames(session.path, realtime=args.speed >= 1.0)
        print(f"replaying {session.name} ({len(session.holds)} holds)")
    elif args.simulate or not (args.port or find_serial_port()):
        config = RadarConfig()
        holds = tuple(tuple(float(v) for v in h.split(",")) for h in args.hold)
        frames = simulated_frames(
            config,
            breaths_per_min=None if args.empty else args.bpm,
            holds=holds,
        )
        print("simulated sensor" + (" (empty room)" if args.empty else f" ({args.bpm:.0f} bpm)"))
    else:
        from main import fit_config
        from respiradar.sources import radar_frames

        port = args.port or find_serial_port()
        config = fit_config(RadarConfig(), args.baudrate)
        frames = radar_frames(port, config=config, baudrate=args.baudrate)
        print(f"XM125 on {port}: {config.frame_rate:.0f} Hz, "
              f"{config.sweeps_per_frame} sweeps x {config.num_points} bins")

    run(frames, config, make_sink())
    return 0


def _on_uno_q() -> bool:
    try:
        import arduino.app_utils  # noqa: F401
    except ImportError:
        return False
    return True


if __name__ == "__main__":
    raise SystemExit(main())
