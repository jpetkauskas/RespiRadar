"""RespiRadar data logger: record raw radar data, labelled, for training a model.

Saves every frame exactly as the sensor sends it: complex I/Q for each range bin (0.3-1.5 m,
6 cm apart) and each sweep. Breathing lives in the phase of those I/Q values, which moves by
~2.4 rad per mm of chest motion, so nothing else needs logging - distance is just the bin
index. Files are Acconeer .h5 recordings, so `python main.py --replay FILE` plays one back.

    python datalog.py sleeping --subject justinas               # 3 minutes, the default
    python datalog.py talking --subject justinas --seconds 300
    python datalog.py breath-hold --subject justinas            # press Enter at each hold start/end
    python datalog.py sleeping --subject justinas --seconds 0   # until Ctrl+C

Suggested labels: sleeping, resting, talking, active, breath-hold, empty.
Press Enter while recording to drop a timestamped marker (e.g. at the start and end of each
breath-hold). Markers, label and subject are stored in the file and in data/sessions.csv.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import threading
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from main import DEFAULT_BAUDRATE, fit_config
from respiradar.breathing import BreathingPipeline
from respiradar.sources import RadarConfig, find_serial_port, radar_frames


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "unnamed"


class Markers(threading.Thread):
    """Timestamps each press of Enter, against the radar's clock."""

    def __init__(self) -> None:
        super().__init__(daemon=True)
        self.now = 0.0  # time of the latest frame, updated by the recording loop
        self.times: list[float] = []

    def run(self) -> None:
        try:
            for _ in sys.stdin:
                self.times.append(self.now)
                print(f"\n  marker {len(self.times)} at {self.now:.1f} s")
        except (OSError, ValueError):  # no usable stdin, e.g. under a test runner
            pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("label", help="what is happening, e.g. sleeping, talking, breath-hold")
    parser.add_argument("--subject", required=True, help="who is being recorded")
    parser.add_argument("--seconds", type=float, default=180, help="0 records until Ctrl+C")
    parser.add_argument("--notes", default="", help="free text stored with the recording")
    parser.add_argument("--countdown", type=int, default=5, help="seconds to get into position")
    parser.add_argument("--out-dir", type=Path, default=Path("data"))
    parser.add_argument("--port", help="serial port; found automatically if omitted")
    parser.add_argument("--baudrate", type=int, default=DEFAULT_BAUDRATE)
    parser.add_argument("--no-flow-control", action="store_true")
    parser.add_argument("--frame-rate", type=float)
    parser.add_argument("--sweeps", type=int)
    parser.add_argument(
        "--mock", action="store_true", help="Acconeer's mock sensor, to test the logger itself"
    )
    args = parser.parse_args()

    port = "mock" if args.mock else (args.port or find_serial_port())
    if port is None:
        print("No radar found. Plug in the XM125, or pass --port.", file=sys.stderr)
        return 1

    overrides = {}
    if args.frame_rate:
        overrides["frame_rate"] = args.frame_rate
    if args.sweeps:
        overrides["sweeps_per_frame"] = args.sweeps
    config = fit_config(RadarConfig(**overrides), args.baudrate)

    label, subject = slug(args.label), slug(args.subject)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    path = args.out_dir / f"{subject}_{label}_{stamp}.h5"

    print(
        f"{subject} / {label}: {config.frame_rate:.0f} Hz, {config.sweeps_per_frame} sweeps x "
        f"{config.num_points} bins ({config.start_m:.2f}-{config.distances_m[-1]:.2f} m)"
    )
    print(f"Saving to {path}")
    print("Press Enter to mark an event, Ctrl+C to stop early.\n")

    frames = radar_frames(
        port,
        config=config,
        baudrate=args.baudrate,
        flow_control=not args.no_flow_control,
        record_to=path,
    )
    # A live breathing readout, so a bad position shows up now rather than at training time.
    pipeline = BreathingPipeline(config)
    markers = Markers()
    count = delayed = 0
    t = 0.0
    last_print = 0.0  # first status line after 1 s, once the frame rate means something
    stopped_early = False

    first_t = None  # first frame of the file
    label_t0 = None  # first frame after the countdown: where the labelled data begins
    start_frame = 0

    try:
        for index, frame in enumerate(frames):
            if first_t is None:
                first_t = frame.t
            # Keep reading during the countdown: pausing would back frames up on the link.
            result = pipeline.process(frame)
            if label_t0 is None:
                remaining = args.countdown - (frame.t - first_t)
                if remaining > 0:
                    print(f"\rStarting in {remaining:.0f}... ", end="", flush=True)
                    continue
                label_t0, start_frame = frame.t, index
                print("\rRecording.          ")
                markers.start()

            t = frame.t - label_t0
            markers.now = t
            count += 1
            delayed += frame.delayed
            if t - last_print >= 1.0:
                last_print = t
                rate = "--" if result.rate_bpm is None else f"{result.rate_bpm:.1f} bpm"
                left = "" if not args.seconds else f" / {args.seconds:.0f} s"
                print(
                    f"\r  {t:6.0f} s{left}   {count / t:4.1f} fps   "
                    f"{delayed} delayed   {result.app_state.value}: {rate}        ",
                    end="",
                    flush=True,
                )
            if args.seconds and t >= args.seconds:
                break
    except KeyboardInterrupt:
        stopped_early = True
    finally:
        frames.close()  # stops the sensor and closes the recording
    print()

    if count == 0:
        path.unlink(missing_ok=True)
        print("No frames recorded.")
        return 1

    # Frames from the countdown stay in the file (they are real data, just unlabelled);
    # `start_frame` is where the labelled part begins, and marker times count from there.
    with h5py.File(path, "a") as f:
        f.attrs["label"] = label
        f.attrs["subject"] = subject
        f.attrs["notes"] = args.notes
        f.attrs["source"] = "mock" if args.mock else "xm125"
        f.attrs["start_frame"] = start_frame
        f.attrs["duration_s"] = t
        f.attrs["markers_s"] = np.asarray(markers.times, dtype=float)

    manifest = args.out_dir / "sessions.csv"
    is_new = not manifest.exists()
    with manifest.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if is_new:
            writer.writerow(
                ["file", "subject", "label", "start_frame", "duration_s", "frames", "delayed",
                 "frame_rate", "sweeps", "markers_s", "notes"]
            )
        writer.writerow(
            [path.name, subject, label, start_frame, f"{t:.1f}", count, delayed, config.frame_rate,
             config.sweeps_per_frame, " ".join(f"{m:.1f}" for m in markers.times), args.notes]
        )

    ending = " (stopped early)" if stopped_early else ""
    print(f"Saved {count} frames, {t:.0f} s{ending}, {len(markers.times)} markers -> {path}")
    if delayed:
        print(f"Warning: {delayed} frames were delayed. Try --frame-rate 15 or --sweeps 4.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
