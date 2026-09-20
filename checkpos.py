#!/usr/bin/env python3
"""Is there actually a person in front of the sensor? Run this BEFORE recording.

Every threshold in this system assumes a body return that stands clear of the noise. The
recordings it was built on have one at 1.02-1.08 m with SNR ~12. A demo session recorded
with nothing beyond 0.72 m above SNR 1 produced phantom breathing rates and no apnea: the
chest-bin tracker had locked onto a bin with SNR 0.4, where the strongest frequency in the
breathing band is noise, and noise never stops. No detector threshold can fix that, so the
check belongs before the run rather than in the post-mortem.

    python checkpos.py
"""
from __future__ import annotations

import sys

import numpy as np

from main import DEFAULT_BAUDRATE, fit_config
from respiradar.sources import RadarConfig, find_serial_port, radar_frames

SECONDS = 6.0
GOOD_SNR = 3.0      # a body return; the working recordings sit at 12
NEAR_FIELD_M = 0.5  # below this is the sensor's own near field, not a person


def main() -> int:
    port = find_serial_port()
    if port is None:
        print("No radar found.", file=sys.stderr)
        return 1
    config = fit_config(RadarConfig(sweeps_per_frame=8), DEFAULT_BAUDRATE)
    amps, noises = [], []
    frames = radar_frames(port, config=config, baudrate=DEFAULT_BAUDRATE)
    print(f"measuring for {SECONDS:.0f} s - sit still, chest towards the sensor ...")
    try:
        for frame in frames:
            amps.append(np.abs(frame.iq.mean(axis=0)))
            noises.append(np.abs(np.diff(frame.iq, axis=0)).mean(axis=0) / np.sqrt(2))
            if frame.t > SECONDS:
                break
    finally:
        frames.close()

    amp = np.asarray(amps).mean(axis=0)
    noise = np.maximum(np.asarray(noises).mean(axis=0), 1e-9)
    snr = amp / noise
    d = config.distances_m

    print()
    for j in range(len(d)):
        bar = "#" * int(min(40, 40 * amp[j] / amp.max()))
        note = "  <-- noise only" if snr[j] < 1 else ""
        print(f"  {d[j]:4.2f} m  SNR {snr[j]:5.1f}  {bar}{note}")

    body = (snr > GOOD_SNR) & (d > NEAR_FIELD_M)
    print()
    if not body.any():
        print("NO BODY RETURN beyond the near field.")
        print("Everything past 0.5 m is noise, so the chest tracker will lock onto noise -")
        print("which reports a breathing rate that never stops. Move to about 1.0 m from")
        print("the sensor, chest facing it, nothing in between, and run this again.")
        return 1
    best = int(np.argmax(np.where(body, snr, 0)))
    print(f"Body return at {d[best]:.2f} m, SNR {snr[best]:.1f}. Good - record from here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
