"""Apnea from the range-spectral structure of the raw IQ, not from one scalar energy.

The shared feature set tracks three range points around the presence peak and collapses
them to a band-limited RMS. On these recordings that goes wrong twice over: the presence
peak sits on the 0.30 m bin (near-field clutter) for most of every session, and the bins
around 0.7-0.9 m carry an amplitude SNR of 2-5, so their unwrapped phase is a random walk
whose 0.1-0.7 Hz content is noise and is indistinguishable from a chest. A hold can look
perfectly alive in `rms_4s` while the subject is not moving at all.

So this detector builds its own representation, per range bin, causally:

1. Unwrap the phase of all 21 range points, not three, and keep the last 20 s of
   displacement for each.
2. Qualify a bin only if its own amplitude SNR - reflected amplitude over the
   sweep-to-sweep noise floor, both running EMAs - exceeds MIN_SNR, so that its phase
   means something. This is the single change that made anything else work.
3. On every frame, take a Hann-windowed, linearly detrended DFT of each qualified bin's
   last N seconds: a running STFT column per range bin, i.e. a range-frequency image.
   `A` is the 0.1-0.7 Hz band RMS, in millimetres, of the loudest qualified bin - a
   maximum over range, so the alarm needs every believable bin to be quiet at once, not
   just the one a tracker happens to have picked.
   A windowed DFT is used rather than an IIR band-pass deliberately: a 0.1 Hz Butterworth
   rings for the better part of ten seconds after the chest stops, and that ringing is
   latency the product pays for. A window only ever carries the past N seconds.
4. Track the person's own normal as the 75th percentile of A over the trailing 90 s. A
   percentile rather than an EMA: one large movement cannot latch it the way the shared
   extractor's asymmetric EMA does, and a hold occupying under a quarter of the window
   cannot pull it down to meet itself.

The alarm is a two-branch rule on (A, A_fast / trailing percentile), each with its own
dwell, with a release hysteresis and a gross-motion gate. See SpectralApneaDetector.

What was tried and did not survive, recorded so nobody repeats it: per-bin phase coherence
with the composite, spectral flatness, the peak-to-band fraction, cepstral peak, rolling
peak-frequency stability and the spatial-profile cosine all separate hold from not-hold at
barely better than chance here. At a 10 s window the 0.1-0.7 Hz band is only about seven
DFT bins wide, so there is almost no spectral shape to measure at a latency anyone wants,
and the periodicity of quiet breathing is simply not that different from the periodicity
of a torso settling. Amplitude, per range bin, with the noisy bins excluded, is the signal.
A logistic regression and a random forest over the whole sorted range-spectral vector were
also tried, fitted leave-one-subject-out; the best of them reached 1/4 holds at zero false
alarms, against 3/4 for the rule below. Range structure is largely subject geometry, and a
model learns the geometry.

Everything here is causal: state is updated frame by frame, windows only look backwards,
and nothing is normalised by a statistic of the whole recording. Feeding the extractor a
truncated recording reproduces the full run's rows exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import DATA, SESSIONS, MM_PER_RADIAN, session_by_name
from respiradar.presence import PresenceDetector
from respiradar.sources import recorded_config, replay_frames

LOW_HZ, HIGH_HZ = 0.1, 0.7
WINDOW_S = 10.0          # STFT window; also the detector's intrinsic lag floor
BUFFER_S = 20.0
MIN_SNR = 12.0           # amplitude SNR a range bin needs before its phase is believed

FEATURE_NAMES = [
    "A",          # band RMS of the loudest qualified bin, 10 s window (mm)
    "A_fast",     # same with a 6 s window - lower lag, noisier
    "base",       # asymmetric EMA baseline of A
    "ratio",      # A / base
    "base_p",     # 75th percentile of A over the trailing 90 s - robust "normal"
    "ratio_p",    # A_fast / base_p, the cross-subject scale-free statistic
    "n_act",      # qualified bins within 6 dB of the loudest
    "peak_hz",
    "intra",      # presence fast-motion score
    "inter",
    "n_qual",     # how many bins currently pass the SNR gate
]

BASE_WINDOW_S = 90.0
BASE_MIN_S = 25.0
BASE_PCT = 75.0


F = {n: i for i, n in enumerate(FEATURE_NAMES)}


class _BinSpectra:
    """Running STFT over the per-range-bin displacement, one column per frame."""

    def __init__(self, fs: float, n_bins: int, window_s: float) -> None:
        self.n = int(window_s * fs)
        self.freqs = np.fft.rfftfreq(self.n, 1 / fs)
        self.band = (self.freqs >= LOW_HZ) & (self.freqs <= HIGH_HZ)
        self.win = np.hanning(self.n)
        self.cg = float((self.win**2).sum())
        self.ramp = np.arange(self.n) - (self.n - 1) / 2
        self.rr = float((self.ramp**2).sum())
        self.band_hz = self.freqs[self.band]

    def __call__(self, buf: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """buf: (>=n, n_bins) displacement. Returns (band_rms, peak_rms, peak_index)."""
        x = buf[-self.n :]
        x = x - x.mean(axis=0)
        x = x - np.outer(self.ramp, (self.ramp @ x) / self.rr)
        power = np.abs(np.fft.rfft(x * self.win[:, None], axis=0)) ** 2
        in_band = power[self.band]
        scale = 2.0 / (self.n * self.cg)
        band_rms = np.sqrt(in_band.sum(axis=0) * scale)
        k = np.argmax(in_band, axis=0)
        idx = np.clip(np.stack([k - 1, k, k + 1]), 0, in_band.shape[0] - 1)
        peak_rms = np.sqrt(np.take_along_axis(in_band, idx, 0).sum(axis=0) * scale)
        return band_rms, peak_rms, k


class SpectralExtractor:
    """Frames in, one causal feature row out. Mirrors what would run on the sensor."""

    def __init__(self, config, baseline_time_const_s: float = 45.0) -> None:
        self.fs = fs = config.frame_rate
        self.presence = PresenceDetector(config)
        self.slow = _BinSpectra(fs, 0, WINDOW_S)
        self.fast = _BinSpectra(fs, 0, 6.0)
        self.buffer_len = int(BUFFER_S * fs)

        self.n_bins: int | None = None
        self.prev_angles: np.ndarray | None = None
        self.unwrapped: np.ndarray | None = None
        self.buf: np.ndarray | None = None
        self.filled = 0

        self.amp: np.ndarray | None = None      # EMA of reflected amplitude per bin
        self.noise: np.ndarray | None = None    # EMA of the per-bin noise floor
        self.a_amp = float(np.exp(-1 / (5.0 * fs)))

        self.base: float | None = None
        self.a_up = 4 / (baseline_time_const_s * fs)
        self.a_down = 0.1 / (baseline_time_const_s * fs)
        self.history: list[float] = []      # trailing A, for the percentile baseline
        self.history_len = int(BASE_WINDOW_S * fs)
        self.history_min = int(BASE_MIN_S * fs)

    def process(self, frame) -> np.ndarray:
        presence = self.presence.process(frame)
        sweeps = frame.iq
        mean_sweep = sweeps.mean(axis=0)
        angles = np.angle(mean_sweep)
        amplitude = np.abs(mean_sweep)
        # Sweep-to-sweep difference is noise: the chest cannot move within one frame.
        if sweeps.shape[0] > 1:
            noise = np.abs(np.diff(sweeps, axis=0)).mean(axis=0) / np.sqrt(2)
            noise = noise / np.sqrt(sweeps.shape[0])
        else:
            noise = np.ones_like(amplitude)
        noise = np.maximum(noise, 1e-9)

        if self.n_bins is None:
            self.n_bins = len(mean_sweep)
            self.unwrapped = np.zeros(self.n_bins)
            self.buf = np.zeros((self.buffer_len, self.n_bins))
            self.amp = amplitude.copy()
            self.noise = noise.copy()
        else:
            self.unwrapped = self.unwrapped + (
                (angles - self.prev_angles + np.pi) % (2 * np.pi) - np.pi
            )
            self.amp = self.a_amp * self.amp + (1 - self.a_amp) * amplitude
            self.noise = self.a_amp * self.noise + (1 - self.a_amp) * noise
        self.prev_angles = angles

        self.buf = np.roll(self.buf, -1, axis=0)
        self.buf[-1] = self.unwrapped * MM_PER_RADIAN
        self.filled += 1

        qualified = (self.amp / self.noise) > MIN_SNR
        if not qualified.any():
            qualified = np.zeros(self.n_bins, dtype=bool)
            qualified[int(np.argmax(self.amp / self.noise))] = True

        if self.filled < self.slow.n:
            # No valid spectrum yet. A == 0 marks the row as not-yet-ready; the ratios read
            # a neutral 1.0 so that nothing downstream mistakes a warmup row for a hold.
            row = np.zeros(len(FEATURE_NAMES))
            row[F["base"]] = 1e-9
            row[F["ratio"]] = 1.0
            row[F["base_p"]] = 1e-9
            row[F["ratio_p"]] = 1.0
            row[F["intra"]] = float(presence.intra.max())
            row[F["inter"]] = float(presence.inter.max())
            row[F["n_qual"]] = float(qualified.sum())
            return row

        band_rms, _, peak_k = self.slow(self.buf)
        band_rms = np.where(qualified, band_rms, 0.0)
        loud = int(np.argmax(band_rms))
        a = float(band_rms[loud])
        peak_hz = float(self.slow.band_hz[peak_k[loud]])
        n_act = float(np.count_nonzero(band_rms > 0.5 * a)) if a > 0 else 0.0

        if self.filled >= self.fast.n:
            fast_rms, _, _ = self.fast(self.buf)
            a_fast = float(np.where(qualified, fast_rms, 0.0).max())
        else:
            a_fast = a

        # Learns this person's normal. Rises quickly, falls very slowly, so a long hold
        # cannot quietly drag "normal" down to meet itself.
        if a > 0:
            if self.base is None:
                self.base = a
            else:
                alpha = self.a_up if a > self.base else self.a_down
                self.base = (1 - alpha) * self.base + alpha * a
        base = self.base if self.base else 1e-9

        # A trailing percentile is a sturdier "normal" than an EMA across subjects: one
        # large movement cannot raise it the way an EMA latches, and a hold occupying less
        # than a quarter of the window cannot lower it.
        self.history.append(a)
        if len(self.history) > self.history_len:
            self.history.pop(0)
        if len(self.history) >= self.history_min:
            base_p = float(np.percentile(self.history, BASE_PCT))
        else:
            base_p = 0.0
        ratio_p = min(a_fast / base_p, 10.0) if base_p > 0 else 1.0

        return np.array(
            [
                a,
                a_fast,
                base,
                min(a / base, 10.0),
                base_p,
                ratio_p,
                n_act,
                peak_hz,
                float(presence.intra.max()),
                float(presence.inter.max()),
                float(qualified.sum()),
            ],
            dtype=float,
        )


CACHE = DATA / "spectral_features.npz"


def build_cache(path: Path = CACHE) -> Path:
    arrays = {}
    for session in SESSIONS:
        extractor = SpectralExtractor(recorded_config(session.path))
        times, rows = [], []
        for frame in replay_frames(session.path):
            times.append(frame.t)
            rows.append(extractor.process(frame))
        arrays[f"{session.name}__t"] = np.asarray(times)
        arrays[f"{session.name}__X"] = np.asarray(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _load(path: Path) -> bool:
    """Populate the in-process cache from disk. False if the file is missing a session."""
    if not path.exists():
        return False
    with np.load(path) as data:
        for session in SESSIONS:
            if f"{session.name}__X" not in data:
                return False  # new recordings have landed since this cache was written
            _CACHE[session.name] = (data[f"{session.name}__t"], data[f"{session.name}__X"])
    return True


def _session_features(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in _CACHE and not _load(CACHE):
        _CACHE.clear()
        build_cache(CACHE)
        _load(CACHE)
    return _CACHE[name]


def features_for(clip: Clip) -> np.ndarray:
    """My own features for a clip, which may be any time slice of a session.

    `clip.X` holds the shared features, so the rows are found by session name and time.
    The features themselves were extracted by a single causal pass over the whole session,
    exactly as the shared cache is, so slicing never reveals anything from the future.
    """
    name = clip.name.split("[")[0]
    session_by_name(name)  # raises for an unknown session rather than guessing
    t, X = _session_features(name)
    lo = int(np.searchsorted(t, clip.t[0] - 1e-9))
    rows = X[lo : lo + len(clip.t)]
    if len(rows) != len(clip.t):  # pragma: no cover - times always line up here
        idx = np.searchsorted(t, clip.t - 1e-9)
        rows = X[np.clip(idx, 0, len(t) - 1)]
    return rows


def _sustained(flags: np.ndarray, need: int) -> np.ndarray:
    """True once `flags` has been continuously true for `need` samples."""
    out = np.zeros(len(flags), dtype=bool)
    run = 0
    for i, f in enumerate(flags):
        run = run + 1 if f else 0
        out[i] = run >= need
    return out


class SpectralApneaDetector:
    """Two branches on the range-STFT, each with its own dwell, plus a release hysteresis.

    * `quiet`    - an absolute floor in millimetres on the 10 s band RMS of the loudest
                   qualified bin. Band-limited chest motion under ~0.35 mm is not breathing
                   for anybody, so this branch needs no knowledge of the subject.
    * `relative` - the 6 s band RMS against the subject's own trailing 75th percentile.
                   This is the branch that survives a change of body. It fires on holds
                   where the subject is still settling and moving half a millimetre - far
                   too much for the absolute branch, but three times below their own
                   normal. It pays for that with a long dwell, because a shallow relative
                   dip is also what a sleeping person's ordinary pause looks like. Eleven
                   and a half seconds is close to the clinical definition of an apnea
                   anyway: a cessation of ten seconds or more.

    Once alarmed, the alarm is released only when motion recovers past `release` times both
    thresholds. Without that, the quiet tail of a hold breaks into several alarm episodes
    and every one after the first scores as a false alarm.
    """

    name = "spectral/range-stft"

    def __init__(
        self,
        quiet_mm: float = 0.35,
        quiet_s: float = 5.0,
        rel_ratio: float = 0.30,
        rel_s: float = 11.5,
        release: float = 2.0,
        intra_gate: float = 4.0,
    ) -> None:
        self.quiet_mm = quiet_mm
        self.quiet_s = quiet_s
        self.rel_ratio = rel_ratio
        self.rel_s = rel_s
        self.release = release
        self.intra_gate = intra_gate

    def fit(self, clips) -> None:
        """Nothing is learned from the labels. With four holds from two bodies a fitted
        threshold would memorise them; the numbers here are millimetres of chest motion and
        multiples of the subject's own running normal, both of which the causal extractor
        supplies without ever having seen this person before."""

    def predict(self, clip: Clip) -> np.ndarray:
        X = features_for(clip)
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)

        a = X[:, F["A"]]
        ratio = X[:, F["ratio_p"]]
        # A == 0 marks a warmup row; base_p == 0 means the percentile has no history yet.
        ready = (a > 0) & (X[:, F["base_p"]] > 0)
        # Something is obviously moving: they are awake, not apnoeic.
        ready &= X[:, F["intra"]] < self.intra_gate

        quiet = _sustained(ready & (a < self.quiet_mm), int(self.quiet_s * fs))
        relative = _sustained(ready & (ratio < self.rel_ratio), int(self.rel_s * fs))
        trigger = ready & (quiet | relative)

        recovered = (a > self.release * self.quiet_mm) & (
            ratio > self.release * self.rel_ratio
        )
        alarms = np.zeros(len(clip.t), dtype=bool)
        on = False
        for i in range(len(alarms)):
            if trigger[i]:
                on = True
            elif on and recovered[i]:
                on = False
            alarms[i] = on
        return alarms


def build():
    return SpectralApneaDetector()


if __name__ == "__main__":  # pragma: no cover
    print("wrote", build_cache())
