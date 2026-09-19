"""Labelled sessions and the causal feature stream every approach shares.

One recording in, one feature row per frame out, plus a label saying whether that frame is
inside a breath hold. Every feature is computed from the past only - the same code can run
live on the sensor. A non-causal feature (a zero-phase filter, a whole-session normalisation)
would score well here and be worthless in the demo, so `test_features_are_causal` pins it.

Ground truth comes from the markers in `data/sessions.csv`. In the breath-hold session the
subject was breathing normally, stopped, breathed normally, stopped - so the four markers are
the two holds' start and end.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import signal

from respiradar.evaluation import Episode
from respiradar.presence import PresenceDetector
from respiradar.sources import WAVELENGTH_M, RadarConfig, recorded_config, replay_frames

DATA = Path(__file__).parent.parent / "data"
MM_PER_RADIAN = WAVELENGTH_M / (4 * np.pi) * 1000

LOW_HZ = 6 / 60
HIGH_HZ = 40 / 60


@dataclass(frozen=True)
class Session:
    name: str
    filename: str
    subject: str = "nishant"
    holds: list[Episode] = field(default_factory=list)
    description: str = ""

    @property
    def path(self) -> Path:
        return DATA / self.filename


SESSIONS: list[Session] = [
    # nishant. The first three names predate the multi-subject recordings and are kept as
    # they are so existing code and detectors continue to resolve them.
    Session("sleeping", "nishant_sleeping_20260919-182058.h5", "nishant",
            description="lying still, breathing normally"),
    Session("noisy", "nishant_noisy_20260919-181623.h5", "nishant",
            description="talking and moving - must never alarm"),
    Session("breath-hold", "nishant_breath-hold_20260919-182741.h5", "nishant",
            holds=[Episode(14.3, 41.4), Episode(73.2, 116.4)],
            description="breathe, hold, breathe, hold"),
    # justinas
    Session("justinas-sleeping-1", "justinas_sleeping_20260919-184132.h5", "justinas"),
    Session("justinas-sleeping-2", "justinas_sleeping_20260919-184458.h5", "justinas"),
    Session("justinas-talking", "justinas_talking_20260919-185444.h5", "justinas",
            description="talking - must never alarm"),
    # The first hold starts at 4.6 s, inside the filter warmup, so its early features are a
    # transient rather than chest motion. Scoring applies a warmup, but expect this hold to
    # look worse than the others for reasons that are not the detector's fault.
    Session("justinas-breath-hold", "justinas_breath-hold_20260919-185023.h5", "justinas",
            holds=[Episode(4.6, 32.8), Episode(67.3, 114.0)],
            description="breathe, hold, breathe, hold"),
    # vishnu - negatives only, but a third subject to train against
    Session("vishnu-sleeping", "vishnu_sleeping_20260919-183707.h5", "vishnu"),
]


def subjects() -> list[str]:
    return sorted({session.subject for session in SESSIONS})


def session_by_name(name: str) -> Session:
    for session in SESSIONS:
        if session.name == name:
            return session
    raise KeyError(name)


FEATURE_NAMES = [
    "rms_4s",  # bandpassed chest motion over the last 4 s - the fast apnea signal
    "rms_8s",
    "rms_16s",
    "baseline",  # slow causal average of rms_8s: this person's own normal
    "ratio_4s",  # rms_4s / baseline - near 0 during a hold, near 1 while breathing
    "ratio_8s",
    "intra",  # presence fast-motion score - high while talking or moving
    "inter",  # presence slow-motion score
    "amplitude",  # reflection strength at the chest, to catch the person leaving
    "disp_std_4s",  # unfiltered motion, including gross movement
    "flatness",  # spectral flatness in band: 1 = noise-like, 0 = one clean tone
    "autocorr",  # strength of the dominant period - breathing is regular, talking is not
]


class FeatureExtractor:
    """Turns frames into the feature row used by every detector in the bake-off."""

    def __init__(self, config: RadarConfig, baseline_time_const_s: float = 45.0) -> None:
        self.config = config
        self.fs = config.frame_rate
        self.presence = PresenceDetector(config)

        nyquist = self.fs / 2
        self.sos = signal.butter(
            2, [LOW_HZ / nyquist, HIGH_HZ / nyquist], btype="bandpass", output="sos"
        )
        self.zi = signal.sosfilt_zi(self.sos) * 0.0

        self.buffer_len = int(16 * self.fs)
        # The band-pass rings for several seconds after it starts. Until the buffer is full
        # those samples are a filter transient, not chest motion, and must not seed the
        # baseline - doing so latches it to a near-zero value and makes every ratio explode.
        self.warm = False
        self.filtered: list[float] = []
        self.raw: list[float] = []

        self.prev_angles: np.ndarray | None = None
        self.unwrapped: np.ndarray | None = None
        self.power: np.ndarray | None = None
        self.baseline: float | None = None
        self.baseline_alpha = 1 / (baseline_time_const_s * self.fs)

    def _window_rms(self, seconds: float) -> float:
        n = int(seconds * self.fs)
        if len(self.filtered) < n:
            return 0.0
        x = np.asarray(self.filtered[-n:])
        return float(np.sqrt(np.mean(x**2)))

    def _flatness_and_autocorr(self) -> tuple[float, float]:
        n = int(16 * self.fs)
        if len(self.filtered) < n:
            return 1.0, 0.0
        x = np.asarray(self.filtered[-n:])
        x = x - x.mean()
        if np.allclose(x, 0):
            return 1.0, 0.0

        spectrum = np.abs(np.fft.rfft(x * np.hanning(len(x)))) ** 2
        freqs = np.fft.rfftfreq(len(x), 1 / self.fs)
        band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        in_band = spectrum[band] + 1e-20
        # Geometric over arithmetic mean: 1 for flat noise, towards 0 for a single peak.
        flatness = float(np.exp(np.mean(np.log(in_band))) / np.mean(in_band))

        norm = np.dot(x, x)
        ac = np.correlate(x, x, mode="full")[len(x) - 1 :] / (norm + 1e-20)
        lo = int(self.fs / HIGH_HZ)
        hi = min(int(self.fs / LOW_HZ), len(ac) - 1)
        autocorr = float(np.max(ac[lo:hi])) if hi > lo else 0.0
        return flatness, autocorr

    def process(self, frame) -> np.ndarray:
        presence = self.presence.process(frame)

        # Track the three range points around the person, weighted by reflected power.
        half = 1
        last = frame.iq.shape[1] - 1
        low = int(np.clip(presence.peak_index - half, 0, max(last - 2, 0)))
        segment = frame.iq[:, low : low + 3].mean(axis=0)
        angles = np.angle(segment)
        amplitude = np.abs(segment)

        if self.prev_angles is None:
            self.unwrapped = np.zeros_like(angles)
            self.power = amplitude**2
        else:
            step = (angles - self.prev_angles + np.pi) % (2 * np.pi) - np.pi
            if len(step) == len(self.unwrapped):
                self.unwrapped = self.unwrapped + step
                self.power = 0.95 * self.power + 0.05 * amplitude**2
            else:  # the tracked range moved; restart the phase integration
                self.unwrapped = np.zeros_like(angles)
                self.power = amplitude**2
        self.prev_angles = angles

        weights = self.power / max(self.power.sum(), 1e-12)
        displacement = float(np.sum(self.unwrapped * weights)) * MM_PER_RADIAN

        # Causal band-pass: sosfilt, never sosfiltfilt, which would look into the future.
        value, self.zi = signal.sosfilt(self.sos, [displacement], zi=self.zi)
        self.filtered.append(float(value[0]))
        self.raw.append(displacement)
        if len(self.filtered) > self.buffer_len:
            self.filtered.pop(0)
            self.raw.pop(0)

        rms_4 = self._window_rms(4.0)
        rms_8 = self._window_rms(8.0)
        rms_16 = self._window_rms(16.0)

        # The baseline learns what this person's normal breathing looks like. It only rises
        # quickly, never falls quickly, so a long hold cannot quietly drag "normal" down to
        # match itself.
        self.warm = self.warm or len(self.filtered) >= self.buffer_len
        if self.warm and rms_8 > 0:
            if self.baseline is None:
                self.baseline = rms_8
            elif rms_8 > self.baseline:
                a = self.baseline_alpha * 4  # adapt up quickly
                self.baseline = (1 - a) * self.baseline + a * rms_8
            else:
                a = self.baseline_alpha * 0.1  # and down only very slowly
                self.baseline = (1 - a) * self.baseline + a * rms_8

        # Before a baseline exists we do not know what this person's normal looks like, so
        # report a neutral ratio of 1.0 ("looks normal") rather than 0.0, which would read as
        # a breath hold and alarm during every start-up.
        if self.baseline is None or self.baseline <= 0:
            base = float(rms_8) if rms_8 > 0 else 1.0
            ratio_4 = ratio_8 = 1.0
        else:
            base = self.baseline
            # Clamp: a ratio of 10 and a ratio of 10,000 mean the same thing (moving a lot),
            # and the unclamped value wrecks any model that scales its inputs.
            ratio_4 = min(rms_4 / base, 10.0)
            ratio_8 = min(rms_8 / base, 10.0)

        n4 = int(4 * self.fs)
        disp_std = float(np.std(self.raw[-n4:])) if len(self.raw) >= n4 else 0.0
        flatness, autocorr = self._flatness_and_autocorr()

        return np.array(
            [
                rms_4,
                rms_8,
                rms_16,
                base,
                ratio_4,
                ratio_8,
                float(presence.intra.max()),
                float(presence.inter.max()),
                float(amplitude.mean()),
                disp_std,
                flatness,
                autocorr,
            ],
            dtype=float,
        )


def extract_session(
    session: Session, max_frames: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, features, labels) for one recording."""
    config = recorded_config(session.path)
    extractor = FeatureExtractor(config)

    times, rows = [], []
    for n, frame in enumerate(replay_frames(session.path)):
        if max_frames is not None and n >= max_frames:
            break
        times.append(frame.t)
        rows.append(extractor.process(frame))

    t = np.asarray(times)
    X = np.asarray(rows)
    y = np.zeros(len(t), dtype=bool)
    for hold in session.holds:
        y |= (t >= hold.start_s) & (t < hold.end_s)
    return t, X, y


CACHE = DATA / "features.npz"


def build_cache(path: Path = CACHE) -> Path:
    """Extract every session once and save it. Feature extraction is the slow part."""
    arrays = {}
    for session in SESSIONS:
        t, X, y = extract_session(session)
        arrays[f"{session.name}__t"] = t
        arrays[f"{session.name}__X"] = X
        arrays[f"{session.name}__y"] = y
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


def load_cached(name: str, path: Path = CACHE):
    """(times, features, labels) for one session, building the cache on first use."""
    if not path.exists():
        build_cache(path)
    with np.load(path) as data:
        return data[f"{name}__t"], data[f"{name}__X"], data[f"{name}__y"]


if __name__ == "__main__":
    print("wrote", build_cache())
