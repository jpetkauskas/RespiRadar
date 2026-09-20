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

import time
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
    # Second recording round: a scripted (30 s breathing, 30 s hold) x 3 per session, so the
    # holds are uniform and - unlike the first round - none of them start inside the filter
    # warmup. These nine holds are worth more than the original four: the in-hold breathing
    # envelope separates at 0.18 of the out-of-hold level, against 0.62-0.71 before.
    Session("nishant-holds-2401", "nishant_breath-hold_20260919-202401.h5", "nishant",
            holds=[Episode(30.0, 60.0), Episode(90.0, 120.0), Episode(149.8, 179.0)],
            description="scripted 30 s hold x 3"),
    Session("nishant-holds-3008", "nishant_breath-hold_20260919-203008.h5", "nishant",
            holds=[Episode(30.0, 60.0), Episode(90.0, 120.0), Episode(149.8, 179.0)],
            description="scripted 30 s hold x 3"),
    # Only five markers: the recording ends during the third hold, so its end is the end of
    # the recording rather than a marked resumption.
    Session("justinas-holds-3515", "justinas_breath-hold_20260919-203515.h5", "justinas",
            holds=[Episode(30.2, 60.1), Episode(90.0, 120.2), Episode(151.9, 180.2)],
            description="scripted 30 s hold x 3"),
    # An empty scene - the sensor pointed at a wall, nobody in the beam. Every other
    # recording contains a person throughout, so until this existed the "does it alarm at an
    # empty room" question was answered only by a simulator, which turned out to understate
    # real clutter badly: simulated presence scores peak at 1.5, this wall reaches 15.2.
    # It is its own subject so leave-one-subject-out gives it a fold of its own.
    Session("wall", "wall_static_20260919-212928.h5", "wall",
            description="pointed at a wall, nobody there - must never alarm"),
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
    "baseline",  # this person's own normal: see REFERENCE MODE in FeatureExtractor
    "ratio_4s",  # rms_4s / baseline - near 0 during a hold, near 1 while breathing
    "ratio_8s",
    "intra",  # presence fast-motion score - high while talking or moving
    "inter",  # presence slow-motion score
    "amplitude",  # reflection strength at the chest, to catch the person leaving
    "disp_std_4s",  # unfiltered motion, including gross movement
    "flatness",  # spectral flatness in band: 1 = noise-like, 0 = one clean tone
    "autocorr",  # strength of the dominant period - breathing is regular, talking is not
    # --- added with the reference/warmup rework; see the class docstring ---
    "ref_q25",  # trailing low quantile of rms_8s - a self-reference that cannot ratchet
    "ratio_4s_q",  # rms_4s / ref_q25
    "ratio_8s_q",  # rms_8s / ref_q25
    "ref_ratchet",  # the OLD asymmetric-EMA baseline, kept for comparison
    "seconds_of_history",  # how long this extractor has been running, in seconds
    "warm",  # 0 while the features are provisional, ramping to 1 once they are settled
]


MIN_RMS_S = 2.0  # shortest window we are willing to call an RMS
MIN_SPECTRUM_S = 6.0  # shortest window we are willing to take a spectrum of
WARM_S = 16.0  # history at which the features are considered fully settled
SEED_BASELINE_S = 4.0  # history at which a provisional self-reference is seeded
INIT_FILTER_STATE = True  # start the band-pass settled at the first sample, not at zero


class FeatureExtractor:
    """Turns frames into the feature row used by every detector in the bake-off.

    REFERENCE MODE. Every useful apnea statistic is *this* chest's motion against *this*
    chest's normal, so the self-reference is the most load-bearing thing in here.

    The original reference was an asymmetric EMA that rose fast and fell ~40x slower, so a
    long hold could not drag "normal" down to meet itself. The side effect was worse than the
    problem: it ratcheted up to the subject's *best* breathing and stayed there, so after one
    burst of movement ordinary shallow breathing read as apnea for minutes. Two bake-off
    entries diagnosed this independently and both worked around it privately, and one measured
    justinas-sleeping-2 sitting at a baseline of 2.28 against an actual rms_4s of 1.2.

    The fix is a TRAILING LOW QUANTILE: the 25th percentile of rms_8s over the last
    `ref_window_s`, computed causally with a zero-order hold (`method="lower"`, never an
    interpolation onto a value the window does not contain). A quantile cannot ratchet - it
    forgets - and a quarter of a two-minute window is still normal breathing for the first
    ~30 s of any hold we have labelled.

    The quantile is ADDED rather than swapped in: `ref_q25`, `ratio_4s_q` and `ratio_8s_q`
    are the non-ratcheting reference and are what new work should use, while `baseline` /
    `ratio_4s` / `ratio_8s` keep the meaning every existing detector was tuned against. That
    is not timidity. The two references answer different questions, and `changepoint` uses
    both deliberately: a trailing quantile for its slow channel, and the ratchet for its
    fastest one *because* a reference that refuses to follow a hold downwards is exactly what
    a fast channel wants. Measured, swapping the meaning of `baseline` for the quantile takes
    `baseline/energy-threshold` from 16 false alarms to 0 and `temporal` from 9 to 5, and
    costs the leading `cusum-bank-conservative` a hold (3/4 -> 2/4) and 8 s of worst-case
    latency. `reference="quantile"` makes the swap for anyone who wants to measure it again.

    WARMUP. Two of the four labelled holds start before 16 s. The extractor used to report
    a literal 0.0 for any RMS whose window was not yet full, which reads as a perfect breath
    hold, and it refused to form a reference at all until 16 s. Now every window is computed
    over whatever history exists past a short minimum, the reference is seeded at
    `SEED_BASELINE_S`, and `warm` / `seconds_of_history` say how provisional the row is so a
    detector can reason about its own reliability instead of guessing.
    """

    def __init__(
        self,
        config: RadarConfig,
        baseline_time_const_s: float = 45.0,
        ref_window_s: float = 120.0,
        ref_quantile: float = 0.25,
        reference: str = "ratchet",  # "ratchet" | "quantile" | "min"
    ) -> None:
        self.config = config
        self.fs = config.frame_rate
        self.presence = PresenceDetector(config)
        # Per-bin state, so the chest is chosen by how much it actually MOVES rather than by
        # how brightly it reflects. See _select_chest_bin.
        self.bin_amp: np.ndarray | None = None
        self.bin_noise: np.ndarray | None = None
        self.bin_angles: np.ndarray | None = None
        self.bin_unwrapped: np.ndarray | None = None
        self.bin_sos = None
        self.bin_zi = None
        self.bin_hist: list[np.ndarray] = []
        self._last_bin = 0  # which range bin was judged to be the chest, for diagnostics
        self.bin_slow: np.ndarray | None = None  # long-run motion per bin, for selection

        nyquist = self.fs / 2
        self.sos = signal.butter(
            2, [LOW_HZ / nyquist, HIGH_HZ / nyquist], btype="bandpass", output="sos"
        )
        # Filled on the first frame from sosfilt_zi scaled by the first displacement, so the
        # filter starts in the steady state for that DC level instead of stepping into it.
        self.zi = None
        self.zi_unit = signal.sosfilt_zi(self.sos)

        self.buffer_len = int(16 * self.fs)
        self.n_frames = 0
        self.filtered: list[float] = []
        self.raw: list[float] = []

        self.prev_angles: np.ndarray | None = None
        self.unwrapped: np.ndarray | None = None
        self.power: np.ndarray | None = None
        self.baseline: float | None = None
        self.baseline_alpha = 1 / (baseline_time_const_s * self.fs)

        self.reference_mode = reference
        self.ref_quantile = ref_quantile
        self.ref_window = max(int(ref_window_s * self.fs), 1)
        self.rms8_hist: list[float] = []  # trailing rms_8s, for the quantile reference

    @property
    def seconds_of_history(self) -> float:
        return self.n_frames / self.fs

    def _window_rms(self, seconds: float) -> float:
        """RMS over the last `seconds`, or over whatever history exists past MIN_RMS_S.

        Returning 0.0 while the window fills - what this used to do - is not "no data", it is
        the exact value a perfect breath hold produces, and it lasted 16 s on a 16 s window.
        """
        n = int(seconds * self.fs)
        have = len(self.filtered)
        if have < int(MIN_RMS_S * self.fs):
            return 0.0
        x = np.asarray(self.filtered[-min(n, have) :])
        return float(np.sqrt(np.mean(x**2)))

    def _flatness_and_autocorr(self) -> tuple[float, float]:
        n = int(16 * self.fs)
        have = len(self.filtered)
        if have < int(MIN_SPECTRUM_S * self.fs):
            return 1.0, 0.0
        x = np.asarray(self.filtered[-min(n, have) :])
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

    def _select_chest_bin(self, frame) -> int:
        """Pick the range bin showing the most breathing-band MOTION, in millimetres.

        The obvious choice - the presence detector's peak - is wrong on this hardware. Bin 0
        sits at 0.30 m in the sensor's near field and reflects ~3200 against the chest's
        ~500. Because the presence score divides by the noise floor, that bright static
        reflection needs only a minuscule drift to outscore a real breathing chest, and it
        won 50-99% of frames in every recording. Selecting on absolute millimetres of
        band-limited motion instead, among bins whose reflection clears the noise, is
        subject-independent physics: 0.15 mm and 1.5 mm differ by 10x whoever is lying there.
        """
        sweeps = frame.iq
        mean_sweep = sweeps.mean(axis=0)
        amplitude = np.abs(mean_sweep)
        noise = (
            np.abs(np.diff(sweeps, axis=0)).mean(axis=0) / np.sqrt(2)
            if sweeps.shape[0] > 1
            else np.ones_like(amplitude)
        )
        noise = np.maximum(noise, 1e-9)

        if self.bin_amp is None:
            n = len(mean_sweep)
            self.bin_amp = amplitude
            self.bin_noise = noise
            self.bin_angles = np.angle(mean_sweep)
            self.bin_unwrapped = np.zeros(n)
            nyq = self.fs / 2
            self.bin_sos = signal.butter(
                2, [LOW_HZ / nyq, HIGH_HZ / nyq], btype="bandpass", output="sos"
            )
            self.bin_zi = np.zeros((self.bin_sos.shape[0], n, 2))
        else:
            self.bin_amp = 0.95 * self.bin_amp + 0.05 * amplitude
            self.bin_noise = 0.95 * self.bin_noise + 0.05 * noise
            angles = np.angle(mean_sweep)
            step = (angles - self.bin_angles + np.pi) % (2 * np.pi) - np.pi
            self.bin_unwrapped = self.bin_unwrapped + step
            self.bin_angles = angles

        # Shape (bins, 1): one new sample per bin, filtered along time (axis=1), so each
        # range bin keeps its own filter state.
        filtered, self.bin_zi = signal.sosfilt(
            self.bin_sos, (self.bin_unwrapped * MM_PER_RADIAN)[:, None], zi=self.bin_zi, axis=1
        )
        self.bin_hist.append(filtered[:, 0])
        if len(self.bin_hist) > int(8 * self.fs):
            self.bin_hist.pop(0)

        recent = np.asarray(self.bin_hist)
        motion_mm = np.sqrt(np.mean(recent**2, axis=0))

        # Selection must be SLOW. Choosing the bin with the most motion right now means that
        # when breathing stops the selector goes hunting for whatever else is moving, so an
        # apnea can never be observed - it simply re-points at a different bin. A 90 s time
        # constant is dominated by normal breathing and barely moves during a 30 s hold.
        #
        # Bias-correcting this EMA's start-up (alpha = max(1/(90 fs), 1/n), which would make
        # it an exact running mean until the window is worth 90 s) was tried and REJECTED: it
        # won cusum-bank-conservative a fourth hold and cost it two false alarms, and false
        # alarms come first. The selector's start-up is left alone.
        if self.bin_slow is None:
            self.bin_slow = motion_mm
        else:
            a = 1 / (90 * self.fs)
            self.bin_slow = (1 - a) * self.bin_slow + a * motion_mm

        # A chest is wider than one 6 cm range bin, so real breathing shows up across several
        # adjacent bins at once. Averaging each bin with its neighbours rewards that and
        # dilutes anything isolated - which is what the near-field clutter at 0.30 m is.
        # Selecting on raw per-bin motion picks that clutter instead; an SNR gate does not
        # help, because the chest's own reflection is weak (SNR < 1 at 0.8 m) while the
        # clutter is bright.
        spread = np.convolve(self.bin_slow, np.ones(3) / 3, mode="same")
        self._last_bin = int(np.argmax(spread))
        return self._last_bin

    def process(self, frame) -> np.ndarray:
        presence = self.presence.process(frame)
        chest_bin = self._select_chest_bin(frame)

        # Track the three range points around the chest, weighted by reflected power.
        half = 1
        last = frame.iq.shape[1] - 1
        low = int(np.clip(chest_bin - half, 0, max(last - 2, 0)))
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
        # Starting from zero state means the first displacement arrives as a step and the
        # filter rings its way out of it for several seconds - which is most of the history
        # a hold starting at 4.6 s ever gets. sosfilt_zi is the state that holds a constant
        # input steady, so scaling it by the first sample starts the filter already settled
        # at that DC level and leaves only genuine motion to respond to.
        if self.zi is None:
            self.zi = self.zi_unit * (displacement if INIT_FILTER_STATE else 0.0)
        value, self.zi = signal.sosfilt(self.sos, [displacement], zi=self.zi)
        self.filtered.append(float(value[0]))
        self.raw.append(displacement)
        if len(self.filtered) > self.buffer_len:
            self.filtered.pop(0)
            self.raw.pop(0)

        rms_4 = self._window_rms(4.0)
        rms_8 = self._window_rms(8.0)
        rms_16 = self._window_rms(16.0)

        self.n_frames += 1
        history_s = self.seconds_of_history
        # A ramp, not a flag: the row is provisional from the moment there is any history and
        # fully trustworthy once the 16 s windows are full. Consumers that want the old
        # boolean can test `warm >= 1`.
        warm = float(np.clip(history_s / WARM_S, 0.0, 1.0))
        seeded = history_s >= SEED_BASELINE_S and rms_8 > 0

        # The old reference: rises fast, falls ~40x slower. Kept for comparison only - it
        # ratchets to this subject's best breathing and never comes back down.
        if seeded:
            if self.baseline is None:
                self.baseline = rms_8
            elif rms_8 > self.baseline:
                a = self.baseline_alpha * 4  # adapt up quickly
                self.baseline = (1 - a) * self.baseline + a * rms_8
            else:
                a = self.baseline_alpha * 0.1  # and down only very slowly
                self.baseline = (1 - a) * self.baseline + a * rms_8

        # The trailing low quantile. Strictly past frames, and `method="lower"` returns a
        # value the window actually contains - never an interpolation towards a neighbour,
        # which at the top of the window would be a sample that has not happened yet.
        if rms_8 > 0:
            self.rms8_hist.append(rms_8)
            if len(self.rms8_hist) > self.ref_window:
                self.rms8_hist.pop(0)
        if seeded and self.rms8_hist:
            ref_q = float(np.quantile(self.rms8_hist, self.ref_quantile, method="lower"))
        else:
            ref_q = 0.0

        ratchet = self.baseline if self.baseline and self.baseline > 0 else 0.0
        if self.reference_mode == "ratchet":
            base = ratchet
        elif self.reference_mode == "quantile":
            base = ref_q
        else:  # "min": the quantile stops the ratcheting, the EMA caps the quantile
            candidates = [v for v in (ratchet, ref_q) if v > 0]
            base = min(candidates) if candidates else 0.0

        # Before a reference exists we do not know what this person's normal looks like, so
        # report a neutral ratio of 1.0 ("looks normal") rather than 0.0, which would read as
        # a breath hold and alarm during every start-up.
        # Clamp: a ratio of 10 and a ratio of 10,000 mean the same thing (moving a lot),
        # and the unclamped value wrecks any model that scales its inputs.
        if base <= 0:
            base = float(rms_8) if rms_8 > 0 else 1.0
            ratio_4 = ratio_8 = 1.0
        else:
            ratio_4 = min(rms_4 / base, 10.0)
            ratio_8 = min(rms_8 / base, 10.0)

        if ref_q > 0:
            ratio_4_q = min(rms_4 / ref_q, 10.0)
            ratio_8_q = min(rms_8 / ref_q, 10.0)
        else:
            ref_q = float(rms_8) if rms_8 > 0 else 1.0
            ratio_4_q = ratio_8_q = 1.0

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
                ref_q,
                ratio_4_q,
                ratio_8_q,
                ratchet if ratchet > 0 else base,
                history_s,
                warm,
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


# Bump when anything changes the meaning of a cached row: the feature list, the extractor's
# behaviour, or the label clock. The cache is committed, so a stale one is not one developer's
# problem - it is everybody's.
CACHE_SCHEMA = 1


def build_cache(path: Path = CACHE, verbose: bool = True) -> Path:
    """Extract every session once and save it. Feature extraction is the slow part.

    Tens of seconds on a laptop and several minutes on a Cortex-A53, and it is reached from
    inside `load_cached` - so when it ran silently, starting the app on the UNO Q was
    indistinguishable from a hang. It says what it is doing.
    """
    names = [session.name for session in SESSIONS]
    arrays = {
        "__schema": np.array(CACHE_SCHEMA),
        "__features": np.array(FEATURE_NAMES),
        "__sessions": np.array(names),
    }
    started = time.monotonic()
    if verbose:
        print(f"building the feature cache: {len(SESSIONS)} sessions", flush=True)
    for i, session in enumerate(SESSIONS, 1):
        mark = time.monotonic()
        t, X, y = extract_session(session)
        arrays[f"{session.name}__t"] = t
        arrays[f"{session.name}__X"] = X
        arrays[f"{session.name}__y"] = y
        if verbose:
            print(f"  [{i:2d}/{len(SESSIONS)}] {session.name:<22} "
                  f"{time.monotonic() - mark:5.1f}s", flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    if verbose:
        print(f"wrote {path} in {time.monotonic() - started:.0f}s", flush=True)
    return path


def cache_is_current(path: Path = CACHE) -> bool:
    """Does this cache match the code about to read it?

    `load_cached` used to rebuild only when the file was MISSING. A cache written before the
    `wall` session was added therefore survived, and surfaced as `KeyError: wall__t` thrown
    from deep inside a detector - a confusing failure a long way from its cause. Now that the
    cache is committed, that same stale file would be distributed to everyone, so the check
    covers the session list and the feature list, not just the file's existence.
    """
    if not path.exists():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            return (
                "__schema" in data
                and int(data["__schema"]) == CACHE_SCHEMA
                and list(data["__features"]) == FEATURE_NAMES
                and list(data["__sessions"]) == [s.name for s in SESSIONS]
            )
    except Exception:
        return False  # unreadable, truncated, written by another numpy - rebuild it


def load_cached(name: str, path: Path = CACHE):
    """(times, features, labels) for one session, rebuilding the cache if it is stale."""
    if not cache_is_current(path):
        build_cache(path)
    with np.load(path) as data:
        return data[f"{name}__t"], data[f"{name}__X"], data[f"{name}__y"]


if __name__ == "__main__":
    print("wrote", build_cache())
