"""Apnea from RHYTHMICITY, not amplitude: is there a spectral line, whatever its size.

The premise every other entry in this bake-off rests on - a held breath is quieter than a
breathing one - is measurably false on this dataset. Median `rms_4s` over the real breath
holds is 1.04. Median `rms_4s` over `sleeping`, where the subject is breathing normally the
whole time, is also 1.04, and half its frames are as quiet as a hold. In
`nishant-holds-3008` the *breathing* stretches are QUIETER than the holds (median 0.77
against 1.04): the chest barely moves while he breathes, and the torso settles and drifts
while he does not. Any statistic derived from band energy inherits this, and the detectors
that survive it do so by measuring a *drop relative to this person's recent normal* - which
is exactly the shape of a false alarm for someone who relaxes, or falls asleep, and starts
breathing more shallowly. That is the failure the user reported watching live.

So this detector never asks how big the motion is. It asks whether the motion is PERIODIC.

    R = the fraction of a range bin's spectral power, over the last 16 s, that falls in
        0.18-0.55 Hz (11-33 breaths per minute), out of everything in 0.03-2.0 Hz.
        The statistic is the largest R over every range bin whose reflection is strong
        enough for its phase to mean anything.

R is a ratio of powers, so it is scale-free by construction: multiply the chest motion by
a hundred or by a hundredth and R does not move. A shallow but regular chest still puts
most of its power on its own breathing line. A held breath has no line at all - the power
goes to drift below the band and to phase noise above it, and R collapses.

THE DECISIVE TEST, on `nishant-holds-3008`, the session where breathing is quieter than
holding. Frames at least one window into a hold, against breathing frames outside the
transitions:

    statistic   AUC     Cohen's d   median breathing   median holding
    R           1.000   +4.20       0.928              0.398
    rms_4s      0.661   +0.70       0.672              0.587

R separates completely: the 5th percentile of breathing frames is 0.791 and the 95th
percentile of hold frames is 0.705, so every threshold in that gap is perfect on this
session. `rms_4s` does not separate at all, and over all frames rather than mature ones it
separates backwards (AUC 0.587 in favour of holds being LOUDER). The shared `autocorr`
feature reads 0.488 - chance - and `flatness` reads 0.159, i.e. inverted.

Within-session AUC on the other labelled sessions, R against rms_4s:

    nishant-holds-2401     0.994   0.807
    justinas-holds-3515    0.999   0.911
    justinas-breath-hold   0.827   0.624
    breath-hold            0.398   0.862   <- the first-round session, where R fails

Three choices matter more than the rest:

1. **The band is NARROW.** The shared features band-pass at 0.10-0.70 Hz, and 0.10 Hz is
   6 breaths/min - below anything a person does. Torso drift lives there and dominates, so
   the shared `autocorr` and `flatness` measure it rather than breathing: they separate
   hold from not-hold at 0.09-0.26 sigma, near enough to nothing. On 0.18-0.55 Hz every
   session shows a plausible 10.8-17.8 bpm line. Recomputing the same ideas on the right
   band is most of what this file does.

2. **Per range bin, then a maximum.** The shared extractor collapses three range points
   around a slowly-chosen chest bin into one waveform. On `nishant-holds-3008` that
   composite carries so little breathing that R computed on it reaches AUC 0.66 - barely
   better than chance - while the best individual bin reaches 1.00. Breathing appears in
   whichever bin the chest happens to illuminate, and that is not always the bin a tracker
   picked. Taking a maximum over bins also means an alarm requires EVERY believable bin to
   be arrhythmic at once, which is much harder for noise to fake than one bin being quiet.

3. **A 16 s window.** The band is 0.37 Hz wide, so at 12 s it is four DFT bins and at 8 s
   barely two; there is no periodicity to measure in a window that holds two breaths.
   16 s costs latency the product pays for at every hold and buys the separation above:
   the same test at 12 s gives AUC 0.986 instead of 1.000.

WHERE IT STOPS WORKING, which is the honest part. R separates superbly WITHIN a recording
and only moderately ACROSS bodies: pooled over every subject, mature hold frames against
the negative sessions, AUC is 0.795. The absolute level of R rides on reflection geometry
and SNR - `sleeping` sits at a median 0.87 while `vishnu-sleeping`, also breathing
normally throughout, sits at 0.55, which is below several sessions' hold medians. So an
absolute threshold does not transfer, and rhythmicity alone (`build_pure`) reaches 2/13
holds at one false alarm, against 12/13 at zero for the energy-based charts. What is
delivered as `build()` is therefore rhythm as a CONFIRMATION on the existing alarm rather
than a replacement for it: see RhythmConfirmedDetector.

Everything is causal: windows look backwards only, state is updated frame by frame, and
nothing is normalised by a statistic of the whole recording.

What was tried and did not survive
----------------------------------
- Autocorrelation peak height at the breathing lag, on the narrow band: AUC 0.56-0.60.
  With a short window the band is a handful of DFT bins wide, so the autocorrelation is
  nearly sinusoidal whatever it is fed, and its peak height measures very little.
- Phase coherence of the dominant line across consecutive windows: 0.64 on the decisive
  session, and it inverts on others. Adjacent Hann windows overlap by 90%, so their phases
  agree whether or not anything real is oscillating.
- Peak-to-band concentration and second-harmonic strength: 0.42-0.72, and both are about
  the SHAPE of the in-band power rather than whether there is any, which turns out to be
  the weaker question.
- Spatial agreement between bins on a common peak frequency ("coh"): 0.27-0.77, worse than
  chance on the session it was built for. Noise bins agree on a frequency as readily as
  chest bins do, because they are all looking at the same drift.
- Median and mean of R across bins instead of the maximum: 0.87-0.95, consistently below
  the maximum. Most bins see nothing even while someone is breathing.
- Logistic regression over (R_max, R_top3, R_med, n50, peak_hz): 0/13 holds and five
  false alarms, leave-one-subject-out. It is kept as `build_model` to show the margin -
  with thirteen events, and an absolute level that is partly geometry, a fitted boundary
  learns the training subjects' geometry and transfers worse than a constant does.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import DATA, MM_PER_RADIAN, SESSIONS
from respiradar.presence import PresenceDetector
from respiradar.sources import RadarConfig, recorded_config, replay_frames

# 11-33 breaths per minute. Below 0.18 Hz is torso drift, not respiration; above 0.55 Hz is
# heartbeat and phase noise. Widening either edge costs separation immediately.
LOW_HZ, HIGH_HZ = 0.18, 0.55
WIDE_LOW_HZ, WIDE_HIGH_HZ = 0.03, 2.0  # the denominator: everything the sensor can resolve

WINDOW_S = 16.0  # ~3-5 breaths, and 0.0625 Hz of resolution. The detector's lag floor.
STRIDE_S = 0.1  # recompute the spectra this often; hold the value in between
MIN_SNR = 6.0  # amplitude SNR a bin needs before its phase is believed at all
MIN_BINS = 3  # if fewer qualify, take the best MIN_BINS anyway rather than nothing

FEATURE_NAMES = [
    "r_max",  # THE statistic: in-band power fraction of the most rhythmic qualified bin
    "r_top3",  # mean of the three most rhythmic - steadier, slightly worse separation
    "r_med",  # median across qualified bins
    "r_n50",  # fraction of qualified bins whose own R exceeds 0.5
    "peak_hz",  # breathing frequency of the most rhythmic bin
    "band_mm",  # in-band RMS of that bin, in mm. Reported, deliberately NOT alarmed on
    "n_qual",  # how many bins passed the SNR gate
    "intra",  # presence fast-motion score
    "inter",  # presence slow-motion score
    "ready",  # 1 once a full window exists, 0 before
]

F = {name: i for i, name in enumerate(FEATURE_NAMES)}


class RhythmExtractor:
    """Frames in, one causal feature row out. This is what would run on the sensor."""

    def __init__(self, config: RadarConfig) -> None:
        self.fs = fs = config.frame_rate
        self.presence = PresenceDetector(config)

        self.n = int(WINDOW_S * fs)
        self.pad = self.n * 2  # zero-padding, for a finer peak-frequency read only
        freqs = np.fft.rfftfreq(self.pad, 1 / fs)
        self.freqs = freqs
        self.band = (freqs >= LOW_HZ) & (freqs <= HIGH_HZ)
        self.wide = (freqs >= WIDE_LOW_HZ) & (freqs <= WIDE_HIGH_HZ)
        self.band_hz = freqs[self.band]
        self.win = np.hanning(self.n)[:, None]
        self.cg = float((np.hanning(self.n) ** 2).sum())
        self.ramp = np.arange(self.n) - (self.n - 1) / 2
        self.rr = float((self.ramp**2).sum())
        self.stride = max(1, int(STRIDE_S * fs))

        self.n_bins: int | None = None
        self.prev_angles: np.ndarray | None = None
        self.unwrapped: np.ndarray | None = None
        self.buf: np.ndarray | None = None
        self.filled = 0

        self.amp: np.ndarray | None = None
        self.noise: np.ndarray | None = None
        self.a_amp = float(np.exp(-1 / (5.0 * fs)))

        self._last: np.ndarray | None = None  # held between strides

    def process(self, frame) -> np.ndarray:
        presence = self.presence.process(frame)
        sweeps = frame.iq
        mean_sweep = sweeps.mean(axis=0)
        angles = np.angle(mean_sweep)
        amplitude = np.abs(mean_sweep)
        # The chest cannot move within one frame, so sweep-to-sweep spread is noise.
        if sweeps.shape[0] > 1:
            noise = np.abs(np.diff(sweeps, axis=0)).mean(axis=0) / np.sqrt(2)
            noise = noise / np.sqrt(sweeps.shape[0])
        else:
            noise = np.ones_like(amplitude)
        noise = np.maximum(noise, 1e-9)

        if self.n_bins is None:
            self.n_bins = len(mean_sweep)
            self.unwrapped = np.zeros(self.n_bins)
            self.buf = np.zeros((self.n, self.n_bins))
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

        intra = float(presence.intra.max())
        inter = float(presence.inter.max())

        if self.filled < self.n:
            # No window yet. R reads a neutral 1.0 ("as rhythmic as it gets") so a warmup
            # row can never be mistaken for an apnea by anything downstream.
            row = np.zeros(len(FEATURE_NAMES))
            row[F["r_max"]] = row[F["r_top3"]] = row[F["r_med"]] = 1.0
            row[F["intra"]], row[F["inter"]] = intra, inter
            return row

        if self._last is None or (self.filled % self.stride) == 0:
            self._last = self._spectra()
        row = self._last.copy()
        row[F["intra"]], row[F["inter"]] = intra, inter
        return row

    def _spectra(self) -> np.ndarray:
        x = self.buf - self.buf.mean(axis=0)
        # Detrended inside the window rather than high-pass filtered. A 0.18 Hz IIR rings
        # for many seconds after the chest stops, and that ringing is latency we would pay
        # for at every hold; a window only ever contains its own past.
        x = x - np.outer(self.ramp, (self.ramp @ x) / self.rr)
        power = np.abs(np.fft.rfft(x * self.win, n=self.pad, axis=0)) ** 2

        in_band = power[self.band]
        total = power[self.wide].sum(axis=0) + 1e-20
        r = in_band.sum(axis=0) / total

        snr = self.amp / self.noise
        qualified = snr > MIN_SNR
        if qualified.sum() < MIN_BINS:
            qualified = np.zeros(self.n_bins, dtype=bool)
            qualified[np.argsort(snr)[-MIN_BINS:]] = True

        rq = r[qualified]
        order = np.argsort(rq)[::-1]
        best_local = int(order[0])
        best = int(np.where(qualified)[0][best_local])

        peak_hz = float(self.band_hz[int(np.argmax(in_band[:, best]))])
        band_mm = float(np.sqrt(in_band[:, best].sum() * 2.0 / (self.n * self.cg)))

        row = np.zeros(len(FEATURE_NAMES))
        row[F["r_max"]] = float(rq[best_local])
        row[F["r_top3"]] = float(np.sort(rq)[::-1][:3].mean())
        row[F["r_med"]] = float(np.median(rq))
        row[F["r_n50"]] = float(np.mean(rq > 0.5))
        row[F["peak_hz"]] = peak_hz
        row[F["band_mm"]] = band_mm
        row[F["n_qual"]] = float(qualified.sum())
        row[F["ready"]] = 1.0
        return row


# ------------------------------------------------------------------------------- the cache

CACHE = DATA / "rhythm_features.npz"


def build_cache(path: Path = CACHE) -> Path:
    arrays = {}
    for session in SESSIONS:
        t, X = extract(replay_frames(session.path), recorded_config(session.path))
        arrays[f"{session.name}__t"] = t
        arrays[f"{session.name}__X"] = X
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


_CACHE: dict[str, tuple[np.ndarray, np.ndarray]] = {}


def _load(path: Path) -> bool:
    if not path.exists():
        return False
    with np.load(path) as data:
        loaded = {}
        for session in SESSIONS:
            key = f"{session.name}__X"
            if key not in data or data[key].shape[1] != len(FEATURE_NAMES):
                return False  # stale: new recordings, or a changed feature set
            loaded[session.name] = (data[f"{session.name}__t"], data[key])
    _CACHE.update(loaded)
    return True


def _session_features(name: str) -> tuple[np.ndarray, np.ndarray]:
    if name not in _CACHE and not _load(CACHE):
        build_cache(CACHE)
        _load(CACHE)
    return _CACHE[name]


def register(name: str, t: np.ndarray, X: np.ndarray) -> None:
    """Supply features for a clip this module cannot look up by name (a live stream)."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected {len(FEATURE_NAMES)} columns, got {X.shape}")
    _CACHE[name] = (np.asarray(t, dtype=float), X)


def extract(frames, config: RadarConfig | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the causal extractor over any iterable of Frames. Returns (times, features)."""
    times, rows, extractor = [], [], None
    for frame in frames:
        if extractor is None:
            extractor = RhythmExtractor(config or RadarConfig())
        times.append(frame.t)
        rows.append(extractor.process(frame))
    if extractor is None:
        return np.zeros(0), np.zeros((0, len(FEATURE_NAMES)))
    return np.asarray(times), np.asarray(rows)


def _features_for(clip: Clip) -> tuple[np.ndarray, np.ndarray]:
    """This detector's own feature rows, aligned to the clip's frames."""
    name = clip.name.split("[")[0]
    t, X = _session_features(name)
    lo, hi = clip.t[0], clip.t[-1]
    mask = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    return t[mask], X[mask]


# ---------------------------------------------------------------------------- the detector


def _trailing_min(x: np.ndarray, n: int) -> np.ndarray:
    """Causal running minimum over the last n samples. Sample i sees only i and earlier."""
    from collections import deque

    out = np.empty(len(x))
    window: deque[int] = deque()
    for i, v in enumerate(x):
        while window and x[window[-1]] >= v:
            window.pop()
        window.append(i)
        while window[0] <= i - n:
            window.popleft()
        out[i] = x[window[0]]
    return out


def _trailing_max(x: np.ndarray, n: int) -> np.ndarray:
    return -_trailing_min(-x, n)


def _trailing_pct(x: np.ndarray, n: int, pct: float, min_n: int, stride: int) -> np.ndarray:
    """Causal trailing percentile, evaluated every `stride` samples and held in between.

    NaN until `min_n` samples exist: before that there is no personal history to compare
    against and the detector must not pretend otherwise.
    """
    out = np.full(len(x), np.nan)
    last = np.nan
    for i in range(len(x)):
        if i >= min_n and i % stride == 0:
            last = float(np.percentile(x[max(0, i - n + 1) : i + 1], pct))
        out[i] = last
    return out


def _dwell(flag: np.ndarray, n: int) -> np.ndarray:
    """True once `flag` has been continuously true for n samples."""
    if n <= 1:
        return flag.copy()
    run = 0
    out = np.zeros(len(flag), dtype=bool)
    for i, f in enumerate(flag):
        run = run + 1 if f else 0
        out[i] = run >= n
    return out


REF_WINDOW_S = 90.0  # how far back "this person's own rhythmicity" looks
REF_PCT = 75.0  # a high percentile: a hold inside the window must not define normal
REF_MIN_S = 25.0


class RhythmDetector:
    """The thesis, standing alone: alarm when nothing in the scene is breathing-periodic.

    Two conditions, both of which must hold for `dwell_s`:

    - `r_max < floor` - an ABSOLUTE arrhythmicity test. R is a power ratio, so this
      threshold at least means the same *thing* on every body, even though its level does
      not transfer perfectly (see below).
    - `r_max < ratio * ref` - R against this person's own trailing 75th percentile of R
      over 90 s. This is a personal reference, and it is worth being precise about why it
      is not the reference that causes the problem this detector exists to solve: it
      normalises RHYTHMICITY, not amplitude. Someone who relaxes into shallower breathing
      drops their amplitude by a factor of several and their R not at all, so this
      reference does not move under them. Someone who stops breathing loses the line, and
      only then does it move.

    Scored leave-one-subject-out, presence-gated: 2/13 holds and one false alarm - and
    that false alarm is on `justinas-breath-hold`, the session whose labels are suspected
    of being ~20 s early, so it may well be a real hold under the wrong label. Either way
    it is far behind the 12/13 at zero the energy-based charts reach, and the reading is
    the one in the module docstring:
    R separates hold from breathing beautifully WITHIN a recording and only moderately
    ACROSS bodies, because its absolute level rides on reflection geometry. `fit` is a
    no-op - every threshold here is a constant, and with thirteen events anything learned
    would be fitted to four of them.
    """

    name = "rhythm/bandfrac"

    def __init__(
        self,
        floor: float = 0.50,
        ratio: float = 0.55,
        dwell_s: float = 12.0,
        motion: float = 40.0,
    ) -> None:
        self.floor = floor
        self.ratio = ratio
        self.dwell_s = dwell_s
        self.motion = motion

    def fit(self, clips) -> None:
        """Nothing to learn. See the class docstring."""

    def statistic(self, clip: Clip):
        """(R, personal reference, feature rows) - the traces the alarm is built on."""
        _, X = _features_for(clip)
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        r = X[:, F["r_max"]]
        ref = _trailing_pct(
            r,
            int(REF_WINDOW_S * fs),
            REF_PCT,
            int(REF_MIN_S * fs),
            max(1, int(0.5 * fs)),
        )
        return r, ref, X

    def predict(self, clip: Clip) -> np.ndarray:
        r, ref, X = self.statistic(clip)
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        have_ref = np.isfinite(ref)
        relative = r < self.ratio * np.where(have_ref, ref, np.inf)
        # Gross movement makes a window broadband, which is arrhythmic, and would read as a
        # hold. This is a veto on alarming while someone thrashes, not evidence of apnea.
        quiet = X[:, F["intra"]] < self.motion
        low = (r < self.floor) & relative & have_ref & quiet & (X[:, F["ready"]] > 0)
        return _dwell(low, max(1, int(self.dwell_s * fs)))


class RhythmConfirmedDetector:
    """The practical entry: an energy detector that may only alarm when the rhythm is gone.

    `gated/best` decides WHEN something changed - sequential change detection is the right
    tool for minimising delay at a fixed false-alarm rate, and nothing here beats it at
    that. What it cannot do is tell a chest that stopped moving from a chest that stopped
    moving MUCH, because both are a drop in band energy against a personal baseline. This
    wrapper adds the second question: is there still a breathing line anywhere in the
    scene? If there is, the alarm is withheld.

    Leave-one-subject-out this scores exactly what `gated/best` scores - 12/13 holds, zero
    false alarms, worst latency 23.7 s, median 18.0 s. It neither gains nor loses a hold,
    which is the point worth being careful about: on THIS dataset the confirmation is free
    but it is not worth anything either, because the false alarm it forecloses is one
    nobody has recorded yet. What can be measured is how much of the negative data it
    would have covered, and that is uneven: on `sleeping` 81% of frames carry enough rhythm
    to veto an alarm outright, on `justinas-sleeping-2` 59%, and on `vishnu-sleeping` only
    6%. It is a safety net with holes in it, and the size of the holes is subject geometry.

    The threshold is FITTED, not chosen on the test data: `fit` takes the 90th percentile
    of R over the training subjects' labelled holds, so the veto is set just above what a
    hold has been seen to look like and cannot suppress one. Across the folds that lands at
    0.77-0.81.
    """

    name = "rhythm/confirmed-cusum"

    def __init__(self, inner=None, veto: float | None = None) -> None:
        from respiradar.detectors import gated

        self.inner = inner or gated.build_best()
        self.veto = veto  # None: learn it in fit()
        self.fitted: float | None = None

    def fit(self, clips) -> None:
        if hasattr(self.inner, "fit"):
            self.inner.fit(clips)
        if self.veto is not None:
            self.fitted = self.veto
            return
        during_holds = []
        for clip in clips:
            _, X = _features_for(clip)
            ready = X[:, F["ready"]] > 0
            if clip.y.any():
                during_holds.append(X[ready & clip.y, F["r_max"]])
        if during_holds:
            self.fitted = float(np.percentile(np.concatenate(during_holds), 90))
        else:
            self.fitted = 0.8

    def predict(self, clip: Clip) -> np.ndarray:
        _, X = _features_for(clip)
        veto = self.fitted if self.fitted is not None else 0.8
        rhythmic = X[:, F["r_max"]] >= veto
        return self.inner.predict(clip) & ~rhythmic


class RhythmModelDetector:
    """Logistic regression over the rhythm features. 0/13 holds, 5 false alarms.

    Kept because the negative result is worth recording: see the module docstring.
    """

    name = "rhythm/logistic"

    def __init__(self, dwell_s: float = 14.0, threshold: float = 0.5) -> None:
        self.dwell_s = dwell_s
        self.threshold = threshold
        self.model = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        cols = [F["r_max"], F["r_top3"], F["r_med"], F["r_n50"], F["peak_hz"]]
        return X[:, cols]

    def fit(self, clips) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler

        rows, labels = [], []
        for clip in clips:
            _, X = _features_for(clip)
            keep = (X[:, F["ready"]] > 0) & (clip.t >= clip.t[0] + 25.0)
            rows.append(self._design(X)[keep])
            labels.append(clip.y[keep])
        self.model = make_pipeline(
            StandardScaler(),
            LogisticRegression(C=0.1, class_weight="balanced", max_iter=2000),
        ).fit(np.vstack(rows), np.concatenate(labels))

    def predict(self, clip: Clip) -> np.ndarray:
        _, X = _features_for(clip)
        fs = 1 / max(float(np.median(np.diff(clip.t))), 1e-6)
        p = self.model.predict_proba(self._design(X))[:, 1]
        low = (p > self.threshold) & (X[:, F["ready"]] > 0)
        return _dwell(low, max(1, int(self.dwell_s * fs)))


def build():
    """The entry for the board: rhythm as the confirmation an energy alarm has to pass."""
    return RhythmConfirmedDetector()


def build_pure():
    """Rhythmicity alone, alarming on its own. The thesis, measured without help.

    Presence-gated like every other entry, so that the empty-room session is not counted
    against a question this detector was told not to answer.
    """
    from respiradar.detectors.gated import PresenceGatedDetector

    return PresenceGatedDetector(inner=RhythmDetector(), name="rhythm/bandfrac+presence")


def build_model():
    return RhythmModelDetector()


if __name__ == "__main__":
    print("wrote", build_cache())
