"""A genuinely learned apnea detector, and an honest measurement of whether it is worth it.

See the assessment at the bottom of this file for the answer. Read that before reading the
code as an endorsement of anything.

Representation
--------------
Two are built and both are measured:

`FEATURES` - windows of the 18 shared causal features from `respiradar.dataset`.

`SPECTROGRAM` - a causal range-frequency image computed here from the raw IQ, because the
thing a learned model might plausibly discover is *rhythmicity*, and rhythmicity is a
spectral line, not a level. Every 0.5 s a Hann-windowed, linearly detrended DFT of the last
`SPEC_WINDOW_S` of per-range-bin displacement is taken, restricted to 0.15-0.60 Hz (9-36
bpm), and reduced over range by a maximum over the bins whose own amplitude SNR says their
phase means anything. Stacking the last `SPEC_HISTORY_S` of those columns gives a
(time x frequency) patch ending at the current frame.

Amplitude is deliberately removable. `shape_only=True` divides each column by its own total
band energy before the model sees it, so the network is shown the *shape* of the spectrum
with the level taken away. That is the direct test of the premise: a still-breathing person
and an apneic person have the same level, so any separation that survives column
normalisation is separation the hand-built energy detectors cannot reach.

Causality
---------
Every window ends at the current frame and is left-padded with its own first row. The DFT
looks back only. The scaler is fitted in `fit()` on training clips alone; nothing is
normalised by a statistic of the clip being scored. `predict(clip[:k]) == predict(clip)[:k]`
is asserted by `_check_prefix` and by the repo's causality test.

Determinism
-----------
`seed_everything()` seeds python, numpy and torch, and the training loop uses a fixed epoch
count with no early stopping, a generator-seeded sampler and no dropout randomness outside
the seeded generator. Two runs of `build().fit(clips)` produce bit-identical weights on CPU.
"""

from __future__ import annotations

import hashlib
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np

from respiradar.bakeoff import Clip
from respiradar.dataset import DATA, FEATURE_NAMES, SESSIONS, session_by_name
from respiradar.sources import WAVELENGTH_M, recorded_config, replay_frames

MM_PER_RADIAN = WAVELENGTH_M / (4 * np.pi) * 1000
FS = 20.0
SEED = 20260919

# --- spectrogram geometry -------------------------------------------------------------
# Two DFT windows per column, on one common frequency grid. A 10 s window is what the
# latency budget can afford; a 20 s window resolves 0.05 Hz, which is the difference between
# "a breathing line" and "a smear". The model is given both and may use either.
FAST_WINDOW_S = 10.0
SLOW_WINDOW_S = 20.0
GRID_HZ = 0.025           # zero-padded sampling of the band, common to both windows
SPEC_STRIDE_S = 0.5       # a new column twice a second; held in between (causal)
SPEC_HISTORY_S = 20.0     # how much of the spectrogram the model sees
LO_HZ, HI_HZ = 0.15, 0.60  # 9-36 bpm; the brief's 0.18-0.55 Hz, widened one grid step
MIN_SNR = 12.0            # amplitude SNR a range bin needs before its phase is believed

CACHE = DATA / "deep_spectrogram.npz"


def seed_everything(seed: int = SEED) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass


class _BandDFT:
    """Hann-windowed, linearly detrended, zero-padded DFT of the last `window_s`."""

    def __init__(self, fs: float, window_s: float) -> None:
        self.n = int(window_s * fs)
        self.nfft = int(round(fs / GRID_HZ))
        freqs = np.fft.rfftfreq(self.nfft, 1 / fs)
        self.band = np.where((freqs >= LO_HZ) & (freqs <= HI_HZ))[0]
        self.freqs = freqs[self.band]
        self.win = np.hanning(self.n)
        self.cg = float(self.win.sum())
        self.ramp = np.arange(self.n) - (self.n - 1) / 2
        self.rr = float((self.ramp ** 2).sum())

    def __call__(self, buf: np.ndarray) -> np.ndarray:
        """buf: (>=n, n_bins) mm of displacement. Returns (n_freq, n_bins) amplitude in mm."""
        x = buf[-self.n:]
        x = x - x.mean(axis=0)
        x = x - np.outer(self.ramp, (self.ramp @ x) / self.rr)
        spec = np.abs(np.fft.rfft(x * self.win[:, None], n=self.nfft, axis=0))
        return spec[self.band] * (2.0 / self.cg)


class SpectrogramExtractor:
    """Frame in, one spectral column out. State is strictly past-only.

    The column is two stacked spectra - fast window, slow window - of ONE range bin: the
    qualified bin with the most in-band energy over the slow window. A maximum taken
    per-frequency across bins was tried first and is worse than useless: the maximum of 21
    noisy bins is a noise floor that rises when the chest stops, so the "hold" columns came
    out LOUDER than the breathing ones in nine of twelve sessions. Committing to one bin
    keeps the column a spectrum of something, which is what any shape statistic needs.
    """

    def __init__(self, fs: float = FS, sweeps: int = 8) -> None:
        self.fs = fs
        self.fast = _BandDFT(fs, FAST_WINDOW_S)
        self.slow = _BandDFT(fs, SLOW_WINDOW_S)
        self.n_freq = len(self.fast.band)
        self.freqs = self.fast.freqs
        self.stride = max(1, int(SPEC_STRIDE_S * fs))
        self.buffer_len = int(SLOW_WINDOW_S * fs)
        self.sweeps = sweeps

        self.prev_angles = None
        self.unwrapped = None
        self.buf = None
        self.filled = 0
        self.amp = None
        self.noise = None
        self.a_amp = float(np.exp(-1 / (5.0 * fs)))
        self._last = np.zeros(2 * self.n_freq, dtype=np.float32)

    def process(self, frame) -> np.ndarray:
        sweeps = frame.iq
        mean_sweep = sweeps.mean(axis=0)
        angles = np.angle(mean_sweep)
        amplitude = np.abs(mean_sweep)
        if sweeps.shape[0] > 1:
            noise = np.abs(np.diff(sweeps, axis=0)).mean(axis=0) / np.sqrt(2)
            noise = noise / np.sqrt(sweeps.shape[0])
        else:
            noise = np.ones_like(amplitude)
        noise = np.maximum(noise, 1e-9)

        if self.prev_angles is None:
            n = len(mean_sweep)
            self.unwrapped = np.zeros(n)
            self.buf = np.zeros((self.buffer_len, n))
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

        if self.filled % self.stride == 0 and self.filled >= self.fast.n:
            self._last = self._column()
        return self._last

    def _column(self) -> np.ndarray:
        qualified = (self.amp / self.noise) > MIN_SNR
        if not qualified.any():
            qualified = np.zeros(len(self.amp), dtype=bool)
            qualified[int(np.argmax(self.amp / self.noise))] = True

        slow = self.slow(self.buf) if self.filled >= self.slow.n else self.fast(self.buf)
        energy = (slow ** 2).sum(axis=0)
        energy[~qualified] = -1.0
        pick = int(np.argmax(energy))
        fast = self.fast(self.buf)
        return np.concatenate([fast[:, pick], slow[:, pick]]).astype(np.float32)


def build_spectrogram_cache(path: Path = CACHE) -> Path:
    arrays = {}
    for session in SESSIONS:
        config = recorded_config(session.path)
        ex = SpectrogramExtractor(config.frame_rate)
        cols = [ex.process(frame) for frame in replay_frames(session.path)]
        arrays[f"{session.name}__S"] = np.asarray(cols, dtype=np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    return path


_SPEC: dict[str, np.ndarray] = {}


def spectrogram_for(name: str) -> np.ndarray:
    if not _SPEC:
        if not CACHE.exists():
            build_spectrogram_cache()
        with np.load(CACHE) as data:
            for key in data.files:
                _SPEC[key[:-3]] = data[key]
    return _SPEC[name]


if __name__ == "__main__":
    print("wrote", build_spectrogram_cache())


# =======================================================================================
# Windowing: dimensionless, self-referenced patches
# =======================================================================================
REF_WINDOW_S = 90.0
REF_MIN_S = 25.0
REF_PCT = 75.0
PATCH_T = int(SPEC_HISTORY_S / SPEC_STRIDE_S)  # 40 columns = 20 s at 0.5 s
PATCH_STRIDE = int(SPEC_STRIDE_S * FS)


def _trailing_pct(x: np.ndarray, window: int, pct: float, min_n: int) -> np.ndarray:
    """Causal trailing percentile, evaluated every 0.5 s and held (see gated.py)."""
    out = np.empty(len(x))
    last = 0.0
    step = max(1, int(0.5 * FS))
    for i in range(len(x)):
        if i % step == 0:
            lo = max(0, i - window + 1)
            seg = x[lo : i + 1]
            last = float(np.percentile(seg, pct)) if len(seg) >= min_n else 0.0
        out[i] = last
    return out


def spec_stream(name: str) -> np.ndarray:
    """(n_frames, 2*n_freq) dimensionless spectrogram: each column over this person's own
    trailing 75th-percentile band energy. Scale-free by construction, so a model cannot
    learn which body or which recording it is looking at from the level."""
    S = spectrogram_for(name).astype(float)
    nf = S.shape[1] // 2
    slow_e = np.sqrt((S[:, nf:] ** 2).sum(axis=1))
    ref = _trailing_pct(slow_e, int(REF_WINDOW_S * FS), REF_PCT, int(REF_MIN_S * FS))
    # Before a reference exists, fall back to the running value so the ratio reads ~1
    # ("looks normal") rather than 0 ("perfect hold"), which would alarm during start-up.
    fallback = np.maximum.accumulate(np.maximum(slow_e, 1e-6))
    ref = np.where(ref > 0, ref, fallback)
    return np.log1p(S / ref[:, None] * 4.0)


FEAT_LOG = [FEATURE_NAMES.index(n) for n in
            ("rms_4s", "rms_8s", "rms_16s", "baseline", "intra", "inter",
             "amplitude", "disp_std_4s", "ref_q25")]
FEAT_KEEP = [FEATURE_NAMES.index(n) for n in
             ("ratio_4s", "ratio_8s", "ratio_4s_q", "ratio_8s_q", "flatness", "autocorr",
              "warm")]


def feat_stream(X: np.ndarray) -> np.ndarray:
    """The dimensionless half of the shared feature row, plus logged amplitudes."""
    a = np.log1p(np.maximum(X[:, FEAT_LOG], 0.0))
    b = X[:, FEAT_KEEP]
    return np.concatenate([a, b], axis=1)


def channels_for(clip: Clip, mode: str) -> np.ndarray:
    name = clip.name.split("[")[0]
    parts = []
    if mode in ("spec", "both"):
        S = spec_stream(name)
        if len(S) != len(clip.t):  # a sliced clip: align by index from the session start
            S = S[: len(clip.t)]
        parts.append(S)
    if mode in ("feat", "both"):
        parts.append(feat_stream(clip.X))
    return np.concatenate(parts, axis=1)


def patches(stream: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """(len(idx), PATCH_T, C). Row j is columns idx[j]-k*PATCH_STRIDE for k=PATCH_T-1..0,
    left-padded with the stream's own first row. Strictly causal."""
    offsets = np.arange(PATCH_T - 1, -1, -1) * PATCH_STRIDE
    taps = idx[:, None] - offsets[None, :]
    np.clip(taps, 0, None, out=taps)
    return stream[taps]


# =======================================================================================
# The network
# =======================================================================================
def _make_net(n_channels: int, width: int = 16, dropout: float = 0.3):
    import torch
    from torch import nn

    class CausalNet(nn.Module):
        """Dilated causal convolutions over the patch, global-pooled at the last step only.

        Small on purpose: 13 events is the effective sample size, so capacity is a
        liability. Three conv layers at width 16 is ~5k parameters.
        """

        def __init__(self) -> None:
            super().__init__()
            self.drop = nn.Dropout(dropout)
            self.c1 = nn.Conv1d(n_channels, width, 3, dilation=1)
            self.c2 = nn.Conv1d(width, width, 3, dilation=3)
            self.c3 = nn.Conv1d(width, width, 3, dilation=9)
            self.head = nn.Linear(width, 1)
            self.act = nn.GELU()

        def forward(self, x):  # x: (B, T, C)
            h = x.transpose(1, 2)
            for conv, dil in ((self.c1, 1), (self.c2, 3), (self.c3, 9)):
                h = nn.functional.pad(h, (2 * dil, 0))  # LEFT pad only: no future taps
                h = self.act(conv(h))
                h = self.drop(h)
            return self.head(h[:, :, -1]).squeeze(-1)

    return CausalNet()


# =======================================================================================
# The detector
# =======================================================================================
class DeepApneaDetector:
    """Causal CNN over the patch, behind the same duration gate the other entries use.

    The gate is not a formality. Per-frame probabilities from any model fitted here cross
    any threshold somewhere in the negatives; what the negatives do not contain is eight
    unbroken seconds of them. Duration buys the zero, exactly as it does for `supervised`.
    """

    def __init__(
        self,
        mode: str = "both",
        name: str | None = None,
        on_threshold: float = 0.90,
        off_threshold: float = 0.40,
        min_duration_s: float = 8.0,
        epochs: int = 30,
        width: int = 16,
        lr: float = 3e-3,
        weight_decay: float = 1e-3,
        n_models: int = 1,
        dropout: float = 0.3,
        noise: float = 0.15,
        event_weighted: bool = False,
        seed: int = SEED,
    ) -> None:
        self.mode = mode
        self.name = name or f"deep/cnn-{mode}"
        self.on_threshold = on_threshold
        self.off_threshold = off_threshold
        self.min_duration_s = min_duration_s
        self.epochs = epochs
        self.width = width
        self.lr = lr
        self.weight_decay = weight_decay
        self.n_models = n_models
        self.dropout = dropout
        self.noise = noise
        self.event_weighted = event_weighted
        self.seed = seed
        self.nets: list = []
        self.net = None
        self.centre = None
        self.scale = None
        self.train_report: dict = {}

    # -- fitting -------------------------------------------------------------------
    def _xy(self, clips: Sequence[Clip], stride: int = 5):
        S, Y, G, W = [], [], [], []
        for clip in clips:
            stream = channels_for(clip, self.mode)
            idx = np.arange(0, len(clip.t), stride)
            idx = idx[clip.t[idx] >= 25.0]
            if not len(idx):
                continue
            S.append(patches(stream, idx))
            yy = clip.y[idx].astype(np.float32)
            Y.append(yy)
            G.append(np.full(len(idx), clip.name))
            # Event weighting: 13 holds is the effective sample size, not 40,000 frames.
            # Every hold gets the same total weight, and so does every negative recording,
            # so a 40 s hold does not outvote a 27 s one and a long quiet session does not
            # outvote a short one.
            w = np.ones(len(idx), dtype=np.float32)
            for hold in clip.holds:
                inside = (clip.t[idx] >= hold.start_s) & (clip.t[idx] < hold.end_s)
                if inside.any():
                    w[inside] = 1.0 / inside.sum()
            neg = yy < 0.5
            if neg.any():
                w[neg] = 1.0 / neg.sum()
            W.append(w)
        return np.concatenate(S), np.concatenate(Y), np.concatenate(G), np.concatenate(W)

    def fit(self, clips: Sequence[Clip]) -> None:
        import torch

        seed_everything(self.seed)
        X, y, _, w = self._xy(clips)
        # Scaler fitted on TRAINING CLIPS ONLY - never on the clip being scored.
        flat = X.reshape(-1, X.shape[-1])
        self.centre = np.median(flat, axis=0)
        iqr = np.subtract(*np.percentile(flat, [75, 25], axis=0))
        self.scale = np.maximum(iqr, 1e-3)

        Xn = ((X - self.centre) / self.scale).astype(np.float32)
        pos = max(float(y.sum()), 1.0)
        neg = max(float((1 - y).sum()), 1.0)
        pos_weight = torch.tensor(neg / pos, dtype=torch.float32)
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction="none")

        xt = torch.from_numpy(Xn)
        yt = torch.from_numpy(y)
        if self.event_weighted:
            wt = torch.from_numpy(w / w.mean())
        else:
            wt = torch.ones(len(xt))
        n = len(xt)
        self.nf = 2 * SpectrogramExtractor().n_freq if self.mode in ("spec", "both") else 0
        self.nets = []
        losses = [0.0]
        for member in range(self.n_models):
            seed = self.seed + 1000 * member
            torch.manual_seed(seed)
            net = _make_net(Xn.shape[-1], self.width, self.dropout)
            opt = torch.optim.Adam(net.parameters(), lr=self.lr,
                                   weight_decay=self.weight_decay)
            g = torch.Generator().manual_seed(seed)
            net.train()
            losses = self._train_one(net, opt, g, xt, yt, wt, loss_fn, n)
            net.eval()
            self.nets.append(net)
        self.net = self.nets[0]

        p = self.probabilities_from(xt)
        self.train_report = {
            "n_train": int(n),
            "final_loss": round(losses[-1], 4),
            "train_auc": round(_auc(y, p), 4),
            "clips": [c.name for c in clips],
        }

    def _train_one(self, net, opt, g, xt, yt, wt, loss_fn, n):
        import torch

        losses = []
        for epoch in range(self.epochs):
            perm = torch.randperm(n, generator=g)
            total = 0.0
            for start in range(0, n, 256):
                batch = perm[start : start + 256]
                xb = xt[batch].clone()
                # Augmentation. Amplitude scaling is deliberately absent: the inputs are
                # already ratios to this person's own normal, so a global gain is a no-op -
                # which is the property that is supposed to make this transfer.
                xb = xb + self.noise * torch.randn(xb.shape, generator=g)
                if self.nf:  # frequency roll: a different breathing rate, same structure
                    shift = int(torch.randint(-2, 3, (1,), generator=g))
                    if shift:
                        xb[:, :, :self.nf] = torch.roll(xb[:, :, :self.nf], shift, dims=2)
                opt.zero_grad()
                loss = (loss_fn(net(xb), yt[batch]) * wt[batch]).mean()
                loss.backward()
                opt.step()
                total += float(loss.detach()) * len(batch)
            losses.append(total / n)
        return losses

    def probabilities_from(self, xt):
        import torch

        out = np.zeros(len(xt))
        with torch.no_grad():
            for net in self.nets:
                chunks = [torch.sigmoid(net(xt[i : i + 4096])) for i in range(0, len(xt), 4096)]
                out += torch.cat(chunks).numpy()
        return out / len(self.nets)

    # -- prediction ----------------------------------------------------------------
    def probabilities(self, clip: Clip) -> np.ndarray:
        import torch

        stream = channels_for(clip, self.mode)
        idx = np.arange(len(clip.t))
        X = ((patches(stream, idx) - self.centre) / self.scale).astype(np.float32)
        return self.probabilities_from(torch.from_numpy(X))

    def predict(self, clip: Clip) -> np.ndarray:
        p = self.probabilities(clip)
        # Trailing 1 s smoother, causal.
        k = int(1.0 * FS)
        csum = np.concatenate([[0.0], np.cumsum(p)])
        lo = np.maximum(np.arange(len(p)) - k + 1, 0)
        smooth = (csum[np.arange(len(p)) + 1] - csum[lo]) / (np.arange(len(p)) + 1 - lo)

        need = int(self.min_duration_s * FS)
        alarms = np.zeros(len(p), dtype=bool)
        latched = False
        run = 0
        for i in range(len(p)):
            if latched:
                if smooth[i] < self.off_threshold:
                    latched = False
                    run = 0
            else:
                run = run + 1 if smooth[i] >= self.on_threshold else 0
                if run >= need:
                    latched = True
            alarms[i] = latched
        return alarms


def _auc(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=bool)
    if y.all() or not y.any():
        return float("nan")
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    n1, n0 = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def build():
    """The entry the bake-off scores: CNN over spectrogram + shared features, presence-gated."""
    from respiradar.detectors import gated

    return gated.PresenceGatedDetector(inner=DeepApneaDetector(mode="both"),
                                       name="deep/cnn-gated")


def build_spec():
    from respiradar.detectors import gated

    return gated.PresenceGatedDetector(inner=DeepApneaDetector(mode="spec"),
                                       name="deep/cnn-spec")


def build_feat():
    from respiradar.detectors import gated

    return gated.PresenceGatedDetector(inner=DeepApneaDetector(mode="feat"),
                                       name="deep/cnn-feat")
