"""Standalone XM125 recording, interval labeling, model inputs, and LSTM baseline.

Run ``python -m respiradar.ml_pipeline --help``. See docs/ml_pipeline.md.
No changes to the dashboard pipeline are required. Predictions are research outputs;
this module ships no trained weights or validated normal/abnormal decision rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import deque
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from respiradar import sources

VERSION = 1
CLASSES = ("normal", "abnormal", "motion", "absent")
FEATURES = ("log_amplitude", "phase_step_rad", "sweep_coherence")
QUALITY_FLAGS = ("data_saturated", "frame_delayed", "calibration_needed")


@dataclass(frozen=True)
class WindowConfig:
    fs: float = sources.FRAME_RATE_HZ
    seconds: float = 30.0
    stride: float = 5.0
    max_gap_periods: float = 2.5

    def __post_init__(self):
        if not all(np.isfinite(x) and x > 0 for x in asdict(self).values()):
            raise ValueError("Window settings must be finite and positive")
        if self.samples < 2:
            raise ValueError("A window needs at least two samples")

    @property
    def samples(self):
        return int(round(self.fs * self.seconds))


def frame_stream(port=None):
    """Yield (Frame, quality flags), using sensor timestamps on real hardware.

    Mirrors sources.radar_frames' sensor configuration, but retains metadata and
    quality flags that its Frame interface currently discards.
    """
    if port is None:
        frames = sources.simulated_frames()
        try:
            for frame in frames:
                yield frame, np.zeros(3, dtype=bool)
        finally:
            frames.close()
        return

    from acconeer.exptool import a121
    from acconeer.exptool.a121.algo import get_distances_m

    start, points = sources._points()
    config = a121.SensorConfig(
        start_point=start, num_points=points, step_length=sources.STEP_LENGTH,
        profile=a121.Profile.PROFILE_3, hwaas=sources.HWAAS,
        sweeps_per_frame=sources.SWEEPS_PER_FRAME, frame_rate=sources.FRAME_RATE_HZ,
    )
    client = a121.Client.open(serial_port=port)
    started = False
    try:
        metadata = client.setup_session(config)
        distances = get_distances_m(config, metadata)
        client.start_session()
        started = True
        first_tick = None
        while True:
            result = client.get_next()
            if first_tick is None:
                first_tick = result.tick_time
            yield sources.Frame(
                t=result.tick_time - first_tick,
                iq=result.frame.copy(), distances_m=distances,
            ), np.array([getattr(result, flag) for flag in QUALITY_FLAGS], dtype=bool)
    finally:
        try:
            if started:
                client.stop_session()
        finally:
            client.close()


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as file:
        np.savez_compressed(file, **arrays)


def read_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def validate_record(data):
    t, iq, distances = data["t"], data["iq"], data["distances_m"]
    if t.ndim != 1 or len(t) < 2 or not np.all(np.isfinite(t)) or np.any(np.diff(t) <= 0):
        raise ValueError("Recording needs at least two finite, increasing timestamps")
    if iq.ndim != 3 or not np.iscomplexobj(iq) or iq.shape[0] != len(t):
        raise ValueError("IQ must be complex [frames, sweeps, range_bins]")
    if min(iq.shape[1:]) < 1 or distances.shape != (iq.shape[2],):
        raise ValueError("Invalid sweep or distance dimensions")
    if not np.all(np.isfinite(distances)) or np.any(np.diff(distances) <= 0):
        raise ValueError("Distance bins must be finite and increasing")
    if data["quality"].shape != (len(t), len(QUALITY_FLAGS)):
        raise ValueError("Quality flags must have shape [frames, 3]")
    metadata = json.loads(str(data["metadata"]))
    if metadata["version"] != VERSION:
        raise ValueError("Unsupported recording version")
    return metadata


def label_path(recording):
    return Path(recording).with_suffix(".labels.csv")


def read_labels(path):
    if not Path(path).exists():
        return []
    with Path(path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    intervals = sorted((float(row["start_s"]), float(row["end_s"]), row["label"]) for row in rows)
    previous_end = -np.inf
    for start, end, label in intervals:
        if not np.isfinite(start + end) or start < 0 or end <= start or start < previous_end:
            raise ValueError("Labels must be finite, positive-length, nonoverlapping intervals")
        if label not in CLASSES:
            raise ValueError(f"Label must be one of {CLASSES}")
        previous_end = end
    return intervals


def record(args):
    if not np.isfinite(args.seconds) or args.seconds <= 0:
        raise ValueError("Recording duration must be positive")
    if args.output.exists() or label_path(args.output).exists():
        raise ValueError("Output recording or label file already exists; choose a new name")
    frames, flags = [], []
    stream = frame_stream(args.port)
    print(f"Recording {'COM port ' + args.port if args.port else 'SIMULATED data'}; Ctrl+C saves captured frames.")
    try:
        for frame, quality in stream:
            frames.append(frame)
            flags.append(quality)
            if frame.t >= args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()
    if len(frames) < 2:
        raise ValueError("Not enough frames captured")
    metadata = {
        "version": VERSION, "subject": args.subject, "session": args.output.stem,
        "source": "radar" if args.port else "simulator", "port": args.port,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "clock": "sensor_tick" if args.port else "simulator",
        "frame_rate_hz": sources.FRAME_RATE_HZ, "quality_flags": QUALITY_FLAGS,
        "sensor_config": {"start_m": sources.START_M, "end_m": sources.END_M,
                          "step_length": sources.STEP_LENGTH, "profile": 3,
                          "hwaas": sources.HWAAS, "sweeps_per_frame": sources.SWEEPS_PER_FRAME},
    }
    save_npz(args.output, t=np.array([f.t for f in frames]),
             iq=np.stack([f.iq for f in frames]).astype(np.complex64),
             distances_m=frames[0].distances_m, quality=np.stack(flags),
             metadata=np.array(json.dumps(metadata)))
    label_path(args.output).write_text("start_s,end_s,label\n", encoding="utf-8")
    print(f"Saved {len(frames)} frames to {args.output}; annotate {label_path(args.output)}")


def label(args):
    data = read_npz(args.recording)
    metadata = validate_record(data)
    end_limit = data["t"][-1] + 1 / metadata["frame_rate_hz"]
    if not (np.isfinite(args.start) and np.isfinite(args.end) and
            data["t"][0] <= args.start < args.end <= end_limit + 1e-6):
        raise ValueError(f"Label interval must lie within [0, {end_limit:.3f}] seconds")
    path = label_path(args.recording)
    intervals = read_labels(path)
    if any(args.start < end and args.end > start for start, end, _ in intervals):
        raise ValueError("Label overlaps an existing interval; edit the CSV to correct it")
    intervals.append((args.start, args.end, args.label))
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(("start_s", "end_s", "label"))
        writer.writerows(sorted(intervals))
    print(f"Labeled [{args.start}, {args.end}) as {args.label}")


def window_features(t, iq, quality, start, config):
    """Return float32 [time, range_bins * 3]; reject unusable windows.

    Keep every range bin. Interpolate log amplitude, unwrapped phase, and
    coherence onto a fixed clock; difference phase only AFTER interpolation.
    Phase steps are a motion proxy, not calibrated chest displacement.
    """
    grid = start + np.arange(config.samples) / config.fs
    lo = max(0, np.searchsorted(t, grid[0], side="right") - 1)
    hi = min(len(t), np.searchsorted(t, grid[-1], side="left") + 1)
    times, values = t[lo:hi], iq[lo:hi]
    if len(times) < 2 or times[0] > grid[0] + 1e-6 or times[-1] < grid[-1] - 1e-6:
        raise ValueError("incomplete_window")
    if np.any(np.diff(times) <= 0) or np.max(np.diff(times)) > config.max_gap_periods / config.fs:
        raise ValueError("timestamp_gap")
    if np.any(quality[lo:hi]) or not np.all(np.isfinite(values)):
        raise ValueError("sensor_quality")
    mean_iq = values.mean(axis=1)
    amplitude = np.abs(values).mean(axis=1)
    if np.max(amplitude) < 1e-8:
        raise ValueError("zero_signal")
    coherence = np.clip(np.abs(mean_iq) / np.maximum(amplitude, 1e-8), 0, 1)
    phase = np.unwrap(np.angle(mean_iq), axis=0)

    def resample(array):
        return np.stack([np.interp(grid, times, array[:, i]) for i in range(array.shape[1])], axis=1)

    log_amplitude = resample(np.log1p(amplitude))
    phase_grid = resample(phase)
    phase_step = np.diff(phase_grid, axis=0, prepend=phase_grid[:1])
    features = np.stack((log_amplitude, phase_step, resample(coherence)), axis=-1)
    return features.reshape(config.samples, -1).astype(np.float32)


def iter_windows(data, config):
    t = data["t"]
    last_start = t[-1] - (config.samples - 1) / config.fs
    count = max(0, int(np.floor((last_start - t[0] + 1e-6) / config.stride)) + 1)
    for index in range(count):
        start = float(t[0] + index * config.stride)
        try:
            x = window_features(t, data["iq"], data["quality"], start, config)
            yield start, x, None
        except ValueError as error:
            yield start, None, str(error)


def prepare(args):
    config = WindowConfig(seconds=args.window, stride=args.stride)
    xs, ys, subjects, sessions, starts, simulated = [], [], [], [], [], []
    distances = sweeps = None
    rejected = {}
    for path in args.recordings:
        data = read_npz(path)
        metadata = validate_record(data)
        if distances is None:
            distances, sweeps = data["distances_m"], data["iq"].shape[1]
        if (data["distances_m"].shape != distances.shape or
                not np.allclose(distances, data["distances_m"], atol=0.003) or
                data["iq"].shape[1] != sweeps or metadata["frame_rate_hz"] != config.fs):
            raise ValueError("Recordings must use matching range bins, sweeps, and frame rate")
        intervals = read_labels(label_path(path))
        for start, x, reason in iter_windows(data, config):
            if x is None:
                rejected[reason] = rejected.get(reason, 0) + 1
                continue
            # Intervals are half-open; require the entire nominal window in one label.
            matches = [name for lo, hi, name in intervals
                       if lo <= start + 1e-6 and start + config.samples / config.fs <= hi + 1e-6]
            xs.append(x)
            ys.append(CLASSES.index(matches[0]) if matches else -1)
            subjects.append(metadata["subject"])
            sessions.append(str(path.resolve()))
            starts.append(start)
            simulated.append(metadata["source"] == "simulator")
    if not xs:
        raise ValueError(f"No usable windows; record at least {config.seconds}s. Rejections: {rejected}")
    x_array = np.stack(xs)
    save_npz(args.output, X=x_array, y=np.array(ys, dtype=np.int64),
             subjects=np.array(subjects), sessions=np.array(sessions), starts=np.array(starts),
             simulated=np.array(simulated), distances_m=distances, sweeps=np.array(sweeps),
             config=np.array(json.dumps(asdict(config))), classes=np.array(CLASSES),
             features=np.array(FEATURES), version=np.array(VERSION))
    print(json.dumps({"shape": list(x_array.shape), "unlabeled": ys.count(-1),
                      "labels": {name: ys.count(i) for i, name in enumerate(CLASSES)},
                      "rejected": rejected}))


def torch_module():
    try:
        import torch
    except ImportError as error:
        raise ValueError("PyTorch is optional. Run with: uv run --with torch python -m respiradar.ml_pipeline ...") from error
    return torch


def make_model(input_size, hidden_size=32):
    torch = torch_module()

    class BreathingLSTM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
            self.head = torch.nn.Linear(hidden_size, len(CLASSES))

        def forward(self, x):
            sequence, _ = self.lstm(x)
            return self.head(sequence.mean(dim=1))

    return BreathingLSTM()


def subject_split(y, subjects, validation_subjects):
    unknown = set(validation_subjects) - set(subjects.tolist())
    if unknown:
        raise ValueError(f"Validation subjects not found: {sorted(unknown)}")
    validation = np.isin(subjects, validation_subjects) & (y >= 0)
    training = ~np.isin(subjects, validation_subjects) & (y >= 0)
    for name, mask in (("training", training), ("validation", validation)):
        if set(y[mask].tolist()) != set(range(len(CLASSES))):
            raise ValueError(f"{name} needs all four classes from separate subjects: {CLASSES}")
    return training, validation


def train(args):
    torch = torch_module()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("Epochs and batch size must be positive")
    if args.output.exists():
        raise ValueError("Checkpoint exists; choose a new output path")
    data = read_npz(args.dataset)
    if int(data["version"]) != VERSION or tuple(data["features"]) != FEATURES or tuple(data["classes"]) != CLASSES:
        raise ValueError("Incompatible dataset schema")
    x, y = data["X"], data["y"]
    if not np.all(np.isfinite(x)):
        raise ValueError("Dataset contains nonfinite features")
    if np.any(data["simulated"]) and not args.allow_simulated:
        raise ValueError("Simulator data is for plumbing tests; use --allow-simulated explicitly")
    training, validation = subject_split(y, data["subjects"], args.validation_subjects)
    # Fit normalization exclusively on training subjects. Keep physical amplitudes
    # comparable across windows rather than normalizing each window separately.
    mean = x[training].mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = np.maximum(x[training].std(axis=(0, 1), dtype=np.float64), 1e-5).astype(np.float32)
    torch.manual_seed(args.seed)
    torch.set_num_threads(2)
    model = make_model(x.shape[-1])
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    counts = np.bincount(y[training], minlength=len(CLASSES))
    weights = torch.tensor(counts.sum() / (len(CLASSES) * counts), dtype=torch.float32)
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)

    def loader(mask, shuffle):
        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(((x[mask] - mean) / std).astype(np.float32)),
            torch.from_numpy(y[mask]),
        )
        return torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=shuffle)

    train_loader, val_loader = loader(training, True), loader(validation, False)
    best_loss = np.inf
    best_state = None
    best_report = None
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        model.eval()
        confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)
        val_loss, total = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                logits = model(xb)
                val_loss += torch.nn.functional.cross_entropy(logits, yb, reduction="sum").item()
                total += len(yb)
                np.add.at(confusion, (yb.numpy(), logits.argmax(dim=1).numpy()), 1)
        report = {"epoch": epoch + 1, "validation_loss": val_loss / total,
                  "balanced_accuracy": float(np.mean(np.diag(confusion) / confusion.sum(axis=1))),
                  "classes": CLASSES, "confusion_true_rows_predicted_columns": confusion.tolist()}
        print(json.dumps(report))
        if val_loss / total < best_loss:
            best_loss, best_report = val_loss / total, report
            best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    checkpoint = {
        "version": VERSION, "state_dict": best_state, "input_size": x.shape[-1],
        "hidden_size": 32, "mean": torch.from_numpy(mean), "std": torch.from_numpy(std),
        "classes": list(CLASSES), "features": list(FEATURES),
        "config": json.loads(str(data["config"])), "distances_m": torch.from_numpy(data["distances_m"]),
        "sweeps": int(data["sweeps"]), "simulated_training": bool(np.any(data["simulated"])),
        "training_subjects": sorted(set(data["subjects"][training].tolist())),
        "validation_subjects": args.validation_subjects, "validation_report": best_report,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as file:
        torch.save(checkpoint, file)
    print(f"Saved research checkpoint to {args.output}")


class Predictor:
    def __init__(self, path, threshold=0.7):
        torch = torch_module()
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (checkpoint["version"] != VERSION or tuple(checkpoint["classes"]) != CLASSES or
                tuple(checkpoint["features"]) != FEATURES):
            raise ValueError("Incompatible checkpoint")
        self.model = make_model(checkpoint["input_size"], checkpoint["hidden_size"])
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.eval()
        self.mean = checkpoint["mean"].numpy()
        self.std = checkpoint["std"].numpy()
        self.distances = checkpoint["distances_m"].numpy()
        self.sweeps = checkpoint["sweeps"]
        self.config = WindowConfig(**checkpoint["config"])
        self.threshold = threshold
        self.simulated_training = checkpoint["simulated_training"]

    def check_geometry(self, distances, sweeps):
        if (distances.shape != self.distances.shape or
                not np.allclose(distances, self.distances, atol=0.003) or sweeps != self.sweeps):
            raise ValueError("Sensor range bins/sweeps differ from the training data")

    def predict(self, x):
        torch = torch_module()
        with torch.no_grad():
            tensor = torch.from_numpy(((x - self.mean) / self.std).astype(np.float32))[None]
            scores = self.model(tensor).softmax(dim=-1)[0].numpy()
        index = int(np.argmax(scores))
        candidate = CLASSES[index]
        status = candidate if candidate in ("normal", "abnormal") and scores[index] >= self.threshold else "unknown"
        return {"status": status, "predicted_class": candidate,
                "scores": dict(zip(CLASSES, map(float, scores))),
                "reason": "low_score" if scores[index] < self.threshold else candidate,
                "research_only": True, "simulated_training": self.simulated_training}


def predict(args):
    predictor = Predictor(args.model, args.threshold)
    if args.recording:
        data = read_npz(args.recording)
        metadata = validate_record(data)
        predictor.check_geometry(data["distances_m"], data["iq"].shape[1])
        if metadata["frame_rate_hz"] != predictor.config.fs:
            raise ValueError("Recording frame rate differs from the training data")
        emitted = False
        for start, x, reason in iter_windows(data, predictor.config):
            emitted = True
            output = predictor.predict(x) if x is not None else {"status": "unknown", "reason": reason}
            print(json.dumps({"start_s": start, "source": metadata["source"], **output}))
        if not emitted:
            print(json.dumps({"status": "unknown", "reason": "incomplete_window"}))
        return
    # Same preprocessing as offline training. Never fill acquisition gaps with zeros.
    frames = deque(maxlen=int(np.ceil(predictor.config.seconds * predictor.config.fs * 2)) + 4)
    flags = deque(maxlen=frames.maxlen)
    next_end = (predictor.config.samples - 1) / predictor.config.fs
    stream = frame_stream(args.port)
    print(json.dumps({"status": "warming_up", "seconds": predictor.config.seconds,
                      "source": args.port or "simulator"}))
    try:
        for frame, quality in stream:
            predictor.check_geometry(frame.distances_m, frame.iq.shape[0])
            frames.append(frame)
            flags.append(quality)
            if frame.t + 1e-6 < next_end:
                continue
            start = frame.t - (predictor.config.samples - 1) / predictor.config.fs
            try:
                x = window_features(np.array([f.t for f in frames]),
                                    np.stack([f.iq for f in frames]), np.stack(flags), start, predictor.config)
                output = predictor.predict(x)
            except ValueError as error:
                output = {"status": "unknown", "reason": str(error)}
            print(json.dumps({"start_s": start, "source": args.port or "simulator", **output}), flush=True)
            next_end = frame.t + predictor.config.stride
    except KeyboardInterrupt:
        pass
    finally:
        stream.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("record", help="Capture raw IQ; omit --port for simulated data")
    p.add_argument("--port")
    p.add_argument("--seconds", type=float, default=60)
    p.add_argument("--subject", required=True, help="Pseudonymous ID, consistent across sessions")
    p.add_argument("--output", type=Path, required=True)
    p.set_defaults(func=record)
    p = commands.add_parser("label", help="Add a reference-labeled time interval")
    p.add_argument("recording", type=Path)
    p.add_argument("--start", type=float, required=True)
    p.add_argument("--end", type=float, required=True)
    p.add_argument("--label", choices=CLASSES, required=True)
    p.set_defaults(func=label)
    p = commands.add_parser("prepare", help="Export [windows, time, features] tensors")
    p.add_argument("recordings", type=Path, nargs="+")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--window", type=float, default=30)
    p.add_argument("--stride", type=float, default=5)
    p.set_defaults(func=prepare)
    p = commands.add_parser("train", help="Train optional PyTorch LSTM with a subject-held-out split")
    p.add_argument("dataset", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--validation-subjects", nargs="+", required=True)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--allow-simulated", action="store_true")
    p.set_defaults(func=train)
    p = commands.add_parser("predict", help="Predict recorded windows or stream from the sensor")
    p.add_argument("--model", type=Path, required=True)
    source = p.add_mutually_exclusive_group()
    source.add_argument("--recording", type=Path)
    source.add_argument("--port", help="Omit both source options to use the simulator")
    p.add_argument("--threshold", type=float, default=0.7, help="Uncalibrated score gate; validate before use")
    p.set_defaults(func=predict)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, OSError, KeyError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
