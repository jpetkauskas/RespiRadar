import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from respiradar import ml_pipeline as ml


def fixture(seconds=40, subject="p001", bins=3):
    t = np.arange(int(seconds * 20) + 1) / 20
    phase = 2 * np.sin(2 * np.pi * 0.25 * t)
    iq = np.broadcast_to(100 * np.exp(1j * phase[:, None, None]), (len(t), 4, bins)).copy()
    return dict(t=t, iq=iq.astype(np.complex64), distances_m=0.3 + np.arange(bins) * 0.06,
                quality=np.zeros((len(t), 3), dtype=bool),
                metadata=np.array(json.dumps(dict(version=ml.VERSION, subject=subject,
                                                  source="simulator", frame_rate_hz=20))))


class FeatureTests(unittest.TestCase):
    def test_shape_physics_and_phase_wrap(self):
        data = fixture()
        # Introduce a static phase rotation that crosses +/- pi; phase steps
        # should remain the same, without artificial 2*pi spikes.
        x = ml.window_features(data["t"], data["iq"], data["quality"], 0, ml.WindowConfig())
        rotated = ml.window_features(data["t"], data["iq"] * np.exp(2j), data["quality"], 0, ml.WindowConfig())
        self.assertEqual(x.shape, (600, 9))
        self.assertEqual(x.dtype, np.float32)
        np.testing.assert_allclose(x[:, 0::3], np.log1p(100), atol=1e-5)
        np.testing.assert_allclose(x[:, 2::3], 1, atol=1e-6)
        np.testing.assert_allclose(x[:, 1::3], rotated[:, 1::3], atol=2e-6)
        expected = np.diff(2 * np.sin(2 * np.pi * 0.25 * data["t"][:600]), prepend=0)
        np.testing.assert_allclose(x[:, 1], expected, atol=1e-6)

    def test_gaps_and_quality_do_not_become_predictions(self):
        for kind in ("gap", "saturation", "nan", "zero"):
            with self.subTest(kind=kind):
                data = fixture()
                if kind == "gap":
                    for key in ("t", "iq", "quality"):
                        data[key] = np.delete(data[key], np.s_[100:110], axis=0)
                elif kind == "saturation":
                    data["quality"][100, 0] = True
                elif kind == "nan":
                    data["iq"][100] = np.nan
                else:
                    data["iq"][:] = 0
                with self.assertRaises(ValueError):
                    ml.window_features(data["t"], data["iq"], data["quality"], 0, ml.WindowConfig())

    def test_jitter_resampling(self):
        data = fixture()
        data["t"][1:-1] += 0.002 * np.sin(np.arange(len(data["t"]) - 2))
        ml.validate_record(data)
        x = ml.window_features(data["t"], data["iq"], data["quality"], 0, ml.WindowConfig())
        self.assertTrue(np.isfinite(x).all())

    def test_invalid_timestamps_and_config(self):
        data = fixture()
        data["t"][1] = data["t"][0]
        with self.assertRaises(ValueError):
            ml.validate_record(data)
        with self.assertRaises(ValueError):
            ml.WindowConfig(stride=0)


class WorkflowTests(unittest.TestCase):
    def test_radar_reader_preserves_sensor_clock_flags_and_cleanup(self):
        client = Mock()
        client.get_next.side_effect = [
            SimpleNamespace(tick_time=123.0 + index * 0.05,
                            frame=np.ones((16, 21), dtype=complex),
                            data_saturated=bool(index), frame_delayed=False, calibration_needed=False)
            for index in range(2)
        ]
        with patch("acconeer.exptool.a121.Client.open", return_value=client) as opened, \
                patch("acconeer.exptool.a121.algo.get_distances_m", return_value=np.arange(21) * 0.06):
            stream = ml.frame_stream("COM6")
            first, flags = next(stream)
            second, second_flags = next(stream)
            stream.close()
            opened.assert_called_once_with(serial_port="COM6")
            self.assertEqual(first.t, 0)
            self.assertAlmostEqual(second.t, 0.05)
            self.assertFalse(flags.any())
            self.assertTrue(second_flags[0])
            client.stop_session.assert_called_once()
            client.close.assert_called_once()

    def test_radar_setup_failure_still_closes_port(self):
        client = Mock()
        client.setup_session.side_effect = RuntimeError("setup failed")
        with patch("acconeer.exptool.a121.Client.open", return_value=client):
            with self.assertRaises(RuntimeError):
                next(ml.frame_stream("COM6"))
        client.close.assert_called_once()
        client.stop_session.assert_not_called()

    def test_labels_and_unlabeled_windows_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "session.npz"
            ml.save_npz(path, **fixture())
            ml.label(argparse.Namespace(recording=path, start=0, end=30, label="normal"))
            with self.assertRaises(ValueError):
                ml.label(argparse.Namespace(recording=path, start=29, end=35, label="motion"))
            output = Path(root) / "dataset.npz"
            ml.prepare(argparse.Namespace(recordings=[path], output=output, window=30, stride=5))
            data = ml.read_npz(output)
            self.assertEqual(data["X"].shape, (3, 600, 9))
            np.testing.assert_array_equal(data["y"], [0, -1, -1])
            self.assertTrue(data["simulated"].all())
            with self.assertRaises(FileExistsError):
                ml.save_npz(path, **fixture())

    def test_subject_split_keeps_all_windows_together(self):
        y = np.tile(np.arange(4), 4)
        subjects = np.repeat(["a", "a", "b", "b"], 4)
        train, val = ml.subject_split(y, subjects, ["b"])
        self.assertEqual(set(subjects[train]), {"a"})
        self.assertEqual(set(subjects[val]), {"b"})
        with self.assertRaises(ValueError):
            ml.subject_split(y, subjects, ["missing"])
        with self.assertRaises(ValueError):
            ml.subject_split(np.zeros(16, dtype=int), subjects, ["b"])

    @unittest.skipUnless(importlib.util.find_spec("torch"), "Optional PyTorch is not installed")
    def test_training_checkpoint_inference_and_training_only_normalization(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            rng = np.random.default_rng(4)
            x = rng.normal(size=(8, 10, 9)).astype(np.float32)
            x[4:] += 50  # validation must not influence fitted normalization
            dataset = root / "dataset.npz"
            ml.save_npz(dataset, X=x, y=np.tile(np.arange(4), 2),
                        subjects=np.repeat(["a", "b"], 4), simulated=np.ones(8, dtype=bool),
                        distances_m=np.array([0.3, 0.36, 0.42]), sweeps=np.array(4),
                        config=np.array(json.dumps(dict(fs=20, seconds=0.5, stride=0.5))),
                        version=np.array(ml.VERSION), classes=np.array(ml.CLASSES), features=np.array(ml.FEATURES))
            model = root / "model.pt"
            args = argparse.Namespace(dataset=dataset, output=model, validation_subjects=["b"],
                                      epochs=1, batch_size=4, seed=42, allow_simulated=False)
            with self.assertRaises(ValueError):
                ml.train(args)
            args.allow_simulated = True
            ml.train(args)
            predictor = ml.Predictor(model, threshold=1)
            np.testing.assert_allclose(predictor.mean, x[:4].mean(axis=(0, 1)), atol=1e-6)
            result = predictor.predict(x[0])
            self.assertEqual(result["status"], "unknown")
            self.assertAlmostEqual(sum(result["scores"].values()), 1, places=5)
            self.assertTrue(result["simulated_training"])
            with self.assertRaises(ValueError):
                predictor.check_geometry(np.array([1, 2, 3]), 4)


if __name__ == "__main__":
    unittest.main()
