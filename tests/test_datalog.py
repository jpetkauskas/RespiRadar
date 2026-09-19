import csv
import sys

import h5py
import pytest

import datalog
from respiradar.sources import replay_frames


def test_records_a_labelled_session_that_replays(tmp_path, monkeypatch):
    pytest.importorskip("acconeer.exptool")
    monkeypatch.setattr(
        sys,
        "argv",
        ["datalog.py", "Breath Hold", "--subject", "Test Person", "--seconds", "2",
         "--countdown", "1", "--mock", "--out-dir", str(tmp_path)],
    )

    assert datalog.main() == 0

    [path] = tmp_path.glob("*.h5")
    assert path.name.startswith("test-person_breath-hold_")
    with h5py.File(path) as f:
        assert f.attrs["label"] == "breath-hold"
        assert f.attrs["subject"] == "test-person"
        start_frame = int(f.attrs["start_frame"])

    frames = list(replay_frames(path))
    assert len(frames) > start_frame + 30  # countdown frames kept, then ~2 s at 20 Hz
    assert frames[0].iq.ndim == 2 and frames[0].iq.dtype.kind == "c"

    rows = list(csv.DictReader((tmp_path / "sessions.csv").open()))
    assert rows[0]["file"] == path.name and rows[0]["label"] == "breath-hold"
