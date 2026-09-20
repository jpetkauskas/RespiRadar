import os
import time
from pathlib import Path

import pytest

# Qt widgets are constructed in the GUI/scope tests; without this they need a display.
# Set here rather than in every test module (and before PySide6 is ever imported) so that a
# bare `pytest` works with no environment set up by hand.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

DATA = Path(__file__).parent / "data"


@pytest.fixture
def sitting_recording() -> Path:
    """38.6 s of a real sitting person breathing, recorded by Acconeer."""
    return DATA / "breathing-sitting.h5"


@pytest.fixture
def no_presence_processor_recording() -> Path:
    """The same sitting person, recorded with Acconeer's presence processor switched off.

    Despite the filename this is NOT an empty room - the embedded config reads
    `use_presence_processor: false`. There is a person in it.
    """
    return DATA / "breathing-sitting-no-presence.h5"


@pytest.fixture
def acconeer_reference() -> Path:
    """Acconeer's own per-frame results for `sitting_recording`."""
    return DATA / "breathing-sitting-controller.h5"


@pytest.fixture(scope="session")
def sleeping_session():
    """Extracted features for one session. Slow, so computed once per test session."""
    from respiradar.dataset import extract_session, session_by_name

    return extract_session(session_by_name("sleeping"))


@pytest.fixture(scope="session", autouse=True)
def feature_cache():
    """Make sure data/features.npz exists before any test uses it.

    The cache is gitignored, so on a fresh checkout the first `load_cached` builds it. Under
    `-n auto` every worker would start building it at once and several would be writing the
    same file - so whichever worker gets the lock builds it and the rest wait.
    """
    from respiradar.dataset import CACHE, build_cache

    if CACHE.exists():
        return CACHE
    lock = CACHE.with_suffix(".lock")
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        deadline = time.time() + 600
        while time.time() < deadline and not CACHE.exists():
            time.sleep(0.5)
        return CACHE
    try:
        build_cache(CACHE)
    finally:
        os.close(fd)
        lock.unlink(missing_ok=True)
    return CACHE


@pytest.fixture(scope="session")
def empty_room_clip():
    """180 s of simulated empty room, as a Clip. ~3 s to make, and four tests want it."""
    import itertools

    import numpy as np

    from respiradar.bakeoff import Clip
    from respiradar.dataset import FeatureExtractor
    from respiradar.sources import RadarConfig, simulated_frames

    seconds = 180
    config = RadarConfig(sweeps_per_frame=8)
    extractor = FeatureExtractor(config)
    times, rows = [], []
    frames = simulated_frames(config, breaths_per_min=None, realtime=False)
    for frame in itertools.islice(frames, seconds * int(config.frame_rate)):
        times.append(frame.t)
        rows.append(extractor.process(frame))
    t = np.asarray(times)
    return Clip("empty-room", t, np.asarray(rows), np.zeros(len(t), bool), [])


@pytest.fixture(scope="session")
def bakeoff_score():
    """`respiradar.bakeoff.score`, memoised on the detector's name.

    Scoring fits and predicts over every fold and every session, so the same detector must
    not be scored twice in one test session.
    """
    from respiradar.bakeoff import score

    cache: dict[str, object] = {}

    def scored(detector):
        key = getattr(detector, "name", repr(detector))
        if key not in cache:
            cache[key] = score(detector)
        return cache[key]

    return scored


# Roughly how long each test takes, worst first. Under `-n auto` xdist hands work out in
# collection order, so a 30 s test collected near the end lands on a worker that then runs
# it alone while everything else has finished - which is what kept the parallel suite at
# ~60 s when the longest test is ~30 s. Running the expensive ones first fills the workers
# up front. It changes scheduling only: every test still runs, and none depends on order.
SLOWEST_FIRST = (
    "test_the_gate_costs_no_detections_and_removes_false_alarms",
    "test_the_shipped_detector_is_not_obviously_broken",
    "test_the_live_detector_finds_holds_in_a_recording",
    "test_the_live_detector_stays_silent_on_a_subject_with_no_holds",
)


def pytest_collection_modifyitems(config, items):
    rank = {name: i for i, name in enumerate(SLOWEST_FIRST)}
    items.sort(key=lambda item: rank.get(item.name, len(rank)))
