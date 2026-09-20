from pathlib import Path

import pytest

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
