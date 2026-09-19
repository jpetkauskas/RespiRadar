import numpy as np
import pytest

from respiradar.dataset import FEATURE_NAMES, SESSIONS, extract_session, session_by_name


def test_every_session_file_is_present():
    for session in SESSIONS:
        assert session.path.exists(), session.path


def test_only_the_breath_hold_session_has_labelled_holds():
    assert session_by_name("breath-hold").holds
    assert not session_by_name("sleeping").holds
    assert not session_by_name("noisy").holds


def test_extracts_one_feature_row_per_frame(sleeping_session):
    t, X, y = sleeping_session

    assert X.shape == (len(t), len(FEATURE_NAMES))
    assert y.shape == (len(t),)


def test_labels_mark_exactly_the_hold_intervals():
    t, _, y = extract_session(session_by_name("breath-hold"))

    assert y[np.argmin(np.abs(t - 30.0))]    # inside hold 1 (14.3-41.4)
    assert not y[np.argmin(np.abs(t - 60.0))]  # breathing between holds
    assert y[np.argmin(np.abs(t - 100.0))]   # inside hold 2 (73.2-116.4)
    assert not y[np.argmin(np.abs(t - 150.0))]  # breathing after


def test_features_are_causal(sleeping_session):
    """A feature row must never depend on frames that arrive later.

    Anything non-causal scores well offline and cannot run live, so this is the one
    property the whole bake-off rests on.
    """
    _, full, _ = sleeping_session
    _, prefix, _ = extract_session(session_by_name("sleeping"), max_frames=600)

    np.testing.assert_allclose(full[:600], prefix, rtol=1e-9, atol=1e-9)


def test_features_are_finite(sleeping_session):
    _, X, _ = sleeping_session

    assert np.isfinite(X).all()
