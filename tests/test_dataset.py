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


def test_ratio_features_never_explode():
    """The baseline used to latch onto a filter transient, sending ratios to ~2.6e8.

    Any model that scales its inputs is wrecked by an outlier that large, and the bad values
    persisted well past the scoring warmup, so this is checked on every session.
    """
    from respiradar.dataset import SESSIONS, load_cached

    ratio = FEATURE_NAMES.index("ratio_4s")
    for session in SESSIONS:
        _, X, _ = load_cached(session.name)
        assert X[:, ratio].max() <= 10.0, session.name


def test_every_subject_is_represented():
    from respiradar.dataset import subjects

    # "wall" is an empty scene rather than a person, and is deliberately its own subject so
    # leave-one-subject-out gives empty-room rejection a fold of its own.
    assert subjects() == ["justinas", "nishant", "vishnu", "wall"]


def test_folds_never_train_on_the_subject_they_test():
    """Leave-one-subject-out is the only split that predicts behaviour on a new person."""
    from respiradar.bakeoff import folds
    from respiradar.dataset import SESSIONS

    subject_of = {s.name: s.subject for s in SESSIONS}
    for train, test in folds():
        trained = {subject_of[c.name.split("[")[0]] for c in train}
        tested = {subject_of[c.name.split("[")[0]] for c in test}

        assert not (trained & tested)


def test_every_subject_is_held_out_by_some_fold():
    """Including subjects with no holds.

    A subject who only ever breathes normally still tests the property that matters most:
    staying quiet on a body the detector has never seen. Keeping such a subject permanently
    in the training set hides their false alarms entirely - two entries in the bake-off
    reported zero false alarms and actually had two and nine.
    """
    from respiradar.bakeoff import folds
    from respiradar.dataset import SESSIONS, subjects

    subject_of = {s.name: s.subject for s in SESSIONS}
    held_out = {
        subject_of[c.name.split("[")[0]] for _, test in folds() for c in test
    }

    assert held_out == set(subjects())


def test_new_ratio_features_never_explode():
    """Same pin as `test_ratio_features_never_explode`, for the quantile-referenced ratios."""
    from respiradar.dataset import SESSIONS, load_cached

    for feature in ("ratio_4s_q", "ratio_8s_q"):
        column = FEATURE_NAMES.index(feature)
        for session in SESSIONS:
            _, X, _ = load_cached(session.name)
            assert X[:, column].max() <= 10.0, (session.name, feature)
            assert X[:, column].min() >= 0.0, (session.name, feature)


def test_the_quantile_reference_reads_less_phantom_apnea_than_the_ratchet():
    """The point of the trailing low quantile, stated as a measurement.

    The asymmetric-EMA reference ratchets up to a subject's *best* breathing, so ordinary
    shallow breathing afterwards sits below it and reads as a hold that is not happening.
    Counting how often each reference does that, over the negatives only, is the whole
    argument for the quantile and does not depend on any detector's tuning.
    """
    from respiradar.dataset import SESSIONS, load_cached

    ratchet = FEATURE_NAMES.index("ratio_4s")
    quantile = FEATURE_NAMES.index("ratio_4s_q")
    worse_for_quantile = []
    for session in SESSIONS:
        t, X, y = load_cached(session.name)
        scored = (t >= 25.0) & ~y  # negatives only, past the scoring warmup
        phantom = [float(np.mean(X[scored, c] < 0.7)) for c in (ratchet, quantile)]
        if phantom[1] > phantom[0] + 1e-9:
            worse_for_quantile.append((session.name, phantom))

    assert not worse_for_quantile


def test_rms_is_reported_before_its_window_has_filled():
    """Two of four labelled holds start before 16 s.

    Reporting 0.0 for an unfilled window is not "no data": 0.0 is the exact value a perfect
    breath hold produces, so the early seconds used to look like the deepest apnea in the
    recording and then jump.
    """
    t, X, _ = extract_session(session_by_name("sleeping"), max_frames=200)  # 10 s at 20 Hz

    rms_8s = X[:, FEATURE_NAMES.index("rms_8s")]
    early = t >= 3.0
    assert (rms_8s[early] > 0).all()


def test_warm_reports_how_provisional_the_row_is():
    t, X, _ = extract_session(session_by_name("sleeping"), max_frames=600)

    warm = X[:, FEATURE_NAMES.index("warm")]
    history = X[:, FEATURE_NAMES.index("seconds_of_history")]

    assert (np.diff(history) > 0).all()          # strictly increasing, never rewound
    assert (np.diff(warm) >= 0).all()            # and warm only ever ramps up
    assert warm[t < 8.0].max() < 1.0             # provisional while the buffers fill
    assert warm[t > 20.0].min() == 1.0           # settled once they have
    assert np.allclose(history, t + 1 / 20.0)    # one frame of history per frame seen
