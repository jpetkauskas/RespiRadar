"""What the LED matrix is allowed to show.

The matrix is the only output on the UNO Q - there is no screen to check it against - so the
properties that make it readable are pinned here rather than looked at once and trusted.
"""

import numpy as np
import pytest

from respiradar.ledmatrix import (
    COLS,
    MAX,
    ROWS,
    STATUS_COL,
    WAVE_COLS,
    Display,
    MatrixRenderer,
    MatrixState,
    to_text,
)


def _drive(renderer, wave, seconds, fs=20.0, t0=0.0, **state):
    """Run a wave through the renderer for `seconds` and return the final frame."""
    frame = None
    for i in range(int(seconds * fs)):
        t = t0 + i / fs
        value = wave(t) if callable(wave) else wave
        frame = renderer.update(MatrixState(t=t, wave_mm=value, **state))
    return frame


def _breathing(amplitude=2.2, bpm=14.0):
    return lambda t: amplitude * np.sin(2 * np.pi * bpm / 60 * t)


LIVE = dict(present=True, alarm=False, settling=False, presence=1.4)


# -- the frame contract -------------------------------------------------


@pytest.mark.parametrize(
    "state",
    [
        dict(present=False, alarm=False),
        dict(present=True, alarm=False),
        dict(present=True, alarm=True),
        dict(present=True, alarm=False, settling=True),
    ],
)
def test_every_frame_fits_the_hardware(state):
    """8x13, whole numbers, 0..7. The sketch runs setGrayscaleBits(3); anything above 7 is
    not a brighter pixel, it is a wrong one."""
    renderer = MatrixRenderer()
    for seconds in (1, 10):
        frame = _drive(renderer, _breathing(), seconds, **state)
        assert frame.shape == (ROWS, COLS)
        assert frame.dtype == np.uint8
        assert frame.min() >= 0 and frame.max() <= MAX


def test_the_frame_is_text_shaped_for_debugging():
    frame = _drive(MatrixRenderer(), _breathing(), 10, **LIVE)
    lines = to_text(frame).splitlines()
    assert len(lines) == ROWS
    assert all(len(line) == COLS for line in lines)


# -- which picture gets drawn -------------------------------------------


def test_absence_outranks_everything():
    """An empty room and a held breath are the same observation to a motion sensor. If the
    gate ever regresses and the detector alarms at a wall, the matrix still must not."""
    state = MatrixState(t=0.0, wave_mm=0.0, present=False, alarm=True)
    assert state.display is Display.ABSENT


def test_an_alarm_outranks_still_warming_up():
    state = MatrixState(t=0.0, wave_mm=0.0, present=True, alarm=True, settling=True)
    assert state.display is Display.APNEA


def test_start_up_does_not_claim_the_room_is_empty():
    """Presence is a 60 s trailing median. For the first seconds it reads low because
    nothing has been measured yet, and flashing the empty-room sweep on every start would be
    a lie about what the sensor knows."""
    state = MatrixState(t=0.1, wave_mm=0.0, present=False, alarm=False, settling=True)
    assert state.display is Display.SETTLING


def test_an_empty_room_still_wins_over_an_alarm_during_warm_up():
    """The one case the rule above must not swallow: nobody there, and a detector that has
    not settled deciding to alarm about it."""
    state = MatrixState(t=0.1, wave_mm=0.0, present=False, alarm=True, settling=True)
    assert state.display is Display.ABSENT


def test_nobody_there_lights_almost_nothing():
    """The idle picture has to be obviously idle: a sweep, not a display full of data."""
    renderer = MatrixRenderer()
    for seconds in range(1, 6):
        frame = _drive(renderer, 0.02, seconds, present=False, alarm=False)
        assert np.count_nonzero(frame) <= 2 * ROWS
        assert frame.max() < MAX


def test_the_sweep_actually_moves():
    renderer = MatrixRenderer()
    seen = {
        to_text(_drive(renderer, 0.02, 0.05, t0=t, present=False, alarm=False))
        for t in (0.0, 0.4, 0.8, 1.2)
    }
    assert len(seen) > 1


# -- the alarm ----------------------------------------------------------


def test_the_alarm_blinks_the_whole_matrix():
    """Measured on the recordings, the wave inside a hold is only ~0.65x the breathing
    around it. That is not visibly flat on eight rows, so noticing an apnea must not depend
    on reading the trace: at some point in every cycle, every LED is on."""
    renderer = MatrixRenderer()
    _drive(renderer, _breathing(), 30, **LIVE)

    frames = [
        renderer.update(MatrixState(t=30 + i / 20, wave_mm=0.5, present=True, alarm=True))
        for i in range(40)  # two seconds, four blink periods
    ]
    assert any((frame == MAX).all() for frame in frames), "no full-brightness flash"
    assert any(not (frame == MAX).all() for frame in frames), "never lets the wave show"


def test_breathing_never_lights_the_whole_matrix():
    """Otherwise the alarm's flash would not be distinguishable from ordinary operation."""
    renderer = MatrixRenderer()
    for seconds in (5, 15, 30):
        frame = _drive(renderer, _breathing(amplitude=8.0), seconds, **LIVE)
        assert not (frame == MAX).all()


# -- the wave -----------------------------------------------------------


def test_the_wave_is_drawn_where_the_chest_is():
    """Row 0 is the top. A chest moving out should draw above the zero line, not below."""
    renderer = MatrixRenderer()
    _drive(renderer, _breathing(), 30, **LIVE)

    def centre_of_mass(frame):
        column = frame[:, WAVE_COLS - 1].astype(float)
        return float(np.average(np.arange(ROWS), weights=column)) if column.sum() else 3.5

    up = centre_of_mass(_drive(renderer, 2.5, 1.0, t0=30, **LIVE))
    down = centre_of_mass(_drive(renderer, -2.5, 1.0, t0=40, **LIVE))
    assert up < 3.5 < down


def test_a_quiet_stretch_is_not_stretched_back_to_full_scale():
    """The scale ratchets, and this is why.

    `dataset.FeatureExtractor` avoids a ratcheting reference because one reads ordinary
    breathing as apnea. A display has the opposite problem: renormalising during a hold would
    amplify whatever noise is left to fill all eight rows and draw a healthy wave out of
    nothing. Breathing, then a hold, must look smaller - not the same.
    """
    renderer = MatrixRenderer()
    _drive(renderer, _breathing(amplitude=3.0), 40, **LIVE)
    breathing = _drive(renderer, _breathing(amplitude=3.0), 10, t0=40, **LIVE)
    held = _drive(renderer, _breathing(amplitude=0.9), 25, t0=50, **LIVE)

    def spread(frame):
        rows, _ = np.nonzero(frame[:, :WAVE_COLS])
        return rows.max() - rows.min() if len(rows) else 0

    assert spread(held) < spread(breathing)


def test_noise_is_not_amplified_into_a_breathing_person():
    """With no floor on the scale, a still chest's residual noise would be auto-ranged up to
    fill the display - the most dangerous thing this screen could do."""
    renderer = MatrixRenderer()
    frame = _drive(renderer, lambda t: 0.03 * np.sin(2 * np.pi * 0.3 * t), 60, **LIVE)
    rows, _ = np.nonzero(frame[:, :WAVE_COLS])
    assert rows.max() - rows.min() <= 2


# -- the status lamp ----------------------------------------------------


def test_the_status_lamp_grows_with_confidence_that_someone_is_there():
    def height(presence):
        renderer = MatrixRenderer()
        frame = _drive(renderer, _breathing(), 10, present=True, alarm=False, presence=presence)
        return np.count_nonzero(frame[:, STATUS_COL])

    assert height(0.85) < height(1.2) < height(2.0)
    assert height(2.0) == ROWS


def test_the_wave_never_writes_into_the_status_column():
    """They mean different things; overlapping them would make both unreadable."""
    renderer = MatrixRenderer()
    frame = _drive(renderer, _breathing(amplitude=20.0), 30, present=True, alarm=False,
                   presence=0.8)
    assert not frame[:, STATUS_COL].any()
