"""Render the breathing signal, presence and the apnea alarm onto the UNO Q's LED matrix.

The board has an 8x13 blue matrix (104 LEDs) wired to the STM32, driven from Linux over the
Router Bridge. This module does no I/O: it turns the pipeline's state into an (8, 13) array of
brightness levels, so the layout can be unit-tested on any machine and previewed in a terminal.
`respiradar/unoq.py` is what pushes the frames to the board.

LAYOUT

    col  0 .. 11   the breathing wave, scrolling left to right, newest at column 11
    col       12   the status lamp: how sure we are that somebody is there

    +-------------------------+---+
    | . . . . . . # # . . . . | . |   row 0   wave high (chest out, inhale)
    | . . . . . # . . # . . . | . |
    | . . . . # . . . . # . . | . |
    | . . . # . . . . . . # . | # |   rows 3/4 are the wave's zero line
    | . . # . . . . . . . . # | # |
    | . # . . . . . . . . . . | # |
    | # . . . . . . . . . . . | # |
    | . . . . . . . . . . . . | # |   row 7   wave low (exhale)
    +-------------------------+---+
      oldest --------> newest   status

WHY THE ALARM TAKES OVER THE WHOLE DISPLAY

The obvious design is to let the wave speak for itself: breathing stops, the trace goes flat,
you can see it. Measured against the recordings, that is not what happens. The 90th percentile
of the band-passed wave inside a labelled breath hold is 0.62-0.67 of its value outside one
(`breath-hold` 1.37 vs 2.21 mm, `justinas-holds-3515` 1.59 vs 2.36, `nishant-holds-3008` 0.74
vs 1.12). A third quieter is not "flat" on eight rows - it is a wave that looks slightly
smaller, which nobody will notice from a bed at 3 a.m.

So the alarm is not a subtlety in the trace. It blinks the entire matrix. The wave is still
shown on alternate half-seconds, because the evidence is worth seeing once you are looking,
but the thing that gets you looking is 104 LEDs at full brightness.

WHY THE WAVE'S SCALE RATCHETS

`dataset.FeatureExtractor` goes to some trouble to avoid a ratcheting reference, because a
baseline that rises to a subject's best breathing and never falls reads ordinary breathing as
apnea. Here the opposite is correct. A display that renormalised during a hold would stretch
whatever noise is left to fill all eight rows, drawing a healthy-looking wave out of nothing.
Holding the scale at the recent peak, and letting it decay only over ~80 s, means a hold is
drawn smaller than the breathing before it. The scale is a display convenience and must never
be fed back into a detector.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from enum import Enum

import numpy as np

ROWS = 8
COLS = 13
LEVELS = 8  # 0..7, matching the sketch's setGrayscaleBits(3)
MAX = LEVELS - 1

WAVE_COLS = 12  # columns 0..11
STATUS_COL = 12

COLUMN_S = 0.4  # seconds of breathing per column -> 4.8 s on screen, about one breath
SCALE_S = 20.0  # window the wave's full scale is measured over
SCALE_DECAY = 0.995  # per column; ~80 s to fall by 1/e, so a 30 s hold barely moves it
WAVE_FLOOR_MM = 0.8  # never amplify quieter than this: stops noise filling the display
WAVE_HEADROOM = 1.1  # a typical breath nearly fills the display; bigger ones clip

BLINK_S = 0.25  # apnea: half a period, so the matrix flashes at 2 Hz
SWEEP_S = 0.2  # absent: seconds per column of the searching sweep

TRACE = MAX  # the wave's own sample
CONNECTOR = 2  # the segment joining one column's sample to the next
OLDEST_FADE = 0.35  # the left edge of the wave is this fraction as bright as the right


class Display(Enum):
    """What the matrix is showing. Not the same as `breathing.AppState`.

    That enum describes the breathing pipeline's internal progress. This one describes a
    picture, and the two do not map one to one: DETERMINE_DISTANCE and
    ESTIMATE_BREATHING_RATE are both just "draw the wave" here.
    """

    ABSENT = "nobody there"
    SETTLING = "warming up"
    BREATHING = "breathing"
    APNEA = "APNEA"


@dataclass
class MatrixState:
    """Everything the matrix needs to know, for one frame."""

    t: float  # seconds since the source started, for the animations
    wave_mm: float  # the band-passed chest displacement
    present: bool  # the 60 s presence median cleared PRESENCE_THRESHOLD
    alarm: bool  # the detector is alarming
    settling: bool = False  # features are still provisional
    presence: float = 1.0  # presence activity / PRESENCE_THRESHOLD; ~1.0 is the boundary

    @property
    def display(self) -> Display:
        # Settling comes first, but only while nothing is alarming. Presence is a 60 s
        # trailing median, so in the first seconds it reads low because nothing has been
        # measured yet, not because the room is empty - and showing the empty-room sweep for
        # half a second every time the app starts is a lie about what the sensor knows.
        if self.settling and not self.alarm:
            return Display.SETTLING
        # Then absence, which outranks the alarm: an empty room and a held breath are the
        # same observation to a motion sensor. The detector is already gated; the display
        # does not depend on that staying true.
        if not self.present:
            return Display.ABSENT
        if self.alarm:
            return Display.APNEA
        return Display.BREATHING


class MatrixRenderer:
    """Turns a stream of `MatrixState` into (8, 13) brightness frames.

    Call `update` as often as you like: the wave advances one column every `column_s`
    regardless, while the blink and sweep animations run at whatever rate you call at.
    """

    def __init__(self, column_s: float = COLUMN_S) -> None:
        self.column_s = column_s
        self.columns: deque[float] = deque([0.0] * WAVE_COLS, maxlen=WAVE_COLS)
        self._recent: deque[float] = deque(maxlen=max(1, int(SCALE_S / column_s)))
        self._scale = WAVE_FLOOR_MM
        self._next_column: float | None = None
        self._bucket_peak = 0.0  # largest excursion since the last column was pushed

    # -- the wave -------------------------------------------------------
    def _advance(self, state: MatrixState) -> None:
        """Accumulate one sample, and push a column when enough time has passed.

        The column takes the bucket's largest excursion rather than its mean. Averaging eight
        samples of a wave would shrink every peak towards zero, which on a display whose whole
        job is "how big is the breathing" is the one distortion we cannot afford.
        """
        if abs(state.wave_mm) > abs(self._bucket_peak):
            self._bucket_peak = state.wave_mm
        if self._next_column is None:
            self._next_column = state.t + self.column_s
            return
        if state.t < self._next_column:
            return

        # A gap longer than one column (a stalled sensor, a replay jump) pushes one column and
        # resyncs, rather than spinning out a burst of identical columns.
        self._next_column = max(state.t + self.column_s, self._next_column + self.column_s)
        self.columns.append(self._bucket_peak)
        self._recent.append(abs(self._bucket_peak))
        self._bucket_peak = 0.0

        peak = float(np.percentile(self._recent, 90)) if self._recent else 0.0
        self._scale = max(WAVE_FLOOR_MM, peak, self._scale * SCALE_DECAY)

    def _row_of(self, value: float) -> float:
        """Where a wave value sits, as a float row. Row 0 is the top, 3.5 is the zero line."""
        span = self._scale * WAVE_HEADROOM
        unit = float(np.clip(value / span, -1.0, 1.0)) if span > 0 else 0.0
        return 3.5 - unit * 3.5

    def _draw_wave(self, frame: np.ndarray) -> None:
        rows = [self._row_of(v) for v in self.columns]
        for col, row in enumerate(rows):
            fade = OLDEST_FADE + (1.0 - OLDEST_FADE) * (col / max(WAVE_COLS - 1, 1))

            # Join this sample to the previous one so the trace reads as a line rather than a
            # scatter of dots. Dim, because the sample itself is what carries the value.
            if col > 0:
                lo, hi = sorted((rows[col - 1], row))
                for r in range(int(np.ceil(lo)), int(np.floor(hi)) + 1):
                    if 0 <= r < ROWS:
                        _lighten(frame, r, col, CONNECTOR * fade)

            # Split the sample across the two rows it falls between, so a wave moving half a
            # pixel still shows as movement. Eight rows is not many; this buys some back.
            top = int(np.floor(row))
            frac = row - top
            if 0 <= top < ROWS:
                _lighten(frame, top, col, TRACE * (1.0 - frac) * fade)
            if 0 <= top + 1 < ROWS:
                _lighten(frame, top + 1, col, TRACE * frac * fade)

    # -- the status lamp ------------------------------------------------
    def _draw_status(self, frame: np.ndarray, state: MatrixState) -> None:
        if state.display is Display.SETTLING:
            # A slow pulse on the zero line: something is happening, nothing is decided yet.
            level = 1 + 2 * (0.5 + 0.5 * np.sin(2 * np.pi * state.t / 2.0))
            _lighten(frame, 3, STATUS_COL, level)
            _lighten(frame, 4, STATUS_COL, level)
            return

        # A bar climbing from the bottom: how far the 60 s presence median sits above the
        # threshold. A wall sits just under it, an occupied room well over. Two lit pixels
        # means "only just somebody"; a full column means there is no doubt.
        height = int(round(float(np.clip((state.presence - 0.8) / 0.8, 0.0, 1.0)) * ROWS))
        for r in range(ROWS - height, ROWS):
            _lighten(frame, r, STATUS_COL, 3)

    # -- modes ----------------------------------------------------------
    def _sweep(self, frame: np.ndarray, t: float) -> None:
        """Nobody there: a dim column walking across, like a radar sweep. Armed, not alarmed."""
        head = int(t / SWEEP_S) % COLS
        for r in range(ROWS):
            _lighten(frame, r, head, 2)
            _lighten(frame, r, (head - 1) % COLS, 1)

    def update(self, state: MatrixState) -> np.ndarray:
        """One (8, 13) uint8 frame, values 0..7."""
        self._advance(state)
        frame = np.zeros((ROWS, COLS), dtype=np.uint8)
        mode = state.display

        if mode is Display.ABSENT:
            self._sweep(frame, state.t)
            return frame

        # Alarm: every other half second the whole matrix goes to full brightness. The other
        # half second shows the wave, so the evidence is there once you look, but nothing
        # about noticing it depends on reading an eight-row trace.
        if mode is Display.APNEA and int(state.t / BLINK_S) % 2 == 0:
            frame[:, :] = MAX
            return frame

        self._draw_wave(frame)
        self._draw_status(frame, state)
        return frame


def _lighten(frame: np.ndarray, row: int, col: int, level: float) -> None:
    """Brightest-wins compositing, clamped to the matrix's range.

    Adding overlapping strokes would let a connector crossing a trace read brighter than the
    trace itself, which inverts the thing the brightness is supposed to mean.
    """
    value = int(np.clip(round(level), 0, MAX))
    if value > frame[row, col]:
        frame[row, col] = value


def to_text(frame: np.ndarray, palette: str = " .:-=+*#") -> str:
    """The frame as text, for terminals and test failures. One character per LED."""
    return "\n".join("".join(palette[int(v)] for v in row) for row in frame)
