# RespiRadar

HackMIT 2026, healthcare track — Nishant Vikramaditya, Vishnu Mangipudi, Justinas Petkauskas.

Ambient nighttime breathing-rate monitoring with the Acconeer A121 (SparkFun XM125 breakout).
Runs on macOS, which Acconeer's own Exploration Tool does not officially support.

## Run it

### Windows (uv)

`uv` reads `pyproject.toml`, picks a compatible Python, builds the environment, and runs —
no venv to create or activate.

```
uv run python main.py
```

That's the whole setup. First run resolves dependencies and refreshes `uv.lock`, which takes
a minute; after that it's instant. Commit the refreshed lock so everyone matches.

```
uv run python main.py --list-ports          # find the COM port
uv run python main.py                       # auto-detects the board by its CH340 USB id
uv run python main.py --port COM6
uv run python main.py --baudrate 115200     # if 230400 proves unstable
uv run python main.py --simulate --bpm 18   # no hardware
uv run python main.py --replay tests/data/breathing-sitting.h5
uv run --extra dev pytest tests/ -q         # the test suite
```

**Close the Acconeer Exploration Tool first** — only one program can hold the port.

230400 is the default, being the rate this board has proven stable at. `main.py` sizes the
sensor config to fit it automatically (20 Hz / 8 sweeps); see the table below.

### macOS (venv)

```
python3.12 -m venv .venv
./.venv/bin/pip install -e .
./.venv/bin/python main.py
```

Python 3.10-3.13. **Not 3.14** — `libusb-package` has no 3.14 wheel yet. This bites on both
platforms; with uv it is handled for you, with plain `venv` you must invoke `python3.12`
explicitly.

`main.py` looks for the sensor and falls back to the simulator if it can't find one, so it
does something useful before the hardware works.

```
python main.py                      # auto-detect, or simulate
python main.py --simulate --bpm 18  # no hardware
python main.py --replay tests/data/breathing-sitting.h5
python main.py --list-ports
python main.py --port /dev/cu.wchusbserial1420 --baudrate 115200  # slower, safer
```

## macOS setup

The XM125 must be running Acconeer's Exploration Server firmware, and nothing else may hold
the port (close the Exploration Tool first).

1. **Install the CH34x driver** from [WCH](https://www.wch-ic.com/downloads/CH34XSER_MAC_ZIP.html)
   and **reboot**. macOS's built-in USB-to-UART driver is unreliable with this board; this is
   the single reason Acconeer calls macOS unsupported. A `/dev/cu.wchusbserial*` appears after.
2. `python main.py --list-ports` to confirm it's there.
3. The default is 230400. If the link misbehaves, drop to `--baudrate 115200`. Acconeer's
   own auto-baud of 2 Mbps exceeds what this board does reliably, and that failure looks like
   corrupted data rather than a clean error.

Symptom when the driver is missing: `LinkError: recv timeout`. We confirmed the board is
completely silent in that state — no bytes at 9600 through 2000000 baud, with flow control on
or off, DTR/RTS high or low, or after a reset pulse. The same board answers immediately on
Windows, so it is the driver, not the firmware. A port named `/dev/cu.usbserial-*` means
Apple's driver is still in charge; WCH's gives you `/dev/cu.wchusbserial-*`.

### Baud rate and frame rate are one decision

Frames are complex int16, so a config costs
`num_points × sweeps_per_frame × 4 bytes × 10 bits × frame_rate`. The reference-app defaults
(21 points, 16 sweeps, 20 Hz) need **269 kbit/s** — more than 230400 baud carries.

`main.py` measures this and backs the config off to fit, preferring to cut sweeps over frame
rate, because breathing needs the sample rate more than it needs per-frame averaging:

| Baud | Auto-selected config | Throughput |
|---|---|---|
| 115200 | 20 Hz, 4 sweeps | 67 kbit/s |
| 230400 | 20 Hz, 8 sweeps | 134 kbit/s |
| 460800 | 20 Hz, 16 sweeps | 269 kbit/s |

Override with `--frame-rate` and `--sweeps`. The status bar reports the share of frames the
sensor flagged as delayed — if that's above zero, the link is oversubscribed.

If connecting hangs, try `--no-flow-control` (RTS/CTS support varies by USB-serial bridge).

## Layout

- `main.py` — entry point
- `respiradar/sources.py` — `RadarConfig`, serial/replay/simulator frame sources, port detection
- `respiradar/presence.py` — is someone there, and where? Intra-frame (fast motion) and
  inter-frame (slow motion) scores, normalised by the sensor's own noise
- `respiradar/breathing.py` — phase → displacement → PSD → rate, plus the app-state machine
- `respiradar/gui.py` — live pyqtgraph dashboard
- `respiradar/server.py` + `static/index.html` — the browser dashboard (`python -m respiradar`)

## Accuracy

The pipeline is our own, not a wrapper around Acconeer's. It's validated against Acconeer's
recorded session of a real sitting person, with their own per-frame output as ground truth:

| | Mean | Range |
|---|---|---|
| Acconeer reference app | 18.36 bpm | 17.40–18.82 |
| RespiRadar | 18.57 bpm | 17.25–19.50 |

```
uv run --extra dev pytest tests/ -q                                  # Windows
QT_QPA_PLATFORM=offscreen ./.venv/bin/pytest tests/ -q              # macOS
```

`tests/data/` contains three recordings from
[acconeer-python-exploration](https://github.com/acconeer/acconeer-python-exploration),
redistributed under the Clear BSD License, Copyright (c) 2018–2022 Acconeer AB.

Note that despite its filename, `breathing-sitting-no-presence.h5` is **not** an empty room —
its embedded config reads `use_presence_processor: false`. It's the same sitting person with
the presence processor disabled. The empty-room test uses the simulator.
