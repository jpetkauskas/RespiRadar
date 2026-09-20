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

## Apnea demo

No training data needed: the first ~25 s of steady breathing becomes the person's own
baseline, and the "Breathing vs. personal baseline" plot is the anomaly score. Breathing
below 30% of baseline for `--apnea-seconds` (default 10) raises the alarm.

```
uv run python main.py --simulate --hold 40:25       # rehearse: 25 s breath-hold at t=40 s
uv run python main.py --baudrate 230400             # live: lie still ~30 s, then hold your breath
```

Expect the alarm about 13 s into the hold: ~3 s for breathing strength to fall, then 10 s
of sustained absence. Hold for 20 s or more; for a snappier demo use `--apnea-seconds 6`.
Stay still during the hold - moving is motion, not apnea. The pipeline assumes the person
has left only after 60 s without any detected presence.

## Recording training data

`datalog.py` saves the raw sensor output, untouched, with a label. That output is complex
I/Q per range bin (21 bins, 0.3-1.5 m) per sweep per frame. Breathing is in the phase of
the I/Q, so this is everything a model needs. Distance is just the bin index.

```
uv run python datalog.py sleeping --subject justinas                 # 3 min (default)
uv run python datalog.py talking --subject justinas --seconds 300
uv run python datalog.py breath-hold --subject justinas              # Enter at each hold start and end
uv run python datalog.py sleeping --subject justinas --seconds 0     # until Ctrl+C
uv run python main.py --replay data/justinas_sleeping_<time>.h5      # watch a recording back
```

Suggested labels: `sleeping`, `resting`, `talking`, `active`, `breath-hold`, `empty` (nobody
there, so a model learns what absence looks like). Each run writes
`data/<subject>_<label>_<time>.h5` (Acconeer's own format, written as it goes) and adds a
row to `data/sessions.csv`. Recordings are git-ignored.

- **Rate:** the UART is the limit, not the sensor. At 230400 baud the logger uses 20 Hz
  x 8 sweeps. That is 20x the fastest breathing, and fast enough for talking and fidgeting.
  `--frame-rate`/`--sweeps` trade one for the other. The live readout warns if frames
  arrive late.
- **Countdown:** the first 5 s (`--countdown`) give you time to get in position. They stay
  in the file but come before `start_frame`, so they are unlabelled.
- **Markers:** Enter drops a timestamp, in seconds from `start_frame`. For breath-holds,
  press it when you stop breathing and again when you start. Hold for 20-40 s at a time
  and breathe normally in between.
- **Checking position:** the live readout runs the breathing pipeline. If it never gets
  past "No presence detected", the chest is out of range or the radar is pointed wrong.
- **Dry run:** `--mock` records from Acconeer's mock sensor to test the logger without
  hardware.

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

## The feature cache is committed

`data/features.npz` holds one causal feature row per frame for every recording, and every
live path needs it before a detector can fit. Building it from the `.h5` files takes ~40 s on
a laptop and **several minutes on the UNO Q's A53s** — long enough that a first run there used
to look like a hang, because it happened silently from inside `load_cached`.

So it is committed, and `python -m respiradar.dataset` reports progress per session.

**Rebuild and re-commit it whenever `SESSIONS` or `FEATURE_NAMES` changes:**

```bash
uv run python -m respiradar.dataset
git add data/features.npz
```

`dataset.cache_is_current` stamps the schema, the feature list and the session list into the
file and checks all three, so a cache that no longer matches the code rebuilds itself. It used
to rebuild only when the file was *missing*, which is how a cache predating the `wall` session
survived and surfaced as `KeyError: wall__t` from inside a detector.

Nothing you look at waits on any of this. The web scope and the LED matrix start immediately
and fit the detector on a background thread; the breathing wave and presence are live
throughout, and the apnea alarm switches on when the detector is ready. The scope says
`detector loading` while that is happening, and the matrix shows its warming-up lamp.

## The scope, over the network

`visualize.py` needs a screen and PySide6. To watch the same eight panels from a phone, a
laptop, or anything else on the network, serve them instead:

```bash
uv run python -m respiradar.webscope                         # sensor if plugged in, else simulator
uv run python -m respiradar.webscope --session breath-hold   # replay a recording, in real time
uv run python -m respiradar.webscope --simulate --hold 35,30 # fake someone who stops breathing
```

It prints the URL to open, on this machine's LAN address. `--host 127.0.0.1` keeps it local;
the default `0.0.0.0` is what makes it reachable from another device.

`GET /snapshot` returns one frame of everything as JSON, which is the quickest way to see
what the pipeline thinks without a browser:

```bash
curl -s http://localhost:8000/snapshot | python -m json.tool | head -30
```

The page is a single self-contained file (`respiradar/static/scope.html`) with no CDN, because
the board serving it may have no route to the internet and neither may the phone looking at it.

What goes over the wire is decimated: the time series drop to 256 points over the last 30 s,
the range-time heatmap sends only its newest column and the browser scrolls its own canvas,
and floats are rounded. That is ~13 KB a tick at 5 Hz. Sending the raw 20 Hz history would be
tens of megabytes a minute, nearly all of it redrawing pixels that did not change.

## The UNO Q LED matrix

The Arduino UNO Q has an 8x13 LED matrix on its STM32, driven from the Linux side over the
Router Bridge. `respiradar/ledmatrix.py` renders the breathing wave, whether anyone is there,
and the apnea alarm onto it; `respiradar/unoq.py` runs the pipeline and pushes frames.

Preview the exact same rendering anywhere, as text:

```bash
uv run python -m respiradar.unoq --selftest --sink terminal   # fixed patterns, no radar
uv run python -m respiradar.unoq --simulate --sink terminal   # breathing
uv run python -m respiradar.unoq --simulate --hold 35,30 --sink terminal   # watch it alarm
uv run python -m respiradar.unoq --session justinas-holds-3515 --sink terminal
```

    +-------------+
    |   :-:      -|    cols 0-11  the breathing wave, newest at the right
    |  .:.::     -|    col 12     the status lamp: how sure we are somebody is there
    |  :   -:    -|
    |  .    :    -|    nobody there  a dim column sweeping across
    | -     =:   -|    apnea         all 104 LEDs flashing at 2 Hz
    |:.      +::#-|
    |.       .== -|
    +-------------+

**On the board**, `unoq/` is an Arduino App Lab app: it flashes `sketch/sketch.ino` to the
STM32 and runs `python/main.py` on Linux. Bring it up in this order, so a failure can only
mean one thing:

```bash
sudo apt install -y python3-numpy python3-scipy     # enough for everything but the radar
RESPIRADAR_SELFTEST=1    # eight fixed patterns - proves the Bridge and the matrix alone
RESPIRADAR_SIMULATE=1    # the whole pipeline, no hardware
                         # then unset both, and plug the XM125 in
```

The radar needs `acconeer-exptool`, which is not in apt:
`pip install --break-system-packages 'acconeer-exptool[algo]==7.18.2'`. Do not `pip install -e .`
on the board - `pyproject.toml` pulls in PySide6 and pyqtgraph, which are a long build on
aarch64 and useless headless. `unoq/python/main.py` puts the checkout on `sys.path` itself.

The XM125 enumerates as `/dev/ttyUSB0` once the CH340 driver is built (`bash build_ch341.sh`),
and you need to be in `dialout`.

### Why the alarm takes over the whole display

The obvious design is to let the wave speak for itself: breathing stops, the trace goes flat.
Measured against these recordings, it does not. The 90th percentile of the band-passed wave
inside a labelled breath hold is 0.62-0.67 of its value outside one - 1.37 vs 2.21 mm on
`breath-hold`, 1.59 vs 2.36 on `justinas-holds-3515`, 0.74 vs 1.12 on `nishant-holds-3008`.
A third quieter is not "flat" on eight rows. So the alarm is not a subtlety in the trace: it
blinks all 104 LEDs, alternating with the wave so the evidence is still there once you look.

## Layout

- `main.py` — entry point
- `respiradar/sources.py` — `RadarConfig`, serial/replay/simulator frame sources, port detection
- `respiradar/presence.py` — is someone there, and where? Intra-frame (fast motion) and
  inter-frame (slow motion) scores, normalised by the sensor's own noise
- `respiradar/breathing.py` — phase → displacement → PSD → rate, plus the app-state machine
- `respiradar/gui.py` — live pyqtgraph dashboard
- `respiradar/server.py` + `static/index.html` — the browser dashboard (`python -m respiradar`)
- `respiradar/webscope.py` + `static/scope.html` — every scope panel, over the network
- `respiradar/ledmatrix.py` — renders the wave, presence and alarm to an 8x13 frame
- `respiradar/unoq.py` + `unoq/` — the UNO Q LED matrix app
- `respiradar/live.py` — runs a bake-off detector on a live sensor, off the frame thread

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
