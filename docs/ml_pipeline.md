# XM125 data collection and breathing ML

The implementation is in `respiradar/ml_pipeline.py`. It is independent of the
existing dashboard and does not modify `breathing.py`, `sources.py`, or `server.py`.
There are no trained weights yet. Collecting reference-labeled data and evaluating
on unseen people is required before normal/abnormal predictions have meaning.

## What the existing repository streams

```text
A121 radar in XM125 → XM125 Exploration Server firmware
  → primary UART → SparkFun CH340 USB bridge → Windows COM port
  → sources.radar_frames(): a121.Client.open(serial_port="COM6")
  → setup_session() → start_session() → get_next().frame
  → breathing.BreathingPipeline.process()
  → server.latest → /ws websocket → browser dashboard
```

`sources.py` requests 20 frames/s, 16 sweeps/frame, Profile 3, HWAAS 32,
and 21 distance points spanning approximately 0.30–1.50 m in 0.06 m steps.
One frame is a complex `[16, 21]` array. I and Q encode amplitude and phase;
this is not a stream of breathing-rate numbers or serial text lines. The
Acconeer library handles the serial protocol. The 16 sweeps are grouped inside
each frame; they are not 16 independent samples at 20 Hz.

The existing breathing pipeline subtracts an estimated static reflection,
selects one range bin, integrates phase changes into displacement, and finds an
FFT peak. Its rate alerts and range selection are explicitly placeholders.
The server reads every frame in a thread but publishes its latest dashboard
snapshot every 100 ms. Record from acquisition, not from the browser websocket.
`radar_monitor.py` is mentioned in the README but was not present in the checkout
when this module was added.

The new module uses the same sensor settings through a separate acquisition
function. It retains sensor-clock timestamps, metadata-derived distances, and
the `data_saturated`, `frame_delayed`, and `calibration_needed` flags that the
existing `Frame` interface omits. It always closes the sensor client on exit.

Hardware references: [SparkFun hardware guide](https://docs.sparkfun.com/SparkFun_Qwiic_Pulsed_Radar_Sensor_XM125/hardware_overview/),
[SparkFun setup and firmware guide](https://docs.sparkfun.com/SparkFun_Qwiic_Pulsed_Radar_Sensor_XM125/single_page/),
[Acconeer A121 Python API](https://docs.acconeer.com/en/latest/exploration_tool/api/a121.html).

## Windows setup and collection

Connect the board using a USB data cable. Install the CH340 driver if Windows
does not expose a COM port, and find its number in Device Manager. The board
needs **XM125 Exploration Server firmware** (`acc_exploration_server_a121.bin`);
the I²C presence/distance firmware does not supply this IQ stream. Close the
Exploration Tool and the existing dashboard before opening the same port.

From the repository directory in PowerShell:

```powershell
uv sync
uv run python -m respiradar.ml_pipeline record --port COM6 --seconds 120 --subject p001 --output data/p001_session01.npz
```

Use a stable pseudonymous subject ID across all recordings of the same person.
The recorder saves a raw NPZ and an empty `.labels.csv` sidecar. It buffers the
recording in RAM, so use short sessions (for example 1–5 minutes). Ctrl+C saves
captured data if at least two frames arrived. Existing files are not overwritten.
An acquisition failure raises an error; it does not currently save partial data.
Recordings can contain sensitive physiological information; store them locally
and outside version control.

To check the recording path without hardware, omit `--port`:

```powershell
uv run python -m respiradar.ml_pipeline record --seconds 35 --subject demo --output data/demo.npz
```

These samples are marked as simulated, and training rejects them unless you
explicitly pass `--allow-simulated`. The simulator generates a regular waveform;
it is useful for software checks, not evidence of classification accuracy.

Raw NPZ fields (load with `np.load(path, allow_pickle=False)`):

| Field | Shape / meaning |
|---|---|
| `iq` | complex64 `[frames, sweeps, bins]`; raw I/Q retained |
| `t` | float seconds since first sensor frame |
| `distances_m` | `[bins]`; actual metadata-derived bin distances |
| `quality` | bool `[frames, 3]`; saturation, delayed frame, calibration needed |
| `metadata` | JSON string: subject, session, source, time, clock, sensor settings |

## Label intervals using independent reference data

Labels refer to seconds since the first recorded radar frame. Synchronize the
reference measurement to this clock and record intervals in the sidecar CSV:

```csv
start_s,end_s,label
0,40,normal
45,80,motion
85,120,absent
```

Or append an interval with the CLI:

```powershell
uv run python -m respiradar.ml_pipeline label data/p001_session01.npz --start 0 --end 40 --label normal
```

Only use the example intervals if they match what actually happened. Labels are
`normal`, `abnormal`, `motion`, and `absent`. Leave uncertain periods unlabeled.
The tool rejects overlapping intervals. Edit the CSV to correct an existing label.
“Abnormal” needs a defined study target and synchronized reference annotation
(for example, independently annotated respiratory events). Do not derive training
labels from this repository's placeholder BPM thresholds. Radar motion alone does
not establish ventilation or diagnose apnea; absence and motion need separate
examples. Do not create medical events just to obtain training examples.

## Convert raw data into model inputs

```powershell
uv run python -m respiradar.ml_pipeline prepare data/p001_session01.npz data/p002_session01.npz --output data/windows.npz
```

List files explicitly; this also works in PowerShell without shell glob expansion.
Use additional recordings as needed. The defaults are 30-second windows with a
5-second stride at 20 Hz. Each usable window becomes float32 `[600, 63]`:

| Feature per distance bin | Calculation |
|---|---|
| Log amplitude | `log1p(mean(abs(IQ), sweeps))` |
| Phase step | Difference of temporally unwrapped mean-IQ phase on the uniform clock |
| Sweep coherence | `abs(mean(IQ, sweeps)) / mean(abs(IQ), sweeps)` |

Features are ordered `[bin0 amplitude, bin0 phase_step, bin0 coherence, bin1 ...]`.
All range bins remain available to the model. Phase steps retain respiratory and
larger movements; static clutter is not explicitly removed, so these are motion
features, not calibrated chest displacement. No respiratory bandpass is applied
because it could hide pauses and artifacts. Small timing jitter is resampled;
gaps over 2.5 nominal frame periods, sensor flags, nonfinite IQ, and entirely zero
signals reject the window. There is no interpolation across long dropouts.
These checks do not establish target presence or physiological signal quality.

The output has `X` shaped `[windows, time, features]`, `y` containing class indices
(`normal=0`, `abnormal=1`, `motion=2`, `absent=3`), plus subject/session/start-time
arrays and preprocessing metadata. A window receives a label only if its full
duration lies inside one annotated interval. Otherwise `y=-1`; it remains in the
export for future self-supervised experiments but is excluded from LSTM training.
Raw IQ is preserved in the original recording if a future model needs individual
sweeps. Existing output datasets are not overwritten.

## Train an LSTM baseline

PyTorch is optional; collection and preprocessing use the existing dependencies.
This command installs it in uv's run environment without changing project files:

```powershell
uv run --with torch python -m respiradar.ml_pipeline train data/windows.npz --validation-subjects p002 --epochs 20 --output models/breathing.pt
```

Training and validation **each need all four classes**, collected from disjoint
subjects. The two-subject command only illustrates syntax; a useful evaluation
needs substantially more diverse people, sessions, positions, and environments.
Every session belonging to a held-out subject stays in validation. Never randomly
split overlapping windows between training and validation. Feature means and
standard deviations are fitted on training subjects only, preserving differences
in amplitude across windows. Use a separate untouched subject cohort for a final
test; validation here selects the checkpoint and is not a final performance claim.

The baseline is a one-layer, 32-unit LSTM with temporal mean pooling and a
four-class output, trained with class-weighted cross entropy. It reports a
validation confusion matrix and balanced accuracy and saves the lowest-validation-
loss checkpoint, including normalization, class order, and window geometry.
The same transformation is used in training and prediction.

An LSTM is a small supervised baseline suitable for testing whether these inputs
contain useful information. JEPA is a possible later representation-learning
experiment using unlabeled windows and masked time/range regions; it would still
need a labeled downstream evaluation for this task. JEPA is not implemented here.
[PyTorch LSTM reference](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html).

## Predict recorded or live windows

```powershell
uv run --with torch python -m respiradar.ml_pipeline predict --model models/breathing.pt --recording data/p003_session01.npz
uv run --with torch python -m respiradar.ml_pipeline predict --model models/breathing.pt --port COM6
```

Live prediction warms up for about 30 seconds, then emits JSON every 5 seconds.
`status` is `normal`, `abnormal`, or `unknown`. Predicted motion/absence, rejected
windows, and top softmax scores below `--threshold` (default 0.7) yield `unknown`.
Scores are **uncalibrated model outputs**, not medical probabilities or a reliable
out-of-distribution detector. Sensor quality gates cannot prevent all false
normal/abnormal results. A pause shorter than the window may be diluted by temporal
pooling; this baseline is not an event-duration detector or an emergency alert.
Validate event-level sensitivity, false alarms, threshold calibration, and behavior
on unseen people before considering any health-facing application.

The existing dashboard remains separate. Its placeholder alerts are not replaced
by the new classifier. [Acconeer's breathing reference application](https://docs.acconeer.com/en/latest/ref_apps/a121/breathing.html)
is also useful for comparing displacement/rate estimates and handling large motion.

Run checks with `uv run python -m unittest discover -s tests`. To include the
optional training/checkpoint test, use `uv run --with torch python -m unittest discover -s tests`.
