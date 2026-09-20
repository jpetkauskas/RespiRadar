# How RespiRadar works

From radio waves to an apnea alarm, then a panel-by-panel guide to the scope.

---

## Part 1 — The physics

### Range bins are "when you listen", not something computed

The A121 is a **pulsed coherent** radar, not FMCW. It emits a very short 60 GHz pulse and
then samples the echo at chosen *delays*. Light covers about 30 cm per nanosecond, so a
delay picks out a distance:

```
distance = c × delay / 2        (÷2 because the signal goes out AND back)
```

One range bin *is* one sampling delay. The chip's delay quantum is 2.5 mm
(`BASE_STEP_M`), which is half the 5 mm wavelength — the natural round-trip step.

This project's configuration (`sources.py`):

```
start_point = 120   →  120 × 2.5 mm =  0.30 m
step_length =  24   →   24 × 2.5 mm =  6 cm between bins
num_points  =  21   →  0.30 m … 1.50 m
```

So the sensor listens at 21 moments after each pulse. Range is a **hardware setting**, not
something recovered from the signal — unlike FMCW radar, where range falls out of an FFT
over beat frequency.

### I and Q are amplitude and phase

At each delay the return is mixed against the transmitter's own oscillator twice — once in
phase, once shifted 90°. Those two products are **I** and **Q**, which together describe
the echo as an arrow:

```
I = x-coordinate          amplitude = √(I² + Q²)      arrow length
Q = y-coordinate          phase     = atan2(Q, I)     arrow angle
```

Same arrow, two notations. In numpy: `np.abs(z)` and `np.angle(z)`.

"Coherent" means the receiver knows the transmitter's phase. That is the whole game.

### Why phase gives sub-millimetre precision

The wave has a crest every 5 mm all the way out and back. The echo lands somewhere in that
repeating pattern — on a crest, in a dip, or between. That landing point is the phase.

Breathe in, chest moves 1 mm closer, and the round trip shortens by **2 mm** — a large
fraction of a 5 mm cycle, so the echo returns at a visibly different point in the wave.

```
chest moves 2.5 mm  →  round trip changes 5 mm  →  one FULL cycle
chest moves 0.5 mm  →  round trip changes 1 mm  →  a fifth of a cycle
```

**This is why a sensor that cannot resolve anything smaller than 6 cm still detects half a
millimetre of chest movement.** It is not measuring size; it is measuring how far through
the ripple pattern the echo landed.

### The two halves carry different information

| | what it tells you | used for |
|---|---|---|
| **amplitude** | how much stuff is there | finding the chest, BODY SNR |
| **phase** | distance, to a fraction of a mm | breathing — the entire detection |

Amplitude is coarse and barely changes when you breathe; your chest does not become more
reflective. Phase is the measurement.

### The failure mode this creates

An empty bin does not return nothing. It returns a tiny random arrow that jitters on its
own — and **phase jitter scales as 1/SNR**, so the weaker the echo, the wilder the arrow
flails.

Rank bins by "which is moving most" and **the emptiest bin wins.** This is not hypothetical:
on three demo recordings the tracker settled at 1.32–1.44 m where SNR is 0.4, while the
subject lay at 0.54 m with SNR 2.8–7.2. Noise has a strongest frequency in the breathing
band, so the scope reported a breathing rate right through a held breath — and noise never
stops, so no hold was ever visible.

**That is what the BODY SNR readout exists for.** See Part 3.

---

## Part 2 — The algorithm

### Stage 1: one frame in

`(8 sweeps × 21 bins)` of complex IQ, 20 times a second.

The **noise floor** comes first. A chest cannot move meaningfully in the ~1 ms between
sweeps, so any disagreement *between sweeps inside one frame* is noise:

```python
noise = |diff(sweeps)|.mean(axis=0) / √2      # per bin
```

Everything downstream divides by this, which is why thresholds mean the same thing
regardless of gain. The 8 sweeps then average into one clean sweep of 21 complex numbers.

### Stage 2: which bin is the chest

For *every* bin, continuously: unwrap phase → millimetres → band-pass → how much is it
moving. Pick the largest, smoothed across 3 neighbours (a torso lights up several adjacent
bins; isolated single-bin spikes are artefacts, and the pulse is physically longer than the
6 cm bin spacing so a real target always bleeds across bins).

The critical detail is that this is averaged with a **90 second time constant**. Choosing
the liveliest bin *right now* means that the moment breathing stops, the selector goes
hunting for whatever else is moving and re-points — **an apnea could never be observed,
because the tracker would run away from it.** It has to be too sluggish to react to the
thing it is trying to detect.

### Stage 3: the breathing trace

```
phase → unwrap → × 0.398 mm/radian → displacement in mm      [panel 5]
      → band-pass 0.18–0.55 Hz (11–33 breaths/min)           [panel 6]
```

The band matters enormously. Raw displacement is dominated by slow drift far *larger* than
the breathing itself. Open the band wider and drift wins — the dominant period reads 6–10
bpm, which is not breathing.

### Stage 4: features, 20 per second

The one that matters is **`rms_4s`** — how much the chest moved over the last 4 seconds.

Alone it is meaningless: 2.0 could be deep breathing or a twitchy sensor. So it is always
measured against **your own recent normal**, two different ways:

- **`ref_ratchet`** — rises fast, falls ~40× slower. Refuses to follow a hold downward, so
  the step stays visible for the hold's whole length. Liability on negatives: one burst of
  movement leaves "normal" inflated for minutes.
- **`ref_q25`** — trailing 25th percentile. Cannot ratchet because it forgets. Its level
  sits in a narrow band across subjects where the ratchet's wanders from 0.14 to 0.79.

Both earn a place; measured on 13 holds, neither wins alone.

Plus `intra` / `inter` (fast and slow motion scores from the presence detector, normalised
by the noise floor) and `disp_std_4s` (unfiltered motion, which catches gross movement).

### Stage 5: the detector — CUSUM

Apnea is not an outlier, it is a **regime change**: chest motion sits at one level while you
breathe and a lower one while you do not. Declaring the second regime as soon as the
evidence justifies it is exactly what **Page's CUSUM** is optimal for.

Each frame, for each chart:

```python
statistic = log(your normal) − log(rms_4s now)    # > 0 means below your baseline
increment = min(statistic − deadband, cap)
total     = max(0, total × decay + increment)
if total ≥ threshold:  ALARM
```

Every piece earns its place:

- **deadband** — without it a chronically shallow breather accumulates a small positive
  drift for a minute and trips the chart
- **cap** — without it a momentary loss of radar lock counts for more than a real hold
- **decay** (0.999) — makes it a *rate* detector: a small persistent drift saturates at a
  low level and never reaches threshold, while a genuine collapse blows straight through
- **threshold** — 200, 30, 120, 30 across the four charts

Because increments accumulate, it needs *sustained* evidence. That is the ~15 s latency:
not slowness by accident, but certainty being bought.

**Four charts in parallel**, because the post-change distribution is not known in advance —
one tuned for a deep fast collapse, another for a shallow long one. Any firing raises the
alarm. Since each chart is individually free of false alarms, so is their union.

### Stage 6: two vetoes

**Motion gate.** Talking and rolling over produce their own energy dips. Those frames are
not merely skipped — `total = 0.0`, discarding all accumulated evidence. So no amount of
fidgeting can accumulate into an alarm.

> Known sharp edge: holding your breath *raises* `disp_std_4s` (bracing and sway, not
> stillness — measured 1.34 breathing vs 1.64 holding). `disp_k` is deliberately loose for
> this reason; at its original value the gate wiped the chart on a third of a hold's frames
> and the alarm covered only 67% of the hold in flickering chunks.

**Presence gate.** An empty room also produces no chest motion. This checks whether
anything has moved slowly in the last **two minutes**, using a 75th percentile of `inter`.

> Known sharp edge: it used to use a 60 s **median**, and a 30 s apnea is half that window —
> so a held breath dragged its own presence evidence under the threshold and the gate
> concluded the subject had left. Measured, the 60 s median during a real hold reached 7.7,
> *below* the most active wall minute at 8.1. A high quantile over a longer window cannot be
> pulled down that way, because the pre-hold breathing stays inside it.

### The whole pipeline

```
IQ → noise floor → pick bin → phase → mm → band-pass → rms_4s
   → ratio against your own normal → accumulate → threshold → APNEA
              ↑                                       ↑
        two references                    motion gate + presence gate
```

### The honest weaknesses

1. **An empty room and a perfectly still person are the same observation.** Nothing in a
   single frame separates them; only time does, because a person accumulates heartbeat and
   sway while a wall does not. Everything awkward in stages 5 and 6 exists because of this.

2. **`inter` is the maximum across all 21 bins** and carries three of the four charts
   (weights 2.0, 0.5, 0.5). The detector therefore leans much harder on "is something in the
   room breathing-ish" than on "is *that specific chest* moving". This is why a recording
   could score 30/30 while tracking an SNR-0.4 noise bin.

3. **The thresholds are fitted to the artifacts of their own inputs.** The 65,000-config
   search ran against features generated by the current bin selector, noise bins included.
   Improving the front end without re-fitting the thresholds therefore *looks* like a
   regression — measured twice. Relatedly, the low false-alarm record is partly an artifact:
   a noise bin gives an almost constant `rms_4s` so the drop statistic never moves, while a
   real chest has natural inter-breath pauses that read as micro-apneas.

---

## Part 3 — The scope, panel by panel

`python visualize.py` — ten panels laid out in pipeline order. Reading left to right, top to
bottom is watching noise turn into a decision.

### Header readouts

| readout | meaning | how to read it |
|---|---|---|
| **BREATHING RATE** | dominant frequency in the breathing band | `--` unless the spectral peak stands 6× clear of the band median. Noise always has a largest peak; requiring prominence stops it inventing a rate from an empty room. |
| **CHEST AT** | distance of the selected range bin | Should be roughly where you are. If it reads 1.44 m while you are at 0.6 m, the tracker is on noise. |
| **CHEST MOTION** | std of the breathing wave over 4 s | Drops during a hold. |
| **BODY SNR** | strongest reflection-to-noise ratio beyond 0.5 m, and where | **The number to maximise before recording.** Green ≥ 7, amber 3–7, red < 3. |
| **PRESENCE** | `person` / `nobody` | The presence gate's verdict. `nobody` silently cancels every alarm. |
| **FRAME RATE** | measured vs configured | Red if off by >15%. The extractor sizes every window from the configured rate; if the sensor cannot sustain it, every time constant in the chain is wrong. |
| **CHART GATE** | `accumulating` / `moving: <which>` / `no reference` | Why the CUSUM charts are being reset. `moving:` names *which* of the four gate conditions is shut — they need opposite fixes. |
| **STATUS** | the headline verdict | `APNEA`, `breathing`, `holding`, `no breathing signal`, `no presence`. Every change is also printed to the terminal with a timestamp. |

**BODY SNR reference points:** nishant lying at 1.08 m in the recordings this was tuned on
reads **12.4**. Three demo-rig runs at the same distance read 7.2, 4.0 and 2.8 — and only
the 7.2 run detected a breath hold. Adjust position, sensor angle and anything covering the
chest (60 GHz is strongly absorbed by fabric) until this is as high as it will go, *then*
record.

---

### 1 — RANGE PROFILE: raw energy vs distance

Amplitude returned from each of the 21 bins. The rawest view there is.

- **Green line** — the bin currently selected as your chest
- **Red band at 0.30 m** — the sensor's own near field, *not* a person

The near-field spike is about six times brighter than a chest. An earlier version of this
pipeline tracked it for 50–99% of frames and measured clutter instead of breathing.

**What good looks like:** a clear bump at your distance, standing well above the floor. If
everything past 0.7 m is flat, there is no body return and nothing downstream can work.

---

### 2 — IQ CONSTELLATION: breathing rotates the phase

The complex return from the chest bin over the last 6 seconds, plotted as I vs Q. **This is
the raw physical measurement everything else derives from.**

- **Blue dots** — the last few seconds of history
- **Green dot** — right now

**What good looks like:** the dot sweeps back and forth along an **arc** as you breathe —
chest moves a few millimetres, round trip changes by a fraction of a wavelength, phase
rotates. During a hold the arc collapses to a **blob**.

If it looks like a random scatter rather than an arc, you are watching noise.

---

### 3 — RANGE vs TIME: the body shows as a bright band

Breathing-band motion at every distance, over time (a waterfall, magma colormap).

**What good looks like:** a bright horizontal band at your range, persisting over time.

This is the panel that shows the sensor is *spatially selective* — it can ignore a moving
curtain at 1.4 m while watching a chest at 0.7 m.

---

### 4 — MOTION PER RANGE BIN: how the chest is chosen

Bar chart of long-run breathing-band motion per bin, with the selected bin marked green.
This is stage 2 made visible.

Not the brightest bin — the one that **moves** most, smoothed across neighbours and averaged
over 90 s.

**What to watch for:** if the tallest bar is at the far end of the range where the range
profile (panel 1) shows nothing, the selector is choosing noise. That was the failure behind
"it said I was breathing while I held my breath".

---

### 5 — PHASE → DISPLACEMENT: breathing buried in drift

Unwrapped phase converted to millimetres, mean-subtracted. The signal *before* cleaning.

Real breathing is in there, buried under slow drift of far larger amplitude than the
breathing itself. This panel exists to show why the band-pass in panel 6 is necessary rather
than cosmetic.

---

### 6 — THE BREATHING WAVE: band-passed, breaths countable

The same trace filtered to 0.18–0.55 Hz. **The "seeing through noise" panel.**

**What good looks like:** individual breaths become countable, and a hold is a visibly flat
stretch. This is the panel your eye can read directly — if the amplitude visibly collapses
when you stop breathing, the signal is good and any failure is downstream.

The band matters: at a wider 0.10 Hz the drift from panel 5 dominates and the dominant
period reads 6–10 bpm, which is not breathing.

---

### 7 — SPECTRUM: a sharp line means breathing

Power across the breathing band over the last 20 s, with the detected peak marked by a
dashed amber line.

- **A breathing person** — a sharp line at their rate
- **A held breath** — no line at all

The peak is only marked, and BREATHING RATE only populated, when it stands **6× above the
band median**. Without that test the readout invents a rate from an empty room, because a
band of noise still has a largest peak.

---

### 8 — DETECTOR: green = real hold, red = alarm

The alarm, against labelled ground truth (replay only).

- **Green shading** — a real, labelled breath hold
- **Red trace** — where the detector alarmed
- **White vertical line** — now

The gap between the start of green and the start of red is the **detection latency**, which
is the number this whole project optimises.

---

### 9 — PRESENCE GATE: alarms are vetoed below the line

Stage 6's presence veto, made visible.

- **Grey line** — the raw slow-motion score (`inter`), per frame
- **Blue line** — its trailing 75th percentile over 120 s, which is what the gate tests
- **Red dashed line** — the threshold (13.0)
- **Red shading** — the gate has concluded nobody is there and is cancelling alarms

**If the detector looks like it should fire and does not, this is the panel that says why.**
When the blue line dips under the red dashes, every alarm is silently vetoed.

---

### 10 — INSIDE THE DETECTOR: each chart's evidence total

Full-width strip along the bottom. Each of the four CUSUM charts' running total as a
fraction of its threshold. **1.0 = that chart is firing** (red dashed line).

| curve | weights | role |
|---|---|---|
| **patient** (green) | `inter` 2.0, `energy` 0.5, `ratio_q` 0.5 | carries most holds |
| **presence** (blue) | `inter` 0.5 | leaky, low threshold — catches what the first misses |
| **energy** (amber) | `ratio` 0.5, `energy` 0.5 | the fast chart, ratcheting reference, no deadband |
| **8s energy** (purple) | `ratio_q8` 0.5, `inter` 0.5 | shortens the median latency |

**How to read it:**

- **A curve creeping upward** — accumulating evidence, working normally
- **A curve pinned at 0** — being reset every frame. Check the CHART GATE readout for which
  condition is shut, or it has no reference yet.
- **Sawtooth** — accumulating then being wiped by the motion gate, repeatedly. This is what
  "it alarmed briefly then went back to normal" looks like from the inside.

This panel is replayed over the same 90 s window the live detector evaluates on, *not* the
30 s the time plots draw. Those must match, or the panel that exists to explain the alarm
computes something the alarm never saw. (It used to be 30 s, of which 20 s was the charts'
own warmup — the `energy` chart needs 15 s at its cap to reach threshold, so its curve was
arithmetically incapable of ever reaching 1.0.)

---

## Quick troubleshooting

| symptom | look at | likely cause |
|---|---|---|
| Breathing rate shown during a hold | **BODY SNR**, panel 1, panel 4 | Tracker on a noise bin. Phase jitter in an empty bin has a strongest frequency in the breathing band and never stops. |
| Never says APNEA | Panel 9, then panel 10 | Presence gate vetoing, or all four charts pinned at 0. |
| APNEA flickers on and off | Panel 10 for sawtooth, CHART GATE | Motion gate wiping the accumulated evidence. |
| Says "no presence" while you are there | Panel 9 | Blue line under the threshold. |
| Everything looks wrong at once | **FRAME RATE** | If red, every window and filter in the chain is sized wrong. |

**Before any run:** `python checkpos.py`, or just watch BODY SNR in the scope. Get it as
high as you can before you start, because no threshold can recover a signal that is not
there.
