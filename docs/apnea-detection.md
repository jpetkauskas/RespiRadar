# Apnea detection: protocol, findings, and what is still uncertain

## How detectors are scored

`respiradar/evaluation.py` and `respiradar/bakeoff.py`. A detector is reduced to one boolean
per frame and judged on three things, in this order:

1. **Zero false alarms.** ~17 minutes of negatives across three people.
2. **Both holds caught while still happening.** An alarm that fires after the subject resumed
   breathing counts as a miss *and* a false alarm, not a late catch.
3. **Lowest worst-case latency from hold onset.**

The split is **leave-one-subject-out**: fit on everyone else, test on the held-out person.
With one subject there is no way to tell a detector that learned breathing from one that
learned *this person's* breathing, and the difference turned out to be the whole problem.

A 25 s warmup is excluded from scoring. Filters, presence tracking and the personal baseline
all need time to settle.

## The main finding: thresholds do not transfer between bodies

The tuned threshold baseline, measured two ways:

| | tuned and tested on nishant | held out across subjects |
|---|---|---|
| Holds found | 2 / 2 | 3 / 4 |
| False alarms | 0 | 10 in 16.8 min |

Nine of those ten false alarms are on *sleeping* and *talking* sessions, with the person
quiet and still. Talking and moving were never the difficulty - removing the motion gate
entirely changes nothing. Generalising across bodies is the difficulty.

For context, the detector this replaced scores **0 of 2** on nishant: it alarms once, 1.4 s
after he resumed breathing.

## Open question: justinas's hold labels look about 20 s early

Comparing the labelled hold windows against the band-passed breathing envelope, and sweeping
a time shift applied to the labels:

| subject | contrast as labelled | best shift | contrast at best shift |
|---|---|---|---|
| nishant | 0.62 | +6 s (marginal) | 0.57 |
| justinas | 0.71 | **+22 s** | 0.55 |

Lower is better - it means the envelope really is quieter inside the labelled hold. Nishant's
labels are essentially optimal where they are. Justinas's improve a lot when moved ~20 s
later, and at that shift his recording is as clean as nishant's, so the data is good and only
the alignment is in doubt.

**This has not been corrected.** Inventing a 22 s shift to improve the numbers would be
fitting the labels to the result. It needs confirming with justinas instead: did the holds
happen when the markers say, or did he press Enter and then take a breath before holding?

What it affects: the per-hold detection and latency figures for justinas. What it does **not**
affect: the false-alarm result above, since those sessions contain no holds under any labelling.

## Known data limitations

- Justinas's first hold starts at **4.6 s** and nishant's at **14.3 s**, both inside the
  filter warmup, so their early features are a transient rather than chest motion. Both will
  score worse than they deserve. Start holds after ~30 s when recording.
- Four labelled holds from two people. Frame counts are large and misleading: frames inside
  one hold are highly correlated, so the effective sample size for "what a hold looks like"
  is four, not thousands. Frame-level accuracy means very little here.
- One posture per subject. Nothing here says how this behaves lying on one side, under a
  duvet, or at a different distance.

## A bug that was live for a while

The personal baseline latched onto the first non-zero `rms_8s`, which early in a recording is
a band-pass transient rather than chest motion. `ratio_4s` reached 2.6e8 and stayed corrupted
well past the scoring warmup. A single subject's 25 s warmup hid it; it only became obvious
with a hold starting at 4.6 s. `test_ratio_features_never_explode` now pins it.

Worth recording because the tuned thresholds from before that fix were fitted to corrupted
features, and looked perfectly reasonable.
