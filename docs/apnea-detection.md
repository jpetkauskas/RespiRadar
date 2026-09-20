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

## Bake-off results

Six approaches, identical folds, identical metrics. Sorted by the priority order above:

| detector | holds | worst latency | median | false alarms |
|---|---|---|---|---|
| **changepoint/cusum-bank-conservative** | **3/4** | **22.0 s** | 20.5 s | **0** |
| spectral/range-stft | 3/4 | 30.8 s | 30.1 s | 0 |
| changepoint/cusum-bank | **4/4** | 23.4 s | 19.2 s | 3 |
| anomaly/one-sided-maha | 2/4 | 33.1 s | 29.8 s | 2 |
| breathgap/time-since-breath | 1/4 | 35.5 s | 35.5 s | 1 |
| temporal/stacked-window | 2/4 | 45.0 s | 33.3 s | 9 |
| baseline/energy-threshold | 2/4 | 8.2 s | 7.6 s | 16 |

**Recommended: `cusum-bank-conservative`.** Sequential change detection is built to minimise
detection delay at a fixed false-alarm rate, which is this problem's objective function
stated exactly. `cusum-bank` catches all four holds but talks three times in 19 minutes.

### What every approach independently agreed on

Four entries arrived at these separately, which is worth more than any single result:

1. **Subject-relative features transfer; absolute ones do not.** Dropping raw amplitudes for
   dimensionless ratios was the largest cross-subject gain every entry found.
2. **The supplied `baseline` ratchets** to a subject's *best* breathing, so shallow stretches
   sit below it forever and read as apnea. A trailing low quantile (25th percentile over
   60-120 s) beat it everywhere. This is a defect in the shared feature and should be fixed.
3. **Simple statistics beat flexible models.** Every multivariate model lost to a 1-D
   statistic. Gradient boosting and logistic regression saturate at probability 1.000 on a
   new subject's *negatives*; only bagged forests kept their ordering. With four events,
   capacity is a liability.
4. **Both first holds are undetectable** and are below the noise, not below a threshold.

### Sub-10 s is not reachable here at zero false alarms

The baseline reaches 8.2 s and pays 16 false alarms. A ratio-only CUSUM chart detects two
holds at 4.0 s and 1.7 s and costs 25. Negatives overlap positives frame-for-frame: a model
fitted on everything scores a labelled *negative* stretch higher than the first six seconds of
a real hold. Duration is the only separator, and duration is latency.

Two of the four holds also start inside the 25 s scoring warmup, which puts a floor of 20.4 s
on justinas's first hold alone. Worst-case latency is close to saturated; median latency is
the number with room left in it.

## A flaw in this harness, found by one of the entries

Folds were originally built only for subjects who had labelled holds, so vishnu - who only
breathes normally - was permanently in the training set and his false alarms were never
counted. Staying quiet on an unseen body is the single most important property, and it was
the one thing not being measured.

Two entries reported zero false alarms and in fact had **two and nine**. Every subject now
gets a fold, whether or not they hold their breath.

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
