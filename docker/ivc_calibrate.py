"""Gated IVC caliper-length calibration — the exact lever that shipped in v24 (882078).

WHAT: if the predicted IVC diameter falls OUTSIDE the training band [p10, p90], replace its
LENGTH with the training median, preserving the segment's midpoint and direction exactly.
Predictions inside the band are returned untouched.

WHY: on the deployed lineage our predicted IVC diameter carries no measurable information
about the true one -- corr +0.0951 with GT, param-MAE 12.14 px against a best-constant of
8.06 px, i.e. **worse than simply predicting a constant**, with 1.72x the GT dispersion.
An over-dispersed estimator with no signal can only be improved in expectation by clamping
its tails to a ground-truth statistic. Corroborated independently: a fine-tuned
EchoNet-Measurements model WITH real diameter skill (corr +0.481) beats this median by only
0.161 param-MAE inside the gate, so the gate already captures nearly all available signal.

EVIDENCE, and its limits.
  CV (5-fold LOFO, n=38 -- the honest out-of-sample measurement):
      dMRE   +0.1452  (3/5 folds favourable)  corrected CI [-0.419, +0.710]
      dparam -3.9870  (4/5 folds favourable)  corrected CI [-16.00, +8.03]
      per-fold dparam [-1.81, -2.60, -3.46, -14.73, +2.66]; fires 10/38
  Official validation (v23 -> v24): fired 3/10; IVC MRE -0.877, param-MAE 13.155 -> 7.978.
  The CI crosses zero and one fold supplies -14.73 of the -3.99 mean, so the MAGNITUDE is
  not trustworthy and the val MRE bonus is val-specific (its 3 fired images had spans
  99.18 / 64.96 / 40.55 px against a training max of 60.85 -- implausible outliers).
  What justifies shipping is bounded loss, not the point estimate: worst observed fold cost
  is +2.66 IVC param (= +0.30 Avg MAE) and +0.60 IVC MRE (= +0.07 Avg MRE), against a
  measured mechanism and a rank-then-aggregate metric that rewards rescuing catastrophes.

CONSTANTS are the FULL-38 training-GT fit -- p10 / p90 / median over the 38 IVC training
ground-truth diameters -- reproduced to <1e-9 from data/folds/folds.csv + the task CSVs on
2026-08-11, and byte-matching submission/v24/candidate_audit.json. They are deliberately NOT
the leave-one-fold-out fits in experiments/results/ivc_length_calibration/result.json: those
exist only to measure the lever out-of-sample and must never be deployed.

EXPECT A DIFFERENT FIRE RATE ON TEST. 21 of the 38 IVC training images are dual-panel
split-screen frames, and the organizers stated the test set contains none. The clamp target
is a GT anatomical statistic, so this is harmless -- but do not read a fire count unlike
val's 3/10 as a fault.
"""
from __future__ import annotations

import math

CALIBRATED_TASK = "IVC"

# p10 / p90 / median of the 38 IVC training GT diameters (px). Frozen; matches v24.
BAND_LO = 15.27421869663953
BAND_HI = 37.62752780503526
TARGET_LEN = 24.65705537396834

# Direction used only if the predicted segment has collapsed to a point, where the
# direction is undefined. Mirrors experiments/ivc_length_calibration.VERTICAL_UNIT.
VERTICAL_UNIT = (0.0, 1.0)


def apply_ivc_calibration(points, task_id, lo=BAND_LO, hi=BAND_HI, target=TARGET_LEN):
    """Return `points` with the gated length calibration applied, or unchanged.

    points  : flat [x1, y1, x2, y2] in ORIGINAL-IMAGE pixels (IVC has 2 landmarks).
    task_id : only "IVC" is ever touched; every other task passes straight through.

    Malformed input (not 4 coordinates) passes through untouched -- the caller's finiteness
    gate is where broken output must fail loud.
    """
    if task_id != CALIBRATED_TASK or len(points) != 4:
        return list(points)

    x0, y0, x1, y1 = (float(v) for v in points)
    dx, dy = x0 - x1, y0 - y1
    length = math.hypot(dx, dy)

    if lo <= length <= hi:
        return [x0, y0, x1, y1]

    ux, uy = VERTICAL_UNIT if length < 1e-9 else (dx / length, dy / length)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    half = target / 2.0
    return [cx + half * ux, cy + half * uy, cx - half * ux, cy - half * uy]


def would_fire(points, task_id, lo=BAND_LO, hi=BAND_HI):
    """True if the gate would act on this row. For logging/auditing only."""
    if task_id != CALIBRATED_TASK or len(points) != 4:
        return False
    length = math.hypot(float(points[0]) - float(points[2]),
                        float(points[1]) - float(points[3]))
    return not (lo <= length <= hi)
