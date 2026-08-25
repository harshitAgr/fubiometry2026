"""Properties the label-geometry projection must satisfy to be safe to ship.

The projection's whole value rests on four claims, so each is asserted directly against
the real training labels rather than against hand-made fixtures:

  1. it is the IDENTITY on ground-truth labels (they already satisfy the constraint),
  2. it preserves each diameter's LENGTH exactly (this is what makes it parameter-neutral),
  3. it COMMUTES with the shipped HC ellipse-scale correction,
  4. it leaves every task other than FA and HC byte-identical.
"""
from __future__ import annotations

import ast
import csv
import math
import os
import random

import pytest

from experiments.geometry_project import (
    PROJECTED_TASKS, project, project_fa, project_hc, project_records,
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Labels are stored as integers, so an invariant exact in the underlying annotation shows
# up with up to half a pixel of slack per coordinate.
ROUND_SLACK = 1.0


def _labels(task, limit=None):
    path = os.path.join(ROOT, "data", "csv", f"{task}_train.csv")
    if not os.path.exists(path):
        pytest.skip(f"{task}_train.csv not present")
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            flat = []
            for i in range(1, 5):
                x, y = ast.literal_eval(r[f"point_{i}_xy"])
                flat += [float(x), float(y)]
            rows.append(flat)
            if limit and len(rows) >= limit:
                break
    return rows


def _d(p, i, j):
    return math.dist((p[2 * i], p[2 * i + 1]), (p[2 * j], p[2 * j + 1]))


def _centre_gap(p):
    c1 = ((p[0] + p[2]) / 2, (p[1] + p[3]) / 2)
    c2 = ((p[4] + p[6]) / 2, (p[5] + p[7]) / 2)
    return math.dist(c1, c2)


def _angle_between(p):
    ux, uy = p[2] - p[0], p[3] - p[1]
    vx, vy = p[6] - p[4], p[7] - p[5]
    c = (ux * vx + uy * vy) / (math.hypot(ux, uy) * math.hypot(vx, vy))
    return math.degrees(math.acos(max(-1.0, min(1.0, abs(c)))))


def _jitter(p, sigma, rng):
    return [v + rng.gauss(0, sigma) for v in p]


# --- 1. identity on the ground truth -----------------------------------------

@pytest.mark.parametrize("task,fn", [("FA", project_fa), ("HC", project_hc)])
def test_identity_on_ground_truth_labels(task, fn):
    for p in _labels(task):
        q = fn(p)
        assert max(abs(a - b) for a, b in zip(p, q)) <= ROUND_SLACK


# --- 2. exact length preservation (the parameter-neutrality guarantee) -------

@pytest.mark.parametrize("task,fn", [("FA", project_fa), ("HC", project_hc)])
def test_diameter_lengths_are_preserved(task, fn):
    rng = random.Random(42)
    for p in _labels(task, limit=200):
        noisy = _jitter(p, 8.0, rng)
        q = fn(noisy)
        assert _d(noisy, 0, 1) == pytest.approx(_d(q, 0, 1), abs=1e-9)
        assert _d(noisy, 2, 3) == pytest.approx(_d(q, 2, 3), abs=1e-9)


# --- 3. the constraint actually holds afterwards -----------------------------

def test_fa_output_is_axis_aligned_and_concentric():
    rng = random.Random(7)
    for p in _labels("FA", limit=200):
        q = project_fa(_jitter(p, 10.0, rng))
        assert q[0] == pytest.approx(q[2], abs=1e-9)      # x0 == x1
        assert q[5] == pytest.approx(q[7], abs=1e-9)      # y2 == y3
        assert _centre_gap(q) == pytest.approx(0.0, abs=1e-9)


def test_hc_output_is_perpendicular_and_concentric():
    rng = random.Random(7)
    for p in _labels("HC", limit=200):
        q = project_hc(_jitter(p, 10.0, rng))
        assert _angle_between(q) == pytest.approx(90.0, abs=1e-7)
        assert _centre_gap(q) == pytest.approx(0.0, abs=1e-9)


# --- 4. idempotence ----------------------------------------------------------

@pytest.mark.parametrize("task,fn", [("FA", project_fa), ("HC", project_hc)])
def test_projection_is_idempotent(task, fn):
    rng = random.Random(11)
    for p in _labels(task, limit=100):
        q = fn(_jitter(p, 12.0, rng))
        assert max(abs(a - b) for a, b in zip(q, fn(q))) < 1e-9


# --- 5. commutation with the shipped HC ellipse-scale correction -------------

def _scale_about_centroid(p, s):
    cx = sum(p[0::2]) / 4.0
    cy = sum(p[1::2]) / 4.0
    out = []
    for i in range(4):
        out += [cx + (p[2 * i] - cx) * s, cy + (p[2 * i + 1] - cy) * s]
    return out


@pytest.mark.parametrize("s", [0.950, 0.975])
def test_hc_projection_commutes_with_scale_correction(s):
    rng = random.Random(3)
    for p in _labels("HC", limit=150):
        noisy = _jitter(p, 8.0, rng)
        a = project_hc(_scale_about_centroid(noisy, s))
        b = _scale_about_centroid(project_hc(noisy), s)
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-6


# --- 6. dispatch leaves every other task untouched ---------------------------

def test_non_ellipse_tasks_pass_through_unchanged():
    pts = [1.5, 2.5, 3.5, 4.5, 5.5, 6.5, 7.5, 8.5]
    for task in ("A4C", "AOP", "IVC", "PLAX", "PSAX", "FUGC", "fetal_femur"):
        assert project(pts, task) == pts
        assert task not in PROJECTED_TASKS


def test_project_records_copies_other_tasks_byte_identically():
    recs = [
        {"image_path": "IVC/0001.png", "task_id": "IVC", "predicted_points_pixels": [1.0, 2.0, 3.0, 4.0]},
        {"image_path": "FA/0001.png", "task_id": "FA",
         "predicted_points_pixels": [10.0, 1.0, 12.0, 21.0, 25.0, 9.0, 3.0, 11.0]},
    ]
    out = project_records(recs)
    assert out[0]["predicted_points_pixels"] == [1.0, 2.0, 3.0, 4.0]
    assert out[1]["predicted_points_pixels"] != recs[1]["predicted_points_pixels"]
    # the input list must not be mutated
    assert recs[1]["predicted_points_pixels"] == [10.0, 1.0, 12.0, 21.0, 25.0, 9.0, 3.0, 11.0]


# --- 7. input validation -----------------------------------------------------

@pytest.mark.parametrize("bad", [[], [1.0, 2.0], [0.0] * 7, [0.0] * 10])
def test_rejects_wrong_landmark_count(bad):
    with pytest.raises(ValueError):
        project_fa(bad)
    with pytest.raises(ValueError):
        project_hc(bad)


def test_hc_handles_a_degenerate_pair_without_crashing():
    p = [5.0, 5.0, 5.0, 5.0, 0.0, 5.0, 10.0, 5.0]      # first pair collapsed to a point
    q = project_hc(p)
    assert len(q) == 8
    assert all(math.isfinite(v) for v in q)
    assert _centre_gap(q) == pytest.approx(0.0, abs=1e-9)
