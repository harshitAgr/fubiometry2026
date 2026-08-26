"""Tests for the two post-processing modules added to the container on 2026-08-11.

`docker/geometry_project.py` is a vendored copy of `experiments/geometry_project.py`, and
`docker/ivc_calibrate.py` freezes the constants that shipped in v24 (882078). Both must be
provably identical to the code/constants behind the officially scored artifacts, so the tests
below check three things: mathematical properties, vendored-copy parity, and byte-level
agreement with the real v24 submission JSON.

Companion: tests/test_hc_scale.py (the third lever, added 2026-08-07).
"""
from __future__ import annotations

import json
import math
import os
import random

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

import sys

sys.path.insert(0, os.path.join(ROOT, "docker"))

from geometry_project import PROJECTED_TASKS, project  # noqa: E402
from hc_scale import HC_SCALE, apply_hc_scale  # noqa: E402
from ivc_calibrate import (  # noqa: E402
    BAND_HI, BAND_LO, TARGET_LEN, apply_ivc_calibration, would_fire,
)


def _pairs(flat):
    return [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]


def _diams(flat):
    p = _pairs(flat)
    return math.dist(p[0], p[1]), math.dist(p[2], p[3])


def _centre(flat):
    p = _pairs(flat)
    return (sum(x for x, _ in p) / len(p), sum(y for _, y in p) / len(p))


def _random_quad(rng):
    return [rng.uniform(50, 950) for _ in range(8)]


# --------------------------------------------------------------- vendored-copy parity


def test_vendored_projection_matches_the_research_implementation_exactly():
    """The container copy must not drift from experiments/geometry_project.py."""
    from experiments.geometry_project import project as research_project

    rng = random.Random(42)
    for _ in range(2000):
        pts = _random_quad(rng)
        for task in ("FA", "HC"):
            assert project(pts, task) == research_project(pts, task)


def test_vendored_projection_agrees_on_degenerate_and_extreme_input():
    from experiments.geometry_project import project as research_project

    cases = [
        [100.0, 100.0, 100.0, 100.0, 200.0, 150.0, 50.0, 150.0],   # collapsed pair 1
        [100.0, 200.0, 100.0, 50.0, 150.0, 150.0, 150.0, 150.0],   # collapsed pair 2
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],                  # fully collapsed
        [1e6, -1e6, -1e6, 1e6, 1e6, 1e6, -1e6, -1e6],              # large magnitudes
    ]
    for pts in cases:
        for task in ("FA", "HC"):
            assert project(pts, task) == research_project(pts, task)


# ------------------------------------------------------------------- projection properties


def test_fa_projection_achieves_axis_alignment_and_concentricity():
    rng = random.Random(7)
    for _ in range(500):
        out = project(_random_quad(rng), "FA")
        p = _pairs(out)
        assert abs(p[0][0] - p[1][0]) < 1e-9, "first pair must be exactly vertical"
        assert abs(p[2][1] - p[3][1]) < 1e-9, "second pair must be exactly horizontal"
        c1 = ((p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2)
        c2 = ((p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2)
        assert math.dist(c1, c2) < 1e-9, "the two diameters must share a centre"


def test_hc_projection_achieves_perpendicularity_and_concentricity():
    rng = random.Random(8)
    for _ in range(500):
        out = project(_random_quad(rng), "HC")
        p = _pairs(out)
        u = (p[1][0] - p[0][0], p[1][1] - p[0][1])
        v = (p[3][0] - p[2][0], p[3][1] - p[2][1])
        cos = (u[0] * v[0] + u[1] * v[1]) / (math.hypot(*u) * math.hypot(*v))
        assert abs(cos) < 1e-9, "diameters must be exactly perpendicular"
        c1 = ((p[0][0] + p[1][0]) / 2, (p[0][1] + p[1][1]) / 2)
        c2 = ((p[2][0] + p[3][0]) / 2, (p[2][1] + p[3][1]) / 2)
        assert math.dist(c1, c2) < 1e-9


@pytest.mark.parametrize("task", ["FA", "HC"])
def test_projection_preserves_both_diameters(task):
    """This is what makes the lever parameter-MAE neutral BY CONSTRUCTION."""
    rng = random.Random(9)
    for _ in range(500):
        pts = _random_quad(rng)
        d0, d1 = _diams(pts)
        e0, e1 = _diams(project(pts, task))
        assert abs(d0 - e0) < 1e-9 and abs(d1 - e1) < 1e-9


@pytest.mark.parametrize("task", ["A4C", "AOP", "FUGC", "IVC", "PLAX", "PSAX", "fetal_femur"])
def test_projection_is_inert_on_every_other_task(task):
    pts = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
    assert project(pts, task) == pts
    assert task not in PROJECTED_TASKS


@pytest.mark.parametrize("pts", [[], [1.0, 2.0], [1.0] * 6, [1.0] * 10, [1.0] * 32])
def test_projection_passes_malformed_rows_through_instead_of_raising(pts):
    """A wrong-length FA/HC row must not raise here; model.py's finiteness/length gate is
    where genuinely broken output has to fail loud."""
    assert project(pts, "FA") == pts
    assert project(pts, "HC") == pts


def test_projection_returns_a_new_list_and_never_mutates_input():
    pts = [10.0, 20.0, 10.0, 5.0, 30.0, 12.0, 1.0, 12.0]
    before = list(pts)
    project(pts, "FA")
    project(pts, "HC")
    assert pts == before


# ------------------------------------------------------------------------ IVC calibration


def test_ivc_constants_are_the_frozen_v24_values():
    """These are the FULL-38 training-GT fit, NOT the LOFO fits used to measure the lever."""
    assert BAND_LO == 15.27421869663953
    assert BAND_HI == 37.62752780503526
    assert TARGET_LEN == 24.65705537396834


@pytest.mark.parametrize("length", [BAND_LO, 20.0, 24.65705537396834, 30.0, BAND_HI])
def test_ivc_inside_the_band_is_returned_untouched(length):
    """The band is CLOSED on both ends: a prediction exactly at p10 or p90 must not fire."""
    pts = [100.0, 100.0, 100.0, 100.0 - length]
    assert apply_ivc_calibration(pts, "IVC") == pts
    assert not would_fire(pts, "IVC")


@pytest.mark.parametrize("length", [0.5, 5.0, 15.2, 37.7, 64.96, 99.18, 500.0])
def test_ivc_outside_the_band_is_clamped_to_the_target_length(length):
    pts = [100.0, 300.0, 100.0 + 0.6 * length, 300.0 + 0.8 * length]   # 3-4-5 direction
    assert would_fire(pts, "IVC")
    out = apply_ivc_calibration(pts, "IVC")
    assert abs(math.dist((out[0], out[1]), (out[2], out[3])) - TARGET_LEN) < 1e-9


@pytest.mark.parametrize("length", [1.0, 10.0, 45.0, 120.0])
def test_ivc_calibration_preserves_midpoint_and_direction_exactly(length):
    """Midpoint shift and direction change were both audited at ~0 in v24; keep it that way."""
    rng = random.Random(11)
    for _ in range(100):
        ang = rng.uniform(0, 2 * math.pi)
        cx, cy = rng.uniform(100, 900), rng.uniform(100, 900)
        half = length / 2.0
        ux, uy = math.cos(ang), math.sin(ang)
        pts = [cx + half * ux, cy + half * uy, cx - half * ux, cy - half * uy]
        out = apply_ivc_calibration(pts, "IVC")
        if not would_fire(pts, "IVC"):
            assert out == pts
            continue
        assert abs((out[0] + out[2]) / 2 - cx) < 1e-9
        assert abs((out[1] + out[3]) / 2 - cy) < 1e-9
        d_in = (pts[0] - pts[2], pts[1] - pts[3])
        d_out = (out[0] - out[2], out[1] - out[3])
        cos = ((d_in[0] * d_out[0] + d_in[1] * d_out[1])
               / (math.hypot(*d_in) * math.hypot(*d_out)))
        assert abs(1.0 - cos) < 1e-12, "direction must be preserved, not flipped"


def test_ivc_collapsed_segment_falls_back_to_vertical():
    pts = [500.0, 400.0, 500.0, 400.0]
    out = apply_ivc_calibration(pts, "IVC")
    assert abs(math.dist((out[0], out[1]), (out[2], out[3])) - TARGET_LEN) < 1e-9
    assert abs(out[0] - 500.0) < 1e-9 and abs(out[2] - 500.0) < 1e-9
    assert abs((out[1] + out[3]) / 2 - 400.0) < 1e-9


@pytest.mark.parametrize("task", ["A4C", "AOP", "FA", "FUGC", "HC", "PLAX", "PSAX",
                                 "fetal_femur"])
def test_ivc_calibration_is_inert_on_every_other_task(task):
    pts = [0.0, 0.0, 500.0, 500.0]                       # length 707, far outside the band
    assert apply_ivc_calibration(pts, task) == pts
    assert not would_fire(pts, task)


@pytest.mark.parametrize("pts", [[], [1.0, 2.0], [1.0] * 6, [1.0] * 8])
def test_ivc_passes_malformed_rows_through_instead_of_raising(pts):
    assert apply_ivc_calibration(pts, "IVC") == pts
    assert not would_fire(pts, "IVC")


def test_ivc_calibration_returns_a_new_list_and_never_mutates_input():
    pts = [0.0, 0.0, 0.0, 100.0]
    before = list(pts)
    apply_ivc_calibration(pts, "IVC")
    assert pts == before


# ------------------------------------------------------- interaction with the HC scale lever


def test_hc_scale_and_projection_commute():
    """Both orderings must agree, because the deployed chain applies scale THEN projection
    while the derivation assumed the opposite. Audited at 2.3e-13 px; assert 1e-9."""
    rng = random.Random(13)
    w, h = 1024, 768                                   # not an HC18 size, so the gate fires
    for _ in range(1000):
        pts = _random_quad(rng)
        a = project(apply_hc_scale(pts, "HC", w, h), "HC")
        b = apply_hc_scale(project(pts, "HC"), "HC", w, h)
        assert max(abs(x - y) for x, y in zip(a, b)) < 1e-9


def test_the_three_levers_own_disjoint_tasks_except_hc():
    """FA gets the projection only; IVC gets the gate only; HC gets scale + projection."""
    quad = [10.0, 20.0, 10.0, 5.0, 30.0, 12.0, 1.0, 12.0]
    seg = [0.0, 0.0, 0.0, 100.0]
    assert apply_hc_scale(quad, "FA", 1024, 768) == quad
    assert apply_ivc_calibration(quad, "FA") == quad
    assert project(seg, "IVC") == seg
    assert apply_hc_scale(seg, "IVC", 1024, 768) == seg


# ------------------------------------------- regression against the officially scored v24

V21 = os.path.join(ROOT, "submission/v21/regression_predictions.json")
V23 = os.path.join(ROOT, "submission/v23/regression_predictions.json")
V24 = os.path.join(ROOT, "submission/v24/regression_predictions.json")

pytestmark_artifacts = pytest.mark.skipif(
    not all(os.path.isfile(p) for p in (V21, V23, V24)),
    reason="v21/v23/v24 submission artifacts not present",
)


def _by_key(path):
    return {(r["task_id"], r["image_path"]): r["predicted_points_pixels"]
            for r in json.load(open(path))}


@pytestmark_artifacts
def test_projection_reproduces_v24_FA_exactly_from_v21():
    """FA is untouched by the HC scale, so v21 -> projection must land on v24's FA rows.
    This pins the vendored module to an artifact with an official score."""
    src, ref = _by_key(V21), _by_key(V24)
    keys = [k for k in ref if k[0] == "FA"]
    assert len(keys) == 188
    for k in keys:
        assert project(src[k], "FA") == ref[k]


@pytestmark_artifacts
def test_gate_reproduces_v24_IVC_exactly_from_v23():
    """IVC is untouched by both the HC scale and the projection, so v23 -> gate must land on
    v24's IVC rows, firing on exactly the 3 images named in v24's audit."""
    src, ref = _by_key(V23), _by_key(V24)
    keys = [k for k in ref if k[0] == "IVC"]
    assert len(keys) == 10
    fired = []
    for k in keys:
        out = apply_ivc_calibration(src[k], "IVC")
        assert out == ref[k]
        if out != src[k]:
            fired.append(k[1])
    assert sorted(fired) == ["IVC/0005.png", "IVC/0008.png", "IVC/0009.png"]


@pytestmark_artifacts
def test_hc_scale_ratio_between_the_ship_and_v24_is_exactly_0975_over_0950():
    """v24 shipped s=0.950; the container ships the externally-fitted s=0.975. Confirm the
    only HC difference is that scalar, i.e. nothing else about HC changed."""
    v21, v24 = _by_key(V21), _by_key(V24)
    expected = HC_SCALE / 0.950
    for k in (k for k in v24 if k[0] == "HC"):
        ship = project(v21[k], "HC")                       # v21 already carries s=0.975
        ref = v24[k]
        cs = _centre(ship)
        cr = _centre(ref)
        rs = math.dist((ship[0], ship[1]), cs)
        rr = math.dist((ref[0], ref[1]), cr)
        if rr > 1e-6:
            assert abs(rs / rr - expected) < 1e-9
