"""Gated HC ellipse-scale correction (docker/hc_scale.py).

The correction ships inside the container, so it is tested here against the exact
properties the derivation relies on: centroid preservation, exact scaling of both
diameters, the in-domain gate, and total inertness on every other task.
"""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docker"))
from hc_scale import HC_SCALE, IN_DOMAIN_SIZES, apply_hc_scale  # noqa: E402

# A realistic HC quadruple: two orthogonal diameters about a shared centre.
HC_PTS = [400.0, 200.0, 400.0, 500.0,          # BPD  (vertical,   len 300)
          250.0, 350.0, 550.0, 350.0]          # OFD  (horizontal, len 300)
OUT_OF_DOMAIN = (1137, 787)                    # a real validation HC size


def _centroid(p):
    n = len(p) // 2
    return (sum(p[2 * i] for i in range(n)) / n, sum(p[2 * i + 1] for i in range(n)) / n)


def _diameters(p):
    return (math.dist(p[0:2], p[2:4]), math.dist(p[4:6], p[6:8]))


def test_out_of_domain_scales_both_diameters_exactly():
    out = apply_hc_scale(HC_PTS, "HC", *OUT_OF_DOMAIN)
    d0, d1 = _diameters(HC_PTS)
    s0, s1 = _diameters(out)
    assert s0 == pytest.approx(d0 * HC_SCALE, rel=1e-12)
    assert s1 == pytest.approx(d1 * HC_SCALE, rel=1e-12)


def test_centroid_is_preserved():
    out = apply_hc_scale(HC_PTS, "HC", *OUT_OF_DOMAIN)
    cx0, cy0 = _centroid(HC_PTS)
    cx1, cy1 = _centroid(out)
    assert (cx1, cy1) == pytest.approx((cx0, cy0), abs=1e-9)


def test_orientation_and_aspect_ratio_unchanged():
    out = apply_hc_scale(HC_PTS, "HC", *OUT_OF_DOMAIN)
    d0, d1 = _diameters(HC_PTS)
    s0, s1 = _diameters(out)
    assert s0 / s1 == pytest.approx(d0 / d1, rel=1e-12)
    ang = lambda p, a, b: math.atan2(p[b + 1] - p[a + 1], p[b] - p[a])
    assert ang(out, 0, 2) == pytest.approx(ang(HC_PTS, 0, 2), abs=1e-12)
    assert ang(out, 4, 6) == pytest.approx(ang(HC_PTS, 4, 6), abs=1e-12)


@pytest.mark.parametrize("size", sorted(IN_DOMAIN_SIZES))
def test_in_domain_sizes_are_untouched(size):
    """HC18 is the training domain: the bias does not exist there, so must not fire."""
    assert apply_hc_scale(HC_PTS, "HC", *size) == HC_PTS


def test_every_other_task_is_inert():
    for task in ("A4C", "AOP", "FA", "FUGC", "IVC", "PLAX", "PSAX", "fetal_femur"):
        assert apply_hc_scale(HC_PTS, task, *OUT_OF_DOMAIN) == HC_PTS


def test_returns_a_copy_never_mutates_input():
    original = list(HC_PTS)
    out = apply_hc_scale(HC_PTS, "HC", *OUT_OF_DOMAIN)
    assert HC_PTS == original
    assert out is not HC_PTS
    assert apply_hc_scale(HC_PTS, "FA", *OUT_OF_DOMAIN) is not HC_PTS


@pytest.mark.parametrize("w,h", [(None, 787), ("", ""), ("abc", "def"), (float("nan"), 787)])
def test_unknown_size_fails_safe_to_no_correction(w, h):
    assert apply_hc_scale(HC_PTS, "HC", w, h) == HC_PTS


@pytest.mark.parametrize("pts", [[], [1.0], [1.0, 2.0, 3.0]])
def test_malformed_points_pass_through(pts):
    assert apply_hc_scale(pts, "HC", *OUT_OF_DOMAIN) == pts


def test_size_is_read_as_width_height_not_height_width():
    """800x540 is in-domain; 540x800 is not. A transposed call must still correct."""
    assert apply_hc_scale(HC_PTS, "HC", 800, 540) == HC_PTS
    assert apply_hc_scale(HC_PTS, "HC", 540, 800) != HC_PTS


def test_scale_of_one_is_the_identity():
    assert apply_hc_scale(HC_PTS, "HC", *OUT_OF_DOMAIN, scale=1.0) == pytest.approx(HC_PTS)


def test_no_validation_hc_image_is_in_domain():
    """The gate must be provably free on validation: if any val HC image were an
    in-domain size, the gated build would score differently from v21/v22."""
    import glob
    import struct
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    paths = sorted(glob.glob(os.path.join(root, "data/val/images/HC/*")))
    if not paths:
        pytest.skip("validation images not present")
    hit = []
    for p in paths:
        with open(p, "rb") as f:
            head = f.read(24)
        if head[1:4] != b"PNG":
            continue
        w, h = struct.unpack(">II", head[16:24])
        if (w, h) in IN_DOMAIN_SIZES:
            hit.append(os.path.basename(p))
    assert hit == [], f"validation HC images at an in-domain size: {hit}"
