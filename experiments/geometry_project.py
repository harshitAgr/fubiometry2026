"""Label-geometry projection for the two ellipse tasks (measured 2026-08-07).

WHAT: snap the 4 predicted FA / HC landmarks onto the exact geometric form that every
ground-truth label obeys, WITHOUT changing either diameter's length.

WHY: the labels are not 4 free points. Across the whole training set:

  FA  (500/500)  pair (p0,p1) is exactly vertical, pair (p2,p3) exactly horizontal, and
                 the two share a centre (max gap 0.707 px = integer rounding).
                 -> 4 free parameters (cx, cy, a, b), not 8.
  HC  (999/999)  the two diameters are exactly perpendicular (worst deviation 0.42 deg)
                 and share a centre. Rotation is free.
                 -> 5 free parameters.

Our predictions do not obey either: FA sits a median 4.95 px off axis-alignment with a
5.81 px centre gap; HC a 6.61 px centre gap and 0.96 deg off perpendicular (p90 5.3 deg).
Because the ground truth always lies on the constraint set, snapping onto it moves us
closer to it.

WHY LENGTH-PRESERVING: the plain orthogonal projection also SHORTENS each diameter by
the cosine of the rotation it applies, and that length is exactly what the clinical
parameter is computed from -- it cost +0.7167 param on HC. Fixing only the angles and
the centre, with each pair keeping its original length, keeps ~91% of the MRE gain at a
parameter delta that is zero BY CONSTRUCTION. That also makes this commute exactly with
the shipped HC ellipse-scale correction (verified to 2.3e-13 px).

MEASURED, paired over the 5 CV folds on the deployed pragmatic_seed42 family route:

    task-mean MRE    -0.1455   5/5 folds   corrected CI [-0.2075, -0.0836]
    task-mean param  +0.0000               corrected CI [-0.0000, +0.0000]
    FA -0.6894, HC -0.6206, the other 7 tasks exactly +0.0000

COORDINATE SPACE: apply in ORIGINAL-IMAGE PIXELS. Normalised space would work for FA
(uniform per-axis scaling preserves axis-alignment) but NOT for HC, where the anisotropic
width/height scaling does not preserve perpendicularity.

Derivation and evidence: experiments/label_geometry_sweep.py,
experiments/results/label_geometry/{sweep,deployed}.json.
"""
from __future__ import annotations

import math

PROJECTED_TASKS = frozenset({"FA", "HC"})


def _centre(pts):
    n = len(pts)
    return (sum(p[0] for p in pts) / n, sum(p[1] for p in pts) / n)


def _as_pairs(points):
    if len(points) != 8:
        raise ValueError(f"expected 8 flat coordinates (4 landmarks); got {len(points)}")
    return [(float(points[2 * i]), float(points[2 * i + 1])) for i in range(4)]


def _flat(pts):
    out = []
    for x, y in pts:
        out += [x, y]
    return out


def project_fa(points):
    """FA: axis-aligned + concentric, each diameter keeping its original length.

    points : flat [x1, y1, ..., x4, y4] in ORIGINAL-IMAGE pixels.
    """
    p = _as_pairs(points)
    cx, cy = _centre(p)
    t = math.copysign(math.dist(p[0], p[1]) / 2.0, p[0][1] - p[1][1])
    w = math.copysign(math.dist(p[2], p[3]) / 2.0, p[2][0] - p[3][0])
    return _flat([(cx, cy + t), (cx, cy - t), (cx + w, cy), (cx - w, cy)])


def project_hc(points):
    """HC: perpendicular + concentric, each diameter keeping its original length.

    The centre separates and lands on the 4-point mean. With A, B the raw half-vectors,
    the orientation minimising |U-A|^2 + |V-B|^2 over U perpendicular to V is

        phi = arg( |A|^2 e^{2i*alpha} - |B|^2 e^{2i*beta} ) / 2

    after which A and B are placed on the frame (cos phi, sin phi) / (-sin phi, cos phi)
    at their ORIGINAL lengths, with the sign that keeps each pair's direction.

    points : flat [x1, y1, ..., x4, y4] in ORIGINAL-IMAGE pixels.
    """
    p = _as_pairs(points)
    cx, cy = _centre(p)
    ax, ay = (p[0][0] - p[1][0]) / 2.0, (p[0][1] - p[1][1]) / 2.0
    bx, by = (p[2][0] - p[3][0]) / 2.0, (p[2][1] - p[3][1]) / 2.0
    na, nb = math.hypot(ax, ay), math.hypot(bx, by)
    if na < 1e-9 or nb < 1e-9:
        # degenerate pair: orientation is undefined, so only enforce the (linear) centre
        return _flat([(cx + ax, cy + ay), (cx - ax, cy - ay),
                      (cx + bx, cy + by), (cx - bx, cy - by)])
    alpha, beta = math.atan2(ay, ax), math.atan2(by, bx)
    zr = na * na * math.cos(2 * alpha) - nb * nb * math.cos(2 * beta)
    zi = na * na * math.sin(2 * alpha) - nb * nb * math.sin(2 * beta)
    phi = 0.5 * math.atan2(zi, zr)
    ux, uy = math.cos(phi), math.sin(phi)
    vx, vy = -math.sin(phi), math.cos(phi)
    sa = math.copysign(na, ax * ux + ay * uy)
    sb = math.copysign(nb, bx * vx + by * vy)
    return _flat([(cx + sa * ux, cy + sa * uy), (cx - sa * ux, cy - sa * uy),
                  (cx + sb * vx, cy + sb * vy), (cx - sb * vx, cy - sb * vy)])


def project(points, task_id):
    """Dispatch on task. Every task other than FA and HC passes through untouched."""
    if task_id == "FA":
        return project_fa(points)
    if task_id == "HC":
        return project_hc(points)
    return list(points)


def project_records(records):
    """Apply in place-ish to a list of submission records; returns a NEW list.

    Each record is {"image_path", "task_id", "predicted_points_pixels", ...}. Records
    for tasks outside PROJECTED_TASKS are copied through byte-identically.
    """
    out = []
    for r in records:
        if r.get("task_id") in PROJECTED_TASKS:
            r = dict(r)
            r["predicted_points_pixels"] = project(r["predicted_points_pixels"], r["task_id"])
        out.append(r)
    return out
