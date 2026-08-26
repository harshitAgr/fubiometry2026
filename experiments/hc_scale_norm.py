"""Gated two-pass HC scale-norm (inference-only): crop-box math + coord mapping for the
larger-than-train HC val images (train HC = 800x540; val HC images run up to 1495px wide for
~205/215 images). A wide FOV makes the head appear too small after the 518 square-resize ->
scale-suppressed mislocalization (the 2026-06-22/26 HC diagnosis; a GT-backed oracle-crop test
recovered -3.36px, `hc_diag.step3_gt_synthetic` / `hc_tta_probe.py`). This module supplies the
crop-box + coord-mapping MATH for a real (non-oracle) two-pass deployment:
    pass 1 = the normal full-image ensemble prediction -> HC head centroid/bbox
    pass 2 = crop back to train-equivalent scale, re-run the SAME ensemble, map back

The driver (experiments/infer_ensemble.py, baseline venv) does the actual crop/resize (cv2) +
re-run; this module is pure numpy-free python (no cv2/torch) so it is unit-tested under the
project .venv, same convention as `sector_crop.py`.

Guaranteed no-op on train-sized-or-smaller images: crop_frac() == 1.0 there (all CV-fold and
train HC images are 800x540), and the driver skips pass 2 entirely when frac == 1.0 — this is
the do-no-harm proof for the CV harness.
"""
from __future__ import annotations

TRAIN_MAX_SIDE = 800.0
DEFAULT_MARGIN = 0.02


def crop_frac(w0: float, h0: float, train_max_side: float = TRAIN_MAX_SIDE) -> float:
    """Fraction of the original image's longer side that reproduces train-time HC scale.
    == 1.0 (a no-op) whenever the image is already <= train_max_side on its longer edge."""
    return min(1.0, train_max_side / max(w0, h0))


def _clamp(v, lo, hi):
    if hi < lo:
        hi = lo
    return max(lo, min(v, hi))


def compute_crop_box(w0, h0, cx, cy, frac, points=None, margin=DEFAULT_MARGIN):
    """(x0,y0,x1,y1) box of size (frac*w0, frac*h0) centered on (cx,cy), clamped to the image.

    frac >= 1.0 -> the identity box (0,0,w0,h0) (callers should already have skipped this case
    via crop_frac() == 1.0; kept here for completeness/safety of direct callers).

    If `points` (a list of (x,y) pass-1 landmark pixels) don't fit inside the centered+clamped
    box, the box is SHIFTED (never resized) to include them with `margin` (a fraction of the box
    side) of slack. Returns None if it still can't contain all points (the head is too large or
    straddles an edge) -- the caller's do-no-harm signal to fall back to the pass-1 prediction.
    """
    if frac >= 1.0:
        return (0.0, 0.0, float(w0), float(h0))
    cw, ch = frac * w0, frac * h0
    x0 = _clamp(cx - cw / 2.0, 0.0, w0 - cw)
    y0 = _clamp(cy - ch / 2.0, 0.0, h0 - ch)
    box = (x0, y0, x0 + cw, y0 + ch)
    if not points:
        return box
    return _shift_to_contain(box, points, w0, h0, margin)


def _shift_to_contain(box, points, w0, h0, margin):
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    pminx, pmaxx, pminy, pmaxy = min(xs), max(xs), min(ys), max(ys)
    mx, my = margin * cw, margin * ch
    need_x0, need_x1 = pminx - mx, pmaxx + mx
    need_y0, need_y1 = pminy - my, pmaxy + my
    if (need_x1 - need_x0) > cw or (need_y1 - need_y0) > ch:
        return None                          # points span more than the box -> can't shift-fit
    if x0 > need_x0:
        x0 = need_x0
    elif x1 < need_x1:
        x0 = need_x1 - cw
    if y0 > need_y0:
        y0 = need_y0
    elif y1 < need_y1:
        y0 = need_y1 - ch
    x0 = _clamp(x0, 0.0, w0 - cw)
    y0 = _clamp(y0, 0.0, h0 - ch)
    x1, y1 = x0 + cw, y0 + ch
    eps = 1e-6
    if pminx < x0 - eps or pmaxx > x1 + eps or pminy < y0 - eps or pmaxy > y1 + eps:
        return None                          # still doesn't fit (near an edge) -> fallback
    return (x0, y0, x1, y1)


def map_crop_to_original_px(coords_norm_flat, box):
    """Normalized coords in CROP space (x,y interleaved, [0,1]) -> ORIGINAL-image pixel coords
    (x,y interleaved): px_x = x0 + nx*cw, px_y = y0 + ny*ch."""
    x0, y0, x1, y1 = box
    cw, ch = x1 - x0, y1 - y0
    out = []
    for j in range(0, len(coords_norm_flat), 2):
        out += [x0 + coords_norm_flat[j] * cw, y0 + coords_norm_flat[j + 1] * ch]
    return out
