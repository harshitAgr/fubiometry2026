"""Pure-numpy sub-pixel heatmap decode + owned affine warp for TTA (Lever 1).

No torch/cv2 import: this module is unit-tested under the project .venv. The driver
(experiments/infer_tta.py, baseline venv) converts torch heatmaps to numpy at the boundary.
Grid convention: 64x64 heatmap, normalize a grid coord by /(W-1) (matches the baseline encode).
"""
from __future__ import annotations
import numpy as np


def _soft_window(m, iy, ix, window):
    r = window // 2
    # reflect-pad keeps an exactly-on-border peak unbiased; a peak 1-2 cells in gets a small
    # outward bias (bounded by the argmax fallback). parabolic/log_parabolic use no window.
    pad = np.pad(m, r, mode="reflect")
    win = pad[iy:iy + 2 * r + 1, ix:ix + 2 * r + 1]
    ys = np.arange(iy - r, iy + r + 1)          # extrapolated labels (may be <0) match the reflect
    xs = np.arange(ix - r, ix + r + 1)
    wsum = win.sum()
    if wsum < 1e-12:
        return float(ix), float(iy)
    x = float((win.sum(axis=0) * xs).sum() / wsum)
    y = float((win.sum(axis=1) * ys).sum() / wsum)
    return x, y


def _parabolic(m, iy, ix, log, eps):
    H, W = m.shape

    def offset(c, l, r):
        if log:
            c, l, r = np.log(c + eps), np.log(l + eps), np.log(r + eps)
        denom = l - 2.0 * c + r
        if denom >= 0:                      # not concave -> no reliable vertex
            return 0.0
        return float(np.clip(0.5 * (l - r) / denom, -0.5, 0.5))

    ox = 0.0 if ix in (0, W - 1) else offset(m[iy, ix], m[iy, ix - 1], m[iy, ix + 1])
    oy = 0.0 if iy in (0, H - 1) else offset(m[iy, ix], m[iy - 1, ix], m[iy + 1, ix])
    return ix + ox, iy + oy


def decode_subpixel(heatmaps, method="soft", window=7, eps=1e-6):
    """[B,K,H,W] post-sigmoid heatmaps -> [B,2K] normalized coords (x,y interleaved)."""
    hm = np.asarray(heatmaps, dtype=np.float64)
    B, K, H, W = hm.shape
    out = np.zeros((B, 2 * K), dtype=np.float64)
    for b in range(B):
        for k in range(K):
            m = hm[b, k]
            iy, ix = np.unravel_index(int(np.argmax(m)), m.shape)
            if method == "argmax":
                x, y = float(ix), float(iy)
            elif (m.max() - np.median(m)) < eps:    # prominence guard (flat map)
                x, y = float(ix), float(iy)
            elif method == "soft":
                x, y = _soft_window(m, iy, ix, window)
            elif method == "parabolic":
                x, y = _parabolic(m, iy, ix, log=False, eps=eps)
            elif method == "log_parabolic":
                x, y = _parabolic(m, iy, ix, log=True, eps=eps)
            else:
                raise ValueError(f"unknown method {method!r}")
            out[b, 2 * k] = x / max(W - 1, 1)
            out[b, 2 * k + 1] = y / max(H - 1, 1)
    return out


def _bilinear(arr, sy, sx):
    H, W = arr.shape
    x0 = np.floor(sx).astype(int); y0 = np.floor(sy).astype(int)
    wx = sx - x0; wy = sy - y0
    x0c, x1c = np.clip(x0, 0, W - 1), np.clip(x0 + 1, 0, W - 1)
    y0c, y1c = np.clip(y0, 0, H - 1), np.clip(y0 + 1, 0, H - 1)
    Ia, Ib = arr[y0c, x0c], arr[y0c, x1c]
    Ic, Id = arr[y1c, x0c], arr[y1c, x1c]
    val = Ia * (1 - wx) * (1 - wy) + Ib * wx * (1 - wy) + Ic * (1 - wx) * wy + Id * wx * wy
    inb = (sx >= 0) & (sx <= W - 1) & (sy >= 0) & (sy <= H - 1)
    return np.where(inb, val, 0.0)


def warp_affine(arr, s, tx=0.0, ty=0.0, inverse=False):
    """Scale `s` about centre + normalized translate. inverse=False renders a forward-augmented
    image (sample source = c+(dest-c-t)/s); inverse=True maps a heatmap back to canonical
    (sample source = c+s*(dest-c)+t). The two compose to identity."""
    arr = np.asarray(arr, dtype=np.float64)
    H, W = arr.shape
    cy, cx = (H - 1) / 2.0, (W - 1) / 2.0
    yy, xx = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    if inverse:
        sx = cx + s * (xx - cx) + tx * (W - 1)
        sy = cy + s * (yy - cy) + ty * (H - 1)
    else:
        sx = cx + (xx - cx - tx * (W - 1)) / s
        sy = cy + (yy - cy - ty * (H - 1)) / s
    return _bilinear(arr, sy, sx)


def average_heatmaps(hms):
    """Mean of a list of [K,H,W] heatmaps (one per TTA view), all already in canonical frame."""
    if len({np.asarray(h).shape for h in hms}) != 1:
        raise ValueError("average_heatmaps: views must share shape")
    stack = np.stack([np.asarray(h, dtype=np.float64) for h in hms], axis=0)
    return stack.mean(axis=0)
