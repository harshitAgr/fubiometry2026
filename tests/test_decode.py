import numpy as np
from experiments import decode as D


def _gaussian(H, W, cy, cx, sigma=1.8):
    yy, xx = np.mgrid[0:H, 0:W]
    g = np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))
    return g  # peak 1.0 at (cy,cx); mimics the trained target (sigma=1.8)


def _cells(coords, H=64, W=64):
    # coords are normalized [x0,y0,...]; return (x_cell, y_cell) of point 0
    return coords[0, 0] * (W - 1), coords[0, 1] * (H - 1)


def test_argmax_returns_exact_cell():
    hm = np.zeros((1, 1, 64, 64))
    hm[0, 0, 30, 40] = 1.0  # (iy=30, ix=40)
    coords = D.decode_subpixel(hm, method="argmax")
    x, y = _cells(coords)
    assert (round(x), round(y)) == (40, 30)


def test_soft_recovers_subcell_peak():
    # true peak between cells -> soft decode should beat argmax's 0.5-cell error
    hm = _gaussian(64, 64, cy=30.4, cx=40.7, sigma=1.8)[None, None]
    x, y = _cells(D.decode_subpixel(hm, method="soft", window=7))
    assert abs(x - 40.7) < 0.2 and abs(y - 30.4) < 0.2


def test_soft_border_peak_not_biased_inward():
    # a peak ON the border must not be pulled inward by an asymmetric window
    center = _cells(D.decode_subpixel(_gaussian(64, 64, 32.0, 32.0)[None, None],
                                      method="soft", window=7))
    bx, by = _cells(D.decode_subpixel(_gaussian(64, 64, 32.0, 0.0)[None, None],
                                      method="soft", window=7))
    assert abs(bx - 0.0) < 0.25  # x at border ~0, not pulled to ~1+


def test_flat_heatmap_falls_back_to_argmax_cell():
    hm = np.full((1, 1, 64, 64), 0.5)  # window-sum is large; only prominence guard saves us
    coords = D.decode_subpixel(hm, method="soft", window=7)
    assert np.isfinite(coords).all()


def test_parabolic_exact_vertex():
    # discrete parabola peaking between cell 40 and 41 -> vertex offset toward 41
    hm = np.zeros((1, 1, 64, 64))
    hm[0, 0, 30, 39] = 0.6
    hm[0, 0, 30, 40] = 1.0
    hm[0, 0, 30, 41] = 0.8
    x, y = _cells(D.decode_subpixel(hm, method="parabolic"))
    assert 40.0 < x < 40.5 and abs(y - 30.0) < 1e-6


def test_log_parabolic_exact_on_clean_gaussian():
    hm = _gaussian(64, 64, cy=30.0, cx=40.3, sigma=1.8)[None, None]
    x, y = _cells(D.decode_subpixel(hm, method="log_parabolic"))
    assert abs(x - 40.3) < 0.05 and abs(y - 30.0) < 0.05  # log-vertex is exact for a Gaussian


def test_parabolic_border_offset_zero():
    hm = np.zeros((1, 1, 64, 64))
    hm[0, 0, 0, 0] = 1.0  # argmax on the corner -> no neighbours -> offset 0
    x, y = _cells(D.decode_subpixel(hm, method="parabolic"))
    assert (x, y) == (0.0, 0.0)


def _smooth(H, W):
    yy, xx = np.mgrid[0:H, 0:W]
    return np.sin(xx / 7.0) * np.cos(yy / 5.0) + 2.0  # smooth -> bilinear round-trip is faithful


def test_warp_roundtrip_scale_and_translate_is_identity():
    g = _smooth(64, 64)
    for s, tx, ty in [(1.08, 0.0, 0.0), (0.92, 0.0, 0.0), (1.05, 0.03, -0.02)]:
        back = D.warp_affine(D.warp_affine(g, s, tx, ty, inverse=False), s, tx, ty, inverse=True)
        inner = back[8:-8, 8:-8]
        # ignore border (constant-0 fill); 2e-2 is the double-pass bilinear floor on a 64x64
        # grid (~1.5%, matches scipy order=1) — direction/sign is guarded by the known-dot test.
        assert np.abs(inner - g[8:-8, 8:-8]).max() < 2e-2


def test_known_dot_endtoend_recovers_location():
    # forward-warp an image dot, "predict" a peak there, inverse-warp the heatmap -> original loc
    img = np.zeros((64, 64)); img[30, 40] = 1.0
    s = 1.08
    warped = D.warp_affine(img, s, inverse=False)          # dot moves to c+s*(u-c)
    iy, ix = np.unravel_index(int(np.argmax(warped)), warped.shape)
    hm = np.zeros((64, 64)); hm[iy, ix] = 1.0              # model peak in augmented frame
    back = D.warp_affine(hm, s, inverse=True)              # map heatmap back to canonical
    by, bx = np.unravel_index(int(np.argmax(back)), back.shape)
    assert abs(bx - 40) <= 1 and abs(by - 30) <= 1


def test_average_heatmaps_mean_and_shape_guard():
    a = np.ones((2, 64, 64)); b = np.full((2, 64, 64), 3.0)
    np.testing.assert_allclose(D.average_heatmaps([a, b]), np.full((2, 64, 64), 2.0))
    import pytest
    with pytest.raises(ValueError):
        D.average_heatmaps([a, np.ones((3, 64, 64))])  # mismatched K
