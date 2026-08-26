from experiments.hc_scale_norm import (
    crop_frac,
    compute_crop_box,
    map_crop_to_original_px,
    TRAIN_MAX_SIDE,
)


def test_crop_frac_is_noop_at_train_size():
    # all CV-fold / train HC images are exactly 800x540 -> the guaranteed no-op case.
    assert crop_frac(800, 540) == 1.0


def test_crop_frac_is_noop_below_train_size():
    # the 640x392 val HC images (smaller than train) must also no-op.
    assert crop_frac(640, 392) == 1.0


def test_crop_frac_shrinks_for_wide_fov():
    # val HC 1137x787 -> frac = 800/1137
    f = crop_frac(1137, 787)
    assert abs(f - 800.0 / 1137.0) < 1e-9
    assert 0.0 < f < 1.0


def test_compute_crop_box_is_full_frame_when_frac_is_one():
    box = compute_crop_box(800, 540, cx=400, cy=270, frac=1.0)
    assert box == (0.0, 0.0, 800.0, 540.0)


def test_compute_crop_box_centered_and_clamped_interior():
    w0, h0 = 1000.0, 1000.0
    frac = 0.5
    box = compute_crop_box(w0, h0, cx=500, cy=500, frac=frac)
    x0, y0, x1, y1 = box
    assert abs((x1 - x0) - frac * w0) < 1e-9
    assert abs((y1 - y0) - frac * h0) < 1e-9
    # centered on (500,500) -> box is [250,750]x[250,750]
    assert abs(x0 - 250) < 1e-9 and abs(x1 - 750) < 1e-9
    assert abs(y0 - 250) < 1e-9 and abs(y1 - 750) < 1e-9


def test_compute_crop_box_clamps_to_image_when_centroid_near_edge():
    w0, h0 = 1000.0, 1000.0
    frac = 0.5  # box side = 500
    box = compute_crop_box(w0, h0, cx=10, cy=10, frac=frac)
    x0, y0, x1, y1 = box
    assert x0 == 0.0 and y0 == 0.0
    assert x1 == 500.0 and y1 == 500.0


def test_compute_crop_box_shifts_to_contain_offcenter_points():
    w0, h0 = 1000.0, 1000.0
    frac = 0.3  # box side = 300
    # centroid near the edge; a couple of points near the opposite edge of a naive centered box
    pts = [(5, 500), (995, 500)]  # span 990px -- far wider than the 300px box
    box = compute_crop_box(w0, h0, cx=500, cy=500, frac=frac, points=pts)
    # points span (990) > box side (300) -> cannot shift-fit -> fallback signal
    assert box is None


def test_compute_crop_box_shifts_when_points_fit_but_off_center():
    w0, h0 = 1000.0, 1000.0
    frac = 0.4  # box side = 400
    pts = [(180, 200), (220, 240)]  # small spread, but centroid guess is off
    box = compute_crop_box(w0, h0, cx=500, cy=500, frac=frac, points=pts, margin=0.02)
    x0, y0, x1, y1 = box
    # box must now contain all points (with margin)
    for px, py in pts:
        assert x0 <= px <= x1
        assert y0 <= py <= y1
    # box size unchanged (shift only, never resize)
    assert abs((x1 - x0) - frac * w0) < 1e-6
    assert abs((y1 - y0) - frac * h0) < 1e-6


def test_compute_crop_box_none_when_points_too_close_to_edge_to_fit():
    w0, h0 = 100.0, 100.0
    frac = 0.2  # box side = 20
    # a point right at the image border with a large margin requirement relative to image size
    pts = [(0, 0), (0, 0)]
    box = compute_crop_box(w0, h0, cx=50, cy=50, frac=frac, points=pts, margin=0.5)
    # margin=0.5 of a 20px box = 10px slack each side -> needed span 20px == box side -> should
    # still just fit (shifted to the corner); assert it's a valid box containing the point.
    assert box is not None
    x0, y0, x1, y1 = box
    assert x0 <= 0.0 <= x1 and y0 <= 0.0 <= y1


def test_map_crop_to_original_px_round_trip():
    box = (100.0, 50.0, 300.0, 250.0)  # cw=200, ch=200
    # normalized crop-space center (0.5, 0.5) -> original px (200, 150)
    out = map_crop_to_original_px([0.5, 0.5], box)
    assert abs(out[0] - 200.0) < 1e-9
    assert abs(out[1] - 150.0) < 1e-9
    # corners round-trip too
    out2 = map_crop_to_original_px([0.0, 0.0, 1.0, 1.0], box)
    assert out2 == [100.0, 50.0, 300.0, 250.0]


def test_map_crop_to_original_px_multi_point():
    box = (0.0, 0.0, 800.0, 800.0)
    out = map_crop_to_original_px([0.25, 0.25, 0.75, 0.75], box)
    assert out == [200.0, 200.0, 600.0, 600.0]


def test_train_max_side_constant_matches_documented_train_hc_size():
    assert TRAIN_MAX_SIDE == 800.0
