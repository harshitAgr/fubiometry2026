"""Unit tests for the pure FUGC scale-normalization geometry."""
import pytest

from experiments.fugc_scale import (
    CROP_MARGIN,
    crop_pred_to_orig_norm,
    needs_scale_norm,
    scale_norm_crop_box,
)


def test_raw_validation_geometry_triggers():
    assert needs_scale_norm((1136, 735), (544, 336)) is True


def test_training_geometry_is_noop():
    assert needs_scale_norm((544, 336), (544, 336)) is False


def test_gate_requires_both_dimensions_to_be_wider():
    assert needs_scale_norm((1136, 336), (544, 336)) is False


def test_gate_threshold_is_tunable():
    assert needs_scale_norm((650, 405), (544, 336)) is False
    assert needs_scale_norm((650, 405), (544, 336), ratio=1.1) is True


def test_crop_box_targets_training_scale_and_is_centered():
    box = scale_norm_crop_box(735, 1136, (544, 336))
    x0, y0, x1, y1 = box
    assert x1 - x0 == round(544 * CROP_MARGIN)
    assert y1 - y0 == round(336 * CROP_MARGIN)
    assert abs(x0 - (1136 - x1)) <= 1
    assert abs(y0 - (735 - y1)) <= 1


def test_crop_box_clamps_to_image():
    assert scale_norm_crop_box(336, 544, (544, 336)) == (0, 0, 544, 336)


def test_crop_center_maps_to_original_center():
    box = scale_norm_crop_box(735, 1136, (544, 336))
    crop_w, crop_h = box[2] - box[0], box[3] - box[1]
    out = crop_pred_to_orig_norm([(crop_w / 2, crop_h / 2)], box, 1136, 735)
    assert out == pytest.approx([0.5, 0.5], abs=1e-3)
