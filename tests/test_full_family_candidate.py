import copy
import json

import pytest

from experiments.full_family_candidate import (
    blend_family_records,
    keyed,
    verify_training_manifests,
)


def records(offset=0.0):
    return [
        {
            "image_path": "HC/a.png", "task_id": "HC", "extra": "base",
            "predicted_points_pixels": [1.0 + offset, 2.0 + offset],
            "predicted_points_normalized": [0.1 + offset, 0.2 + offset],
        },
        {
            "image_path": "IVC/b.png", "task_id": "IVC", "extra": "base",
            "predicted_points_pixels": [3.0 + offset, 4.0 + offset],
            "predicted_points_normalized": [0.3 + offset, 0.4 + offset],
        },
    ]


def test_uniform3_formula_and_exact_ivc_passthrough():
    base, small, head = records(0), records(3), records(6)
    head[1] = copy.deepcopy(base[1])
    output, audit = blend_family_records(base, small, head)
    assert output[0]["predicted_points_pixels"] == [4.0, 5.0]
    assert output[0]["predicted_points_normalized"] == pytest.approx([3.1, 3.2])
    assert output[0]["extra"] == "base"
    assert output[1] == base[1]
    assert audit["passed"] is True
    assert audit["ivc_record_values_exact"] is True
    assert audit["max_coordinate_formula_residual"] == 0.0


def test_pragmatic_route_duplicates_five_seed_base_outside_hc():
    base, small, head = records(0), records(3), records(60)
    base[0]["task_id"] = small[0]["task_id"] = head[0]["task_id"] = "FA"
    output, audit = blend_family_records(
        base, small, head, mode="pragmatic_seed42")
    assert output[0]["predicted_points_pixels"] == [2.0, 3.0]
    assert audit["routes"]["other_tasks"] == "(2 * base + hcsmall) / 3"


def test_pragmatic_hc_still_uses_refined_head_family():
    output, _ = blend_family_records(
        records(0), records(3), records(6), mode="pragmatic_seed42")
    assert output[0]["predicted_points_pixels"] == [4.0, 5.0]


def test_symmetric_rejects_hchead_divergence_outside_hc():
    base, small, head = records(0), records(3), records(0)
    base[0]["task_id"] = small[0]["task_id"] = head[0]["task_id"] = "FA"
    head[0]["predicted_points_pixels"][0] += 0.01
    with pytest.raises(ValueError, match="diverges from base outside HC"):
        blend_family_records(base, small, head, mode="symmetric_five_seed")


def test_input_records_are_not_mutated():
    inputs = [records(0), records(3), records(6)]
    inputs[2][1] = copy.deepcopy(inputs[0][1])
    before = copy.deepcopy(inputs)
    blend_family_records(*inputs)
    assert inputs == before


def test_key_mismatch_is_rejected():
    small = records(3)
    small.pop()
    with pytest.raises(ValueError, match="keys differ"):
        blend_family_records(records(), small, records(6))


def test_duplicate_and_nonfinite_coordinates_are_rejected():
    duplicate = records() + [records()[0]]
    with pytest.raises(ValueError, match="duplicate"):
        keyed(duplicate, label="x")
    bad = records()
    bad[0]["predicted_points_pixels"][0] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        keyed(bad, label="x")


def test_landmark_shape_mismatch_is_rejected():
    small = records(3)
    small[0]["predicted_points_pixels"] += [4.0, 5.0]
    small[0]["predicted_points_normalized"] += [0.4, 0.5]
    with pytest.raises(ValueError, match="landmark shape"):
        blend_family_records(records(), small, records(6))


def test_training_manifest_set_is_exact_before_artifact_use(tmp_path):
    paths = []
    for family, seed in (("hcsmall", 42), ("hchead", 43)):
        path = tmp_path / f"{family}{seed}.json"
        path.write_text(json.dumps({"passed": True, "family": family, "seed": seed}))
        paths.append(path)
    with pytest.raises(ValueError, match="manifest set differs"):
        verify_training_manifests(paths, "pragmatic_seed42")
