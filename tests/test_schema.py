import json
import numpy as np
import pytest
from scoring import schema

VALID = [
    {"image_path": "A4C/x.jpg", "task_id": "A4C",
     "predicted_points_normalized": [0.1, 0.2, 0.3, 0.4],
     "predicted_points_pixels": [10.0, 20.0, 30.0, 40.0]},
]

def test_validate_accepts_valid(tmp_path):
    p = tmp_path / "pred.json"
    p.write_text(json.dumps(VALID))
    records = schema.load_submission(p)
    schema.validate_submission(records)  # must not raise
    assert records[0]["task_id"] == "A4C"

def test_points_to_array_is_Kx2():
    arr = schema.points_to_array([10.0, 20.0, 30.0, 40.0])
    assert arr.shape == (2, 2)
    assert np.allclose(arr, [[10, 20], [30, 40]])

def test_validate_rejects_odd_length():
    bad = [{"image_path": "a", "task_id": "A4C",
            "predicted_points_normalized": [0.1, 0.2, 0.3],
            "predicted_points_pixels": [1.0, 2.0, 3.0]}]
    with pytest.raises(schema.SchemaError):
        schema.validate_submission(bad)

def test_validate_rejects_missing_key():
    with pytest.raises(schema.SchemaError):
        schema.validate_submission([{"task_id": "A4C",
            "predicted_points_normalized": [], "predicted_points_pixels": []}])
