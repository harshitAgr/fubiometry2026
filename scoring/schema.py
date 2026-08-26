"""Load and validate the FU_Biometry submission JSON (verified flat-array schema)."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

REQUIRED_KEYS = ("image_path", "task_id",
                 "predicted_points_normalized", "predicted_points_pixels")

class SchemaError(ValueError):
    pass

def load_submission(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text())
    if not isinstance(data, list):
        raise SchemaError(f"submission root must be a JSON array, got {type(data).__name__}")
    return data

def points_to_array(flat: list[float]) -> np.ndarray:
    a = np.asarray(flat, dtype=float)
    if a.ndim != 1 or a.size % 2 != 0:
        raise SchemaError(f"point list must be flat even-length, got size {a.size}")
    return a.reshape(-1, 2)

def validate_submission(records: list[dict]) -> None:
    if not isinstance(records, list):
        raise SchemaError("records must be a list")
    for i, r in enumerate(records):
        for k in REQUIRED_KEYS:
            if k not in r:
                raise SchemaError(f"record {i} missing key {k!r}")
        for k in ("predicted_points_normalized", "predicted_points_pixels"):
            v = r[k]
            if not isinstance(v, list) or len(v) % 2 != 0:
                raise SchemaError(f"record {i} field {k!r} must be flat even-length list")
