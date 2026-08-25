import json, glob, numpy as np, pandas as pd, pytest
from scoring import derive, param_specs


def _load_task_points(task_id):
    csv = glob.glob(f"data/csv/{task_id}_train.csv")
    if not csv:
        pytest.skip(f"{task_id} CSV not present")
    df = pd.read_csv(csv[0])
    pcols = sorted([c for c in df.columns if c.startswith("point_")],
                   key=lambda c: int(c.split("_")[1]))
    out = []
    for _, r in df.iterrows():
        pts = [json.loads(r[c]) if isinstance(r[c], str) else r[c]
               for c in pcols if pd.notna(r[c])]
        out.append(np.asarray(pts, float))
    return out


def test_aop_angle_in_clinical_range():
    pts = _load_task_points("AOP")
    aops = [derive.derive_parameters("AOP", p).get("aop") for p in pts]
    aops = [a for a in aops if a is not None]
    med = float(np.median(aops))
    assert 90.0 <= med <= 180.0, f"AOP median {med:.1f} outside clinical 90-180"


def test_hc_fa_perimeters_positive_and_sane():
    for tid in ("HC", "FA"):
        pts = _load_task_points(tid)
        vals = [list(derive.derive_parameters(tid, p).values())[0] for p in pts[:50]]
        assert all(v > 0 for v in vals)
        assert np.median(vals) > 100  # px-perimeter of a fetal head/abdomen is large
