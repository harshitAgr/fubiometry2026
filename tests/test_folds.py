import pandas as pd
from experiments import folds as F


def _df():
    rows = []
    rows += [{"image_path": f"AOP/{i:05d}.jpg", "task_id": "AOP"} for i in range(1, 101)]
    for clip in range(5):
        rows += [{"image_path": f"A4C/DCM_IM_{clip:04d}_frame{f:03d}.png", "task_id": "A4C"}
                 for f in range(5)]
    return pd.DataFrame(rows)


def test_deterministic_same_seed():
    a = F.make_folds(_df(), k=5, guard=2, seed=0)
    b = F.make_folds(_df(), k=5, guard=2, seed=0)
    assert (a["fold"].values == b["fold"].values).all()


def test_groups_never_split_across_folds():
    out = F.make_folds(_df(), k=5, guard=2, seed=0)
    card = out[out.task_id == "A4C"]
    for grp, sub in card.groupby("group"):
        assert sub["fold"].nunique() == 1, f"group {grp} split across folds"


def test_each_task_spans_all_folds():
    out = F.make_folds(_df(), k=5, guard=2, seed=0)
    for tid in ("AOP", "A4C"):
        folds_used = set(out[(out.task_id == tid) & (out.fold >= 0)]["fold"])
        assert folds_used == {0, 1, 2, 3, 4}


def test_aop_has_zero_adjacent_frame_leak():
    out = F.make_folds(_df(), k=5, guard=2, seed=0)
    assert F.aop_adjacent_leak(out) == 0.0


def test_no_clip_split_by_true_group_key():
    from experiments.groups import group_key
    out = F.make_folds(_df(), k=5, guard=2, seed=0)
    card = out[out.task_id == "A4C"].copy()
    card["true_group"] = [group_key("A4C", p) for p in card["image_path"]]
    for gk, sub in card.groupby("true_group"):
        assert sub["fold"].nunique() == 1, f"clip {gk} split across folds {set(sub.fold)}"
