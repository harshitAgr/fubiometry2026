import numpy as np
from experiments import semisup as S


def _gauss(H, W, cy, cx, sigma=1.8, amp=1.0):
    yy, xx = np.mgrid[0:H, 0:W]
    return amp * np.exp(-((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * sigma ** 2))


def test_prominence_weight_high_for_sharp_peak():
    # one sharp landmark -> prominence (max-median) is large -> weight 1.0
    hm = _gauss(64, 64, 30, 40)[None]                 # [K=1, H, W]
    w = S.prominence_weight(hm, floor=0.05)
    assert w.shape == (1,)
    assert abs(w[0] - 1.0) < 1e-9


def test_prominence_weight_zero_below_floor():
    # a near-flat map: max-median tiny -> below floor -> weight 0
    hm = np.full((1, 64, 64), 0.5)
    hm[0, 0, 0] += 1e-4                                # max barely above median
    w = S.prominence_weight(hm, floor=0.05)
    assert w[0] == 0.0


def test_prominence_weight_monotone_in_prominence():
    # two landmarks, different peak amplitudes -> the sharper one gets >= weight
    hm = np.stack([_gauss(64, 64, 30, 40, amp=1.0), _gauss(64, 64, 10, 10, amp=0.2)], 0)
    w = S.prominence_weight(hm, floor=0.0)
    assert w[0] >= w[1]
    assert w[1] >= 0.0


def test_prominence_weight_per_task_floor_lookup():
    # the gating helper resolves a per-task floor (dict) with a default fallback
    hm = np.stack([_gauss(64, 64, 30, 40, amp=0.1)], 0)   # modest prominence
    floors = {"AOP": 0.30, "_default": 0.02}
    w_aop = S.prominence_weight(hm, floor=S.task_floor(floors, "AOP"))   # high floor -> 0
    w_hc = S.prominence_weight(hm, floor=S.task_floor(floors, "HC"))     # default -> kept
    assert w_aop[0] == 0.0
    assert w_hc[0] > 0.0


def test_lambda_ramp_zero_at_start_max_at_end():
    assert S.lambda_ramp(0, ramp_steps=100, lam_max=1.0) < 0.02     # near 0 at step 0
    assert abs(S.lambda_ramp(100, ramp_steps=100, lam_max=1.0) - 1.0) < 1e-6  # clamped to max
    assert abs(S.lambda_ramp(500, ramp_steps=100, lam_max=2.0) - 2.0) < 1e-6  # held past ramp


def test_lambda_ramp_monotone_nondecreasing():
    vals = [S.lambda_ramp(t, ramp_steps=50, lam_max=1.0) for t in range(0, 60, 5)]
    assert all(b >= a - 1e-9 for a, b in zip(vals, vals[1:]))


def test_lambda_ramp_midpoint_is_half_max():
    # sigmoid centred on the ramp midpoint -> ~half of lam_max at t = ramp_steps/2
    mid = S.lambda_ramp(50, ramp_steps=100, lam_max=1.0)
    assert abs(mid - 0.5) < 1e-6


def test_ema_alpha_ramp_endpoints_and_clamp():
    assert abs(S.ema_alpha(0, ramp_steps=100) - S.EMA_ALPHA_START) < 1e-9
    assert abs(S.ema_alpha(100, ramp_steps=100) - S.EMA_ALPHA_END) < 1e-9
    assert abs(S.ema_alpha(9999, ramp_steps=100) - S.EMA_ALPHA_END) < 1e-9   # held after ramp


def test_ema_alpha_is_linear_midpoint():
    mid = S.ema_alpha(50, ramp_steps=100)
    assert abs(mid - (S.EMA_ALPHA_START + S.EMA_ALPHA_END) / 2) < 1e-9


import pandas as pd


def test_normalize_labeled_path_strips_data_images_prefix():
    assert S.normalize_labeled_path("data/images/A4C/x.png") == "A4C/x.png"
    assert S.normalize_labeled_path("A4C/x.png") == "A4C/x.png"   # already-normalized passthrough


def test_fold_val_phashes_selects_only_fold_k_val_rows():
    folds = pd.DataFrame({
        "image_path": ["A4C/a.png", "A4C/b.png", "AOP/c.png"],
        "task_id": ["A4C", "A4C", "AOP"], "fold": [0, 1, 0], "group": ["", "", ""]})
    labeled = pd.DataFrame({
        "image_path": ["data/images/A4C/a.png", "data/images/A4C/b.png", "data/images/AOP/c.png"],
        "task_id": ["A4C", "A4C", "AOP"], "phash": [10, 20, 30]})
    ph = S.fold_val_phashes(folds, labeled, fold=0)
    assert sorted(ph) == [10, 30]          # a.png (fold0) + c.png (fold0); b.png is fold1


def test_filter_pool_for_fold_drops_near_val_matches():
    # pool has 3 images; one (phash 10) exactly matches a fold-0-val labeled phash -> dropped
    pool = pd.DataFrame({"task_id": ["A4C", "A4C", "AOP"],
                         "image_path": ["A4C/p0.png", "A4C/p1.png", "AOP/p2.png"],
                         "phash": [10, 999, 8000]})
    folds = pd.DataFrame({"image_path": ["A4C/a.png"], "task_id": ["A4C"], "fold": [0], "group": [""]})
    labeled = pd.DataFrame({"image_path": ["data/images/A4C/a.png"], "task_id": ["A4C"], "phash": [10]})
    kept = S.filter_pool_for_fold(pool, folds, labeled, fold=0, thresh=0)
    assert list(kept["phash"]) == [999, 8000]          # exact match dropped; others kept
    assert "A4C/p0.png" not in set(kept["image_path"])


def test_filter_pool_for_fold_is_leak_free_assertion_helper():
    # after filtering, ZERO kept pool phashes are within thresh of any fold-K-val phash
    pool = pd.DataFrame({"task_id": ["A4C", "A4C"], "image_path": ["A4C/p0.png", "A4C/p1.png"],
                         "phash": [0, 7]})           # 7 = hamming 3 from 0
    folds = pd.DataFrame({"image_path": ["A4C/a.png"], "task_id": ["A4C"], "fold": [0], "group": [""]})
    labeled = pd.DataFrame({"image_path": ["data/images/A4C/a.png"], "task_id": ["A4C"], "phash": [0]})
    kept = S.filter_pool_for_fold(pool, folds, labeled, fold=0, thresh=2)   # drop within 2 of 0
    refs = S.fold_val_phashes(folds, labeled, fold=0)
    assert S.count_near(kept["phash"].tolist(), refs, thresh=2) == 0        # the leak assertion == 0


def test_task_balanced_indices_groups_by_task():
    man = pd.DataFrame({"task_id": ["A4C", "A4C", "A4C", "AOP"],
                        "image_path": ["A4C/0", "A4C/1", "A4C/2", "AOP/0"], "phash": [0, 1, 2, 3]})
    by_task = S.indices_by_task(man)
    assert set(by_task) == {"A4C", "AOP"}
    assert sorted(by_task["A4C"]) == [0, 1, 2]
    assert by_task["AOP"] == [3]


def test_task_balanced_batches_are_task_uniform_and_recycle_small_tasks():
    # A4C huge, AOP tiny: over many batches each task is chosen ~equally (uniform over TASKS,
    # not images), so AOP's one image recycles often -> A4C volume does not swamp it.
    man = pd.DataFrame({"task_id": ["A4C"] * 100 + ["AOP"] * 1,
                        "image_path": [f"A4C/{i}" for i in range(100)] + ["AOP/0"],
                        "phash": list(range(101))})
    import random
    random.seed(0)
    batches = S.task_balanced_batches(man, batch_size=2, steps=400)
    assert len(batches) == 400
    assert all(len(b) == 2 for b in batches)
    # all indices valid
    assert all(0 <= i < len(man) for b in batches for i in b)
    # every index in a batch belongs to ONE task (a batch never mixes tasks)
    task_of = man["task_id"].tolist()
    assert all(len({task_of[i] for i in b}) == 1 for b in batches)
    # AOP appears in ~half the batches (uniform over the 2 tasks), despite being 1/101 of images
    aop_batches = sum(1 for b in batches if task_of[b[0]] == "AOP")
    assert 140 < aop_batches < 260      # ~200 of 400; wide band for seed robustness


def test_task_balanced_skips_absent_tasks():
    # fetal_femur has no rows in the manifest -> never sampled (no KeyError)
    man = pd.DataFrame({"task_id": ["HC", "HC"], "image_path": ["HC/0", "HC/1"], "phash": [0, 1]})
    import random
    random.seed(1)
    batches = S.task_balanced_batches(man, batch_size=2, steps=10)
    task_of = man["task_id"].tolist()
    assert all(task_of[b[0]] == "HC" for b in batches)
