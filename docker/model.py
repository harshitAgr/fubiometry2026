"""FU_Biometry test-phase inference — `class Model` required by the organizer entry script.

Reproduces the validated `experiments/infer_ensemble.py` path operation-for-operation:

    per GROUP: N checkpoints x V scale-TTA views -> per-view inverse warp back to canonical
    frame -> mean over all N*V heatmaps -> ONE sub-pixel soft-argmax decode (9x9 window)
    -> normalized coords -> original-image pixels;
    then a per-task weighted mean over groups in COORDINATE space (ROUTES), and finally the
    three post-processing levers, in this order (the order v22 -> v23 -> v24 was built and
    officially scored in; the scale and the projection commute to 2.3e-13 px anyway):

        1. size-gated HC ellipse-scale correction         hc_scale.py         (HC only)
        2. length-preserving label-geometry projection     geometry_project.py (FA + HC)
        3. gated IVC caliper-length calibration            ivc_calibrate.py    (IVC only)

    Each module carries its own evidence and its own reason for being trusted; see their
    docstrings. NOTE this composite was never itself submitted: v24 (882078, 23.4231 MRE /
    27.2313 MAE) used HC scale s=0.950, and we deliberately ship the externally-fitted
    s=0.975 instead. Deterministic val equivalent of what ships: ~23.436 / ~27.320.

Deliberate differences from the research driver, each for a container-specific reason:

* No albumentations / pandas. `A.Resize` and `A.Normalize` are replicated exactly in cv2+numpy
  (verified numerically against the reference pipeline) and the metadata is read with the stdlib
  csv module. Fewer pinned deps == less to break in an offline image we cannot fully test on the
  target GPU.
* One output row per metadata row, always. The baseline `InferenceDataset` silently recurses to
  the NEXT index when an image fails to load, which emits a duplicate and drops the failed row;
  the challenge's evaluation policy penalizes missing outputs (Evaluation 4.2, "missing outputs
  are penalized"), so an unreadable image falls back to the image centre instead of vanishing.
* Images are batched within a task (the driver ran one image at a time), with an automatic
  halve-and-retry on CUDA OOM. BatchNorm is in eval mode, so batching is numerically equivalent.
* Checkpoints are loaded one at a time and canonical heatmaps are accumulated on CPU. This
  preserves the exact ensemble calculation while only one ViT-B occupies GPU/RAM at a time.
* `mem_frac` capping is dropped -- the container owns the whole GPU.
* HC scale-norm is NOT wired: tested on validation (v13, 836592) and rejected as an MRE lever.
  (Distinct from the HC ellipse-scale CORRECTION in hc_scale.py, which IS wired and was
  confirmed officially by the v19 -> v21 A/B. See that module's docstring.)
"""
from __future__ import annotations

import csv
import json
import math
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fub_decode as D                                          # noqa: E402
from fub_arch import INPUT_SIZE, NORM_MEAN, NORM_STD, TASK_SPEC, load_member  # noqa: E402
from geometry_project import project                             # noqa: E402
from hc_scale import apply_hc_scale                             # noqa: E402
from ivc_calibrate import apply_ivc_calibration                 # noqa: E402

# /app in the container. Overridable ONLY so the identical code can be numerically diffed against
# experiments/infer_ensemble.py outside Docker (see docker/verify_against_reference.py); the
# organizer entry script never sets it, so the container always uses /app.
APP_DIR = os.environ.get("FUB_APP_DIR", "/app")

# Model FAMILIES, in load order. predict.py copies ONLY /app/best_model.pth into the work dir,
# so every other checkpoint is read straight from /app (explicitly allowed by the guide).
#
# The deployed family is the full-data continuation of the audited DINOv2 ViT-B checkpoint:
#   base     = 5 continued-DINO seeds, averaged in HEATMAP space, decoded once (window 9)
#   hcsmall  = the continued-DINO geo_v1_hcsmall model, seed 42, decoded on its own
#   hchead   = the continued-DINO HC-head-refined model, seed 42, decoded on its own
# and the three are then combined in COORDINATE space per task (see ROUTES).
MEMBER_GROUPS = {
    "base": ["best_model.pth", "model_s43.pth", "model_s44.pth",
             "model_s45.pth", "model_s46.pth"],
    "hcsmall": ["model_hcsmall.pth"],
    "hchead": ["model_hchead.pth"],
}

# Per-task coordinate-space combination weights over (base, hcsmall, hchead).
# Exactly the `pragmatic_seed42` realization in experiments/full_family_candidate.py:
#   IVC -> exact base passthrough; HC -> (base + hcsmall + hchead)/3;
#   every other task -> (2*base + hcsmall)/3.
# R42's non-HC tensors equal the single seed-42 base, not the lower-variance 5-seed family,
# which is why hchead is routed to HC only.
ROUTE_DEFAULT = {"base": 2.0 / 3.0, "hcsmall": 1.0 / 3.0}
ROUTES = {
    "IVC": {"base": 1.0},
    "HC": {"base": 1.0 / 3.0, "hcsmall": 1.0 / 3.0, "hchead": 1.0 / 3.0},
}

# Adopted inference recipe: heatmap-space scale-TTA + soft decode.
# Window 9 (not 7): every artifact from v15 onward uses it, including v19/v21/v22.
TTA_SCALES = (0.92, 1.08)
DECODE_METHOD = "soft"
DECODE_WINDOW = 9

# Hard cap on images per forward, independent of the organizer's batch_size=8. 8 images x 3
# views = 24 tensors at 518^2 through a ViT-B is the ONE memory figure never measured on the
# sm_86 eval host, and runtime has ~10x slack (21 forwards/img ~= 1.4 s on a 3080, so even
# 5,000 test images fit ~2 h of the 5 h budget). There is no upside to risking the OOM path.
MAX_IMAGES_PER_FORWARD = 4

# Crash-loud thresholds for centre fallbacks. Rationale: the test phase grants 10 retryable
# attempts but locks in the FIRST SUCCESSFUL run, so a crash is cheap and a run that
# "succeeds" with a silently centre-filled task is unrecoverable. One straggler per task
# stays tolerated (a single corrupt file should not throw away a good submission); a
# systematic read failure -- a path/naming surprise like the Drive PSAK typo -- must abort.
MAX_FALLBACK_FRACTION_PER_TASK = 0.01
MAX_FALLBACK_ABS_PER_TASK = 1
MAX_FALLBACK_TOTAL = 5

# cv2 with 4 cores and a 7 GB cap: keep OpenCV from oversubscribing threads.
cv2.setNumThreads(2)


def _resize(img, size=INPUT_SIZE):
    """Equivalent to albumentations A.Resize(size, size) (default INTER_LINEAR)."""
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)


def _to_tensor(img_uint8):
    """Equivalent to A.Compose([A.Normalize(MEAN, STD), ToTensorV2()]).

    albumentations Normalize is (img/255 - mean)/std at max_pixel_value=255; ToTensorV2 is a
    HWC->CHW float32 transpose.
    """
    x = img_uint8.astype(np.float32) / 255.0
    x = (x - np.asarray(NORM_MEAN, np.float32)) / np.asarray(NORM_STD, np.float32)
    return torch.from_numpy(np.ascontiguousarray(x.transpose(2, 0, 1)))


def _scaled_view(img_uint8, s):
    """Forward-warp a uint8 RGB image by scale `s` about its centre (matches infer_ensemble)."""
    if s == 1.0:
        return img_uint8
    warped = np.stack(
        [D.warp_affine(img_uint8[..., c].astype(np.float64), s, inverse=False) for c in range(3)],
        axis=-1,
    )
    return np.clip(warped, 0, 255).astype(np.uint8)


def _read_rgb(path):
    """BGR->RGB uint8, or None. Mirrors the baseline's cv2.imread + cvtColor."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _resolve(data_root, rel_path):
    """Baseline path resolution: try {data_root}/images/<rel> then {data_root}/<rel>."""
    cleaned = os.path.normpath(rel_path)
    while cleaned.startswith(".." + os.sep):
        cleaned = cleaned[3:]
    for root in (os.path.join(data_root, "images"), data_root):
        cand = os.path.normpath(os.path.join(root, cleaned))
        if os.path.isfile(cand):
            return cand
    return None


class Model:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[model] device={self.device}", flush=True)

        self.groups = {}
        for gname, names in MEMBER_GROUPS.items():
            paths = []
            for name in names:
                path = os.path.join(APP_DIR, name)
                if not os.path.isfile(path):
                    # Every checkpoint is COPY'd in at build time, so absence means a broken
                    # image, not a runtime condition. Fail loud: a retryable crash costs one
                    # of 10 attempts, whereas a quietly degraded ensemble would SUCCEED and
                    # lock in permanently as our one scored submission.
                    raise FileNotFoundError(
                        f"ensemble member missing from the image: {path}. Expected all of "
                        f"{ {g: v for g, v in MEMBER_GROUPS.items()} }."
                    )
                paths.append(path)
            self.groups[gname] = paths

        missing = [g for g in MEMBER_GROUPS if g not in self.groups]
        if missing:
            raise FileNotFoundError(f"ensemble groups failed to load: {missing}")

        self.view_scales = [1.0] + list(TTA_SCALES)
        n = sum(len(v) for v in self.groups.values())
        print(f"[model] {n} sequential members in {len(self.groups)} group(s) x "
              f"{len(self.view_scales)} views = {n * len(self.view_scales)} forwards/image",
              flush=True)

    def _route(self, task_id):
        """Coordinate-space weights for this task, normalised to sum to 1.

        __init__ now hard-requires every group, so the filter below can never drop one; it is
        kept as a normalisation invariant (the ROUTES literals sum to 1 only up to float
        rounding, e.g. 3 x 1/3) rather than as a degradation path."""
        w = {g: x for g, x in ROUTES.get(task_id, ROUTE_DEFAULT).items() if g in self.groups}
        if not w:
            raise KeyError(f"no ensemble group available for task {task_id!r}")
        total = sum(w.values())
        return {g: x / total for g, x in w.items()}

    # ---------------------------------------------------------------- inference

    def _forward_member(self, model, views, task_id, n_img, n_view):
        """Return one member's canonical heatmap sum over all TTA views."""
        hm = torch.sigmoid(model(views, task_id=task_id)).float().cpu().numpy()
        k, hh, ww = hm.shape[1], hm.shape[2], hm.shape[3]
        hm = hm.reshape(n_img, n_view, k, hh, ww)
        for vi, s in enumerate(self.view_scales):
            if s == 1.0:
                continue
            for bi in range(n_img):
                hm[bi, vi] = np.stack(
                    [D.warp_affine(hm[bi, vi, c], s, inverse=True) for c in range(k)], 0
                )
        return hm.sum(axis=1)

    def _predict_group(self, paths, items, task_id, n_per_fwd):
        """Predict one family with only one checkpoint resident at a time.

        The per-image CPU accumulators preserve model-order float32 additions. Decoding happens
        once after division by ``members * views``, exactly as in the research driver.
        """
        n_view = len(self.view_scales)
        heatmap_acc = [None] * len(items)
        group_n_per_fwd = n_per_fwd
        for path in paths:
            model = load_member(path, self.device)
            print(f"[model] loaded {task_id}/{os.path.basename(path)}", flush=True)
            i = 0
            while i < len(items):
                chunk = items[i:i + group_n_per_fwd]
                tens = []
                for _, img in chunk:
                    for s in self.view_scales:
                        tens.append(_to_tensor(_scaled_view(img, s)))
                batch = torch.stack(tens, 0).to(self.device)
                try:
                    with torch.no_grad():
                        summed = self._forward_member(
                            model, batch, task_id, len(chunk), n_view)
                except torch.cuda.OutOfMemoryError:
                    del batch
                    torch.cuda.empty_cache()
                    if group_n_per_fwd == 1:
                        raise
                    group_n_per_fwd = max(1, group_n_per_fwd // 2)
                    print(f"[model] CUDA OOM -> retrying at {group_n_per_fwd} "
                          f"images/forward", flush=True)
                    continue
                del batch
                for offset, member_sum in enumerate(summed):
                    slot = i + offset
                    heatmap_acc[slot] = (
                        member_sum if heatmap_acc[slot] is None
                        else heatmap_acc[slot] + member_sum
                    )
                i += len(chunk)
            del model
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
            print(f"[model] released {task_id}/{os.path.basename(path)}", flush=True)

        if any(h is None for h in heatmap_acc):
            raise RuntimeError(f"{task_id}: internal unfilled heatmap accumulator")

        out = []
        denom = float(len(paths) * n_view)
        for i in range(0, len(items), group_n_per_fwd):
            chunk = items[i:i + group_n_per_fwd]
            avg = np.stack(heatmap_acc[i:i + len(chunk)], 0) / denom
            coords = D.decode_subpixel(avg, method=DECODE_METHOD, window=DECODE_WINDOW)
            for (rec, _), c in zip(chunk, coords):
                h0, w0 = rec["hw"]
                px = []
                for j in range(0, len(c), 2):
                    px += [float(c[j]) * w0, float(c[j + 1]) * h0]
                out.append((rec, px))
        return out

    def predict(self, data_root: str, output_dir: str, batch_size: int = 8):
        meta_path = os.path.join(data_root, "csv", "test_metadata.csv")
        with open(meta_path, newline="") as f:
            records = list(csv.DictReader(f))
        print(f"[model] {len(records)} rows in test_metadata.csv", flush=True)

        # Group by task: one head / heatmap size per forward.
        by_task = {}
        for idx, r in enumerate(records):
            by_task.setdefault(r["task_id"], []).append(idx)
        print(f"[model] tasks: { {k: len(v) for k, v in sorted(by_task.items())} }", flush=True)

        # batch_size is the organizer's image budget; each image expands to n_view tensors.
        n_per_fwd = max(1, min(int(batch_size), MAX_IMAGES_PER_FORWARD))
        if n_per_fwd != int(batch_size):
            print(f"[model] batch_size {batch_size} -> {n_per_fwd} images/forward "
                  f"(x{len(self.view_scales)} views), capped for VRAM headroom", flush=True)

        unknown = sorted(t for t in by_task if t not in TASK_SPEC)
        if unknown:
            # A task we have no head for cannot be predicted, only centre-filled -- which
            # would SUCCEED and lock in a worst-possible score for that whole task. Abort
            # instead and spend a retry on fixing the name mapping.
            raise KeyError(
                f"unknown task_id(s) in test_metadata.csv: {unknown}. Known: "
                f"{sorted(TASK_SPEC)}. Refusing to centre-fill an entire task."
            )

        n_fallback_total = 0
        results = [None] * len(records)
        for task_id, idxs in sorted(by_task.items()):
            loaded, n_fail, failed_paths = [], 0, []
            for i in idxs:
                r = records[i]
                abs_path = _resolve(data_root, r["image_path"])
                img = _read_rgb(abs_path) if abs_path else None
                if img is None:
                    n_fail += 1
                    if len(failed_paths) < 10:
                        failed_paths.append(r["image_path"])
                    results[i] = self._fallback(r)
                    continue
                h0, w0 = img.shape[:2]
                loaded.append(({"i": i, "row": r, "hw": (h0, w0)}, _resize(img)))

            n_fallback_total += n_fail
            budget = max(MAX_FALLBACK_ABS_PER_TASK,
                         int(MAX_FALLBACK_FRACTION_PER_TASK * len(idxs)))
            if n_fail > budget or n_fallback_total > MAX_FALLBACK_TOTAL:
                raise IOError(
                    f"{task_id}: {n_fail}/{len(idxs)} images unreadable (budget {budget}; "
                    f"{n_fallback_total} total, cap {MAX_FALLBACK_TOTAL}). This is a "
                    f"systematic path/naming failure, not a corrupt file. First failures: "
                    f"{failed_paths}"
                )
            if n_fail:
                print(f"[model] WARNING: {task_id}: {n_fail} unreadable image(s) -> centre "
                      f"fallback (within budget {budget}): {failed_paths}", flush=True)

            # Each group is decoded independently, then combined in COORDINATE space.
            route = self._route(task_id)
            acc = {}
            for gname, weight in sorted(route.items()):
                for rec, px in self._predict_group(
                        self.groups[gname], loaded, task_id, n_per_fwd):
                    slot = acc.get(rec["i"])
                    if slot is None:
                        slot = acc[rec["i"]] = (rec, [0.0] * len(px))
                    elif len(slot[1]) != len(px):
                        raise ValueError(f"{task_id}: group {gname} returned {len(px)} coords, "
                                         f"expected {len(slot[1])}")
                    for j, v in enumerate(px):
                        slot[1][j] += weight * v

            n_scaled = n_projected = n_calibrated = 0
            for rec, px in acc.values():
                h0, w0 = rec["hw"]
                # Post-processing, in the officially-scored order. Each step is a no-op for
                # tasks it does not own, so the chain is safe to run unconditionally.
                px_out = apply_hc_scale(px, task_id, w0, h0)
                n_scaled += px_out != px
                px_proj = project(px_out, task_id)
                n_projected += px_proj != px_out
                px_final = apply_ivc_calibration(px_proj, task_id)
                n_calibrated += px_final != px_proj

                if len(px_final) != 2 * TASK_SPEC[task_id] or not all(
                        math.isfinite(v) for v in px_final):
                    raise ValueError(
                        f"{task_id}/{rec['row']['image_path']}: post-processing produced "
                        f"{len(px_final)} coords (expected {2 * TASK_SPEC[task_id]}) or a "
                        f"non-finite value: {px_final}"
                    )
                results[rec["i"]] = {
                    "image_path": rec["row"]["image_path"],
                    "task_id": task_id,
                    "predicted_points_pixels": px_final,
                }
            steps = [(n_scaled, "HC scale"), (n_projected, "projection"),
                     (n_calibrated, "IVC gate")]
            fired = ", ".join(f"{lbl} on {n}/{len(loaded)}" for n, lbl in steps if n)
            print(f"[model] {task_id}: route={ {g: round(w, 4) for g, w in sorted(route.items())} }"
                  + (f", {fired}" if fired else ""), flush=True)
            print(f"[model] {task_id}: {len(loaded)} done", flush=True)

        assert all(r is not None for r in results), "internal: unfilled prediction slot"
        self._save(results, output_dir)
        return results

    @staticmethod
    def _fallback(record):
        """Centre-of-image prediction for a row we could not run. Uses the metadata h/w so it
        works even when the image itself is unreadable."""
        task_id = record["task_id"]
        k = TASK_SPEC.get(task_id)
        if k is None:
            k = int(record.get("num_classes") or 1)
        try:
            h0, w0 = float(record["height"]), float(record["width"])
        except (KeyError, TypeError, ValueError):
            h0, w0 = 1.0, 1.0
        return {
            "image_path": record["image_path"],
            "task_id": task_id,
            "predicted_points_pixels": [w0 / 2.0, h0 / 2.0] * k,
        }

    @staticmethod
    def _save(results, output_dir):
        # allow_nan=False makes a NaN/Inf a retryable CRASH rather than the literal `NaN`
        # token, which is invalid JSON: the organizers' scorer would either reject the file
        # (losing everything) or, worse, accept the run as successful and lock it in.
        for r in results:
            bad = [v for v in r["predicted_points_pixels"] if not math.isfinite(v)]
            if bad:
                raise ValueError(f"non-finite coordinate in {r['task_id']}/{r['image_path']}")
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "regression_predictions.json")
        with open(path, "w") as f:
            json.dump(results, f, allow_nan=False)
        print(f"[model] wrote {len(results)} predictions to {path}", flush=True)
