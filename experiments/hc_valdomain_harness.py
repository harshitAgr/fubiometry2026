#!/usr/bin/env python3
"""Val-domain HC/femur evaluator built from pixel-identical public images.

A large fraction of the challenge's HC and fetal_femur VALIDATION images are
pixel-identical to images in the public annotated pool already vendored at
external/ucl_multicentre/... (FETAL_PLANES_DB, "FP"). Matching them by decoded
pixel hash gives a LABELLED test set drawn from the validation distribution --
which our 5-fold CV cannot represent (every CV HC image is single-centre HC18
at 800x540; no val HC image is).

EVALUATION ONLY. Never train on the matched images: doing so would both destroy
this harness and inflate the hidden test score unmeasurably.

Usage:  hc_valdomain_harness.py [--pred FILE ...] [--out JSON]
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, math, os, statistics, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
POOL = os.path.join(ROOT, 'external/ucl_multicentre/FetalBiometry-Multicentre-Landmarks-2026')
TASK_POOL = {'HC': ('FP/Head.csv', 'images/FP/Head'), 'fetal_femur': ('FP/Femur.csv', 'images/FP/Femur')}


def phash_exact(path):
    """md5 of the DECODED grayscale pixels (container/metadata independent)."""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    return hashlib.md5(im.tobytes()).hexdigest(), im.shape


def ellipse_perimeter(axis_a, axis_b):
    a, b = axis_a / 2, axis_b / 2
    return math.pi * (3 * (a + b) - math.sqrt((3 * a + b) * (a + 3 * b)))


def canonical_head(row):
    """Public CSV endpoint order is arbitrary; the challenge's convention is
    BPD -> smaller y first, OFD -> larger x first (holds in 999/999 HC train)."""
    b = [(float(row['bpd_1_x']), float(row['bpd_1_y'])), (float(row['bpd_2_x']), float(row['bpd_2_y']))]
    o = [(float(row['ofd_1_x']), float(row['ofd_1_y'])), (float(row['ofd_2_x']), float(row['ofd_2_y']))]
    b.sort(key=lambda p: p[1])            # smaller y first
    o.sort(key=lambda p: -p[0])           # larger x first
    return [b[0], b[1], o[0], o[1]]


def canonical_femur(row):
    p = [(float(row['fl_1_x']), float(row['fl_1_y'])), (float(row['fl_2_x']), float(row['fl_2_y']))]
    p.sort(key=lambda q: q[0])            # smaller x first (gate-verified below)
    return p


def load_preds(path, task):
    out = {}
    for r in json.load(open(path)):
        if r['task_id'] == task:
            v = r['predicted_points_pixels']
            out[r['image_path'].split('/')[-1]] = [(v[2 * i], v[2 * i + 1]) for i in range(len(v) // 2)]
    return out


def mre(pred, gt):
    return statistics.mean(math.dist(p, g) for p, g in zip(pred, gt))


def scale_about_centroid(pts, s):
    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    return [(cx + (x - cx) * s, cy + (y - cy) * s) for x, y in pts]


def build_index(task):
    csv_rel, img_rel = TASK_POOL[task]
    rows = {r['image_name']: r for r in csv.DictReader(open(os.path.join(POOL, 'annotations', csv_rel)))}
    idx = {}
    for p in glob.glob(os.path.join(POOL, img_rel, '*')):
        h = phash_exact(p)
        if h and os.path.basename(p) in rows:
            idx.setdefault(h[0], []).append(os.path.basename(p))
    return rows, idx


def match_val(task):
    rows, idx = build_index(task)
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, 'data/val/images/%s/*' % task))):
        h = phash_exact(p)
        if h and h[0] in idx:
            out[os.path.basename(p)] = idx[h[0]][0]
    return rows, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', nargs='*', default=[
        'submission/vitb_ens/regression_predictions.json',
        'submission/v14/regression_predictions.json',
        'submission/full_family_postdrop_seed42/regression_predictions.json'])
    ap.add_argument('--scales', nargs='*', type=float, default=[1.0, 0.99, 0.985, 0.98, 0.975, 0.97, 0.965])
    ap.add_argument('--out', default='experiments/results/hc_valdomain/harness.json')
    a = ap.parse_args()
    res = {}

    # --- gate 1: does the canonical-order rule reproduce our HC TRAIN GT? ----
    hc18 = {r['image_name']: r for r in csv.DictReader(
        open(os.path.join(POOL, 'annotations/HC18/Head.csv')))}
    ours = {}
    for r in csv.DictReader(open(os.path.join(ROOT, 'data/csv/HC_train.csv'))):
        ours[r['image_path'].split('/')[-1]] = [tuple(map(float, eval(r['point_%d_xy' % i]))) for i in (1, 2, 3, 4)]
    d = [mre(canonical_head(hc18[k]), ours[k]) for k in ours if k in hc18]
    d.sort()
    res['gate_canonical_order'] = dict(n=len(d), mean=statistics.mean(d), median=d[len(d) // 2],
                                       frac_le_1p5=sum(1 for x in d if x <= 1.5) / len(d),
                                       passed=statistics.mean(d) < 1.5)

    # --- gate 1b: femur canonical order vs our femur TRAIN GT ---------------
    fp_fem = {r['image_name']: r for r in csv.DictReader(
        open(os.path.join(POOL, 'annotations/FP/Femur.csv')))}
    ourf = {}
    for r in csv.DictReader(open(os.path.join(ROOT, 'data/csv/fetal_femur_train.csv'))):
        ourf[r['image_path'].split('/')[-1]] = [tuple(map(float, eval(r['point_%d_xy' % i]))) for i in (1, 2)]
    shared = [k for k in ourf if k in fp_fem]
    if shared:
        df = sorted(mre(canonical_femur(fp_fem[k]), ourf[k]) for k in shared)
        res['gate_femur_order'] = dict(n=len(df), mean=statistics.mean(df), median=df[len(df) // 2],
                                       frac_le_5px=sum(1 for x in df if x <= 5) / len(df),
                                       note='public-vs-challenge annotation disagreement on IDENTICAL pixels')

    # --- match validation images to the public pool -------------------------
    for task in ('HC', 'fetal_femur'):
        rows, m = match_val(task)
        n_val = len(glob.glob(os.path.join(ROOT, 'data/val/images/%s/*' % task)))
        res.setdefault('matching', {})[task] = dict(val_images=n_val, matched=len(m),
                                                    frac=len(m) / n_val if n_val else 0)
        res['matching'][task]['example'] = dict(list(m.items())[:3])
        canon = canonical_head if task == 'HC' else canonical_femur
        gt = {v: canon(rows[src]) for v, src in m.items()}
        per_pred = {}
        for pf in a.pred:
            if not os.path.exists(os.path.join(ROOT, pf)):
                continue
            P = load_preds(os.path.join(ROOT, pf), task)
            keys = [k for k in gt if k in P]
            if not keys:
                continue
            e = {}
            for s in ([1.0] if task != 'HC' else a.scales):
                errs = [mre(scale_about_centroid(P[k], s), gt[k]) for k in keys]
                if task == 'HC':
                    pm = [abs(ellipse_perimeter(math.dist(*scale_about_centroid(P[k], s)[:2]),
                                                math.dist(*scale_about_centroid(P[k], s)[2:]))
                              - ellipse_perimeter(math.dist(*gt[k][:2]), math.dist(*gt[k][2:]))) for k in keys]
                else:
                    pm = [abs(math.dist(*P[k]) - math.dist(*gt[k])) for k in keys]
                e['s=%.3f' % s] = dict(mre=statistics.mean(errs), param_mae=statistics.mean(pm))
            # predicted/GT size ratio at s=1
            ratio = []
            for k in keys:
                if task == 'HC':
                    ratio.append((math.dist(*P[k][:2]) + math.dist(*P[k][2:])) /
                                 (math.dist(*gt[k][:2]) + math.dist(*gt[k][2:])))
                else:
                    ratio.append(math.dist(*P[k]) / math.dist(*gt[k]))
            e['n'] = len(keys); e['pred_over_gt_size'] = dict(mean=statistics.mean(ratio),
                                                              median=sorted(ratio)[len(ratio) // 2])
            per_pred[pf] = e
        res.setdefault('scored', {})[task] = per_pred

    os.makedirs(os.path.join(ROOT, os.path.dirname(a.out)), exist_ok=True)
    with open(os.path.join(ROOT, a.out), 'w') as f:
        json.dump(res, f, indent=2)
    print(json.dumps(res, indent=2)[:4000])
    print('\nwrote', a.out)


if __name__ == '__main__':
    main()
