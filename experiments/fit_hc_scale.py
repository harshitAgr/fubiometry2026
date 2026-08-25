#!/usr/bin/env python3
"""Honest out-of-sample fit of the HC ellipse-scale correction.

FIT on 1,484 patient-disjoint FETAL_PLANES_DB head images (val domain, never
evaluated, unseen in training). FREEZE the scale. APPLY to the 151 challenge
validation images that are pixel-identical to public FP images.

No training. Inference-only. Writes experiments/results/hc_valdomain/scale_fit.json.
"""
from __future__ import annotations
import csv, json, math, os, statistics, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from experiments.hc_valdomain_harness import (canonical_head, ellipse_perimeter,  # noqa: E402
                                              load_preds, match_val, mre, scale_about_centroid)
POOL = os.path.join(ROOT, 'external/ucl_multicentre/FetalBiometry-Multicentre-Landmarks-2026')


def metrics(P, G, s):
    keys = sorted(set(P) & set(G))
    e, pm, ratio = [], [], []
    for k in keys:
        p = scale_about_centroid(P[k], s); g = G[k]
        e.append(mre(p, g))
        pm.append(abs(ellipse_perimeter(math.dist(*p[:2]), math.dist(*p[2:]))
                      - ellipse_perimeter(math.dist(*g[:2]), math.dist(*g[2:]))))
        ratio.append((math.dist(*p[:2]) + math.dist(*p[2:])) / (math.dist(*g[:2]) + math.dist(*g[2:])))
    ratio.sort()
    return dict(n=len(keys), mre=statistics.mean(e), param_mae=statistics.mean(pm),
                size_ratio_mean=statistics.mean(ratio), size_ratio_median=ratio[len(ratio) // 2],
                size_ratio_p10=ratio[int(.1 * len(ratio))], size_ratio_p90=ratio[int(.9 * len(ratio))],
                note='mean is outlier-sensitive (a collapsed prediction can give ratio>100); use the median')


def main():
    res = {}
    # ---- FIT set: the 1,484 held-out FP heads --------------------------------
    fit_gt = {}
    for r in csv.DictReader(open(os.path.join(ROOT, 'data/_fpheads/gt.csv'))):
        fit_gt[r['image_path'].split('/')[-1]] = [tuple(map(float, eval(r['point_%d_xy' % i]))) for i in (1, 2, 3, 4)]
    fit_P = load_preds(os.path.join(ROOT, 'submission/fp_head_probe/regression_predictions.json'), 'HC')
    grid = [1.0 - i * 0.0025 for i in range(25)]
    curve = {('%.4f' % s): metrics(fit_P, fit_gt, s) for s in grid}
    best_mae = min(curve.items(), key=lambda kv: kv[1]['param_mae'])
    best_mre = min(curve.items(), key=lambda kv: kv[1]['mre'])
    s_star = float(best_mae[0])
    res['fit'] = dict(set='FP heads, patient-disjoint from val, unseen in training',
                      n=curve['1.0000']['n'], baseline=curve['1.0000'],
                      s_star_param_mae=s_star, at_s_star=best_mae[1],
                      s_star_mre=float(best_mre[0]), best_mre=best_mre[1],
                      curve={k: v for k, v in curve.items() if float(k) >= 0.95})

    # ---- EVAL set: the 151 val-matched images, frozen s ----------------------
    rows, m = match_val('HC')
    ev_gt = {v: canonical_head(rows[src]) for v, src in m.items()}
    res['eval'] = {}
    for pf in ['submission/vitb_ens/regression_predictions.json',
               'submission/v14/regression_predictions.json',
               'submission/full_family_postdrop_seed42/regression_predictions.json']:
        P = load_preds(os.path.join(ROOT, pf), 'HC')
        base, corr = metrics(P, ev_gt, 1.0), metrics(P, ev_gt, s_star)
        res['eval'][pf] = dict(baseline=base, at_frozen_s=corr,
                               delta_mre=corr['mre'] - base['mre'],
                               delta_param_mae=corr['param_mae'] - base['param_mae'])
    # per native-size subgroup sign consistency (uses the deployed artifact)
    import cv2
    P = load_preds(os.path.join(ROOT, 'submission/full_family_postdrop_seed42/regression_predictions.json'), 'HC')
    groups = {}
    for k in ev_gt:
        im = cv2.imread(os.path.join(ROOT, 'data/val/images/HC', k), cv2.IMREAD_GRAYSCALE)
        groups.setdefault('%dx%d' % (im.shape[1], im.shape[0]), []).append(k)
    res['subgroups'] = {}
    for g, ks in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        sub_P = {k: P[k] for k in ks if k in P}; sub_G = {k: ev_gt[k] for k in ks}
        b, c = metrics(sub_P, sub_G, 1.0), metrics(sub_P, sub_G, s_star)
        loc = min(((s, metrics(sub_P, sub_G, s)['param_mae']) for s in grid), key=lambda x: x[1])[0]
        res['subgroups'][g] = dict(n=b['n'], delta_mre=c['mre'] - b['mre'],
                                   delta_param_mae=c['param_mae'] - b['param_mae'],
                                   local_optimum_s=loc, size_ratio_at_1=b['size_ratio_median'])
    os.makedirs(os.path.join(ROOT, 'experiments/results/hc_valdomain'), exist_ok=True)
    with open(os.path.join(ROOT, 'experiments/results/hc_valdomain/scale_fit.json'), 'w') as f:
        json.dump(res, f, indent=2)

    print('FIT (n=%d, out of sample):  s=1.000 -> MRE %.3f / paramMAE %.3f / size(median) %.4f'
          % (res['fit']['n'], res['fit']['baseline']['mre'], res['fit']['baseline']['param_mae'],
             res['fit']['baseline']['size_ratio_median']))
    print('   argmin paramMAE s* = %.4f  -> MRE %.3f / paramMAE %.3f' % (
        s_star, best_mae[1]['mre'], best_mae[1]['param_mae']))
    print('   argmin MRE      s  = %.4f  -> MRE %.3f / paramMAE %.3f' % (
        float(best_mre[0]), best_mre[1]['mre'], best_mre[1]['param_mae']))
    print('\nEVAL on the 151 val-matched images with FROZEN s=%.4f:' % s_star)
    for pf, v in res['eval'].items():
        print('  %-44s MRE %7.3f -> %7.3f (%+.3f) | paramMAE %7.3f -> %7.3f (%+.3f)' % (
            pf.split('/')[1], v['baseline']['mre'], v['at_frozen_s']['mre'], v['delta_mre'],
            v['baseline']['param_mae'], v['at_frozen_s']['param_mae'], v['delta_param_mae']))
    print('\nper native-size subgroup (deployed artifact):')
    for g, v in res['subgroups'].items():
        print('  %-10s n=%3d  dMRE %+7.3f  dParamMAE %+8.3f  local s* %.4f  size@1 %.4f' % (
            g, v['n'], v['delta_mre'], v['delta_param_mae'], v['local_optimum_s'], v['size_ratio_at_1']))


if __name__ == '__main__':
    main()
