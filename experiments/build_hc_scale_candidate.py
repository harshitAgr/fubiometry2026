#!/usr/bin/env python3
"""Build the HC ellipse-scale A/B candidate from an existing scored artifact.

Single variable: HC landmarks are scaled about their own centroid by a frozen
factor (fitted out of sample -- see experiments/fit_hc_scale.py). All other
tasks are copied through byte-identically so the official delta is attributable
to the HC scale alone.

  build_hc_scale_candidate.py --base <scored.json> --scale 0.9750 --out submission/vN
"""
from __future__ import annotations
import argparse, hashlib, json, math, os, statistics


def scale_about_centroid(flat, s):
    pts = [(flat[2 * i], flat[2 * i + 1]) for i in range(len(flat) // 2)]
    cx = sum(p[0] for p in pts) / len(pts); cy = sum(p[1] for p in pts) / len(pts)
    out = []
    for x, y in pts:
        out += [cx + (x - cx) * s, cy + (y - cy) * s]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', default='submission/full_family_postdrop_seed42/regression_predictions.json')
    ap.add_argument('--scale', type=float, required=True)
    ap.add_argument('--task', default='HC')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    base = json.load(open(a.base))
    out, audit = [], dict(base=a.base, base_md5=hashlib.md5(open(a.base, 'rb').read()).hexdigest(),
                          scale=a.scale, task=a.task)
    max_resid, ratios, n_touched = 0.0, [], 0
    for r in base:
        r2 = dict(r)
        if r['task_id'] == a.task:
            r2['predicted_points_pixels'] = scale_about_centroid(r['predicted_points_pixels'], a.scale)
            n_touched += 1
            # exactness: recompute independently and compare
            chk = scale_about_centroid(r['predicted_points_pixels'], a.scale)
            max_resid = max(max_resid, max(abs(x - y) for x, y in zip(chk, r2['predicted_points_pixels'])))
            p = r['predicted_points_pixels']; q = r2['predicted_points_pixels']
            d0 = math.dist(p[0:2], p[2:4]) + math.dist(p[4:6], p[6:8])
            d1 = math.dist(q[0:2], q[2:4]) + math.dist(q[4:6], q[6:8])
            if d0 > 0:
                ratios.append(d1 / d0)
        out.append(r2)

    # ---- audit -----------------------------------------------------------
    kb = {(r['image_path'], r['task_id']): tuple(r['predicted_points_pixels']) for r in base}
    ko = {(r['image_path'], r['task_id']): tuple(r['predicted_points_pixels']) for r in out}
    changed = {k for k in kb if kb[k] != ko[k]}
    tasks_changed = sorted({k[1] for k in changed})
    counts = {}
    for r in out:
        counts[r['task_id']] = counts.get(r['task_id'], 0) + 1
    finite = all(all(isinstance(v, float) and math.isfinite(v) for v in r['predicted_points_pixels']) for r in out)
    lens_ok = all(len(r['predicted_points_pixels']) % 2 == 0 and len(r['predicted_points_pixels']) > 0 for r in out)
    audit.update(
        records=len(out), records_base=len(base), task_counts=counts,
        records_touched=n_touched, tasks_changed=tasks_changed,
        other_tasks_byte_identical=(tasks_changed == [a.task]),
        keys_identical=(set(kb) == set(ko)),
        scale_formula_max_residual=max_resid,
        realized_size_ratio=dict(mean=statistics.mean(ratios), min=min(ratios), max=max(ratios)),
        realized_matches_requested=abs(statistics.mean(ratios) - a.scale) < 1e-9,
        all_finite=finite, all_lengths_valid=lens_ok)
    audit['PASS'] = bool(audit['other_tasks_byte_identical'] and audit['keys_identical'] and finite and lens_ok
                         and audit['realized_matches_requested'] and audit['records'] == audit['records_base'])

    os.makedirs(a.out, exist_ok=True)
    pj = os.path.join(a.out, 'regression_predictions.json')
    with open(pj, 'w') as f:
        json.dump(out, f)
    audit['out_md5'] = hashlib.md5(open(pj, 'rb').read()).hexdigest()
    with open(os.path.join(a.out, 'candidate_audit.json'), 'w') as f:
        json.dump(audit, f, indent=2)
    print(json.dumps(audit, indent=2))


if __name__ == '__main__':
    main()
