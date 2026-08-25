#!/usr/bin/env python3
"""Build the OUT-OF-SAMPLE fitting set for the HC ellipse-scale probe.

FETAL_PLANES_DB head images that are (a) NOT pixel-identical to any challenge
validation image and (b) from patients with no val-matched image at all, so no
patient straddles the fit/eval split. These images are in the val DOMAIN but are
never used for evaluation, so a scale fitted here is genuinely out of sample.

They are also unseen by our models: HC training is 100% HC18, and the rejected
external lever only ingested FA/femur, never heads.

Writes data/_fpheads/{images/HC/*, split.csv, gt.csv}.
"""
from __future__ import annotations
import csv, glob, hashlib, os, re, sys
import cv2

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
POOL = os.path.join(ROOT, 'external/ucl_multicentre/FetalBiometry-Multicentre-Landmarks-2026')
OUT = os.path.join(ROOT, 'data/_fpheads')


def h(path):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return None if im is None else hashlib.md5(im.tobytes()).hexdigest()


def patient(name):
    m = re.match(r'(Patient\d+)_', name)
    return m.group(1) if m else name


def main():
    rows = {r['image_name']: r for r in csv.DictReader(open(os.path.join(POOL, 'annotations/FP/Head.csv')))}
    val_hashes = {h(p) for p in glob.glob(os.path.join(ROOT, 'data/val/images/HC/*'))}
    pool = sorted(glob.glob(os.path.join(POOL, 'images/FP/Head/*')))
    matched_patients, cand = set(), []
    for p in pool:
        n = os.path.basename(p)
        if n not in rows:
            continue
        if h(p) in val_hashes:
            matched_patients.add(patient(n))
        else:
            cand.append((n, p))
    keep = [(n, p) for n, p in cand if patient(n) not in matched_patients]
    print('FP head pool %d | val-matched patients excluded %d | fitting set %d images / %d patients'
          % (len(pool), len(matched_patients), len(keep), len({patient(n) for n, _ in keep})))

    os.makedirs(os.path.join(OUT, 'images/HC'), exist_ok=True)
    for n, p in keep:
        dst = os.path.join(OUT, 'images/HC', n)
        if not os.path.lexists(dst):
            os.symlink(os.path.abspath(p), dst)
    with open(os.path.join(OUT, 'split.csv'), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['image_path'])
        for n, _ in keep:
            w.writerow(['HC/' + n])
    # GT in the challenge's canonical endpoint order (gate-verified at 0.985 px on HC train)
    with open(os.path.join(OUT, 'gt.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_path', 'task_name', 'task_id', 'num_classes'] + ['point_%d_xy' % i for i in (1, 2, 3, 4)])
        for n, _ in keep:
            r = rows[n]
            b = [(float(r['bpd_1_x']), float(r['bpd_1_y'])), (float(r['bpd_2_x']), float(r['bpd_2_y']))]
            o = [(float(r['ofd_1_x']), float(r['ofd_1_y'])), (float(r['ofd_2_x']), float(r['ofd_2_y']))]
            b.sort(key=lambda q: q[1]); o.sort(key=lambda q: -q[0])
            w.writerow(['HC/' + n, 'Regression', 'HC', 4] + ['[%r, %r]' % (p[0], p[1]) for p in b + o])
    print('wrote', OUT)


if __name__ == '__main__':
    main()
