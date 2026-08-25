# Training on the FU_Biometry 2026 data

How to obtain the challenge data, lay it out, and retrain every model the deployed container
ships. Inference-only users do not need any of this — see the [README](README.md).

## 1. Get the data

The FU_Biometry 2026 data is **CC BY-NC** and is not redistributable, so it is not in this
repository. Register for the challenge and download it from the organizers:

- Codabench validation competition: <https://www.codabench.org/competitions/15590/>
- Codabench final-test competition: <https://www.codabench.org/competitions/17560/>
- Organizer baseline: <https://github.com/lijiake2408/Foundation-Model-for-Ultrasound-Biometry>

You receive, per task, a labelled archive with a landmark CSV and — for every task — an
**unlabelled** archive. **Semi-supervised use of the unlabelled pool is required by Participation
Rule §2.1**, and every checkpoint in our ranked container depends on it.

### Data sources and licences

| Source | Used for | Amount | Licence |
|---|---|---|---|
| **FU_Biometry 2026** labelled (organizer) | stage-2 training, all selection | 6,743 images across 9 tasks | CC BY-NC |
| **FU_Biometry 2026** unlabelled (organizer) | stage-1 DINO continuation | 24,467 images after de-duplication | CC BY-NC |
| **DINOv2 ViT-B/14** `lvd142m` (public weights) | stage-1 initialization | — | Apache-2.0 |
| **Fetal Biometry multicentre landmarks** (external, FP subset) | HC ellipse-scale fit + labelled val-domain proxy | 1,484 head images | see below |

The external landmark dataset is used for **calibration and evaluation only** — never trained on,
and no annotations of it were created by us. It supplies the single out-of-sample fit behind the
HC scale constant. Its own README defers licensing to a LICENSE file that was not present in the
copy we obtained, so we assert no terms for it; obtain it and its terms from its authors. Only
`experiments/build_fp_head_probe.py` and `experiments/hc_valdomain_harness.py` read it, and both
are optional — the constant is already frozen in `docker/hc_scale.py`.

### Two organizer data corrections that are baked into this pipeline

- **25 mirrored/flipped fetal-femur frames were disregarded** by the organizers (message
  2026-07-16). `scripts/prepare_data.py` drops them; the eligible femur count is 702, not 727.
  Every shipped checkpoint was trained post-drop.
- **The cardiac left–right "correction" was retracted.** The organizers shipped an A4C/PSAX
  endpoint swap on 2026-07-17 and withdrew it the next day — the original CSVs were correct. The
  confirmed standard is *vertical*: within each odd–even landmark pair the first point is above the
  second. `scripts/prepare_data.py` uses the original raw CSVs.

## 2. Lay it out

Put the downloaded per-task archives under `data/drive_raw/<TASK>/` exactly as the organizers ship
them (note the organizers' directory for PSAX is spelled `PSAK`; the scripts handle it), then:

```bash
uv run python scripts/prepare_data.py    # -> data/csv/<task>_train.csv + data/images/<task>/
uv run python scripts/prepare_val.py     # -> data/val/csv/ + data/val/images/  (inference only)
```

```
data/
├── drive_raw/<TASK>/{labeled,unlabeled}/     as downloaded
├── images/<task>/<basename>                  flattened labelled images
├── csv/<task>_train.csv                      image_path, task_name, task_id, num_classes, point_i_xy
├── folds/folds.csv                           built in §3
├── unlabeled/                                built in §4
│   ├── <task>/                               kept images
│   └── manifest.csv                          task_id, image_path, phash
└── val/{csv,images}/                         validation inference root
```

**Coordinates in the CSVs are ORIGINAL-IMAGE PIXELS.** The training code normalizes them; the
scorer in `scoring/` compares them directly against `predicted_points_pixels`. Do not denormalize.

## 3. Leak-free folds

All model selection used a **group-aware five-fold split** — `k=5`, guard band `2`, `seed=0` —
grouping by clip / patient / series so no clip straddles a fold, with a hard assertion that the
AOP frame-adjacency leak is exactly zero.

```bash
uv run python experiments/make_folds_core.py            # verify: reproduces the folds (6,743 rows)
uv run python experiments/make_folds_core.py --write    # write data/folds/folds.csv
```

6,743 rows: A4C 108, AOP 4000, FA 500, FUGC 260, HC 999, IVC 38, PLAX 87, PSAX 49, fetal_femur 702.
16 rows are assigned `fold == -1` and excluded, leaving **6,727** eligible training images and
~1,340 per validation fold.

Use the script, not a `glob('data/csv/*.csv')` one-liner. A bare glob sweeps in the external-data
rows from a rejected lever and silently reshuffles the FA and femur fold assignments — the folds
directory is gitignored, so that break is invisible.

## 4. Stage 1 — the unlabelled pool and continued DINO

This is the rules-mandatory semi-supervised stage and the initialization of **every** ranked
checkpoint.

```bash
# 1. decode + perceptually hash every unlabelled, labelled and validation image (slow, once)
uv run python scripts/prepare_unlabeled.py --phase hash

# 2. de-duplicate and screen for leakage at hash distance 2 (fast, re-tunable)
uv run python scripts/prepare_unlabeled.py --phase materialize --thresh 2
#    -> data/unlabeled/manifest.csv, 24,467 images
```

Phase 2 does two things: it collapses near-duplicates within the pool, and it **drops any pool
image that perceptually matches a labelled-train or validation image**, so the pretraining set
cannot contain an evaluation image.

```bash
baseline/.venv-baseline/bin/python experiments/dino_pretrain.py \
    --encoder dinov2_vitb --epochs 3 --batch-size 16 --lr 3.333333333e-5 \
    --out runs/dino_ssl_vitb --mem-frac 0.45 --log-every 50 \
    --results-json experiments/results/dino_ssl_vitb_5fold/pretrain_log.json
```

Continued DINO CLS-token self-distillation from public DINOv2 ViT-B/14: 518 px global crops, 2
global + 4 local (168 px) crops, a projection head with 2048-d hidden layer, 256-d bottleneck and
4096-d output, student temperature 0.1, teacher temperature 0.04, teacher EMA momentum 0.996
(cosine to 1.0), centering momentum 0.9, AdamW weight decay 0.04, 50-step warm-up, seed 42. Only
**transformer blocks 10–11 and the final norm** are unfrozen. 3 epochs = **4,587 steps**, final
EMA loss 3.667, **834 s** wall time.

Output `runs/dino_ssl_vitb/encoder.pth`, SHA-256
`13edc7ddaa0690c0b3bcf3bcb15b49999f21a4903d9eccab5316aefbbc886c3f`, consumed by every supervised
run below as `--encoder-init`.

**One export subtlety.** Repeated float32 EMA updates introduce deterministic round-off even in
tensors that are mathematically frozen. The exporter therefore restores the nominally frozen
tensors from the unchanged student copy, so the result is provably a *minimal* edit of public
DINOv2: 174 tensors total, 144 bit-identical to the public weights, exactly 30 changed and all
confined to the unfrozen blocks. Asserted in
`experiments/results/dino_ssl_vitb_5fold/encoder_export_audit.json`.

### The two rejected alternatives on the same pool

Both are included so the comparison can be rerun, and neither is deployed:

- `experiments/ssl_train.py` — mean-teacher consistency trained on labelled and unlabelled images
  *simultaneously*. Regressed seven of nine tasks; the shared encoder pays a localization penalty.
- `experiments/mae_pretrain.py` — masked-autoencoder pretraining on the same pool. Also rejected.

## 5. Stage 2 — supervised training

The recipe, identical for every base member: DINOv2 ViT-B/14 at 518², nine per-task heatmap heads
at uniform 64² with Gaussian σ=1.8, heatmap MSE, `geo_v1` augmentation, AdamW with **encoder LR
2e-5 / head LR 1e-3**, batch 4, linear warm-up 3 epochs then cosine to zero over 40 epochs, seed 42,
fp32 (no AMP). `geo_v1` is keypoint-aware photometric plus conformal affine, applied through
`experiments/kp_aug_dataset.py`, which co-transforms landmarks and reject-samples draws that put a
landmark out of frame.

**Five-fold evidence.** Trains all five folds (three on GPU 0, two on GPU 1), scores each held-out
fold with the deployed decoder, and prints the paired per-fold delta against the control:

```bash
bash experiments/reproduce_dino_vitb_5fold.sh                 # add --skip-pretrain to reuse §4
bash experiments/reproduce_dino_vitb_specialists_5fold.sh     # HC-small + HC-head, five folds
```

**Full-data members.** Seven checkpoints, gated on the five-fold decision JSON — it refuses to run
without one:

```bash
bash experiments/reproduce_dino_vitb_full_family.sh
baseline/.venv-baseline/bin/python experiments/audit_dino_vitb_full_family.py
```

| member | recipe | epochs |
|---|---|---|
| `vitb_full_dino_corr{,_s43,_s44,_s45,_s46}` | `--full-data --aug geo_v1 --warmup 3 --cosine`, seeds 42–46 | 40 |
| `vitb_full_dino_hcsmall_corr` | same, `--aug geo_v1_hcsmall`, seed 42 | 40 |
| `vitb_full_dino_hchead_corr` | continues the seed-42 base, `--aug geo_v1_hcsmall --train-task HC --head-lr 1e-4` | 5 |

The HC-head member is a **head-only** refinement: `experiments/audit_head_refinement.py` asserts
that exactly the 14 HC-head tensors changed and all 286 others are bit-identical to its parent. If
that audit fails, the checkpoint is not usable.

## 6. Inference, calibration and the container

Research-path inference is `experiments/infer_tta.py` (single checkpoint) and
`experiments/infer_ensemble.py` (families and routing). Both decode with a 9×9 intensity-weighted
sub-pixel centroid under heatmap-space scale TTA `{0.92, 1.0, 1.08}`.

Three bounded post-processors run after decoding, in this fixed order. Each is a small frozen
constant, not a learned module:

1. **`docker/hc_scale.py`** — shrink the four HC landmarks about their centroid by `s = 0.975`, but
   only for images whose native size is *not* an HC18 training size (`800×540`, `800×542`). The
   ~2.5 % ellipse over-prediction is domain-conditional, so the gate matters. Fitted out of sample
   on 1,484 patient-disjoint external head images: `experiments/build_fp_head_probe.py`,
   `experiments/hc_valdomain_harness.py`, `experiments/fit_hc_scale.py`, driver
   `experiments/reproduce_hc_scale_valdomain.sh`.
2. **`docker/geometry_project.py`** — length-preserving projection of the FA and HC landmark pairs
   onto the annotation protocol's own geometry. Both derived diameters are preserved to 2.3e-13 px,
   so it is parameter-neutral by construction. Sweep: `experiments/reproduce_label_geometry.sh`.
3. **`docker/ivc_calibrate.py`** — gated IVC caliper-length calibration, constants from the
   full-38 training-GT fit (p10 15.274 / p90 37.628 / median 24.657). Measurement:
   `experiments/reproduce_ivc_length_calibration.sh`.

> The LOFO fits in `experiments/results/ivc_length_calibration/result.json` exist **only** to
> measure that lever out of sample. Never deploy them; the container uses the full-38 fit.

Stage the checkpoints and build:

```bash
baseline/.venv-baseline/bin/python experiments/stage_dino_vitb_docker.py   # pins every source hash
docker build -t fu-biometry:final docker
bash docker/selfcheck.sh                                   # organizers' exact invocation, CPU
bash docker/selfcheck.sh --gpu                             # REQUIRED on an Ampere-class host
baseline/.venv-baseline/bin/python docker/verify_against_reference.py --all
```

`verify_against_reference.py --all` diffs the container against the research path over all 619
validation images; ours passed at worst **1.85e-4 px**, well inside the 1e-3 px tolerance.
`docker/selfcheck_guards.py` confirms all six guard behaviours, four of which must raise — under
one-successful-run-locks-in, a crash is cheap and a silently degraded success is unrecoverable.

## 7. Running your own experiments

The evaluation protocol matters more than the metric here.

**Judge on paired per-fold deltas, not the mean.** Fold-0 alone is several pixels noisy. Every
adopted lever cleared a pre-registered gate on the five-fold paired delta *and* sign consistency
across folds. An external-data lever that looked like a −0.56 win on the raw mean died exactly
here: 5/5 sign consistency did not hold and the interval covered zero.

**The raw task-mean over-weights the tiny cardiac tasks.** IVC has 38 images and PSAX 49 — roughly
8–10 per fold, where a single case swings the task mean by tens of pixels. Read adoption off the
stable tasks (AOP, HC, FA, FUGC, fetal_femur), the pooled cardiac group, and the per-task intervals
that `experiments/aggregate_cv.py` prints; not the headline number. The `geo_v1` result illustrates
the gap: of its −4.68 headline, roughly two thirds is tiny-task noise.

**Use corrected intervals.** Cross-validation folds are not independent, so ordinary paired-t
intervals are too narrow. Use the Nadeau–Bengio correction. The strongest ensemble candidate here
looked significant under the naive interval and stopped looking so under the corrected one — and the
stage-1 DINO gain itself has a corrected interval that crosses zero, which is why the README calls
it an expected-score selection rather than a significant effect.

**Compare against a matched same-session control.** Training is deterministic given seed 42 and the
documented fold build, but run-to-run variation of roughly 1 px remains (DataLoader workers, GPU
non-determinism). Regenerate the control in the same session rather than differencing against a
recorded number from an earlier one.

**Two fold-based runs cannot run concurrently.** They share the `data/_cvfold{0..4}` scratch CSVs.
Full-data runs are `_cvfold`-safe and can be parallelized freely.

**Long runs must be launched fully detached** (`setsid nohup … </dev/null &`). Jobs tied to an
interactive session were reaped at ~60 minutes on this machine, mid-training and without a
traceback.

## 8. Runtime

On a single NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB), with two workers sharing the card:

| Stage | Cost |
|---|---|
| Stage 1 — DINO continuation (3 epochs, 4,587 steps, 24,467 images) | **834 s** |
| Stage 2 — one CV fold (40 epochs, 5,381 images) | ~1 h 56 m |
| Scoring one held-out fold (decode + scale TTA) | ~2 min |
| Full five-fold sweep, 3+2 split across two GPUs | ~6 h |
| One full-data base member (40 epochs, 6,727 images) | ~2 h 25 m |
| HC-head member (5 epochs, head-only) | a few minutes |
| All seven full-data members, two at a time | ~7.5 h |

Inference is far cheaper than training, and the 5–6 h evaluation budget was never the binding
constraint. The binding constraints on the evaluation host are **10 GB VRAM and 7 GB container
RAM**, which is why `docker/model.py` loads the seven members one at a time and accumulates
canonical heatmaps on CPU rather than holding them resident.
