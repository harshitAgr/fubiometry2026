# FU_Biometry 2026 — challenge entry

Training and inference source code for our final submission to the **FU_Biometry** challenge
(Foundation Model for Ultrasound Biometry, MICCAI 2026) — Codabench
[15590](https://www.codabench.org/competitions/15590/) (validation) and
[17560](https://www.codabench.org/competitions/17560/) (final test).

Team **hars25** — Harshit Agrawal, AIATELLA Oy, Helsinki, Finland.

One shared model localizes landmarks across **nine task IDs** — prenatal head circumference (HC),
fetal abdomen (FA), fetal femur, cervical length (FUGC), intrapartum angle of progression (AOP),
and the cardiac A4C, PLAX, PSAX and IVC views — and the clinical parameters are derived from those
landmarks. Scoring is 50 % landmark mean radial error (MRE) and 50 % parameter MAE.

| Artifact | Avg MRE | Avg MAE |
|---|---|---|
| Organizer DINOv2 ViT-S baseline (validation) | 38.882 | 37.075 |
| Our best validation submission | 23.423 | 27.231 |
| **Our final-test container** (Codabench 887569) | **21.392** | **22.709** |

Per task, on the hidden test set:

| | A4C | AOP | FA | femur | FUGC | HC | IVC | PLAX | PSAX |
|---|---|---|---|---|---|---|---|---|---|
| MRE | 22.017 | 18.553 | 24.318 | 11.463 | 14.467 | 23.536 | 21.406 | 16.443 | 40.327 |
| MAE | 16.555 | 7.583 | 100.075 | 10.939 | 9.925 | 28.462 | 7.773 | 7.150 | 15.922 |

Ranked **1st of 31 teams** on both of the organizers' preliminary leaderboards (preliminary
results, 21 Aug 2026; final rankings confirmed 5 Sept 2026).

Model weights and challenge data are not included. See [TRAINING.md](TRAINING.md) to obtain the
data and retrain, and [weights/README.md](weights/README.md) for the checkpoint layout the
container expects.

## Method

**Two stages, both on challenge data only.**

**Stage 1 — continued DINO self-distillation on the unlabeled pool.** Participation rule §2.1
requires that all unlabeled data be used, and **every one of the seven checkpoints in the scored
container descends from this encoder** — none is initialized from public weights alone.
`scripts/prepare_unlabeled.py` builds the pool from the per-task unlabeled archives: it decodes
every image, perceptually de-duplicates it, and drops any image that perceptually matches a
labeled-train or validation image, yielding a manifest of **24,467 images**.
`experiments/dino_pretrain.py` then continues DINO CLS-token self-distillation from public DINOv2
ViT-B/14 on that manifest — 518 px global crops, two global and four local (168 px) crops, 4096-d
projection head, student/teacher temperature 0.1/0.04, teacher EMA momentum 0.996, transformer
blocks 10–11 plus the final norm unfrozen, 3 epochs / 4,587 steps, batch 16, LR 3.333e-5, seed 42.
The exported encoder is audited tensor by tensor: all 174 expected tensors present, the 144
nominally frozen ones bit-identical to public DINOv2, and exactly 30 changed tensors confined to
the unfrozen blocks.

**Stage 2 — supervised fine-tuning** of that shared encoder plus nine per-task heatmap heads.
DINOv2 ViT-B/14 at 518², uniform 64² heatmaps with Gaussian σ=1.8, heatmap MSE; `geo_v1`
keypoint-aware photometric + conformal affine augmentation; linear warm-up (3) + cosine over 40
epochs; seed 42. All selection used a leak-free group-aware five-fold split.

**Ensemble and decoding.** Seven checkpoints: five base seeds (42–46), one HC-small-augmentation
specialist, one HC-head-refined specialist. Heatmaps are averaged within each family and decoded
once with a 9×9 intensity-weighted sub-pixel centroid (argmax fallback) under heatmap-space scale
TTA at {0.92, 1.0, 1.08}; the three families are then combined per task in **coordinate** space —
IVC = base, HC = the three-way mean, the other seven tasks = (2·base + HC-small)/3.

**Bounded post-processing**, in this fixed order: size-gated HC ellipse-scale correction (s=0.975,
applied only outside the HC18 training resolutions), length-preserving FA/HC ellipse-geometry
projection, and a gated IVC vessel-length calibration.

Paired five-fold effect of stage 1, with the supervised recipe, folds, seed, decoder and TTA held
fixed: task-mean MRE **22.889 → 22.586** (four of five folds favorable) and approximate
parameter-MAE **17.848 → 17.797** (four of five). Both Nadeau–Bengio corrected 95 % intervals
cross zero, so this was selected on expected score for the single ranked attempt and is **not**
claimed as a statistically significant gain. Rebuilding the whole seven-member route from the
encoder changed out-of-fold MRE **22.447 → 22.107**. Reports live under
`experiments/results/dino_ssl_vitb_5fold/`.

Rule §2.1 asks for semi-supervised methods. We also built the joint alternative — a mean-teacher
consistency branch over labeled and unlabeled images (`experiments/ssl_train.py`) — and it
regressed seven of nine tasks, consistent with a shared-encoder localization penalty. Stage 1
above is what we deployed instead.

## Layout

```
experiments/  training, semi-supervised pretraining, fold construction, ensembling, calibration
scripts/      data preparation for the labeled sets, the unlabeled pool and validation
scoring/      local implementation of the challenge metric (MRE + derived parameters)
docker/       the inference container — the build context of the scored image
baseline/     organizer baseline modules our trainers import (vendored, byte-identical)
tests/        unit tests for everything above
env/          pinned dependencies for all three environments
weights/      expected checkpoint layout + SHA-256 provenance of the scored artifacts
TRAINING.md   data, layout, environments, both stages, evaluation protocol, runtime
NOTICE        licence scope: what is ours vs vendored third-party
```

## Training

Data, layout, environments, both semi-supervised and supervised stages, and the evaluation protocol
are in **[TRAINING.md](TRAINING.md)**. In short:

```bash
# data and the leak-free group-aware folds
uv run python scripts/prepare_data.py
uv run python experiments/make_folds_core.py --write

# stage 1 — the unlabeled pool, then continued DINO on it
uv run python scripts/prepare_unlabeled.py --phase hash
uv run python scripts/prepare_unlabeled.py --phase materialize --thresh 2
baseline/.venv-baseline/bin/python experiments/dino_pretrain.py \
    --encoder dinov2_vitb --epochs 3 --batch-size 16 --lr 3.333333333e-5 \
    --out runs/dino_ssl_vitb

# stage 2 — five-fold evidence, then the seven full-data members
bash experiments/reproduce_dino_vitb_5fold.sh
bash experiments/reproduce_dino_vitb_specialists_5fold.sh
bash experiments/reproduce_dino_vitb_full_family.sh
```

## Inference

Stage the seven checkpoints into `docker/weights/` — `experiments/stage_dino_vitb_docker.py` does
this with every source hash pinned in advance — then:

```bash
docker build -t fu-biometry:final docker

docker run --rm --gpus all --network none -m 7g --cpus 4 --shm-size 2g \
  -v /path/to/input:/input:ro -v /path/to/output:/output \
  fu-biometry:final
```

`docker/predict.py` is the organizer's entry point, shipped unmodified; `docker/model.py` provides
the required `class Model`. The image is fully offline: every checkpoint carries all 174 DINOv2
encoder tensors, so the backbone is built with `pretrained=False` and nothing is fetched at
runtime. Members are loaded **one at a time** with canonical heatmaps accumulated on CPU, to fit
the 10 GB VRAM / 7 GB RAM evaluation host.

Three checks ship with it. `docker/selfcheck.sh` runs the organizers' exact invocation and contract.
`docker/verify_against_reference.py` diffs the container against the research inference path over
all 619 validation images — worst deviation **1.85e-4 px**, well inside the 1e-3 px tolerance.
`docker/selfcheck_guards.py` verifies all six guard behaviours — four of them must raise, one
asserts the batch cap, and one checks that a lone unreadable image stays tolerated: the first
*successful* run locks in the final-test score, so an unknown `task_id`, a missing ensemble member,
too many unreadable images or a non-finite coordinate must crash rather than silently degrade.

**Known gap, disclosed.** The scored image's CUDA path was never executed before submission. The
evaluation host is CUDA 12.1 / sm_86, the development GPU is Blackwell sm_120, and the organizers'
pinned base image has no kernels for the latter — so `--gpus all` cannot run locally for reasons
unrelated to this code. Correctness was established on CPU and by the 619-image numeric diff above.
If you rebuild, run `docker/selfcheck.sh --gpu` on an Ampere-class card first.

## Release scope

This is the code behind the submitted system: both training stages, the container, the local
scorer, fold construction, the reproduction scripts for every adopted lever, and the result JSONs
behind the numbers quoted here. It is a curated release, not the full research repository — the
~200 scripts and protocols belonging to rejected levers, the chronological experiment log, and all
analysis of other teams' entries are not published.

Several rejected levers are nonetheless present, each because a shipped file imports or pins it.
`experiments/run_config.py` — the trainer behind every deployed checkpoint — imports
`adaptive_wing`, `marginal_loss`, `param_loss` and `model_ema` as selectable options, and
`experiments/infer_ensemble.py` imports `fugc_scale` and `hc_scale_norm` the same way; none is
reachable from the container. The layer-wise-LR-decay preflight
(`experiments/preflight_llrd.py`, `preflight_inverse_llrd_full_seeds.py` and their runners) is
here for a different reason: `experiments/{full_family_candidate,audit_vitb5_candidates}.py` —
which define and audit the deployed route — hash a frozen source list at run time that includes
its report, so omitting it would leave shipped provenance code unable to execute.

Several result JSONs pin the SHA-256 of each source file **as of the run that produced them**.
Eight of those files no longer match their recorded digest:

- `experiments/{decode,encoders,infer_tta,kp_aug_dataset,per_task_model,run_config}.py` drifted in
  the research repository after those runs, one of them through a head-variant rename.
- `experiments/make_folds_core.py` and `experiments/train_ensemble_vitb_corr.sh` were edited *for
  this release*, to remove comment pointers into internal documents that are not published.

Every other pinned path still matches, and every pinned path resolves — so the provenance code
runs; it simply reports the drift above rather than a clean match.

Challenge data, model checkpoints, run directories and logs are **not** in this repository. Obtain
the data through the organizers, then follow [TRAINING.md](TRAINING.md).

## License and attribution

Code here is **Apache-2.0** (`LICENSE`), *except* the vendored third-party files below, which keep
their own terms — see [`NOTICE`](NOTICE) for the exact scope. Weights trained by this code are
**non-commercial**: the challenge data is CC BY-NC.

- Organizer baseline (`baseline/baseline/`):
  <https://github.com/lijiake2408/Foundation-Model-for-Ultrasound-Biometry>, commit `603263b`.
  `dataset.py`, `model.py`, `model_factory.py`, `train.py`, `utils.py` are byte-identical upstream;
  `docker/predict.py` is the organizers' unmodified entry point; `docker/fub_arch.py` is our
  derivative of upstream `model_factory.py`. Upstream publishes no LICENSE file.
- **Data.** FU_Biometry 2026 challenge data (CC BY-NC), from the organizers. The out-of-sample fit
  of the single HC ellipse-scale constant additionally uses a public multicentre fetal-biometry
  landmark dataset (FP subset), for **calibration and evaluation only** — never trained on. No data
  is redistributed here; see [TRAINING.md](TRAINING.md).
- Built on DINOv2 (Apache-2.0), timm (Apache-2.0), PyTorch (BSD-3-Clause), albumentations (MIT),
  OpenCV (Apache-2.0).

## Citation

```bibtex
@inproceedings{
agrawal2026multitask,
title={Multi-Task Ultrasound Biometry Landmark Detection},
author={Harshit Agrawal},
booktitle={The 1st MICCAI Workshop on Medical World Models},
year={2026},
url={https://openreview.net/forum?id=LKW9RPveuI}
}
```
