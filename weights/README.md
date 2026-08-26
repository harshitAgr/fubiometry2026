# Model weights

**Weights are not distributed in this release.** This directory documents the layout the container
expects and records the exact artifacts that produced the scored result, so they can be identified
unambiguously. Train them with [`../TRAINING.md`](../TRAINING.md).

## Expected layout

`docker/weights/` is the build context the `COPY` lines in `docker/Dockerfile` read from; each file
lands at `/app/<name>` in the image. The organizer entry script copies only `/app/best_model.pth`
into its work directory, so `docker/model.py` loads the other six straight from `/app` — explicitly
allowed by the submission guide.

```
docker/weights/
├── best_model.pth        base seed 42   ─┐
├── model_s43.pth         base seed 43    │  "base" group: heatmap-averaged,
├── model_s44.pth         base seed 44    │  then decoded once (9x9 window)
├── model_s45.pth         base seed 45    │
├── model_s46.pth         base seed 46   ─┘
├── model_hcsmall.pth     HC-small-augmentation specialist, seed 42
└── model_hchead.pth      HC-head-refined specialist, seed 42
```

Each file is a 466,006,205-byte fp32 `state_dict` holding the shared DINOv2 ViT-B/14 encoder (all
174 tensors) plus the nine per-task heatmap heads. Because the encoder tensors travel with every
checkpoint, the container builds the backbone with `pretrained=False` and never touches the
network — see [`../env/README.md`](../env/README.md).

Also required to *retrain*, but not shipped in the image:

```
runs/dino_ssl_vitb/encoder.pth    stage-1 unlabeled-pool encoder; the --encoder-init
                                  of all seven checkpoints above
```

## Combination

The three groups are averaged in **heatmap** space within a group, decoded independently, and then
combined per task in **coordinate** space (`ROUTES` in `docker/model.py`):

| task | route |
|---|---|
| IVC | `base` — exact passthrough |
| HC | `(base + hcsmall + hchead) / 3` |
| A4C, AOP, FA, FUGC, PLAX, PSAX, fetal_femur | `(2·base + hcsmall) / 3` |

So a base seed carries 1/5 of the base vote, and on the seven routed tasks the base group as a
whole carries 2/3. Three bounded post-processing steps then run in this fixed order: gated HC
ellipse scale → FA/HC label-geometry projection → gated IVC length calibration.

## Provenance of the scored artifacts

[`SHA256SUMS`](SHA256SUMS) records the stage-1 encoder and the seven checkpoints baked into the
submitted image.

| item | value |
|---|---|
| Codabench submission | **887569**, phase 29264 ("Final Test"), 2026-08-13, status Finished |
| Score | **Avg MRE 21.3920553114 / Avg MAE 22.7094187043** |
| Image digest | `sha256:9d716176261d1c5d3132a35f81fc5e478206f73a3d590bf95a63aea8a9763815` |
| Stage-1 encoder | `13edc7ddaa0690c0b3bcf3bcb15b49999f21a4903d9eccab5316aefbbc886c3f` |

The Docker Hub coordinates of the submitted image are deliberately not published here: the image
is private, it carries no code that is not in `docker/`, and the weights it contains are covered by
the CC BY-NC term above. The digest identifies it unambiguously to anyone who already holds it.

The staging step that populated `docker/weights/` pins every source hash in advance and refuses to
run on a mismatch — `experiments/stage_dino_vitb_docker.py`, manifest
`experiments/results/dino_ssl_vitb_docker/staging_manifest.json`. If you retrain, regenerate that
manifest rather than copying checkpoints by hand.

## Licensing, if you later distribute weights

Any model trained on this data is **non-commercial**: the FU_Biometry challenge data is CC BY-NC,
and that term propagates to derived checkpoints. The stage-1 encoder additionally derives from
public DINOv2 ViT-B/14 (Apache-2.0), which imposes no further restriction. See
[`../NOTICE`](../NOTICE) before distributing any trained artifact.
