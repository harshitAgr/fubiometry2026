# Environments and dependencies

Three environments are involved. They are deliberately separate — the two training environments
track current CUDA for the development GPU, while the inference container pins the older CUDA line
the evaluation hardware requires.

| Environment | Used for | Pin file | Python |
|---|---|---|---|
| **Model** | all training and inference (DINO stage 1, supervised stage 2, ensembling) | [`requirements-model.txt`](requirements-model.txt) | 3.11 |
| **Project** | local scorer, fold construction, pure analysis units, tests | [`requirements-scoring.txt`](requirements-scoring.txt) | 3.12 |
| **Container** | the scored inference image | [`../docker/requirements.txt`](../docker/requirements.txt) | 3.10 (base image) |

```bash
# project env — pyproject.toml + uv.lock are in the repository root
uv sync
uv run pytest -q

# model env — torch/torchvision from the cu130 index, everything else from PyPI
python3.11 -m venv baseline/.venv-baseline
baseline/.venv-baseline/bin/pip install torch==2.12.0+cu130 torchvision==0.27.0+cu130 \
    --index-url https://download.pytorch.org/whl/cu130
baseline/.venv-baseline/bin/pip install -r env/requirements-model.txt
```

Anything that touches the model runs under `baseline/.venv-baseline/bin/python`. Anything that only
scores, aggregates or builds folds runs under `uv run python`.

## Key versions

| Package | Model env | Project env | Container | Note |
|---|---|---|---|---|
| `torch` | 2.12.0+**cu130** | — | 2.3.1+**cu121** (base image) | The evaluation host is an RTX 3080 (**sm_86**) on CUDA 12.1; the development GPU is Blackwell (**sm_120**) and needs cu130. Neither wheel runs on the other's hardware — see the pitfall below. |
| `torchvision` | 0.27.0+cu130 | — | 0.18.1+cu121 | follows `torch` |
| `timm` | 1.0.27 | — | 1.0.27 | same pin both sides; it builds the DINOv2 graph |
| `numpy` | 1.26.4 | 2.4.6 | 1.26.4 | the model env is deliberately held on 1.x for albumentations 2.0.8 |
| `albumentations` | 2.0.8 | — | **not installed** | training-only; the container replicates `A.Resize`/`A.Normalize` in cv2+numpy |
| `opencv` | 4.11.0.86 | — | 4.11.0.86 (headless) | |
| `pandas` | 2.2.3 | 3.0.3 | **not installed** | the container reads metadata with the stdlib `csv` module |
| `scipy` | 1.17.1 | 1.17.1 | — | |

The two training environments are genuinely different interpreters with different numpy majors;
that is the point of keeping them apart, not an inconsistency. Read each column against its own
pin file.

`requirements-model.txt` is a full freeze (133 packages) captured from the machine that produced the
submitted checkpoints, so it includes transitive dependencies. `requirements-scoring.txt` is the
resolved set for the project venv; `pyproject.toml` + `uv.lock` in the repository root are the
authoritative source for it.

## Three things that will bite you

**1. The container base image cannot run on a Blackwell GPU.** The organizers' evaluation host is
CUDA 12.1 / sm_86, so `docker/Dockerfile` pins `pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime`.
Those kernels do not exist for sm_120, so `docker run --gpus all` fails on the development machine
with *"no kernel image is available for execution on the device"* — for reasons unrelated to this
code. The consequence is real and is disclosed in [`../README.md`](../README.md): the shipped image
was verified on CPU and by a full numeric diff against the research path, but its CUDA path was
never executed before submission. If you rebuild it, run `docker/selfcheck.sh --gpu` on an
Ampere-class card first.

**2. The container must build and run offline.** Evaluation runs with `--network none`, so nothing
may be fetched at runtime. `timm`'s usual `pretrained=True` path would hard-fail there. It is not
needed: every shipped checkpoint carries all 174 DINOv2 encoder tensors, so `docker/fub_arch.py`
constructs the backbone with `pretrained=False` and loads everything from our own `state_dict`s.
Nothing is downloaded, and no pretrained-weight `COPY` is required.

**3. The container's dependency surface is deliberately smaller than the trainer's.**
`albumentations` and `pandas` are not installed in the image. `A.Resize` and `A.Normalize` are
replicated exactly in cv2+numpy and verified numerically against the research pipeline
(`docker/verify_against_reference.py`, worst deviation 1.85e-4 px over all 619 validation images).
Fewer pins means less to break in an image whose GPU path cannot be fully smoke-tested locally.

## Hardware

Training was done on a single NVIDIA RTX PRO 6000 Blackwell Max-Q (96 GB). A supervised ViT-B fold
peaks well inside that, and two folds run concurrently. The evaluation target is far smaller — an
RTX 3080 with **10 GB VRAM**, **7 GB container RAM**, 4 CPU cores, 2 GB shm and a 5–6 h budget for
the whole hidden test set. That memory ceiling, not runtime, is what shapes the inference code: the
seven ensemble members are loaded **one at a time** with canonical heatmaps accumulated on CPU, so
only one ViT-B is ever resident.
