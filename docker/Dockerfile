# FU_Biometry final-test container.
# Base pinned by the organizers' guide: CUDA 12.1 runtime, RTX 3080 (sm_86) on the eval host.
# NOTE: this base CANNOT run on the dev box (Blackwell sm_120 needs cu13x), so the mandatory
# `--gpus all` self-check must be done on an Ampere-class GPU. The CPU self-check
# (selfcheck.sh, without --gpu) exercises every code path except the CUDA kernels.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# System dependencies for OpenCV.
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# No pretrained-weight COPY is needed: the fine-tuned checkpoints carry all 174 DINOv2 encoder
# keys, so fub_arch builds the backbone with pretrained=False and loads everything from our own
# state_dicts. Nothing is fetched at runtime (the container is --network none).

# Inference code. `model.py` MUST be our Model wrapper -- predict.py does `from model import
# Model` -- so the vendored architecture lives in fub_arch.py.
COPY predict.py model.py fub_arch.py fub_decode.py \
     hc_scale.py geometry_project.py ivc_calibrate.py /app/

# Ensemble members. best_model.pth is seed 42 (the one predict.py copies to the work dir).
# 5 continued-DINO base seeds (heatmap-averaged) + continued-DINO hcsmall/hchead members
# (coordinate-combined per task). See MEMBER_GROUPS / ROUTES in model.py. On top of that:
# gated HC ellipse scale s=0.975 + FA/HC label-geometry projection + gated IVC length
# calibration -- i.e. the v24 lever set at the externally-fitted HC scale, not val's argmin.
COPY weights/best_model.pth   /app/best_model.pth
COPY weights/model_s43.pth    /app/model_s43.pth
COPY weights/model_s44.pth    /app/model_s44.pth
COPY weights/model_s45.pth    /app/model_s45.pth
COPY weights/model_s46.pth    /app/model_s46.pth
COPY weights/model_hcsmall.pth /app/model_hcsmall.pth
COPY weights/model_hchead.pth  /app/model_hchead.pth

# Fail fast at BUILD time if any member cannot strict-load on CPU. The list expression releases
# each model immediately, matching the one-resident-at-a-time runtime contract.
RUN python3 -c "\
import gc, torch, fub_arch; \
names=['best_model.pth','model_s43.pth','model_s44.pth','model_s45.pth','model_s46.pth','model_hcsmall.pth','model_hchead.pth']; \
[(lambda m: print('build-check OK:', n, sorted(m.heads.keys())))(fub_arch.load_member('/app/'+n, torch.device('cpu'))) for n in names]; \
gc.collect()"

ENTRYPOINT ["python3", "/app/predict.py"]
