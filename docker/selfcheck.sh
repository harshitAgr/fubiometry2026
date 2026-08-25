#!/usr/bin/env bash
# Local self-check for the FU_Biometry final-test container.
#
# Simulates the platform runtime: /input read-only, /output writable, --network none,
# --memory 7g --cpus 4 --shm-size 2g, GU_INPUT_DIR / GU_OUTPUT_DIR, organizer ENTRYPOINT.
#
# ⚠️ CPU-ONLY BY DEFAULT, and that is a real limitation, not laziness. The eval host is CUDA 12.1
# / RTX 3080 (sm_86); this dev box is Blackwell sm_120, which the organizers' cuda12.1 base image
# cannot drive ("no kernel image is available"). So `--gpus all` CANNOT be exercised here. Pass
# --gpu ONLY on an Ampere-class machine. Everything except the CUDA kernels is covered here:
# entrypoint wiring, offline operation, file layout, task coverage, output schema.
#
# Usage:
#   bash docker/selfcheck.sh                 # build + CPU run over a small val subset
#   bash docker/selfcheck.sh --per-task 5    # bigger subset (slower on CPU)
#   bash docker/selfcheck.sh --gpu           # ONLY on an sm_86-compatible GPU host
#   bash docker/selfcheck.sh --no-build      # reuse the existing image
set -euo pipefail
cd "$(dirname "$0")/.."

IMAGE="${IMAGE:-fu-biometry:selfcheck}"
PER_TASK=2
GPU_FLAG=""
DO_BUILD=1
while [ $# -gt 0 ]; do
  case "$1" in
    --per-task) PER_TASK="$2"; shift 2 ;;
    --gpu)      GPU_FLAG="--gpus all"; shift ;;
    --no-build) DO_BUILD=0; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

WORK="$(pwd)/scratch_tmp/fub_selfcheck"
IN="$WORK/input"; OUT="$WORK/output"
rm -rf "$WORK"; mkdir -p "$IN" "$OUT"

# The staging + validation steps need cv2; only the baseline venv has it (the project .venv does
# not). Everything INSIDE the container uses the image's own interpreter.
PY_HOST="${PY_HOST:-baseline/.venv-baseline/bin/python}"
[ -x "$PY_HOST" ] || { echo "no host python with cv2 at $PY_HOST" >&2; exit 1; }

echo "=== [1/4] staging a $PER_TASK-image/task subset of the validation set into $IN ==="
"$PY_HOST" - "$IN" "$PER_TASK" <<'PY'
import csv, glob, os, shutil, sys
inp, per_task = sys.argv[1], int(sys.argv[2])
rows = []
for p in sorted(glob.glob("data/val/csv/*_val.csv")):
    recs = list(csv.DictReader(open(p)))[:per_task]
    for r in recs:
        task = r["task_id"]
        dst = os.path.join(inp, f"{task}_test")          # platform naming: <TASK>_test
        os.makedirs(dst, exist_ok=True)
        base = os.path.basename(r["image_path"])
        src = os.path.join("data/val/images", task, base)
        shutil.copy2(src, os.path.join(dst, base))
        import cv2
        h, w = cv2.imread(src).shape[:2]
        rows.append({"image_path": r["image_path"], "task_name": "Regression",
                     "task_id": task, "num_classes": r["num_classes"],
                     "height": h, "width": w})
with open(os.path.join(inp, "test_metadata.csv"), "w", newline="") as f:
    wtr = csv.DictWriter(f, fieldnames=["image_path", "task_name", "task_id",
                                        "num_classes", "height", "width"])
    wtr.writeheader(); wtr.writerows(rows)
print(f"staged {len(rows)} images across {len(set(r['task_id'] for r in rows))} tasks")
PY

if [ "$DO_BUILD" = "1" ]; then
  echo "=== [2/4] docker build (linux/amd64) ==="
  for f in best_model.pth model_s43.pth model_s44.pth; do
    [ -f "docker/weights/$f" ] || { echo "MISSING docker/weights/$f -- populate it first" >&2; exit 1; }
  done
  docker build --platform linux/amd64 -t "$IMAGE" docker/
else
  echo "=== [2/4] skipping build (--no-build) ==="
fi

echo "=== [3/4] running the container (network none, 7g, 4 cpus, 2g shm) ${GPU_FLAG:-[CPU]} ==="
set +e
docker run --rm $GPU_FLAG \
  --network none \
  --memory 7g --cpus 4 --shm-size 2g \
  -v "$IN":/input:ro \
  -v "$OUT":/output:rw \
  -e GU_INPUT_DIR=/input \
  -e GU_OUTPUT_DIR=/output \
  "$IMAGE"
RC=$?
set -e
echo "container exit code: $RC"
[ "$RC" -eq 0 ] || { echo "SELF-CHECK FAILED: non-zero exit" >&2; exit 1; }

echo "=== [4/4] validating $OUT/regression_predictions.json against the output contract ==="
"$PY_HOST" - "$OUT/regression_predictions.json" "$IN/test_metadata.csv" <<'PY'
import csv, json, sys
pred_path, meta_path = sys.argv[1], sys.argv[2]
preds = json.load(open(pred_path))
meta = list(csv.DictReader(open(meta_path)))
errs = []
if not isinstance(preds, list):
    errs.append("top level must be a JSON list")
want = {(m["task_id"], m["image_path"]): int(m["num_classes"]) for m in meta}
got = {}
for p in preds:
    if set(p.keys()) != {"image_path", "task_id", "predicted_points_pixels"}:
        errs.append(f"unexpected field set: {sorted(p.keys())}"); break
for p in preds:
    k = (p["task_id"], p["image_path"])
    got[k] = p["predicted_points_pixels"]
    n = want.get(k)
    if n is None:
        errs.append(f"prediction for a row not in metadata: {k}")
    elif len(p["predicted_points_pixels"]) != 2 * n:
        errs.append(f"{k}: expected {2*n} coords, got {len(p['predicted_points_pixels'])}")
    elif not all(isinstance(v, (int, float)) for v in p["predicted_points_pixels"]):
        errs.append(f"{k}: non-numeric coordinate")
for k in want:
    if k not in got:
        errs.append(f"MISSING prediction for {k} (rule 3.3 penalizes this with the worst score)")
print(f"rows in metadata : {len(want)}")
print(f"predictions      : {len(preds)}")
print(f"tasks covered    : {sorted({t for t, _ in got})}")
if errs:
    print("\nFAILURES:")
    for e in errs[:20]:
        print("  -", e)
    sys.exit(1)
print("\nOUTPUT CONTRACT OK")
PY
echo "=== SELF-CHECK PASSED (${GPU_FLAG:-CPU mode}) ==="
[ -z "$GPU_FLAG" ] && echo "NOTE: GPU path NOT exercised -- rerun with --gpu on an sm_86 host before submitting."
