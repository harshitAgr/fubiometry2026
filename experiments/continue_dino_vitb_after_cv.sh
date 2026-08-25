#!/usr/bin/env bash
# Detached fail-closed controller: wait for the active CV orchestrator, evaluate the strict gate,
# conditionally train/audit the full-data family, and STOP before Docker.
set -euo pipefail
cd "$(dirname "$0")/.."
CV_PID="${CV_PID:?set CV_PID to the detached five-fold orchestrator PID}"
while kill -0 "$CV_PID" 2>/dev/null; do
  command_line="$(ps -o args= -p "$CV_PID")"
  [[ "$command_line" == *"reproduce_dino_vitb_5fold.sh"* ]] || break
  sleep 30
done

uv run python experiments/evaluate_dino_vitb_5fold.py \
  --out experiments/results/dino_ssl_vitb_5fold/decision.json

if [[ "$(jq -r '.deploy' experiments/results/dino_ssl_vitb_5fold/decision.json)" == true ]]; then
  echo "$(date -u) strict continued-DINO gate PASSED; starting full-data checkpoint family"
  bash experiments/reproduce_dino_vitb_full_family.sh \
    experiments/results/dino_ssl_vitb_5fold/decision.json
else
  echo "$(date -u) strict continued-DINO gate FAILED; no full-data training or Docker work"
fi
