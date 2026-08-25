#!/usr/bin/env bash
# CPU-only IVC centroid-preserving diameter LENGTH calibration, LOFO-fitted panel.
set -euo pipefail

cd "$(dirname "$0")/.."
uv run python experiments/ivc_length_calibration.py
