#!/usr/bin/env bash
set -euo pipefail
docker run --rm -p 8000:8000 \
  -e MODEL_WEIGHTS="${MODEL_WEIGHTS:-yolo11n.pt}" \
  -e CONF_THRESHOLD="${CONF_THRESHOLD:-0.20}" \
  -e IMGSZ="${IMGSZ:-832}" \
  -e ALLOWED_CLASSES="${ALLOWED_CLASSES:-person,car,bicycle,motorcycle,airplane,boat,truck,bus,train}" \
  localdetectionservice-vibe:cpu
