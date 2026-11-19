#!/usr/bin/env bash
set -euo pipefail
: "${CONFIG:?Set CONFIG to the reviewed runtime JSON}"
: "${OUTPUT:?Set OUTPUT to a fresh run directory}"
torchrun --standalone --nproc_per_node="${TRAIN_GPUS:-8}" -m ops_repro.train \
  --config "$CONFIG" --method "${METHOD:-rlsd}" --output "$OUTPUT"
