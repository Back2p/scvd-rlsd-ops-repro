#!/usr/bin/env bash
# Starts only when the platform explicitly invokes this script. Not run by packaging/tests.
set -euo pipefail
MODEL_PATH=${MODEL_PATH:?Set MODEL_PATH}
MODEL_REVISION=${MODEL_REVISION:?Set the pinned MODEL_REVISION}
RUN_ROOT=${RUN_ROOT:?Set a dedicated shared output root}
WORKERS=${WORKERS:-8}
BASE_PORT=${BASE_PORT:-8100}
HOST=${WORKER_HOST:-127.0.0.1}
DEVICE_OFFSET=${WORKER_DEVICE_OFFSET:-0}
mkdir -p "$RUN_ROOT/worker-logs"
pids=()
stop_children() { for pid in "${pids[@]}"; do kill "$pid" 2>/dev/null || true; done; }
trap stop_children EXIT INT TERM
for (( index=0; index<WORKERS; index++ )); do
  auth=()
  if [[ "$HOST" != "127.0.0.1" && "$HOST" != "localhost" ]]; then
    : "${RUNNER_API_KEY:?Set RUNNER_API_KEY before exposing workers over the cluster network}"
    auth=(--api-key-env RUNNER_API_KEY)
  fi
  CUDA_VISIBLE_DEVICES="$((DEVICE_OFFSET+index))" python -m ops_repro.worker --model "$MODEL_PATH" --revision "$MODEL_REVISION" \
    --allowed-reload-root "$RUN_ROOT" --host "$HOST" --port "$((BASE_PORT+index))" "${auth[@]}" \
    >"$RUN_ROOT/worker-logs/worker-$index.log" 2>&1 &
  pids+=("$!")
done
wait -n "${pids[@]}"
# If any worker exits, fail the supervisor and stop only its own child PIDs.
exit 1
