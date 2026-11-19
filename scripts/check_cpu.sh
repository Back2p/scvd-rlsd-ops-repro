#!/usr/bin/env bash
set -euo pipefail
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
if [[ -z "${QWEN_TOKENIZER_DIR:-}" && -f /platform/models/Qwen3.5-9B/tokenizer.json ]]; then
  export QWEN_TOKENIZER_DIR=/platform/models/Qwen3.5-9B
fi
runtime_config="${CONFIG:-configs/runtime.example.json}"
python -m compileall -q ops_repro src scripts tests
python scripts/audit_sources.py
python scripts/run_cpu_checks.py
python -m ops_repro.doctor --config "$runtime_config" --runtime
python -m ops_repro.train --config "$runtime_config" --method scvd --output /platform/output/PLAN --plan
python -m ops_repro.train --config "$runtime_config" --method rlsd --output /platform/output/PLAN --plan
# No worker, full model, download, teacher request, optimizer update or training command starts here.
