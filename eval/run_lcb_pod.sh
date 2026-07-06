#!/usr/bin/env bash
# Run LCB completo sul pod (vLLM gia' attivo): baseline 1, 3, 4 su livecodebench.
# Uso: bash eval/run_lcb_pod.sh [LIMIT=100]
set -uo pipefail
cd /workspace/horizon
export HF_HOME=/workspace/hf_cache
export PATH=/workspace/horizon/.venv-eval/bin:$PATH
LIMIT="${1:-100}"
curl -sf http://localhost:8000/v1/models >/dev/null || { echo "vLLM non attivo"; exit 1; }
for b in 1_base_naked 3_base_verifier 4_full_system; do
  echo "=== $b su livecodebench @$LIMIT $(date -u +%H:%M) ==="
  .venv-eval/bin/python -m eval.run_benchmarks --baseline "$b" \
    --benchmarks livecodebench --limit "$LIMIT" --workers 8 2>&1 | tail -3
done
echo "LCB_LOCAL_DONE $(date -u +%H:%M)"
