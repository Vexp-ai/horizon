#!/usr/bin/env bash
# Fase 5 — Tier A locale (CPU 16GB envelope): misura hardware + qualita' Q4.
# Uso: bash eval/tier_a_local.sh [N_PROBLEMI] [BENCH...]
set -uo pipefail
cd /home/nicola/progetti/horizon
LIMIT="${1:-30}"; shift 2>/dev/null || true
BENCHES="${*:-math500 humaneval}"
LC=tools/llama.cpp/build/bin
BASE_GGUF=models/gguf/DeepSeek-R1-Distill-Qwen-7B-Q4_K_M.gguf
LOG=results/runs/tier_a_hw.log
mkdir -p results/runs

echo "=== [1/4] misure di carico: load time + RAM + tok/s single ===" | tee $LOG
/usr/bin/time -v $LC/llama-cli -m $BASE_GGUF -no-cnv -n 128 -t $(nproc) \
  -p "Compute 17*23 step by step." 2>&1 | grep -E "eval time|prompt eval|Maximum resident|Elapsed" | tee -a $LOG

echo "=== [2/4] llama-server (base Q4, -np 8 batching v2) ===" | tee -a $LOG
$LC/llama-server -m $BASE_GGUF --port 8080 -c 49152 -np 6 -t $(nproc) \
  > results/runs/llamacpp_server.log 2>&1 &
SRV=$!
for i in $(seq 1 60); do sleep 5; curl -sf http://localhost:8080/health >/dev/null 2>&1 && break; done
curl -sf http://localhost:8080/health >/dev/null || { echo "server non pronto"; kill $SRV; exit 1; }
echo "server pronto (pid $SRV)" | tee -a $LOG

echo "=== [3/4] qualita' Q4: baseline 3 (base+verifier) su subset @$LIMIT ===" | tee -a $LOG
export VLLM_BASE_URL="http://localhost:8080/v1"
export HORIZON_HTTP_TIMEOUT=7200
for b in $BENCHES; do
  .venv/bin/python -m eval.run_benchmarks --baseline 3_base_verifier \
    --benchmarks $b --limit $LIMIT --workers 6 --best-of-n 8 \
    --out results/runs/tierA_3_base_verifier__$b.json 2>&1 | tail -3 | tee -a $LOG
done

echo "=== [4/4] RAM di picco server ===" | tee -a $LOG
grep VmHWM /proc/$SRV/status 2>/dev/null | tee -a $LOG
kill $SRV 2>/dev/null
echo "TIER_A_DONE" | tee -a $LOG
