#!/usr/bin/env bash
# Eval del gate sul pod, da lanciare A TRAINING FINITO (ALL_TRAINING_DONE in train.log).
# Usa il venv-eval SEPARATO (vLLM) — non tocca il venv del training.
#
# Fa: serve vLLM multi-LoRA -> baseline locali 1,2,3,4 -> (opz) soffitti API -> tabelle.
# Uso:  bash eval/run_eval_pod.sh [LIMIT] [--with-ceilings]
set -uo pipefail
cd /workspace/horizon
export HF_HOME=/workspace/hf_cache
# il JIT di vLLM/flashinfer invoca `ninja` via subprocess: serve nel PATH
export PATH="/workspace/horizon/.venv-eval/bin:$PATH"

LIMIT="${1:-100}"
WITH_CEILINGS="${2:-}"
BENCHES="gsm8k math500 humaneval mbpp"
PY=.venv-eval/bin/python

echo "=== [1/5] verifica adapter ==="
for s in math code science; do
  test -f "adapters/horizon-$s-lora/adapter_config.json" \
    && echo "  OK horizon-$s-lora" || { echo "  MANCA horizon-$s-lora — abort"; exit 1; }
done

echo "=== [2/5] avvio vLLM multi-LoRA (log: vllm_server.log) ==="
# path assoluto del vllm del venv-eval; --lora-modules name=path per i 3 adapter (§7.2)
nohup .venv-eval/bin/vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
  --host 0.0.0.0 --port 8000 \
  --enable-lora --max-loras 3 --max-lora-rank 64 \
  --max-model-len 16384 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --lora-modules \
    horizon-math-lora=adapters/horizon-math-lora \
    horizon-code-lora=adapters/horizon-code-lora \
    horizon-science-lora=adapters/horizon-science-lora \
  > vllm_server.log 2>&1 &
VLLM_PID=$!
echo "  vLLM pid=$VLLM_PID — attendo readiness (max 15 min: primo load compila kernel)"
for i in $(seq 1 90); do
  sleep 10
  curl -sf http://localhost:8000/v1/models >/dev/null 2>&1 && { echo "  vLLM PRONTO"; break; }
  kill -0 $VLLM_PID 2>/dev/null || { echo "  vLLM MORTO — vedi vllm_server.log"; exit 1; }
  [ $i -eq 90 ] && { echo "  timeout vLLM"; exit 1; }
done

echo "=== [3/5] baseline locali 1,2,3,4 su: $BENCHES (limit=$LIMIT, workers=10) ==="
# workers=10: vLLM fa continuous batching -> wall-time ~/10 vs sequenziale
for b in 1_base_naked 2_base_specialist 3_base_verifier 4_full_system; do
  echo "--- baseline $b $(date -u +%H:%M) ---"
  $PY -m eval.run_benchmarks --baseline "$b" --benchmarks $BENCHES --limit "$LIMIT" \
    --workers 10 2>&1 | tail -5
done

if [ "$WITH_CEILINGS" = "--with-ceilings" ]; then
  echo "=== [4/5] soffitti API: V4 Flash + R1, single + boN ==="
  $PY -m eval.deepseek_ceiling --benchmarks $BENCHES --limit "$LIMIT" \
    --ceiling both --mode both 2>&1 | tail -8
else
  echo "=== [4/5] soffitti API: SALTATI (lancia con --with-ceilings o da locale) ==="
fi

echo "=== [5/5] tabelle ==="
$PY -m eval.build_tables
echo "EVAL_DONE $(date -u)"
