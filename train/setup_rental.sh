#!/usr/bin/env bash
# Setup of a rented GPU pod (RunPod/Vast/Lambda) to train the 3 7B LoRAs.
# Target: 24-48 GB GPU (A10 24GB minimum, A100 40GB comfortable) — see train/RENTAL.md.
# Usage (on the pod, inside the repo folder):  bash train/setup_rental.sh
set -euo pipefail

# HF cache on the persistent VOLUME (/workspace on RunPod) so a Community Cloud
# pod preemption doesn't force you to re-download the 7B (~15GB). Change it if you don't use /workspace.
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
mkdir -p "$HF_HOME"
echo "HF_HOME=$HF_HOME"

echo "=== GPU on the pod ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || {
  echo "!! no GPU: pick a pod with a CUDA GPU before proceeding"; exit 1; }

PY=${PYTHON:-python3}
echo "=== venv ==="
$PY -m venv .venv 2>/dev/null || $PY -m venv .venv --without-pip
if [ ! -x .venv/bin/pip ]; then
  curl -fsSL https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py
  .venv/bin/python /tmp/get-pip.py -q
fi
. .venv/bin/activate
pip install -U -q pip

echo "=== training stack ==="
# Unsloth brings along torch(CUDA)/transformers/peft/trl/bitsandbytes/xformers,
# optimized for single-GPU QLoRA (§2/§6). If the pod already has torch, it leaves it alone.
pip install -q "unsloth" || echo "(unsloth not installed: the TRL+PEFT fallback will be used)"
pip install -q "transformers>=4.48" "trl>=0.13" "peft>=0.14" "datasets>=3.2" \
               "accelerate>=1.3" "bitsandbytes>=0.45" pyyaml python-dotenv \
               "math-verify" sympy tabulate tqdm

echo "=== CUDA check ==="
.venv/bin/python - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0),
          f"| {torch.cuda.mem_get_info()[1]/1e9:.0f} GB")
PY

echo
echo "OK. HF_HOME=$HF_HOME (persistent). Now upload the jsonl files to data/processed/ and:"
echo "  # calibrate the time/epoch before launching (§Phase 2):"
echo "  .venv/bin/python -m train.train_lora --specialist code --max-steps 40"
echo "  # then the real training (uses the *.filtered.jsonl if present, seq 8192):"
echo "  for s in math code science; do .venv/bin/python -m train.train_lora --specialist \$s --tier a; done"
