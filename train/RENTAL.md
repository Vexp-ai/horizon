# Training the 3 7B LoRAs on a rented GPU

Data prep runs **locally** (CPU, free) and produces `data/processed/*.filtered.jsonl`:
decontaminated (§4) and **length-filtered** (only traces that fit WHOLE within 8192
tokens — see `data/filter_length.py`), with the `<think>` reasoning preserved. You upload
these to the pod, not the ~4.4 GB of raw datasets nor the HF cache. You pay for the GPU only
for the training. **Estimate 4090 @ seq 8192, 1 epoch: ~$15–25 total** (calibration on the pod gives the exact number).

> **Seq length already decided:** `max_seq_length: 8192` in the configs (safe on the 4090's 24 GB).
> *Code* traces beyond 8192 tokens are **filtered out** (not truncated) upstream, so it never
> trains on mutilated traces. The `*.filtered.jsonl` files are already prepared for this threshold.

## 1. Pick the pod (recommended RunPod 4090 config)
- **GPU:** 1× **RTX 4090 24 GB** (just one: the 3 LoRAs train sequentially).
  7B QLoRA at seq 8192 uses ~10–14 GB → fits comfortably. The configs are already at seq 8192.
- **Cloud:** Community Cloud is fine (~$0.34–0.44/h). It is *interruptible*: the persistent
  volume + saving the adapter after each LoRA cover you. Secure Cloud = zero risk, more expensive.
- **Container disk: 50 GB** (torch+CUDA+unsloth+xformers weigh ~10–15 GB; the RunPod default is not enough).
- **Volume disk: 40 GB** mounted on `/workspace` (repo + data ~2.2 GB + HF cache ~15 GB + adapters).
- **Image:** PyTorch CUDA 12.x, e.g. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
  (unsloth installs its own torch anyway; you only need CUDA ≥12.1, required by the 4090).
- **SSH:** enabled (port 22) for `scp` + launching. Expose port **8000** only if you also run the eval here.
- **System RAM:** ≥16 GB (4090 pods typically come with 32–64 GB), needed for data loading.

> Summary: 1× RTX 4090 24GB · Community Cloud · PyTorch CUDA 12.4 template ·
> **container 50GB · volume 40GB (/workspace)** · SSH on.

## 2. Bring the repo to the pod (without .venv, cache, adapters)
From the local machine:
```bash
cd /home/nicola/progetti
# include only the *.filtered.jsonl (the ones used by training); exclude heavy raw/clean files.
tar --exclude='horizon/.venv' --exclude='horizon/.git' \
    --exclude='**/__pycache__' --exclude='horizon/adapters' \
    --exclude='horizon/data/cache' --exclude='horizon/data/processed/*.raw.jsonl' \
    --exclude='horizon/data/processed/*.clean.jsonl' -czf horizon.tgz horizon/
# copy to the pod (replace the pod's host/port):
scp -P <PORT> horizon.tgz root@<POD_IP>:/workspace/
```
The tar keeps the `*.filtered.jsonl` (the training data) + the decontamination logs.
On the pod:
```bash
cd /workspace && tar xzf horizon.tgz && cd horizon
```

## 3. Environment setup
```bash
bash train/setup_rental.sh          # installs the stack + checks CUDA
```
Also create `.env` on the pod if you need HF to download the 7B base:
```bash
echo "HF_TOKEN=hf_..." > .env
```

## 4. Train the 3 adapters (§6)
```bash
. .venv/bin/activate
# 4a. CALIBRATION: 40 steps to estimate the full-epoch time before committing
python -m train.train_lora --specialist code --max-steps 40   # code = the heaviest

# 4b. Real TRAINING (automatically uses the *.filtered.jsonl, seq 8192, <think> preserved)
for s in math code science; do
  python -m train.train_lora --specialist $s --tier a
done
# output: adapters/horizon-{math,code,science}-lora/  (~100-300 MB each)
```
A quick sanity check per adapter is in the plan §6/§Fase 2 (beats the naked base on the dev set).

## 5. (Option) Eval on the pod
The 7B doesn't fit in the 4 GB available locally, so it's best to run the gate eval **on the pod**
(big GPU) in the same session:
```bash
pip install vllm lighteval
python -m serve.vllm_server                     # 7B base + 3 adapters (multi-LoRA)
# in another shell:
for b in 1_base_naked 2_base_specialist 3_base_verifier 4_full_system; do
  python -m eval.run_benchmarks --baseline $b --benchmarks math500 humaneval gsm8k mbpp
done
python -m eval.deepseek_ceiling --benchmarks math500 humaneval gsm8k mbpp   # ceiling (§9)
python -m eval.build_tables
```

## 6. Bring the adapters back home (for the local Tier B floor)
```bash
# from the local machine:
scp -P <PORT> -r root@<POD_IP>:/workspace/horizon/adapters ./adapters
```
With the adapters locally you can run the **Tier B floor** (1.5B/CPU, §Fase 5) and the
DeepSeek ceiling calls without a pod.

## Costs (§13)
Training 3 LoRAs on a spot 4090 ~**$15–25**, DeepSeek ceiling ~$10–20 via OpenRouter,
GPU eval on the same pod ~$10–20 → **~$40–65** for a credible gate.
Remember to **shut down the pod** as soon as you're done (Community Cloud bills for time powered on).
