"""Tier A — multi-LoRA vLLM server (§7.2).

Loads the base ONCE and mounts the 3 adapters S-LoRA-style; requests
select the adapter via the `model` field (e.g. "horizon-math-lora").

This script builds and prints/runs the `vllm serve` command with the right
flags taken from config/base.yaml. You can also launch it by hand (see README).

Usage:
  python -m serve.vllm_server            # start
  python -m serve.vllm_server --print    # only print the command
"""
from __future__ import annotations

import argparse
import subprocess
import sys

from common.config import ADAPTER_DIR, base_config


def build_command(tier: str = "a") -> list[str]:
    bc = base_config()
    base_model = bc["base_model"]["tier_a" if tier == "a" else "tier_b"]
    v = bc["serving"]["vllm"]
    adapters = bc["adapters"]

    cmd = [
        "vllm", "serve", base_model,
        "--host", str(v["host"]),
        "--port", str(v["port"]),
        "--enable-lora",
        "--max-loras", str(v["max_loras"]),
        "--max-lora-rank", str(v["max_lora_rank"]),
        "--max-model-len", str(v["max_model_len"]),
        "--gpu-memory-utilization", str(v["gpu_memory_utilization"]),
        "--dtype", v["dtype"],
    ]
    # Register every adapter present on disk as "name=path" (§7.2).
    for spec, name in adapters.items():
        path = ADAPTER_DIR / name
        if path.exists():
            cmd += ["--lora-modules", f"{name}={path}"]
        else:
            print(f"[vllm] WARNING: adapter '{name}' not found in {path} — not registered",
                  file=sys.stderr)
    return cmd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tier", choices=["a", "b"], default="a")
    ap.add_argument("--print", dest="print_only", action="store_true")
    ap.add_argument("--speculative", action="store_true",
                    help="v2 (§2, §7.2): speculative decoding with a 1.5B draft (lossless, "
                         "LOGGED). VERIFY speculative+LoRA compatibility in the vLLM "
                         "version in use (§15.10): if incompatible, use only for runs without adapters.")
    args = ap.parse_args()

    cmd = build_command(args.tier)
    if args.speculative:
        import json as _json

        bc = base_config()
        spec_cfg = {"model": bc["base_model"]["tier_b"], "num_speculative_tokens": 5}
        cmd += ["--speculative-config", _json.dumps(spec_cfg)]  # VERIFY current flag
        print(f"[vllm] SPECULATIVE ON (draft={spec_cfg['model']}) — log it in the runs (§15.10)")
    else:
        print("[vllm] speculative off")
    print("[vllm] command:\n  " + " ".join(cmd))
    if args.print_only:
        return
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
