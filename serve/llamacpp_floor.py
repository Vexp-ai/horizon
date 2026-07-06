"""Tier B floor — llama.cpp (GGUF Q4_K_M) + LoRA (§3, §7.2).

Helper to: (1) convert base+LoRA to GGUF, (2) launch llama-server with an
adapter. The floor is "deliberate/slow" on CPU/8GB (§3, §15.4): it measures
quality and latency, without expecting parity.

Assumes llama.cpp is compiled and its Python scripts are available
(convert_hf_to_gguf.py, convert_lora_to_gguf.py). Pass the path with --llama-cpp.

Usage:
  # 1) conversion (one-off)
  python -m serve.llamacpp_floor convert --llama-cpp /path/llama.cpp --specialist math
  # 2) serve
  python -m serve.llamacpp_floor serve --specialist math --llama-cpp /path/llama.cpp
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from common.config import ADAPTER_DIR, ROOT, base_config, specialist_config

GGUF_DIR = ROOT / "models" / "gguf"


def cmd_convert(args) -> None:
    bc = base_config()
    base_model = bc["base_model"]["tier_b"]
    qtype = bc["quant"]["gguf_type"]
    lc = Path(args.llama_cpp)
    GGUF_DIR.mkdir(parents=True, exist_ok=True)

    base_out = GGUF_DIR / f"base-{qtype}.gguf"
    print(f"[floor] converting base {base_model} -> {base_out}")
    subprocess.run(
        ["python", str(lc / "convert_hf_to_gguf.py"), base_model,
         "--outtype", qtype, "--outfile", str(base_out)],
        check=True,
    )

    cfg = specialist_config(args.specialist)
    lora_dir = ADAPTER_DIR / cfg["output_name"]
    lora_out = GGUF_DIR / f"{cfg['output_name']}.gguf"
    print(f"[floor] converting LoRA {lora_dir} -> {lora_out}")
    subprocess.run(
        ["python", str(lc / "convert_lora_to_gguf.py"), str(lora_dir),
         "--base", base_model, "--outfile", str(lora_out)],
        check=True,
    )
    print(f"[floor] done: {base_out}, {lora_out}")


def cmd_serve(args) -> None:
    bc = base_config()
    qtype = bc["quant"]["gguf_type"]
    cfg = specialist_config(args.specialist)
    base_out = GGUF_DIR / f"base-{qtype}.gguf"
    lora_out = GGUF_DIR / f"{cfg['output_name']}.gguf"
    lc = Path(args.llama_cpp)

    cmd = [
        str(lc / "llama-server"),
        "-m", str(base_out),
        "--lora", str(lora_out),
        "--port", str(args.port),
        "-c", str(bc["serving"]["vllm"]["max_model_len"]),
        # v2 (§3, §7.2): best-of-N on CPU must be measured in BATCH — the N
        # parallel sequences amortize reading the weights (decode is bandwidth-bound).
        "-np", str(args.parallel),
    ]
    if args.gpu_layers is not None:
        # partial offload for small GPUs (e.g. A2000 4GB: some layers on GPU, rest on CPU)
        cmd += ["-ngl", str(args.gpu_layers)]
    if args.draft:
        # v2: speculative decoding (lossless). ALWAYS LOG when active (§2, §15.10).
        cmd += ["-md", str(args.draft)]
    print(f"[floor] speculative={'ON draft=' + args.draft if args.draft else 'off'} "
          f"batch_np={args.parallel} gpu_layers={args.gpu_layers}")
    print("[floor] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)

    c = sub.add_parser("convert")
    c.add_argument("--llama-cpp", required=True)
    c.add_argument("--specialist", required=True, choices=["math", "code", "science"])
    c.set_defaults(func=cmd_convert)

    s = sub.add_parser("serve")
    s.add_argument("--llama-cpp", required=True)
    s.add_argument("--specialist", required=True, choices=["math", "code", "science"])
    s.add_argument("--port", type=int, default=8000)
    s.add_argument("--parallel", type=int, default=8,
                   help="v2: parallel sequences (-np) for batched best-of-N")
    s.add_argument("--gpu-layers", type=int, default=None,
                   help="partial offload (-ngl) for small GPUs, e.g. A2000 4GB")
    s.add_argument("--draft", default=None,
                   help="v2: draft GGUF for speculative decoding (-md), e.g. 1.5B Q4; logged")
    s.set_defaults(func=cmd_serve)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
