"""Aggregates results/runs/*.json into the master table (§10) and applies the gate (§12).

Produces:
  - results/master_table.md : rows = 6 baselines, columns = benchmarks; values =
    'final' metric + (for the system) % of the DeepSeek score.
  - report/gate_decision.md  : per-benchmark verdict according to the §12 criteria.

Usage:
  python -m eval.build_tables
"""
from __future__ import annotations

import json

from common.config import ROOT, RUNS_DIR

BASELINE_ORDER = [
    "1_base_naked", "2_base_specialist", "3_base_verifier",
    "3b_verifier_repair",  # 8d: agentic repair loop (no router)
    "4_full_system",
    "4b_full_system_repair",  # everything on: router + specialists + repair
    "5_deepseek_single", "6_deepseek_boN",
    "5r1_single", "6r1_boN",  # additional ceiling: R1 "the father"
]
CORE = ["math500", "livecodebench"]  # §0.2 verifiable core


def load_runs() -> dict[tuple[str, str], dict]:
    runs = {}
    if not RUNS_DIR.exists():
        return runs
    for f in sorted(RUNS_DIR.glob("*.json")):
        # tierA_*/smoke_* runs (Q4 CPU, smoke tests) go into the hardware table,
        # NOT here: they share the `baseline` field with GPU runs and would collide.
        # a2a_/a2b_ replication runs (other base families) likewise: they reuse
        # the same baseline ids and belong to the README replication table.
        if f.name.startswith(("tierA_", "smoke_", "a2a_", "a2b_")) or "_partial" in f.name:
            continue
        d = json.loads(f.read_text())
        if "quality" in d:
            runs[(d["baseline"], d["benchmark"])] = d
    return runs


def build_master(runs) -> str:
    benches = sorted({b for (_, b) in runs})
    lines = ["# Master table (§10)\n",
             "Value = `final` score (maj@k math / post-sel code). "
             "In parentheses: % of the DeepSeek single-shot score (baseline 5).\n"]
    header = "| Baseline | " + " | ".join(benches) + " |"
    sep = "|" + "---|" * (len(benches) + 1)
    lines += [header, sep]

    v4_single = {b: runs.get(("5_deepseek_single", b), {}).get("quality", {}).get("final")
                 for b in benches}
    r1_single = {b: runs.get(("5r1_single", b), {}).get("quality", {}).get("final")
                 for b in benches}
    for base in BASELINE_ORDER:
        cells = []
        for b in benches:
            r = runs.get((base, b))
            if not r:
                cells.append("—")
                continue
            val = r["quality"]["final"]
            pct = ""
            if base == "4_full_system":
                parts = []
                if v4_single.get(b):
                    parts.append(f"{round(100*val/v4_single[b])}% V4")
                if r1_single.get(b):
                    parts.append(f"{round(100*val/r1_single[b])}% R1")
                if parts:
                    pct = f" ({', '.join(parts)})"
            cells.append(f"{val:.1f}{pct}")
        lines.append(f"| {base} | " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def apply_gate(runs) -> str:
    def q(base, b):
        r = runs.get((base, b))
        return r["quality"]["final"] if r else None

    lines = ["# Gate decision (§12)\n",
             "> Draft auto-generated from the runs. Complete with qualitative analysis (§Phase 6).\n"]

    # 4 vs 5 on the core (§12)
    lines.append("## Verifiable core: 4 vs 5 (parity with the single-shot ceiling)\n")
    core_ok = []
    for b in CORE:
        f4, f5 = q("4_full_system", b), q("5_deepseek_single", b)
        if f4 is None or f5 is None:
            lines.append(f"- **{b}**: missing data (4={f4}, 5={f5}).")
            core_ok.append(None)
            continue
        pct = round(100 * f4 / f5) if f5 else 0
        ok = f4 >= f5
        core_ok.append(ok)
        lines.append(f"- **{b}**: system={f4:.1f} vs DeepSeek={f5:.1f} → {pct}% "
                     f"{'✅ ≥100%' if ok else '❌ <100%'}")

    # 4 vs 1/2/3 (architecture merit, §12) — on all available benchmarks
    lines.append("\n## Architecture merit: 4 vs 1/2/3\n")
    benches = sorted({b for (_, b) in runs})
    for b in benches:
        f4, f1, f2, f3 = (q("4_full_system", b), q("1_base_naked", b),
                          q("2_base_specialist", b), q("3_base_verifier", b))
        if f4 is None:
            continue
        def delta(x):
            return f"+{f4-x:.1f}" if x is not None else "n/a"
        d1 = (f4 - f1) if f1 is not None else None
        d2 = (f4 - f2) if f2 is not None else None
        mark = "✅" if (d1 is not None and d1 >= 5 and d2 is not None and d2 >= 5) else "⚠️"
        lines.append(f"- **{b}** {mark}: vs base(1)={delta(f1)}, vs specialist(2)={delta(f2)}, "
                     f"vs verifier(3)={delta(f3)}  (threshold ≥ +5 over 1 and 2)")

    # 4 vs 6 (honesty)
    lines.append("\n## Honesty: 4 vs 6 (DeepSeek with verification as well)\n")
    for b in benches:
        f4, f6 = q("4_full_system", b), q("6_deepseek_boN", b)
        if f4 is not None and f6 is not None:
            lines.append(f"- **{b}**: system={f4:.1f} vs DeepSeek+boN={f6:.1f} "
                         f"(Δ={f4-f6:+.1f})")

    # vs the father R1 (additional ceiling — the identity claim)
    lines.append("\n## The father: 4 vs R1 (does the system on consumer hardware beat the teacher?)\n")
    for b in benches:
        f4, fr1 = q("4_full_system", b), q("5r1_single", b)
        if f4 is not None and fr1 is not None:
            mark = "✅ BEATS" if f4 >= fr1 else "❌ below"
            lines.append(f"- **{b}**: system={f4:.1f} vs R1={fr1:.1f} "
                         f"(Δ={f4-fr1:+.1f}) {mark}")
        fr1b = q("6r1_boN", b)
        if f4 is not None and fr1b is not None:
            lines.append(f"    - apples-to-apples: vs R1+boN={fr1b:.1f} (Δ={f4-fr1b:+.1f})")

    # summary verdict
    lines.append("\n## Verdict (auto, §12)\n")
    if core_ok and all(x for x in core_ok if x is not None) and any(x for x in core_ok):
        lines.append("- Core ≥ parity with DeepSeek single-shot on all available benchmarks. "
                     "Now check the architecture merit (≥+5 vs 1 and 2) and Tier A "
                     "→ candidate **Promising**.")
    else:
        lines.append("- Core NOT at parity on all benchmarks → **Iterate / Rethink**: "
                     "analyze the bottleneck (router? specialists? N? quant?).")
    lines.append("\n*(The final decision requires the hardware table §Phase 5 and the "
                 "human judgment §Phase 6.)*")
    return "\n".join(lines) + "\n"


def main() -> None:
    runs = load_runs()
    if not runs:
        print(f"[tables] no runs in {RUNS_DIR}. Run eval.run_benchmarks first.")
        return
    master = build_master(runs)
    (ROOT / "results" / "master_table.md").write_text(master)
    print(f"[tables] wrote results/master_table.md ({len(runs)} runs)")
    gate = apply_gate(runs)
    (ROOT / "report" / "gate_decision.md").write_text(gate)
    print("[tables] wrote report/gate_decision.md")


if __name__ == "__main__":
    main()
