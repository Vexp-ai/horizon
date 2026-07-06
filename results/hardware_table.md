# Tabella hardware (§3, §Fase 5)

Macchina locale: **i7-11850H (8C/16T), 15,6 GB RAM, WSL2** — envelope **Tier A "CPU-16GB"**.
GPU locale A2000 **4 GB** (sotto l'envelope 8GB; utilizzabile per offload parziale, non misurato — richiede CUDA toolkit).

> ⚠️ **Misure di velocità rimandate — ROOT CAUSE DIAGNOSTICATA (2026-07-03 02:40):**
> banda RAM misurata **3,1 GB/s copy / 10 GB/s read** (normale: 25–40 GB/s) → il decode
> LLM (bandwidth-bound) crawla a ~0,7 tok/s mentre la CPU è sana (matmul 135 GFLOPS,
> cache-resident) e il disco veloce (1,4 GB/s). Causa più probabile: **laptop in
> risparmio energetico/batteria** (uncore/RAM downclockati). FIX (2 min, lato utente):
> collegare l'alimentatore + profilo Windows "Prestazioni elevate", opz. `wsl --shutdown`.
> Escl. verificate: build llama.cpp (rebuild statico ok), disco, load host, swap, KV-quant.
> **RISOLTO col riavvio del PC (2026-07-03 10:00): pp 47,6 / tg 9,2 tok/s = sani.** Nota: il
> test numpy single-thread segna ~3,4 GB/s anche da sana (baseline della macchina); la
> degradazione notturna era 0,6-0,9.
> Gotcha llama.cpp scoperti: (1) senza `-c` usa il contesto di training del modello
> (131k → 7,5 GB KV → OOM su 15 GB); (2) `-c` è il totale diviso per gli slot `-np`;
> (3) KV q8_0 senza flash-attention degrada ~100×; (4) `llama-cli` deprecato ("use
> llama-completion"). Il retest è automatico al prossimo risveglio del monitor.

## Tier A — CPU 16GB (7B Q4_K_M + LoRA)

| Metrica | Valore | Note |
|---|---|---|
| File modello Q4_K_M | 4,7 GB | + 3 adapter GGUF f16 da 309 MB l'uno |
| RSS server (49k ctx f16 KV, 6 slot) | **~8,5 GB** ✅ | misurato; entra nell'envelope 16 GB con margine |
| RSS llama-cli picco | ~8,1 GB | misurato (prima del rallentamento host) |
| Tempo load (primo avvio, disco freddo) | ~40–60 s | misurato ~1 min incl. warmup |
| tok/s prompt processing | **47,6 ± 14** | llama-bench pp128, 8 thread |
| tok/s generazione (single) | **9,2 ± 0,3** | llama-bench tg32 — in linea con l'atteso |
| tok/s generazione (batch -np 6) | → rerun | il best-of-N ammortizza (v2 §3) |
| Latenza/problema con N=8 | → rerun | dai json tierA quando il run completa |
| **Qualità Q4 vs bf16 (stessi problemi)** | → rerun | baseline 3 @25 math500 / @20 humaneval vs pod (93/98 @100) |

## Tier B — floor 8GB (1.5B Q4_K_M, CPU)

| Metrica | Valore | Note |
|---|---|---|
| File modello | **1,12 GB** ✅ | scaricato; + KV → **entra largamente in 8 GB** |
| tok/s generazione | **38,9 ± 0,9** (pp 273,9) | llama-bench — quasi interattivo su CPU |
| Qualità | non misurata | floor "deliberato" dichiarato (§3); qualità < Tier A per costruzione |

## Leve di velocità disponibili (Fase 7, post-gate — misurare con tabella prima/dopo)
1. **Speculative decoding** draft 1.5B (`-md`) — lossless, atteso 1.3–2×
2. **Offload parziale A2000 4GB** (`-ngl`) — atteso 1.5–2× (richiede CUDA toolkit, sudo)
3. **Early-stop self-consistency** — riduce N effettivo ~1.5–2× (flag off nelle baseline)
4. KV q8_0 **con** flash-attention (senza `-fa` degrada 100×: gotcha misurato)
