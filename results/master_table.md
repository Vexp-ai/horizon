# Master table (§10)

Value = `final` score (maj@k math / post-sel code). In parentheses: % of the DeepSeek single-shot score (baseline 5).

| Baseline | aime2024 | gsm8k | humaneval | livecodebench | math500 | mbpp | mmlu_stem |
|---|---|---|---|---|---|---|---|
| 1_base_naked | 43.3 | 85.0 | 79.0 | 25.0 | 85.0 | 72.0 | 89.3 |
| 2_base_specialist | — | 91.0 | 74.0 | — | 87.0 | 71.0 | — |
| 3_base_verifier | — | 90.0 | 98.0 | 38.0 | 93.0 | 92.0 | — |
| 3b_verifier_repair | — | — | — | 44.0 | — | — | — |
| 4_full_system | 53.3 (67% V4, 80% R1) | 95.0 (96% V4, 99% R1) | 99.0 (103% V4, 105% R1) | 39.0 (57% V4, 85% R1) | 92.0 (94% V4, 95% R1) | 92.0 (99% V4, 99% R1) | 90.0 (94% V4) |
| 4b_full_system_repair | — | — | — | 42.0 | — | — | — |
| 5_deepseek_single | 80.0 | 99.0 | 96.0 | 69.0 | 98.0 | 93.0 | 96.0 |
| 6_deepseek_boN | — | 96.0 | 97.0 | — | 98.0 | 96.0 | — |
| 5r1_single | 66.7 | 96.0 | 94.0 | 46.0 | 97.0 | 93.0 | — |
| 6r1_boN | — | — | — | — | — | — | — |
