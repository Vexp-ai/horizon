# Quarantined runs (never deleted, never in tables)

Runs that produced invalid data due to harness defects, preserved for audit:

- `a2b_ministral_3b_verifier_repair__livecodebench_template400.json` —
  77/100 problems errored (HTTP 400: strict Mistral chat template rejected
  consecutive user messages in repair turns). The reported quality field
  (52.17) is computed over the 23 surviving problems and is meaningless.
  Fixed in harness (user-turn merge) and fully re-run: honest result 21.0/100.
- `a2b_ministral_3_base_verifier__mbpp_8err_template400.json` — same defect,
  8/100 problems errored (88.0 over 92 problems). Re-run clean: 79.0/100.
