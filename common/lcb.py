"""LiveCodeBench (code_generation_lite) — loader + I/O test format (§8 core).

The dataset is script-based (not loadable with datasets>=5): we read the raw
jsonl files from the repo (test.jsonl..test6.jsonl, incremental releases).

Row format (VERIFY — may change between releases):
  question_content, starter_code, difficulty, contest_date, question_id,
  public_test_cases: JSON [{input, output, testtype}],
  private_test_cases: JSON as above OR base64(zlib(pickle(json))) for large ones,
  metadata: JSON {func_name?} — func_name present => "call" mode, else "stdin".

Anti-contamination (§8/§15.1): filter by contest_date >= date_from (default
2024-08-01, after the R1-distill base cutoff).
"""
from __future__ import annotations

import base64
import json
import pickle
import zlib
from pathlib import Path

from common.config import ROOT, env
from common.benchmarks import Problem

CACHE = ROOT / "data" / "cache" / "lcb"
RELEASE_FILES = ["test.jsonl", "test2.jsonl", "test3.jsonl", "test4.jsonl",
                 "test5.jsonl", "test6.jsonl"]


def _decode_tests(raw: str) -> list[dict]:
    """Large private_test_cases are base64(zlib(pickle(json_string)))."""
    if not raw:
        return []
    try:
        return json.loads(raw)
    except Exception:
        try:
            blob = pickle.loads(zlib.decompress(base64.b64decode(raw)))
            return json.loads(blob) if isinstance(blob, str) else blob
        except Exception:
            return []


def _row_to_problem(row: dict, i: int) -> Problem:
    pub = _decode_tests(row.get("public_test_cases", ""))
    priv = _decode_tests(row.get("private_test_cases", ""))
    meta_raw = row.get("metadata") or "{}"
    try:
        meta = json.loads(meta_raw) if isinstance(meta_raw, str) else dict(meta_raw)
    except Exception:
        meta = {}
    fn = meta.get("func_name") or None
    starter = row.get("starter_code") or ""
    prompt = row["question_content"]
    if starter.strip():
        prompt += ("\n\nUse this starter code (keep the same signature):\n"
                   f"```python\n{starter}\n```")
    return Problem(
        id=f"lcb-{row.get('question_id', i)}",
        domain="code",
        prompt=prompt,
        tests=[],                     # unused: scoring via io_tests in meta
        public_tests=[],
        meta={
            "lcb": True,
            "exec_mode": "call" if fn else "stdin",
            "fn_name": fn,
            "starter_code": starter,
            "io_public": pub,
            "io_private": priv,
            "difficulty": row.get("difficulty"),
            "date": row.get("contest_date"),
        },
    )


def load_lcb(limit: int | None = None, date_from: str = "2024-08-01",
             releases: list[str] | None = None,
             difficulty: str | None = None) -> list[Problem]:
    from huggingface_hub import hf_hub_download

    tok = env("HF_TOKEN")
    problems: list[Problem] = []
    seen: set[str] = set()
    for fname in (releases or RELEASE_FILES):
        try:
            path = Path(hf_hub_download("livecodebench/code_generation_lite", fname,
                                        repo_type="dataset", token=tok,
                                        local_dir=str(CACHE)))
        except Exception as e:
            print(f"[lcb] skip {fname}: {str(e)[:80]}")
            continue
        with path.open() as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                row = json.loads(line)
                date = str(row.get("contest_date", ""))[:10]
                if date and date < date_from:
                    continue
                if difficulty and str(row.get("difficulty", "")).lower() != difficulty:
                    continue
                qid = str(row.get("question_id", f"{fname}-{i}"))
                if qid in seen:
                    continue
                seen.add(qid)
                problems.append(_row_to_problem(row, i))
    problems.sort(key=lambda p: (p.meta.get("date") or "", p.id))
    if limit:
        problems = problems[:limit]
    print(f"[lcb] {len(problems)} problems (date >= {date_from}"
          + (f", difficulty={difficulty}" if difficulty else "") + ")")
    return problems
