"""Config loading + project paths (§14).

A single place to: resolve the repo root, load the YAML files in `config/`,
and read env vars (with optional `.env` via python-dotenv).
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

try:  # .env is optional
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = ROOT / "config"
ADAPTER_DIR = ROOT / "adapters"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
RUNS_DIR = RESULTS_DIR / "runs"


def load_yaml(path: str | Path) -> dict[str, Any]:
    p = Path(path)
    if not p.is_absolute():
        p = CONFIG_DIR / p
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@lru_cache(maxsize=1)
def base_config() -> dict[str, Any]:
    return load_yaml("base.yaml")


def specialist_config(specialist: str) -> dict[str, Any]:
    """math|code|science -> config/lora_<specialist>.yaml"""
    return load_yaml(f"lora_{specialist}.yaml")


def env(name: str, default: str | None = None, required: bool = False) -> str | None:
    val = os.environ.get(name, default)
    if required and not val:
        raise RuntimeError(
            f"Missing required env var: {name}. Copy .env.example to .env and fill it in."
        )
    return val


def adapter_path(specialist: str) -> Path:
    """Local path of the LoRA adapter for a specialist."""
    name = base_config()["adapters"][specialist]
    return ADAPTER_DIR / name
