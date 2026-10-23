"""Central configuration for EdgeSplit-Bench.

Everything that is "a fact about the setup" lives here: container names, ports,
model profiles (layer count / hidden size) and result paths.

Set EDGESPLIT_MOCK=1 to run the whole pipeline against a built-in simulator
(no Docker needed). Mock results go to results_mock/ and are watermarked in the
plots. NEVER use mock numbers in a CV, report or interview.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

CLIENT = "es-client"
WORKERS = {"w1": "es-w1", "w2": "es-w2", "w3": "es-w3"}
RPC_PORT = 50052
ALL_CONTAINERS = [CLIENT, *WORKERS.values()]

# Verify n_layers / n_embd in the llama.cpp startup log (llama-cli prints
# "n_layer" and "n_embd") - model families differ!
MODEL_PROFILES = {
    "dev": {  # small: fast iteration while developing
        "n_layers": 24,
        "n_embd": 896,
        "files": {"q4_k_m": "qwen2.5-0.5b-instruct-q4_k_m.gguf"},
    },
    "main": {  # used for the real experiments
        "n_layers": 28,
        "n_embd": 1536,
        "files": {
            "q2_k": "qwen2.5-1.5b-instruct-q2_k.gguf",
            "q4_k_m": "qwen2.5-1.5b-instruct-q4_k_m.gguf",
            "q8_0": "qwen2.5-1.5b-instruct-q8_0.gguf",
        },
    },
}

# Only used in mock mode when the GGUF file is not on disk.
_MOCK_SIZES = {"q2_k": 0.68e9, "q4_k_m": 1.12e9, "q8_0": 1.89e9}


def mock() -> bool:
    return os.environ.get("EDGESPLIT_MOCK") == "1"


def results_dir() -> Path:
    d = ROOT / ("results_mock" if mock() else "results")
    d.mkdir(parents=True, exist_ok=True)
    return d


def results_csv() -> Path:
    return results_dir() / "results.csv"


def fig_dir() -> Path:
    d = results_dir() / "figures"
    d.mkdir(parents=True, exist_ok=True)
    return d


def profile(name: str) -> dict:
    if name not in MODEL_PROFILES:
        raise SystemExit(f"unknown profile {name!r}; choose from {list(MODEL_PROFILES)}")
    return MODEL_PROFILES[name]


def model_file(profile_name: str, quant: str) -> str:
    files = profile(profile_name)["files"]
    if quant not in files:
        raise SystemExit(f"profile {profile_name!r} has no quant {quant!r}; choose from {list(files)}")
    return files[quant]


def container_model_path(profile_name: str, quant: str) -> str:
    return f"/models/{model_file(profile_name, quant)}"


def model_size_bytes(profile_name: str, quant: str) -> float:
    p = ROOT / "models" / model_file(profile_name, quant)
    if p.exists():
        return float(p.stat().st_size)
    if mock():
        return _MOCK_SIZES.get(quant, 1.0e9)
    raise SystemExit(f"{p} not found - run scripts/download_models.sh first")
