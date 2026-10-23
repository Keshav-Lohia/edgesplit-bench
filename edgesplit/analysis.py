"""Load results, fit the communication model, print summaries."""
from __future__ import annotations

import argparse

import pandas as pd

from . import config as C
from .model import TgParams, fit_tg


def load() -> pd.DataFrame:
    p = C.results_csv()
    if not p.exists():
        raise SystemExit(f"{p} does not exist - run experiments first")
    df = pd.read_csv(p)
    for col in ("rtt_ms", "rate_mbit", "pp_ts", "tg_ts", "load_s", "pred_tg_ts"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def fit_from_results(profile: str, quant: str = "q4_k_m") -> TgParams | None:
    """Fit the decode model on all E1-E3 RPC points of one model; None if not enough data."""
    p = C.results_csv()
    if not p.exists():
        return None
    df = load()
    d = df[(df["profile"] == profile) & (df["quant"] == quant) & (df["n_nodes"] >= 1)
           & df["experiment"].isin(["e1", "e2", "e3"]) & df["tg_ts"].notna()]
    if len(d) < 6 or d["n_nodes"].nunique() < 2:
        return None
    n_embd = C.profile(profile)["n_embd"]
    return fit_tg(d["n_nodes"], d["rtt_ms"], [None if pd.isna(x) else x for x in d["rate_mbit"]],
                  d["tg_ts"], n_embd)


def cmd_fit(args) -> None:
    prm = fit_from_results(args.profile, args.quant)
    if prm is None:
        raise SystemExit("not enough E1-E3 data to fit (need >= 6 points with >= 2 node counts)")
    print(f"t0 (single-node compute)   : {prm.t0 * 1000:8.2f} ms/token")
    print(f"o  (per-node RPC overhead) : {prm.o * 1000:8.2f} ms/token")
    print(f"k  (round trips per hop)   : {prm.k:8.2f}")
    print(f"beta (data volume factor)  : {prm.beta:8.2f}")
    print(f"R^2 = {prm.r2:.4f}   MAPE = {prm.mape * 100:.1f} %")
    print("\nInterpretation: k > 1 means llama.cpp's RPC backend needs several synchronous round "
          "trips per node and token - that is why decode degrades so quickly with latency.")


def cmd_summary(args) -> None:
    df = load()
    cols = ["experiment", "quant", "n_nodes", "rtt_ms", "rate_mbit", "split", "pp_ts", "tg_ts", "load_s", "note"]
    print(df[[c for c in cols if c in df]].to_string(index=False, float_format=lambda x: f"{x:.2f}"))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="analyse EdgeSplit-Bench results")
    ap.add_argument("command", choices=["fit", "summary"])
    ap.add_argument("--profile", default="main")
    ap.add_argument("--quant", default="q4_k_m")
    args = ap.parse_args(argv)
    {"fit": cmd_fit, "summary": cmd_summary}[args.command](args)


if __name__ == "__main__":
    main()
