"""Create all figures from results.csv.   Usage: python -m edgesplit.plots"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from . import config as C  # noqa: E402
from .analysis import fit_from_results, load  # noqa: E402
from .model import predict_tg_ts  # noqa: E402


def _finish(fig, ax, name: str, title: str) -> None:
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if C.mock():
        fig.text(0.5, 0.5, "SIMULATED - MOCK DATA", ha="center", va="center", fontsize=28,
                 color="red", alpha=0.25, rotation=25)
    fig.tight_layout()
    out = C.fig_dir() / f"{name}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_e2(df, profile, quant) -> None:
    d = df[(df.experiment == "e2") & (df["profile"] == profile) & (df.quant == quant)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for n, g in d.groupby("n_nodes"):
        g = g.sort_values("rtt_ms")
        ax.errorbar(g.rtt_ms, g.tg_ts, yerr=g.tg_std, marker="o", capsize=3, label=f"{n} node(s)")
    ax.set_xlabel("round-trip time between nodes (ms)")
    ax.set_ylabel("decode speed (tokens/s)")
    ax.legend()
    _finish(fig, ax, "e2_latency_decode", "Decode speed vs. network latency")


def plot_e3(df, profile, quant) -> None:
    d = df[(df.experiment == "e3") & (df["profile"] == profile) & (df.quant == quant)].copy()
    if d.empty:
        return
    d["bw"] = d.rate_mbit.fillna(1000.0)  # unshaped docker bridge drawn at 1000 as a stand-in
    d = d.sort_values("bw")
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.plot(d.bw, d.pp_ts, "o-", label="prefill (pp)")
    ax.set_xscale("log")
    ax.set_xlabel("link rate per node (Mbit/s; 1000 = unshaped)")
    ax.set_ylabel("prefill speed (tokens/s)")
    ax2 = ax.twinx()
    ax2.plot(d.bw, d.tg_ts, "s--", color="tab:orange", label="decode (tg)")
    ax2.set_ylabel("decode speed (tokens/s)")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="best")
    _finish(fig, ax, "e3_bandwidth", "Bandwidth sensitivity of prefill and decode")


def plot_e4(df, profile) -> None:
    d = df[(df.experiment == "e4") & (df["profile"] == profile)]
    if d.empty:
        return
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0))
    un = d[d.note == "unshaped"].sort_values("tg_ts")
    axes[0].bar(un.quant, un.tg_ts, color="tab:blue")
    axes[0].set_ylabel("decode speed (tokens/s)")
    axes[0].set_title("Decode speed by quantisation")
    ld = d[d.note.str.startswith("load@")] if "note" in d else d.iloc[0:0]
    if not ld.empty:
        axes[1].bar(ld.quant, ld.load_s, color="tab:red")
        axes[1].set_ylabel("model load time (s)")
        axes[1].set_title(f"Load time over a {int(ld.rate_mbit.iloc[0])} Mbit/s link")
    for a in axes:
        a.grid(alpha=0.3)
    _finish(fig, axes[0], "e4_quantisation", "Decode speed by quantisation")


def plot_e5(df, profile) -> None:
    d = df[(df.experiment == "e5") & (df["profile"] == profile)]
    if d.empty:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    x = range(len(d))
    ax.bar([i - 0.2 for i in x], d.tg_ts, width=0.4, label="measured")
    if d.pred_tg_ts.notna().any():
        ax.bar([i + 0.2 for i in x], d.pred_tg_ts, width=0.4, label="planner prediction", alpha=0.6)
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"{n.split(' ')[0]}\n{s}" for n, s in zip(d.note, d.split)], fontsize=8)
    ax.set_ylabel("decode speed (tokens/s)")
    ax.legend()
    _finish(fig, ax, "e5_partitioning", "Layer-split strategies on heterogeneous nodes (4/2/1 CPUs)")


def plot_fit(df, profile, quant) -> None:
    prm = fit_from_results(profile, quant)
    if prm is None:
        return
    d = df[(df["profile"] == profile) & (df.quant == quant) & (df.n_nodes >= 1)
           & df.experiment.isin(["e1", "e2", "e3"]) & df.tg_ts.notna()]
    n_embd = C.profile(profile)["n_embd"]
    pred = [predict_tg_ts(prm, int(r.n_nodes), r.rtt_ms, None if r.rate_mbit != r.rate_mbit else r.rate_mbit, n_embd)
            for r in d.itertuples()]
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.scatter(d.tg_ts, pred, alpha=0.8)
    m = max(max(d.tg_ts), max(pred)) * 1.05
    ax.plot([0, m], [0, m], "k--", lw=1)
    ax.set_xlabel("measured decode speed (tokens/s)")
    ax.set_ylabel("model prediction (tokens/s)")
    ax.text(0.05, 0.9, f"R²={prm.r2:.3f}  k={prm.k:.2f}", transform=ax.transAxes)
    _finish(fig, ax, "model_fit", "Analytical model vs. measurement")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default="main")
    ap.add_argument("--quant", default="q4_k_m")
    a = ap.parse_args(argv)
    df = load()
    plot_e2(df, a.profile, a.quant)
    plot_e3(df, a.profile, a.quant)
    plot_e4(df, a.profile)
    plot_e5(df, a.profile)
    plot_fit(df, a.profile, a.quant)


if __name__ == "__main__":
    main()
