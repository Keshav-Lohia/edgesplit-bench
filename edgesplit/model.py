"""Analytical communication/latency model for layer-split (pipeline) LLM inference.

Decode (token generation, batch=1) for a model split over `n` RPC nodes:

    t_token = t0 + n * o + n * k * RTT + n * beta * (8 * D) / BW

    t0    compute time of the whole model on one node (s)
    o     fixed per-node RPC overhead (s)
    k     round trips per node-hop per token (fitted! a pure pipeline would give 1)
    beta  data-volume factor (1 = one transfer of D bytes per node-hop; 2 = up + down)
    D     bytes of the hidden state per token = n_embd * dtype_bytes (f32 -> 4)
    BW    link bandwidth (bit/s)

Prefill of P prompt tokens is bandwidth-bound instead: the volume is P * D per hop.
The point of the model: decode is LATENCY-bound, prefill is BANDWIDTH-bound.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

UNSHAPED_BW_BPS = None  # docker bridge without tc: treated as "infinite" (term = 0)


def hidden_bytes(n_embd: int, dtype_bytes: int = 4) -> int:
    return n_embd * dtype_bytes


def bw_term(n_embd: int, rate_mbit: float | None, tokens: int = 1, dtype_bytes: int = 4) -> float:
    """Seconds needed to push `tokens` hidden states over one hop (0 if unshaped)."""
    if not rate_mbit:
        return 0.0
    return 8.0 * hidden_bytes(n_embd, dtype_bytes) * tokens / (rate_mbit * 1e6)


@dataclass
class TgParams:
    t0: float
    o: float
    k: float
    beta: float
    r2: float = float("nan")
    mape: float = float("nan")

    def hop_cost(self, rtt_ms: float, rate_mbit: float | None, n_embd: int) -> float:
        """Cost (s/token) of adding one more node to the pipeline."""
        return self.o + self.k * rtt_ms / 1000.0 + self.beta * bw_term(n_embd, rate_mbit)


def predict_tg_time(p: TgParams, n_nodes: int, rtt_ms: float, rate_mbit: float | None, n_embd: int) -> float:
    return p.t0 + n_nodes * p.hop_cost(rtt_ms, rate_mbit, n_embd)


def predict_tg_ts(p: TgParams, n_nodes: int, rtt_ms: float, rate_mbit: float | None, n_embd: int) -> float:
    return 1.0 / predict_tg_time(p, n_nodes, rtt_ms, rate_mbit, n_embd)


def fit_tg(n_nodes, rtt_ms, rate_mbit, tg_ts, n_embd: int) -> TgParams:
    """Least-squares fit of (t0, o, k, beta) from measured decode speeds (tokens/s).

    Inputs are equally long sequences; rate_mbit may contain None/NaN for unshaped links.
    """
    n = np.asarray(n_nodes, dtype=float)
    rtt_s = np.asarray(rtt_ms, dtype=float) / 1000.0
    rate = np.array([np.nan if (r is None) else float(r) for r in rate_mbit], dtype=float)
    bw = np.array([0.0 if np.isnan(r) else bw_term(n_embd, r) for r in rate])
    y = 1.0 / np.asarray(tg_ts, dtype=float)

    X = np.column_stack([np.ones_like(n), n, n * rtt_s, n * bw])
    # beta is unidentifiable if no row is bandwidth-limited -> drop that column
    cols = [0, 1, 2, 3] if np.any(bw > 0) else [0, 1, 2]
    coef, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
    full = np.zeros(4)
    full[cols] = coef
    pred = X @ full
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2)) or 1e-12
    mape = float(np.mean(np.abs((y - pred) / y)))
    return TgParams(t0=full[0], o=full[1], k=full[2], beta=full[3], r2=1 - ss_res / ss_tot, mape=mape)


def predict_pp_ts(t_pp_compute_per_tok: float, n_nodes: int, rtt_ms: float, rate_mbit: float | None,
                  n_embd: int, n_prompt: int, k: float = 1.0, beta: float = 1.0) -> float:
    """Prefill throughput (tokens/s): compute + per-hop latency + P hidden states per hop."""
    t = (n_prompt * t_pp_compute_per_tok
         + n_nodes * (k * rtt_ms / 1000.0 + beta * bw_term(n_embd, rate_mbit, tokens=n_prompt)))
    return n_prompt / t
