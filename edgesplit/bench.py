"""Run llama-bench inside the client container (or the simulator in mock mode)."""
from __future__ import annotations

import json
import random
import zlib
import subprocess
import time
from dataclasses import dataclass, field

from . import config as C
from . import netem, nodes


class BenchError(RuntimeError):
    pass


@dataclass
class BenchConfig:
    model_path: str                      # path INSIDE the container, e.g. /models/x.gguf
    rpc: list[str] = field(default_factory=list)   # ["w1:50052", ...]; empty = local CPU only
    tensor_split: str | None = None      # "16/11/1" (order = order of rpc)
    n_prompt: int = 64
    n_gen: int = 32
    reps: int = 3
    threads: int = 4                     # threads of the client process
    # simulator-only context (ignored by the real run)
    profile: str = "main"
    quant: str = "q4_k_m"
    node_names: list[str] = field(default_factory=list)


def build_cmd(cfg: BenchConfig) -> list[str]:
    cmd = ["llama-bench", "-m", cfg.model_path, "-p", str(cfg.n_prompt), "-n", str(cfg.n_gen),
           "-r", str(cfg.reps), "-t", str(cfg.threads), "-o", "json"]
    if cfg.rpc:
        cmd += ["--rpc", ",".join(cfg.rpc), "-ngl", "99"]
        if cfg.tensor_split:
            cmd += ["-ts", cfg.tensor_split]
    else:
        cmd += ["-ngl", "0"]
    return cmd


def parse_bench_json(text: str) -> dict:
    """Extract prefill (pp) and decode (tg) speeds from llama-bench JSON output."""
    start = text.find("[")
    if start < 0:
        raise BenchError(f"no JSON in llama-bench output:\n{text[:500]}")
    rows = json.loads(text[start:])
    out = {"pp_ts": float("nan"), "pp_std": float("nan"), "tg_ts": float("nan"), "tg_std": float("nan")}
    for r in rows:
        if r.get("n_gen", 0) == 0 and r.get("n_prompt", 0) > 0:
            out["pp_ts"], out["pp_std"] = r["avg_ts"], r.get("stddev_ts", float("nan"))
        elif r.get("n_prompt", 0) == 0 and r.get("n_gen", 0) > 0:
            out["tg_ts"], out["tg_std"] = r["avg_ts"], r.get("stddev_ts", float("nan"))
    return out


def run(cfg: BenchConfig, timeout: float = 3600) -> dict:
    """Run one benchmark point. Returns pp/tg stats and the wall time in seconds."""
    if C.mock():
        return _simulate(cfg)
    cmd = ["docker", "exec", C.CLIENT] + build_cmd(cfg)
    t0 = time.time()
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    wall = time.time() - t0
    if p.returncode != 0:
        raise BenchError(f"llama-bench failed ({p.returncode}):\n{p.stderr[-2000:]}")
    res = parse_bench_json(p.stdout)
    res["wall_s"] = wall
    return res


def measure_load_time(cfg: BenchConfig, timeout: float = 3600) -> float:
    """Wall time of a minimal run (1 prompt token, 1 generated token, 1 rep).

    This is dominated by loading the model - and with RPC the weights are SHIPPED OVER THE
    NETWORK to the workers, so load time ~ model_size / bandwidth.
    """
    mini = BenchConfig(**{**cfg.__dict__, "n_prompt": 1, "n_gen": 1, "reps": 1})
    return run(mini, timeout=timeout)["wall_s"]


# --------------------------------------------------------------------------- simulator
def _simulate(cfg: BenchConfig) -> dict:
    """Toy performance model used ONLY in mock mode to test the pipeline without Docker."""
    key = repr((cfg.model_path, cfg.rpc, cfg.tensor_split, cfg.n_prompt, netem.STATE["rtt_ms"],
                netem.STATE["rate_mbit"], dict(nodes.CPUS)))
    rng = random.Random(zlib.crc32(key.encode()))  # deterministic noise
    size = C.model_size_bytes(cfg.profile, cfg.quant)
    n_embd = C.profile(cfg.profile)["n_embd"]
    names = cfg.node_names or [e.split(":")[0] for e in cfg.rpc]
    rtt = netem.STATE["rtt_ms"] / 1000.0
    rate = netem.STATE["rate_mbit"]
    mem_bw = 6e9  # bytes/s per "4 CPU" node (decode is memory-bound)

    if names:
        if cfg.tensor_split:
            w = [float(x) for x in cfg.tensor_split.split("/")]
        else:
            w = [1.0] * len(names)
        fr = [x / sum(w) for x in w]
        speeds = [nodes.CPUS[n] / 4.0 for n in names]
        t_tok = sum(f * size / (mem_bw * s) for f, s in zip(fr, speeds) if f > 0)
        active = sum(1 for f in fr if f > 0)
        bits = 8.0 * n_embd * 4
        hop_tg = 0.003 + 1.5 * rtt + (bits / (rate * 1e6) if rate else 0.0)
        t_tg = t_tok + active * hop_tg
        hop_pp = 0.003 + rtt + (bits * cfg.n_prompt / (rate * 1e6) if rate else 0.0)
        t_pp = cfg.n_prompt * t_tok / 8.0 + active * hop_pp
        load = 1.0 + (8.0 * size / (rate * 1e6) if rate else 8.0 * size / 8e9)
    else:
        s = nodes.CPUS["client"] / 4.0
        t_tg = size / (mem_bw * s)
        t_pp = cfg.n_prompt * t_tg / 8.0
        load = 1.0

    j = lambda: 1.0 + rng.uniform(-0.02, 0.02)  # noqa: E731
    pp, tg = cfg.n_prompt / t_pp * j(), 1.0 / t_tg * j()
    wall = load + cfg.reps * (cfg.n_prompt / pp + cfg.n_gen / tg)
    return {"pp_ts": pp, "pp_std": 0.02 * pp, "tg_ts": tg, "tg_std": 0.02 * tg, "wall_s": wall}
