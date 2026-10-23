"""Experiment runner.  Usage:  python -m edgesplit.experiments <check|e1|e2|e3|e4|e5|reset> [options]

e1  baseline: local vs. RPC with 1/2/3 nodes (no network shaping)
e2  latency sweep (RTT) for 1/2/3 nodes             -> decode is latency-bound
e3  bandwidth sweep for 2 nodes, long prompt        -> prefill is bandwidth-bound
e4  quantisation: speed + model load time (weights travel over the network!)
e5  heterogeneous nodes: profile -> plan -> compare split strategies
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt

from . import config as C
from . import netem, nodes
from . import partition as P
from .bench import BenchConfig, measure_load_time
from .bench import run as bench_run
from .model import TgParams

FIELDS = ["timestamp", "source", "experiment", "profile", "quant", "nodes", "n_nodes", "rtt_ms",
          "rate_mbit", "split", "n_prompt", "n_gen", "threads", "pp_ts", "pp_std", "tg_ts", "tg_std",
          "load_s", "pred_tg_ts", "note"]

ALL_NODES = ["w1", "w2", "w3"]


def record(row: dict) -> None:
    p = C.results_csv()
    new = not p.exists()
    with p.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


def point(exp: str, args, node_names: list[str], *, quant: str | None = None, rtt_ms: float = 0.0,
          rate_mbit: float | None = None, split: str | None = None, n_prompt: int | None = None,
          n_gen: int | None = None, note: str = "", load: bool = False,
          pred_tg_ts: float | None = None) -> dict:
    quant = quant or args.quant
    n_prompt = n_prompt or args.n_prompt
    n_gen = n_gen or args.n_gen
    netem.apply_all(rtt_ms, rate_mbit)
    cfg = BenchConfig(model_path=C.container_model_path(args.profile, quant),
                      rpc=nodes.rpc_endpoints(node_names), tensor_split=split, n_prompt=n_prompt,
                      n_gen=n_gen, reps=args.reps, threads=args.threads, profile=args.profile,
                      quant=quant, node_names=node_names)
    res = bench_run(cfg)
    load_s = measure_load_time(cfg) if load else ""
    row = {
        "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
        "source": "mock" if C.mock() else "real", "experiment": exp, "profile": args.profile,
        "quant": quant, "nodes": ",".join(node_names) or "local", "n_nodes": len(node_names),
        "rtt_ms": rtt_ms, "rate_mbit": "" if rate_mbit is None else rate_mbit, "split": split or "",
        "n_prompt": n_prompt, "n_gen": n_gen, "threads": args.threads, "pp_ts": res["pp_ts"],
        "pp_std": res["pp_std"], "tg_ts": res["tg_ts"], "tg_std": res["tg_std"], "load_s": load_s,
        "pred_tg_ts": "" if pred_tg_ts is None else pred_tg_ts, "note": note,
    }
    record(row)
    print(f"[{exp}] nodes={row['nodes']:<9} rtt={rtt_ms:>5}ms rate={str(rate_mbit or '-'):>5} "
          f"split={split or '-':<8} pp={res['pp_ts']:8.2f} t/s  tg={res['tg_ts']:7.2f} t/s  "
          f"{('load=%.1fs' % load_s) if load else ''} {note}")
    return row


def prepare(cpus: dict[str, float] | None = None, names: list[str] | None = None) -> None:
    names = names or ALL_NODES
    cpus = cpus or {n: 4.0 for n in names}
    for n in names:
        nodes.set_cpus(n, cpus[n])
    nodes.start_workers(names, threads={n: max(1, int(round(cpus[n]))) for n in names})
    netem.clear_all()


# ----------------------------------------------------------------------------- commands
def cmd_check(args) -> None:
    prepare()
    print("workers up. netem 20 ms RTT test:")
    netem.apply_all(rtt_ms=20)
    if C.mock():
        print("[mock] skipping ping")
    else:
        for w in ALL_NODES:
            out = nodes.docker_exec(C.CLIENT, f"ping -c 3 -q {w}").stdout.strip().splitlines()[-1]
            print(f"  client -> {w}: {out}")
    netem.clear_all()
    print("smoke benchmark (1 node, tiny):")
    small = argparse.Namespace(**{**vars(args), "reps": 1, "n_prompt": 16, "n_gen": 8})
    point("check", small, ["w1"])


def cmd_e1(args) -> None:
    prepare()
    for names in ([], ["w1"], ["w1", "w2"], ["w1", "w2", "w3"]):
        point("e1", args, names, note="local" if not names else "")


def cmd_e2(args) -> None:
    prepare()
    for n in args.node_counts:
        for rtt in args.rtts:
            point("e2", args, ALL_NODES[:n], rtt_ms=rtt)
    netem.clear_all()


def cmd_e3(args) -> None:
    prepare()
    names = ALL_NODES[:args.n]
    for rate in [None] + args.rates:
        point("e3", args, names, rtt_ms=1.0, rate_mbit=rate, n_prompt=args.long_prompt)
    netem.clear_all()


def cmd_e4(args) -> None:
    prepare()
    names = ALL_NODES[:args.n]
    for q in C.profile(args.profile)["files"]:
        point("e4", args, names, quant=q, load=True, note="unshaped")
        for rate in args.load_rates:
            point("e4", args, names, quant=q, rate_mbit=rate, load=True, note=f"load@{rate}Mbit")
    netem.clear_all()


def cmd_e5(args) -> None:
    from . import analysis
    prof = C.profile(args.profile)
    L, n_embd = prof["n_layers"], prof["n_embd"]
    cpus = {"w1": 4.0, "w2": 2.0, "w3": 1.0}
    cap_frac = {"w1": 0.6, "w2": 0.4, "w3": 0.4}  # "memory": no node can hold the whole model
    try:
        prepare(cpus)
        # 1) profile every node ALONE
        speeds = {}
        for n in ALL_NODES:
            row = point("e5-profile", args, [n], rtt_ms=args.rtt, note=f"cpus={cpus[n]}")
            speeds[n] = row["tg_ts"]
        # 2) hop cost from the fitted model (or a conservative default)
        fitted = analysis.fit_from_results(args.profile, args.quant)
        prm = fitted or TgParams(t0=0.0, o=0.003, k=1.5, beta=1.0)
        hop = prm.hop_cost(args.rtt, None, n_embd)
        print(f"hop cost = {hop * 1000:.2f} ms/token ({'fitted' if fitted else 'default'})")
        specs = [P.Node(n, max((1.0 / speeds[n] - hop) / L, 1e-6), int(cap_frac[n] * L)) for n in ALL_NODES]
        for s in specs:
            print(f"  {s.name}: {s.sec_per_layer * 1000:.2f} ms/layer, max {s.max_layers} layers")
        # 3) candidate strategies
        strategies = {
            "even": P.even_split(L, 3),
            "proportional": P.proportional_split(specs, L),
            "plan-latency": P.plan(specs, L, hop, "latency").layers,
            "plan-throughput": P.plan(specs, L, hop, "throughput").layers,
        }
        for label, layers in strategies.items():
            ok = all(l <= s.max_layers for l, s in zip(layers, specs))
            pred = 1.0 / P.latency_cost(layers, specs, hop)
            point("e5", args, ALL_NODES, rtt_ms=args.rtt, split="/".join(map(str, layers)),
                  pred_tg_ts=pred, note=label + ("" if ok else " (violates memory caps)"))
    finally:
        netem.clear_all()
        prepare()  # restore homogeneous nodes


def cmd_reset(args) -> None:
    prepare()
    print("nodes reset: 4 CPUs each, rpc-server restarted, netem cleared")


# ----------------------------------------------------------------------------- CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["check", "e1", "e2", "e3", "e4", "e5", "reset"])
    ap.add_argument("--profile", default="main", choices=list(C.MODEL_PROFILES))
    ap.add_argument("--quant", default="q4_k_m")
    ap.add_argument("--threads", type=int, default=4, help="threads of the client process")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--n-prompt", type=int, default=64)
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--rtts", type=float, nargs="+", default=[0, 1, 2, 5, 10, 20, 50], help="e2: RTTs in ms")
    ap.add_argument("--node-counts", type=int, nargs="+", default=[1, 2, 3], help="e2: nodes")
    ap.add_argument("--rates", type=float, nargs="+", default=[500, 200, 100, 50, 20, 10], help="e3: Mbit/s")
    ap.add_argument("--long-prompt", type=int, default=128, help="e3: prompt length")
    ap.add_argument("--n", type=int, default=2, help="e3/e4: number of worker nodes")
    ap.add_argument("--load-rates", type=float, nargs="+", default=[100], help="e4: link rates for load time")
    ap.add_argument("--rtt", type=float, default=2.0, help="e5: RTT in ms")
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    {"check": cmd_check, "e1": cmd_e1, "e2": cmd_e2, "e3": cmd_e3, "e4": cmd_e4, "e5": cmd_e5,
     "reset": cmd_reset}[args.command](args)


if __name__ == "__main__":
    main()
