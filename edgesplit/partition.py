"""Profiling-driven layer partitioning for heterogeneous edge nodes.

Given per-node speed (seconds per layer), memory capacity (max layers) and the cost of
adding a node to the pipeline (`hop_cost`), find the layer split that minimises
  - "latency":    sum_i l_i * c_i + hop_cost * (#active nodes)      (single stream, batch 1)
  - "throughput": max_i ( l_i * c_i + hop_cost )                    (pipelined, many requests)

Key insight: for ONE request the stages run one after another, so splitting never makes
decoding faster - you split only because the model does not fit on one node. When the model
must be split, put as many layers as possible on the fastest nodes. A balanced split
(equal stage times) is optimal only if you can keep all stages busy with several requests.

The search is exhaustive over all compositions of n_layers into len(nodes) parts, which is
cheap for <= 5 nodes and <= 80 layers (a few 10^4 candidates).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class Node:
    name: str
    sec_per_layer: float
    max_layers: int


@dataclass(frozen=True)
class Plan:
    layers: tuple[int, ...]
    objective: str
    cost: float

    def tensor_split(self) -> str:
        """Value for llama.cpp's --tensor-split / llama-bench -ts (proportions, '/'-separated)."""
        return "/".join(str(x) for x in self.layers)


def compositions(total: int, caps: list[int]) -> Iterator[tuple[int, ...]]:
    """All ways to distribute `total` layers over nodes with per-node caps."""
    n = len(caps)

    def rec(i: int, remaining: int, acc: tuple[int, ...]):
        if i == n - 1:
            if remaining <= caps[i]:
                yield acc + (remaining,)
            return
        # layers the later nodes can still absorb
        rest_cap = sum(caps[i + 1:])
        lo = max(0, remaining - rest_cap)
        hi = min(caps[i], remaining)
        for l in range(lo, hi + 1):
            yield from rec(i + 1, remaining - l, acc + (l,))

    yield from rec(0, total, ())


def latency_cost(layers: tuple[int, ...], nodes: list[Node], hop_cost: float) -> float:
    return sum(l * nd.sec_per_layer + (hop_cost if l > 0 else 0.0) for l, nd in zip(layers, nodes))


def stage_cost(layers: tuple[int, ...], nodes: list[Node], hop_cost: float) -> float:
    return max((l * nd.sec_per_layer + hop_cost) if l > 0 else 0.0 for l, nd in zip(layers, nodes))


def plan(nodes: list[Node], n_layers: int, hop_cost: float, objective: str = "latency") -> Plan:
    if sum(nd.max_layers for nd in nodes) < n_layers:
        raise ValueError("model does not fit: total node capacity < number of layers")
    cost_fn = {"latency": latency_cost, "throughput": stage_cost}[objective]
    best = None
    for layers in compositions(n_layers, [nd.max_layers for nd in nodes]):
        c = cost_fn(layers, nodes, hop_cost)
        key = (round(c, 12), sum(1 for l in layers if l > 0), layers)  # tie-break: fewer nodes
        if best is None or key < best[0]:
            best = (key, layers, c)
    return Plan(layers=best[1], objective=objective, cost=best[2])


def even_split(n_layers: int, n_nodes: int) -> tuple[int, ...]:
    base, extra = divmod(n_layers, n_nodes)
    return tuple(base + (1 if i < extra else 0) for i in range(n_nodes))


def proportional_split(nodes: list[Node], n_layers: int) -> tuple[int, ...]:
    """Layers proportional to node speed (largest-remainder rounding), ignoring memory caps."""
    speeds = [1.0 / nd.sec_per_layer for nd in nodes]
    total = sum(speeds)
    raw = [n_layers * s / total for s in speeds]
    floor = [int(r) for r in raw]
    order = sorted(range(len(nodes)), key=lambda i: raw[i] - floor[i], reverse=True)
    for i in order[: n_layers - sum(floor)]:
        floor[i] += 1
    return tuple(floor)
