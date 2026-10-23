"""Network emulation with Linux tc/netem, applied to the egress of every container.

If every container delays its own egress by RTT/2, the round-trip time between ANY
two nodes is RTT. `rate` caps the egress bandwidth of each node.
"""
from __future__ import annotations

from . import config as C
from .nodes import docker_exec

STATE: dict = {"rtt_ms": 0.0, "rate_mbit": None}


def _apply_one(container: str, rtt_ms: float, rate_mbit: float | None, jitter_ms: float, iface: str) -> None:
    parts = ["tc", "qdisc", "replace", "dev", iface, "root", "netem"]
    if rtt_ms > 0:
        parts += ["delay", f"{rtt_ms / 2:.3f}ms"]
        if jitter_ms > 0:
            parts += [f"{jitter_ms:.3f}ms"]
    if rate_mbit:
        parts += ["rate", f"{rate_mbit}mbit"]
    docker_exec(container, " ".join(parts))


def clear_all(iface: str = "eth0") -> None:
    STATE.update(rtt_ms=0.0, rate_mbit=None)
    if C.mock():
        return
    for c in C.ALL_CONTAINERS:
        docker_exec(c, f"tc qdisc del dev {iface} root 2>/dev/null; true", check=False)


def apply_all(rtt_ms: float = 0.0, rate_mbit: float | None = None, jitter_ms: float = 0.0,
              iface: str = "eth0") -> None:
    """Set one-way delay rtt/2 and a rate limit on all nodes; (0, None) removes shaping."""
    if rtt_ms <= 0 and not rate_mbit:
        clear_all(iface)
        return
    STATE.update(rtt_ms=float(rtt_ms), rate_mbit=rate_mbit)
    if C.mock():
        return
    for c in C.ALL_CONTAINERS:
        _apply_one(c, rtt_ms, rate_mbit, jitter_ms, iface)


def show(iface: str = "eth0") -> str:
    if C.mock():
        return f"[mock] {STATE}"
    return "\n".join(f"{c}: " + docker_exec(c, f"tc qdisc show dev {iface}").stdout.strip()
                     for c in C.ALL_CONTAINERS)
