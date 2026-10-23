"""Thin wrapper around the Docker CLI to control the emulated edge nodes."""
from __future__ import annotations

import subprocess
import time

from . import config as C

# Bookkeeping of the CPU quota per node (also the state of the mock simulator).
CPUS: dict[str, float] = {"client": 4.0, "w1": 4.0, "w2": 4.0, "w3": 4.0}
THREADS: dict[str, int] = {"w1": 4, "w2": 4, "w3": 4}


class DockerError(RuntimeError):
    pass


def run(cmd: list[str], check: bool = True, timeout: float | None = None) -> subprocess.CompletedProcess:
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and p.returncode != 0:
        raise DockerError(f"command failed: {' '.join(cmd)}\n{p.stderr.strip()}")
    return p


def docker_exec(container: str, shell_cmd: str, detach: bool = False, check: bool = True,
                timeout: float | None = None) -> subprocess.CompletedProcess:
    cmd = ["docker", "exec"] + (["-d"] if detach else []) + [container, "sh", "-c", shell_cmd]
    return run(cmd, check=check, timeout=timeout)


def start_worker(name: str, threads: int = 4) -> None:
    """(Re)start rpc-server inside worker `name` with `threads` CPU threads.

    PID 1 of every container is `sleep infinity`, so we can restart rpc-server freely.
    """
    THREADS[name] = threads
    if C.mock():
        return
    c = C.WORKERS[name]
    docker_exec(c, "pkill -x rpc-server; true")
    time.sleep(0.5)
    docker_exec(
        c,
        f"exec rpc-server -H 0.0.0.0 -p {C.RPC_PORT} -t {threads} > /tmp/rpc-server.log 2>&1",
        detach=True,
    )


def wait_ready(name: str, timeout: float = 30.0) -> None:
    if C.mock():
        return
    c = C.WORKERS[name]
    t0 = time.time()
    while time.time() - t0 < timeout:
        p = docker_exec(c, f"bash -c 'exec 3<>/dev/tcp/127.0.0.1/{C.RPC_PORT}'", check=False)
        if p.returncode == 0:
            return
        time.sleep(0.5)
    log = docker_exec(c, "tail -n 20 /tmp/rpc-server.log", check=False).stdout
    raise DockerError(f"rpc-server on {name} not ready after {timeout}s. Log:\n{log}")


def start_workers(names: list[str], threads: int | dict[str, int] = 4) -> None:
    for n in names:
        t = threads[n] if isinstance(threads, dict) else threads
        start_worker(n, t)
    for n in names:
        wait_ready(n)


def set_cpus(name: str, cpus: float) -> None:
    """Limit the CPU quota of a node (emulates a weaker edge device)."""
    CPUS[name] = cpus
    if C.mock():
        return
    container = C.CLIENT if name == "client" else C.WORKERS[name]
    run(["docker", "update", f"--cpus={cpus}", container])


def rpc_endpoints(names: list[str]) -> list[str]:
    return [f"{n}:{C.RPC_PORT}" for n in names]
