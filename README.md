# EdgeSplit-Bench

Network-aware benchmarking and layer-partitioning for **distributed LLM inference on emulated
industrial edge nodes**, built on [llama.cpp](https://github.com/ggml-org/llama.cpp)'s RPC backend.

A client and three worker nodes run as Docker containers. A quantised GGUF model is split layer-wise
over the workers; Linux `tc/netem` emulates industrial network conditions (latency, bandwidth); CPU quotas
emulate weak devices. Python tooling automates the experiments, fits an analytical communication
model, and plans the layer split.

## What it answers
| # | Question | Experiment |
|---|----------|-----------|
| E1 | What does RPC splitting cost vs. one node? | local vs. 1/2/3 nodes |
| E2 | How does decode speed react to latency? | RTT sweep 0-50 ms |
| E3 | How does prefill react to bandwidth? | 10-500 Mbit/s sweep |
| E4 | What does quantisation change (speed, **load time over the network**)? | Q2_K / Q4_K_M / Q8_0 |
| E5 | Does profiling-driven partitioning beat naive splits? | 4/2/1-CPU nodes, memory caps |

## Layout
```
docker/          Dockerfile (llama.cpp + RPC + tc) and docker-compose (client + 3 workers)
edgesplit/       config, nodes (docker control), netem, bench (llama-bench wrapper + parser),
                 model (analytical model + fit), partition (planner), experiments (E1-E5),
                 analysis (fit/summary), plots
tests/           unit tests + full mock-mode pipeline test (no Docker needed)
scripts/         download_models.sh
```

## Quick start
```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v     # sanity check, no Docker
./scripts/download_models.sh                # ~4 GB
make build && make up
python -m edgesplit.experiments check       # smoke test
python -m edgesplit.experiments e2          # latency sweep, etc.
python -m edgesplit.analysis fit
python -m edgesplit.plots                   # figures in results/figures
```
`EDGESPLIT_MOCK=1` runs everything against a built-in simulator for development. Mock data is stored
separately and watermarked - it must never be reported as a measurement.

## Results (summary)

Measured on one laptop (i9-14900HX, Docker, llama.cpp b6000, Qwen2.5-1.5B). Full report with figures and limitations: **[REPORT.md](REPORT.md)**.

- Latency: at 20 ms RTT decode drops by 63 % / 75 % / 82 % with 1 / 2 / 3 nodes.
- Bandwidth: at 100 Mbit/s decode keeps 34 % of its speed, prefill 92 %.
- Quantisation: Q2_K decodes 2.1x faster than Q8_0 on a fast link, but all variants give 7-8 tokens/s on a 100 Mbit/s link.
- Planner: the profiling-driven split was 11 % faster (decode) than an even split on heterogeneous nodes.
