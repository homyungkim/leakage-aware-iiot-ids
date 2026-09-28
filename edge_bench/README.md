# Edge benchmark under emulated gateway budgets

Batch-one inference latency of the trained models in a resource-limited Docker container. The host is x86-64; only the core count and memory are limited, so the results are **emulated budgets, not measurements on ARM devices**. The paper labels them "X-class budget".

| Profile | CPUs | Memory (no swap) | Reference device |
|---|---|---|---|
| OrinNano-class | 6 | 4 GB | Jetson Orin Nano (CPU-only deployment) |
| RPi4-class | 4 | 2 GB | Raspberry Pi 4 Model B |
| Gateway-class | 2 | 2 GB | small industrial gateway |

Each model is measured with as many threads as the profile has cores and with one thread. Default: 100 warm-up calls, then 10,000 timed calls, repeated 5 times; reported are p50/p95/p99 latency with 95% confidence intervals, model size, load time, and peak memory (cgroup).

## Run (Windows, Docker Desktop)
1. Put the models into `edge_bench/models/`: `*.onnx`, `*_LightGBM.lgb.txt` (e.g. from `models/edge-iiotset/`), and, if you have run notebook 01, its `bench_inputs_*.npz`.
2. Plug in the power adapter, set the power mode to best performance, and close other programs.
3. From this folder:
```powershell
# quick check (a few minutes)
powershell -ExecutionPolicy Bypass -File .\run_profiles.ps1 -Iters 500 -Runs 2 -Cooldown 5
# full measurement
powershell -ExecutionPolicy Bypass -File .\run_profiles.ps1
```
Results are written to `edge_bench/results/` (`edge_<profile>.json`, `host.json`). Use `-Filter PALT` to measure selected models only.
On Linux, build the image with `docker build -t palt-edge .` and run `bench.py` with `docker run --cpus N --cpuset-cpus ... --memory ... --memory-swap ...` as in `run_profiles.ps1`.
