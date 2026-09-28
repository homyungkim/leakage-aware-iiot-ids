#!/usr/bin/env python3
"""Edge-profile latency benchmark (runs inside a resource-limited Docker container).

For every model in /models (*.onnx, and LightGBM *.lgb.txt) it measures batch-1 latency
with the thread count set to the container's CPU budget, repeated R runs x N iterations
after a warm-up, plus model load time, model size and peak memory.

Inputs: /models/bench_inputs.npz (X, C, P arrays of real test records, exported by the
Kaggle notebook). If absent, synthetic inputs with the right shapes are used and the
result is flagged "synthetic_inputs".

Usage (inside the container):
  python bench.py --profile RPi4-class --threads 4 --iters 10000 --runs 5 --out /results
"""
import argparse, glob, json, os, platform, resource, time
import numpy as np
import onnxruntime as ort


def cgroup_value(name):
    for p in (f"/sys/fs/cgroup/{name}",):
        try:
            return open(p).read().strip()
        except OSError:
            pass
    return None


def cpu_model():
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0  # Linux: KB -> MB


def load_inputs(model_dir, model_file):
    """Real test records exported by the Kaggle notebook: bench_inputs_<tag>.npz where the model
    file is <tag>_<model>.onnx; falls back to bench_inputs.npz; None -> synthetic."""
    base = os.path.basename(model_file)
    for f in sorted(glob.glob(os.path.join(model_dir, "bench_inputs_*.npz")), key=len, reverse=True):
        tag = os.path.basename(f)[len("bench_inputs_"):-4]
        if base.startswith(tag + "_"):
            z = np.load(f); return {k: z[k] for k in z.files}, os.path.basename(f)
    f = os.path.join(model_dir, "bench_inputs.npz")
    if os.path.exists(f):
        z = np.load(f); return {k: z[k] for k in z.files}, os.path.basename(f)
    return None, None


def synth_for(sess, n):
    rng = np.random.default_rng(0); out = {}
    for i in sess.get_inputs():
        shape = [n if (d is None or isinstance(d, str)) else d for d in i.shape]
        if "int64" in i.type:
            out[i.name] = np.zeros(shape, np.int64)
        elif i.name == "P":
            out[i.name] = (rng.random(shape) > 0.5).astype(np.float32)
        else:
            out[i.name] = rng.standard_normal(shape).astype(np.float32)
    return out


def stats(lat):
    lat = np.asarray(lat)
    return dict(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                p99=float(np.percentile(lat, 99)), mean=float(lat.mean()), std=float(lat.std()))


def bench_callable(fn, rows, iters, runs, warmup):
    n = len(rows)
    for i in range(warmup):
        fn(rows[i % n])
    per_run = []
    for r in range(runs):
        lat = []
        for i in range(iters):
            x = rows[i % n]
            t = time.perf_counter(); fn(x); lat.append((time.perf_counter() - t) * 1e3)
        per_run.append(stats(lat))
    p99s = np.array([s["p99"] for s in per_run]); p50s = np.array([s["p50"] for s in per_run])
    ci = lambda a: float(1.96 * a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else 0.0
    return dict(per_run=per_run, p50_mean=float(p50s.mean()), p50_ci95=ci(p50s),
                p99_mean=float(p99s.mean()), p99_ci95=ci(p99s),
                throughput_rec_per_s=float(1e3 / np.mean([s["mean"] for s in per_run])))


def _ver(pkg):
    try:
        return __import__(pkg).__version__
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--threads", required=True, help="comma list, e.g. 4,1 (first = profile budget)")
    ap.add_argument("--iters", type=int, default=10000)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--models", default="/models")
    ap.add_argument("--out", default="/results")
    ap.add_argument("--filter", default="", help="substring; only models whose file name contains it")
    ap.add_argument("--exclude", default="", help="substring; skip models whose file name contains it")
    ap.add_argument("--suffix", default="", help="appended to the output file name, e.g. _rf")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    thread_list = [int(t) for t in str(a.threads).split(",")]
    env = dict(profile=a.profile, threads=thread_list, iters=a.iters, runs=a.runs, warmup=a.warmup,
               cpu_model=cpu_model(), visible_cpus=os.cpu_count(),
               affinity=sorted(os.sched_getaffinity(0)), cgroup_cpu_max=cgroup_value("cpu.max"),
               cgroup_memory_max=cgroup_value("memory.max"), onnxruntime=ort.__version__,
               numpy=np.__version__, lightgbm=_ver("lightgbm"), xgboost=_ver("xgboost"), python=platform.python_version(), kernel=platform.release())
    results = dict(environment=env, models={})

    files = sorted(glob.glob(os.path.join(a.models, "*.onnx"))) + sorted(glob.glob(os.path.join(a.models, "*.lgb.txt"))) \
        + sorted(glob.glob(os.path.join(a.models, "*.xgb.json")))
    files = [f for f in files if a.filter in os.path.basename(f) and not (a.exclude and a.exclude in os.path.basename(f))]
    for f, th in [(f, t) for f in files for t in thread_list]:
        a.threads = th
        name = f"{os.path.basename(f)}@{th}t"; rec = dict(size_kb=os.path.getsize(f) / 1024, threads=th)
        data, src_name = load_inputs(a.models, f)
        rec["inputs"] = src_name or "synthetic"
        try:
            if f.endswith(".onnx"):
                so = ort.SessionOptions(); so.intra_op_num_threads = a.threads; so.inter_op_num_threads = 1
                so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
                t = time.perf_counter()
                s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
                rec["load_ms"] = (time.perf_counter() - t) * 1e3
                names = [i.name for i in s.get_inputs()]
                if names == ["input"]:   # tree ensemble exported as ONNX: one float matrix [numeric fields | categorical codes]
                    if data is None: raise RuntimeError("tree models need bench_inputs.npz")
                    XC = np.hstack([data["X"], data["C"].astype(np.float32)]).astype(np.float32) if "C" in data else data["X"]
                    src = {"input": XC}
                else:
                    src = data if data is not None else synth_for(s, 1000)
                n = len(next(iter(src.values())))
                rows = [{k: src[k][j:j + 1] for k in names} for j in range(min(n, 2000))]
                rec.update(bench_callable(lambda x: s.run(None, x), rows, a.iters, a.runs, a.warmup))
                del s
            elif f.endswith(".xgb.json"):  # XGBoost native predictor, same thread budget
                import xgboost as xgb
                t = time.perf_counter(); b = xgb.Booster(); b.load_model(f); b.set_param({"nthread": a.threads})
                rec["load_ms"] = (time.perf_counter() - t) * 1e3
                if data is None:
                    raise RuntimeError("XGBoost needs bench_inputs.npz")
                XC = np.hstack([data["X"], data["C"].astype(np.float32)]).astype(np.float32) if "C" in data else data["X"]
                rows = [XC[j:j + 1] for j in range(min(len(XC), 2000))]
                rec.update(bench_callable(lambda x: b.inplace_predict(x), rows, a.iters, a.runs, a.warmup))
            else:  # LightGBM text model; native predictor, same thread budget
                import lightgbm as lgb
                t = time.perf_counter(); b = lgb.Booster(model_file=f); rec["load_ms"] = (time.perf_counter() - t) * 1e3
                if data is None:
                    raise RuntimeError("LightGBM needs bench_inputs.npz")
                XC = np.hstack([data["X"], data["C"].astype(np.float32)]) if "C" in data else data["X"]
                rows = [XC[j:j + 1] for j in range(min(len(XC), 2000))]
                rec.update(bench_callable(lambda x: b.predict(x, num_threads=a.threads), rows, a.iters, a.runs, a.warmup))
        except Exception as e:
            rec["error"] = repr(e)
        rec["process_peak_rss_mb"] = peak_rss_mb()
        rec["cgroup_memory_peak_mb"] = (int(cgroup_value("memory.peak")) / 2**20) if (cgroup_value("memory.peak") or "").isdigit() else None
        results["models"][name] = rec
        msg = rec.get("error") or f"p50={rec['p50_mean']:.3f}ms p99={rec['p99_mean']:.3f}±{rec['p99_ci95']:.3f}ms"
        print(f"[{a.profile}] {name:45s} {msg}", flush=True)

    out = os.path.join(a.out, f"edge_{a.profile}{a.suffix}.json")
    with open(out, "w") as fh:
        json.dump(results, fh, indent=2)
    print("saved", out)


if __name__ == "__main__":
    main()
