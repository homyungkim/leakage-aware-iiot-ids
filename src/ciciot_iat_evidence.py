# %% [markdown]
# # CICIoT2023 — does the IAT feature record capture time? (Kaggle, CPU is enough)
# Code for "Beyond Shortcuts: A Leakage-Aware Audit and Benchmark of Intrusion Detectors for Smart-Factory IIoT Gateways".
#
# The shortcut audit flagged `IAT`: alone it reaches 59.2% macro-F1 over 34 classes, and removing it costs LightGBM
# 13.8 points (neural detectors < 1). This notebook collects evidence on *why*, over all 46.7 M rows:
# 1. per-class range, quantiles and number of distinct values of IAT (and of two control features);
# 2. rank correlation of IAT with the row position in the file, per class and overall (a clock-like field rises with it);
# 3. separability of same-protocol DoS/DDoS pairs by IAT alone (AUC) versus by the control features;
# 4. a plot of IAT against row position.
#
# **How to run:** open the CICIoT2023 dataset page used for the benchmark (`akashdogra/ciciot23csv`, one merged CSV)
# → Code → New Notebook → File → Import Notebook; Accelerator None (CPU); Save Version → Save & Run All.
# Download `ciciot_iat_evidence.zip` from the Output tab. Run time ≈ 30–45 min.

# %%
import os, glob, json, time, zipfile
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, mannwhitneyu

OUT = "/kaggle/working/iat_evidence" if os.path.isdir("/kaggle/working") else "iat_evidence"
os.makedirs(OUT, exist_ok=True)
CSV = os.environ.get("CIC_CSV")
if CSV is None:
    cands = [f for f in glob.glob("/kaggle/input/**/*.csv", recursive=True)]
    cands = [f for f in cands if {"IAT", "label"} <= set(pd.read_csv(f, nrows=2).columns.str.strip())]
    if not cands: raise FileNotFoundError("CICIoT2023 CSV with IAT and label columns not found under /kaggle/input")
    CSV = max(cands, key=os.path.getsize)
print("file:", CSV)
FEATS = ["IAT", "flow_duration", "Tot size"]          # IAT + two controls (flow timing, packet size)
CHUNK = 2_000_000

# %% pass 1: class counts
t0 = time.time(); counts = {}
for ch in pd.read_csv(CSV, usecols=["label"], chunksize=CHUNK):
    for k, v in ch["label"].str.strip().value_counts().items(): counts[k] = counts.get(k, 0) + int(v)
n_all = sum(counts.values()); print(f"{n_all:,} rows, {len(counts)} classes ({time.time()-t0:.0f}s)")

# %% pass 2: exact per-class min/max, sample of up to 60,000 rows per class (row position kept)
KEEP = 60_000
rng = np.random.default_rng(0)
prob = {k: min(1.0, KEEP / v) for k, v in counts.items()}
stats = {k: {f: [np.inf, -np.inf] for f in FEATS} for k in counts}
samples = []; row0 = 0; t0 = time.time()
for ch in pd.read_csv(CSV, usecols=FEATS + ["label"], chunksize=CHUNK):
    ch.columns = ch.columns.str.strip(); ch["label"] = ch["label"].str.strip()
    ch["row"] = np.arange(row0, row0 + len(ch)); row0 += len(ch)
    for k, g in ch.groupby("label"):
        for f in FEATS:
            s = stats[k][f]; s[0] = min(s[0], float(g[f].min())); s[1] = max(s[1], float(g[f].max()))
    keep = rng.random(len(ch)) < ch["label"].map(prob).values
    samples.append(ch.loc[keep, FEATS + ["label", "row"]])
    print(f"  {row0:,} rows ({time.time()-t0:.0f}s)")
S = pd.concat(samples, ignore_index=True); del samples
S.to_csv(os.path.join(OUT, "iat_sample.csv.gz"), index=False, compression="gzip")

# %% evidence
res = {"file": os.path.basename(CSV), "rows": n_all, "class_counts": counts, "sample_rows": int(len(S))}
per = {}
for k, g in S.groupby("label"):
    d = {"n_sample": int(len(g))}
    for f in FEATS:
        q = np.percentile(g[f], [1, 25, 50, 75, 99])
        rho = spearmanr(g["row"], g[f]).correlation if g[f].nunique() > 1 else float("nan")
        d[f] = dict(min=stats[k][f][0], max=stats[k][f][1], q01=q[0], q25=q[1], median=q[2], q75=q[3], q99=q[4],
                    distinct_share=float(g[f].nunique() / len(g)), spearman_vs_row=float(rho))
    per[k] = d
res["per_class"] = per
res["overall_spearman_vs_row"] = {f: float(spearmanr(S["row"], S[f]).correlation) for f in FEATS}
res["median_abs_spearman_vs_row_within_class"] = {f: float(np.nanmedian([abs(per[k][f]["spearman_vs_row"]) for k in per])) for f in FEATS}
pairs = [("DoS-SYN_Flood", "DDoS-SYN_Flood"), ("DoS-TCP_Flood", "DDoS-TCP_Flood"), ("DoS-UDP_Flood", "DDoS-UDP_Flood"),
         ("DoS-HTTP_Flood", "DDoS-HTTP_Flood"), ("Recon-OSScan", "Recon-PortScan"), ("BenignTraffic", "Recon-PortScan")]
auc = {}
for a, b in pairs:
    if a in per and b in per:
        xa, xb = S.loc[S["label"] == a], S.loc[S["label"] == b]
        auc[f"{a} vs {b}"] = {f: float(max(u := mannwhitneyu(xa[f], xb[f]).statistic / (len(xa) * len(xb)), 1 - u)) for f in FEATS}
res["pair_auc_single_feature"] = auc
json.dump(res, open(os.path.join(OUT, "iat_evidence.json"), "w"), indent=1, default=float)

# %% plots
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sub = S.sample(min(200_000, len(S)), random_state=0)
for ax, f in zip(axes, ["IAT", "flow_duration"]):
    ax.scatter(sub["row"], sub[f], s=0.2, alpha=0.3, c=pd.factorize(sub["label"])[0], cmap="tab20")
    ax.set_xlabel("row position in file"); ax.set_ylabel(f); ax.set_yscale("symlog"); ax.set_title(f"{f} vs row position")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "iat_vs_row.png"), dpi=150)
fig, ax = plt.subplots(figsize=(8, 9))
order = sorted(per, key=lambda k: per[k]["IAT"]["median"])
for i, k in enumerate(order):
    d = per[k]["IAT"]; ax.plot([d["q01"], d["q99"]], [i, i], color="#bbbbbb"); ax.plot([d["q25"], d["q75"]], [i, i], lw=4, color="#2a78d6")
ax.set_yticks(range(len(order))); ax.set_yticklabels(order, fontsize=7); ax.set_xscale("symlog"); ax.set_xlabel("IAT (1-99% and 25-75% ranges)")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "iat_ranges_by_class.png"), dpi=150)

print(json.dumps({k: res[k] for k in ["overall_spearman_vs_row", "median_abs_spearman_vs_row_within_class", "pair_auc_single_feature"]}, indent=1))
zp = os.path.join(os.path.dirname(OUT), "ciciot_iat_evidence.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(os.path.join(OUT, "*")): z.write(f, os.path.basename(f))
print("Download:", zp)
