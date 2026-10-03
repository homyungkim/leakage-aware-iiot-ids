"""Paired Wilcoxon signed-rank tests on macro-F1 (fixed class set), pooling the seeds of both splits.

Usage: python paired_tests.py macro_f1_fixed_classes.csv
With five seeds per split, the smallest two-sided p-value of a per-split test is 0.0625, so runs of
the random and chronological splits are paired by (split, seed) and pooled (n = 10).
The runs "main", "robust_b", and "robust_c" use identical split indices and seeds (Tables III, VI),
so their models can be paired directly. Holm's step-down correction is applied within each family.
"""
import sys
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv(sys.argv[1])
df = df[(df.features == "strict") & (df.run.isin(["main", "robust_b", "robust_c"])) & (~df.weighted.astype(bool))]
wide = df.pivot_table(index=["split", "seed"], columns="model", values="macro_f1")

FAMILIES = {
    "vs LightGBM (Table III)": [("XGBoost", "LightGBM"), ("RandomForest", "LightGBM"), ("CatBoost", "LightGBM"),
                               ("MLP-PLR", "LightGBM"), ("PALT", "LightGBM")],
    # PALT-Teacher and PALT-KD are auxiliary runs of notebook 01; they stay in this pre-specified family.
    "vs PALT (Table III)": [("PALT", "FTTransformer"), ("PALT", "MLP"), ("PALT", "CNN1D"), ("PALT-Teacher", "PALT"),
                           ("PALT-KD", "PALT"), ("MLP-PLR", "PALT"), ("LightGBM", "PALT")],
    "value resolution (Sec. V-D)": [("MLP-PLR", "MLP"), ("MLP-PLR", "FTTransformer"), ("PALT-PLR", "PALT")],
    "PALT controls and variants (Table VI)": [("PALT-RandGroup", "PALT"), ("PALT-NoLocal", "PALT"),
                                                ("PALT-FullAttn", "PALT"), ("PALT-ZeroAbsent", "PALT"),
                                                ("PALT-PLR", "PALT"), ("PALT-PLR-RandGroup", "PALT"),
                                                ("PALT-PLR", "PALT-PLR-RandGroup"), ("PALT-PLR", "MLP-PLR"),
                                                ("PALT-PLR", "LightGBM"), ("PALT-PLR", "FTTransformer")],
}

def holm(p):
    order = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * p[i])); adj[i] = run
    return adj

for fam, pairs in FAMILIES.items():
    print(f"\n== {fam}")
    rows = []
    for a, b in pairs:
        d = (wide[a] - wide[b]).dropna()
        by = d.groupby(level="split").mean().round(2).to_dict()
        rows.append((a, b, d.mean(), int((d > 0).sum()), len(d), wilcoxon(d).pvalue, by))
    adj = holm(np.array([r[5] for r in rows]))
    for (a, b, m, w, n, p, by), q in zip(rows, adj):
        print(f"{a:>18} - {b:<18} mean {m:+6.2f}  wins {w}/{n}  p = {p:.4f}  Holm p = {q:.4f}  per split {by}")
