"""Recompute macro-F1 over a fixed class set from results.json files.

sklearn's average="macro" only includes classes that occur in y_true or y_pred, so a class with no
test vectors (Fingerprinting under the chronological split) enters some runs and not others. This
script averages the stored per-class F1 on distinct test vectors over one fixed set of classes:
every class with at least MIN_VECTORS distinct test vectors in every setting (14 classes for
Edge-IIoTset; Fingerprinting is reported separately).

Usage: python recompute_macro_f1.py results/edge-iiotset/*/results.json
Writes macro_f1_fixed_classes.csv next to this script and prints mean ± std per split/features/model.
"""
import json, sys, os
import pandas as pd

EXCLUDE = ["Fingerprinting"]      # < 10 distinct test vectors in at least one setting (see paper, Sec. VI-A)

rows = []
for path in sys.argv[1:]:
    run = os.path.basename(os.path.dirname(os.path.abspath(path)))
    for rec in json.load(open(path)):
        s, pc = rec["split"], rec["test_unique"]["per_class_f1"]
        cls = [c for c in pc if c not in EXCLUDE]
        rows.append(dict(run=run, split=s["mode"], features=s["feature_set"], weighted=bool(s.get("weighted", False)),
                         model=rec["model"], seed=s["seed"],
                         macro_f1=100 * sum(pc[c] for c in cls) / len(cls), n_classes=len(cls)))
df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "macro_f1_fixed_classes.csv")
df.to_csv(out, index=False)
print(df.groupby(["run", "split", "features", "weighted", "model"])["macro_f1"].agg(["mean", "std", "count"]).round(2).to_string())
print("saved", out)
