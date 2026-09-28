"""Check that a local copy of a dataset matches the one used in the paper (row count and per-class counts).

The datasets are not redistributed here. Download them from the original distributors (see DATA.md), then run
    python tools/verify_data.py edge-iiotset "path/to/DNN-EdgeIIoT-dataset.csv"
    python tools/verify_data.py xiiotid      "path/to/X-IIoTID dataset.csv"
    python tools/verify_data.py ciciot2023   path/to/merged.csv            # or a folder with the part-*.csv files
It also prints the SHA-256 of each file so that you can record the exact copy you used.
"""
import sys, os, glob, json, hashlib
from collections import Counter
import pandas as pd

name, path = sys.argv[1], sys.argv[2]
exp = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "expected_counts.json")))[name]
files = sorted(glob.glob(os.path.join(path, "*.csv"))) if os.path.isdir(path) else [path]

counts = Counter()
for f in files:
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    print(f"sha256  {h.hexdigest()}  {f}")
    for chunk in pd.read_csv(f, usecols=[exp["label"]], dtype=str, chunksize=1_000_000):
        counts.update(chunk[exp["label"]].str.strip())

rows = sum(counts.values())
ok = rows == exp["rows"] and dict(counts) == exp["classes"]
print(f"rows {rows:,} (expected {exp['rows']:,}); classes {len(counts)} (expected {len(exp['classes'])})")
for k in sorted(set(counts) | set(exp["classes"])):
    if counts.get(k, 0) != exp["classes"].get(k, 0):
        print(f"  mismatch {k}: {counts.get(k, 0)} vs expected {exp['classes'].get(k, 0)}")
print("MATCH" if ok else "DIFFERENT COPY: results may not reproduce exactly")
