#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 0 - Edge-IIoTset Data Audit
=================================
Edge-IIoTset data audit (Phase 0) for the PALT study.

Purpose
-------
Produce verified facts about the dataset BEFORE any modelling decision:
  1. Shape, columns, dtypes, class counts (Attack_type / Attack_label)
  2. Protocol-group coverage (which protocol fields are really present, per class)
     -> decides the Protocol-Field Tokenization (PFT) groups, and checks e.g. CoAP
  3. Leakage audit: identifier / payload / timestamp columns, and a
     single-feature "purity" score that flags columns that alone reveal the label
  4. Duplicate audit: exact duplicates, duplicates after dropping leakage columns,
     and label conflicts (same features, different labels)
  5. Time-order audit (frame.time): can records be ordered in time? are classes
     captured in separate time blocks? -> decides whether temporal windows are valid

Usage
-----
  python 00_audit.py --csv /path/to/DNN-EdgeIIoT-dataset.csv --out results/audit
  (optional) --sample 0      : use all rows (default)
             --sample 500000 : random subsample for a quick run

All outputs go to --out:
  audit_summary.md        <- human-readable summary of the audit
  class_counts.csv
  column_profile.csv
  protocol_coverage.csv
  leakage_purity.csv
  duplicates.json
  time_audit.json / time_by_class.csv
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

LABEL_TYPE = "Attack_type"
LABEL_BIN = "Attack_label"

# Drop list used in the dataset authors' public preprocessing (Kaggle notebook).
# The audit only CHECKS these; the final drop list is decided from the outputs.
AUTHOR_DROP_LIST = [
    "frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4",
    "arp.dst.proto_ipv4", "http.file_data", "http.request.full_uri",
    "icmp.transmit_timestamp", "http.request.uri.query", "tcp.options",
    "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg",
]

# Protocol groups by column-name prefix (checked against the real columns).
PROTOCOL_PREFIXES = {
    "frame_ip": ["frame.", "ip."],
    "arp": ["arp."],
    "icmp": ["icmp."],
    "tcp": ["tcp."],
    "udp": ["udp."],
    "http": ["http."],
    "dns": ["dns."],
    "mqtt": ["mqtt."],
    "modbus_tcp": ["mbtcp.", "modbus."],
    "coap": ["coap."],
}

EMPTY_TOKENS = {"", "0", "0.0", "0.00", "nan", "NaN", "None", "-", "0x00000000"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def load(csv_path, sample):
    log(f"Loading {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False, dtype=str, keep_default_na=False)
    df.columns = [c.strip() for c in df.columns]
    log(f"Loaded shape={df.shape}")
    if sample and sample < len(df):
        df = df.sample(n=sample, random_state=42).reset_index(drop=False).rename(
            columns={"index": "_orig_row"})
        log(f"Subsampled to {len(df)} rows (random_state=42)")
    else:
        df["_orig_row"] = np.arange(len(df))
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()
    return df


def is_empty_series(s):
    return s.isin(EMPTY_TOKENS)


def column_profile(df, feat_cols):
    rows = []
    for c in feat_cols:
        s = df[c]
        num = pd.to_numeric(s, errors="coerce")
        numeric_ratio = float(num.notna().mean())
        nunique = int(s.nunique(dropna=False))
        vals = s.value_counts(dropna=False).head(3)
        rows.append({
            "column": c,
            "numeric_ratio": round(numeric_ratio, 6),
            "empty_or_zero_ratio": round(float(is_empty_series(s).mean()), 6),
            "nunique": nunique,
            "unique_ratio": round(nunique / max(len(s), 1), 6),
            "top3_values": " ; ".join(f"{str(k)[:40]}:{v}" for k, v in vals.items()),
            "in_author_drop_list": c in AUTHOR_DROP_LIST,
        })
    return pd.DataFrame(rows)


def protocol_coverage(df, feat_cols, classes):
    groups = {}
    for g, prefixes in PROTOCOL_PREFIXES.items():
        groups[g] = [c for c in feat_cols if any(c.startswith(p) for p in prefixes)]
    assigned = set(sum(groups.values(), []))
    groups["unassigned"] = [c for c in feat_cols if c not in assigned]

    rows = []
    for g, cols in groups.items():
        row = {"group": g, "n_columns": len(cols), "columns": ";".join(cols)}
        if cols:
            active = ~is_empty_series(df[cols]).all(axis=1)
            row["active_ratio_all"] = round(float(active.mean()), 6)
            for k in classes:
                m = df[LABEL_TYPE] == k
                row[f"active_{k}"] = round(float(active[m].mean()), 4) if m.any() else np.nan
        else:
            row["active_ratio_all"] = 0.0
        rows.append(row)
    return pd.DataFrame(rows), groups


def leakage_purity(df, feat_cols, max_card=200000):
    """Single-feature label purity: sum over values of the majority-label count / N.
    A column near 1.0 with many unique values is a strong identifier-leak suspect.
    Baseline = majority-class share (a constant column scores this)."""
    y = df[LABEL_TYPE]
    n = len(df)
    base = float(y.value_counts(normalize=True).iloc[0])
    rows = []
    for c in feat_cols:
        nun = df[c].nunique()
        if nun > max_card:
            rows.append({"column": c, "nunique": int(nun), "purity": np.nan,
                         "lift_over_majority": np.nan, "note": "skipped: cardinality"})
            continue
        ct = pd.crosstab(df[c], y)
        purity = float(ct.max(axis=1).sum() / n)
        rows.append({"column": c, "nunique": int(nun), "purity": round(purity, 6),
                     "lift_over_majority": round(purity - base, 6), "note": ""})
    out = pd.DataFrame(rows).sort_values("purity", ascending=False)
    return out, base


def duplicate_audit(df, feat_cols, drop_cols):
    res = {}
    res["n_rows"] = int(len(df))
    full = feat_cols + [LABEL_TYPE]
    res["exact_duplicate_rows(features+label)"] = int(df.duplicated(subset=full).sum())

    kept = [c for c in feat_cols if c not in drop_cols]
    res["n_features_after_author_drop"] = len(kept)
    dup_kept = df.duplicated(subset=kept + [LABEL_TYPE])
    res["duplicate_rows_after_author_drop"] = int(dup_kept.sum())
    res["duplicate_ratio_after_author_drop"] = round(float(dup_kept.mean()), 6)

    # label conflicts: same feature vector, >1 distinct label
    g = df.groupby(kept, sort=False, dropna=False)[LABEL_TYPE].nunique()
    conflict_keys = int((g > 1).sum())
    res["feature_vectors_with_label_conflict"] = conflict_keys
    res["unique_feature_vectors_after_drop"] = int(len(g))

    # per-class counts after de-duplication
    dedup = df.loc[~dup_kept]
    res["class_counts_after_dedup"] = {
        str(k): int(v) for k, v in dedup[LABEL_TYPE].value_counts().items()}
    return res


def time_audit(df):
    res = {"has_frame_time": "frame.time" in df.columns}
    if not res["has_frame_time"]:
        return res, None
    raw = df["frame.time"]
    t = pd.to_datetime(raw, errors="coerce", utc=False)
    if t.notna().mean() < 0.5:
        # Edge-IIoTset strings sometimes look like " 2021 11:44:10.081753000 "
        t = pd.to_datetime(raw.str.replace(r"\s+", " ", regex=True),
                           errors="coerce", format="mixed")
    res["parse_ratio"] = round(float(t.notna().mean()), 6)
    res["examples"] = raw.head(5).tolist()
    if res["parse_ratio"] < 0.5:
        res["note"] = "frame.time not parseable -> temporal windows NOT defensible"
        return res, None
    tt = t.copy()
    order = df["_orig_row"].values
    s = pd.Series(tt.values, index=order).sort_index()
    diffs = s.diff().dropna()
    res["file_order_monotonic_ratio"] = round(float((diffs >= pd.Timedelta(0)).mean()), 6)
    res["time_min"] = str(t.min())
    res["time_max"] = str(t.max())
    by = (pd.DataFrame({"t": t, "y": df[LABEL_TYPE]})
          .dropna().groupby("y")["t"].agg(["min", "max", "count"]))
    by["span_hours"] = (by["max"] - by["min"]).dt.total_seconds() / 3600
    # overlap of each class window with Normal window
    if "Normal" in by.index:
        nmin, nmax = by.loc["Normal", "min"], by.loc["Normal", "max"]
        by["overlaps_normal_period"] = (by["min"] <= nmax) & (by["max"] >= nmin)
    return res, by.reset_index()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="results/audit")
    ap.add_argument("--sample", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    df = load(args.csv, args.sample)
    for need in (LABEL_TYPE, LABEL_BIN):
        if need not in df.columns:
            sys.exit(f"Column '{need}' not found. Columns: {list(df.columns)}")

    feat_cols = [c for c in df.columns if c not in (LABEL_TYPE, LABEL_BIN, "_orig_row")]
    classes = df[LABEL_TYPE].value_counts().index.tolist()

    # 1. class counts
    cc = df[LABEL_TYPE].value_counts().rename_axis("Attack_type").reset_index(name="count")
    cc["ratio_%"] = (cc["count"] / cc["count"].sum() * 100).round(4)
    cc.to_csv(f"{args.out}/class_counts.csv", index=False)
    bin_counts = df[LABEL_BIN].value_counts().to_dict()
    log("Class counts done")

    # 2. column profile
    prof = column_profile(df, feat_cols)
    prof.to_csv(f"{args.out}/column_profile.csv", index=False)
    log("Column profile done")

    # 3. protocol coverage
    pc, groups = protocol_coverage(df, feat_cols, classes)
    pc.to_csv(f"{args.out}/protocol_coverage.csv", index=False)
    log("Protocol coverage done")

    # 4. leakage purity
    lp, base = leakage_purity(df, feat_cols)
    lp.to_csv(f"{args.out}/leakage_purity.csv", index=False)
    log("Leakage purity done")

    # 5. duplicates
    drop_present = [c for c in AUTHOR_DROP_LIST if c in df.columns]
    drop_missing = [c for c in AUTHOR_DROP_LIST if c not in df.columns]
    dup = duplicate_audit(df, feat_cols, drop_present)
    dup["author_drop_present"] = drop_present
    dup["author_drop_missing"] = drop_missing
    with open(f"{args.out}/duplicates.json", "w", encoding="utf-8") as f:
        json.dump(dup, f, indent=2, ensure_ascii=False)
    log("Duplicate audit done")

    # 6. time
    ta, tby = time_audit(df)
    with open(f"{args.out}/time_audit.json", "w", encoding="utf-8") as f:
        json.dump(ta, f, indent=2, ensure_ascii=False, default=str)
    if tby is not None:
        tby.to_csv(f"{args.out}/time_by_class.csv", index=False)
    log("Time audit done")

    # summary markdown
    L = []
    L.append("# Edge-IIoTset Audit Summary\n")
    L.append(f"- File: `{os.path.basename(args.csv)}`  |  rows used: {len(df):,}"
             f"  |  sample={'all' if not args.sample else args.sample}")
    L.append(f"- Columns: {len(df.columns) - 1} (features {len(feat_cols)} + 2 labels)")
    L.append(f"- Attack_label counts: {bin_counts}\n")
    L.append("## 1. Class counts (Attack_type)\n")
    L.append(cc.to_markdown(index=False))
    L.append("\n## 2. Protocol groups\n")
    L.append(pc[["group", "n_columns", "active_ratio_all", "columns"]].to_markdown(index=False))
    L.append(f"\n- CoAP columns present: **{len(groups.get('coap', [])) > 0}**")
    L.append("\n## 3. Leakage suspects (top 20 single-feature purity)\n")
    L.append(f"Majority-class baseline purity = {base:.4f}\n")
    L.append(lp.head(20).to_markdown(index=False))
    L.append("\n## 4. Duplicates\n")
    L.append("```json\n" + json.dumps(dup, indent=2, ensure_ascii=False) + "\n```")
    L.append("\n## 5. Time audit\n")
    L.append("```json\n" + json.dumps(ta, indent=2, ensure_ascii=False, default=str) + "\n```")
    if tby is not None:
        L.append(tby.to_markdown(index=False))
    L.append("\n## 6. Column profile (full)\n")
    L.append(prof.to_markdown(index=False))
    with open(f"{args.out}/audit_summary.md", "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    log(f"Done. Share: {args.out}/audit_summary.md")


if __name__ == "__main__":
    main()
