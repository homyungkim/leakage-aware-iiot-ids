# %% [markdown]
# # PALT v5 robustness experiments — Edge-IIoTset (Kaggle GPU)
# Code for "Beyond Shortcuts: Leakage-Aware Evaluation and Protocol-Aware Lightweight Intrusion Detection for Smart-Factory IIoT Gateways".
# Reviewer-driven checks on top of the v4 benchmark (same preprocessing, same grouped splits and seeds).
#
# **Set `STAGE` in the first code cell before running:**
# * `"robust_a"` (≈3–4 h) — (1) **session-grouped split**: all packets of a session (bidirectional address/port pair,
#   UDP stream, or host pair) stay in one set; scored on all distinct test vectors and on test vectors never seen in
#   training; LightGBM, PALT, FT-Transformer, MLP × 5 seeds. (2) **frequency-weighted training** (log(1+multiplicity))
#   on the grouped random split; LightGBM and PALT × 3 seeds. (3) **tcp.seq / tcp.ack removed** (strict-noseq);
#   grouped random split; 4 models × 3 seeds.
# * `"robust_b"` (≈3–4 h) — (4) **more baselines** under the v4 splits (random + chronological, 5 seeds):
#   XGBoost, CatBoost, random forest, MLP with periodic numerical embeddings (MLP-PLR).
#   (5) **PALT controls**: random assignment of fields to the eight tokens (same group sizes), no local path,
#   full attention in every block, zero vector instead of the learned absent token.
# * `"robust_c"` (≈1.5–2 h) — (6) **PALT-PLR**: PALT whose protocol tokens are built from periodic numerical
#   embeddings of each field (as in MLP-PLR), plus the same model with random field-to-token assignment;
#   v4 splits (random + chronological), 5 seeds.
# * `"robust_d"` (≈2–3 h) — (7) **deployment export** for the edge benchmark: LightGBM, XGBoost, random forest,
#   PALT, MLP-PLR, PALT-PLR trained on the grouped random split (seed 0, same split as v4) and exported
#   (ONNX FP32, XGBoost JSON, LightGBM text; random forest via skl2onnx) with 2,000 real test inputs.
#   (8) **tcp.seq / tcp.ack removed** (strict-noseq) for XGBoost, random forest, MLP-PLR, PALT-PLR; 3 seeds.
#   Two zips: `palt_v5_robust_d_results.zip` (logs) and `palt_v5_robust_d_models.zip` (models for Docker).
# * `"robust_e"` (≈6–7 h) — reviewer checks. (9) **session-grouped split** for random forest, XGBoost, MLP-PLR,
#   PALT-PLR (5 seeds). (10) **tcp.seq / tcp.ack interventions that keep the split fixed** (grouped random split,
#   3 seeds; LightGBM, XGBoost, random forest, MLP, MLP-PLR, PALT, PALT-PLR): (a) the two fields permuted across the
#   test vectors of each model trained normally; (b) the two fields coarsened to 16 quantile bins (fit on training)
#   before training and testing. (11) **vector-level oracle**: majority label of each test vector, scored with the
#   same fixed-class macro-F1 on distinct test vectors.
#   `"robust_e_cpu"` (Accelerator None, ≈4 h) runs the tree ensembles and the oracle; `"robust_e_gpu"` (GPU, ≈3.5 h)
#   runs MLP, MLP-PLR, PALT, PALT-PLR. Together they equal `"robust_e"`.
#
# **How to run:** Add Input → "Edge-IIoTset Cyber Security Dataset of IoT & IIoT"; GPU T4 x2; Internet ON;
# Save Version → Save & Run All. Download `palt_v5_<stage>_results.zip` from the Output tab.

# %%
import subprocess, sys, importlib.util
for pkg in ["onnxruntime", "onnx", "lightgbm", "xgboost", "catboost", "skl2onnx", "onnxmltools"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import os, gc, glob, json, math, time, random, platform, warnings, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, matthews_corrcoef, balanced_accuracy_score,
                             confusion_matrix, precision_recall_fscore_support)
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")

CFG = dict(
    data_csv=None,                 # auto-detected below; set a path to override
    out_dir="/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results",
    max_rows=None,                 # e.g. 300_000 for a quick dry run
    # (split mode, feature set) runs. split: random = stratified after de-dup; temporal = per-class chronological.
    # feature set: author = drop list shipped with the dataset; strict = author + identifier-like fields
    # raw-author = author drop list on the raw strings (reproduces common practice, keeps format shortcuts);
    # author/strict = canonicalized values ("0" == "0.0" == "0x00000000"), see Section 2.
    epochs=40, batch_size=2048, lr=1e-3, lr_teacher=5e-4, weight_decay=1e-4, patience=8,
    clip=5.0,                      # standardized inputs clipped to [-clip, clip] (outliers up to 31 sigma broke INT8)
    d_student=32, d_teacher=128, heads=4, blocks_student=2, blocks_teacher=4,
    kd_T=4.0, kd_alpha=0.5, cb_beta=0.9999, focal_gamma=2.0,
    cat_vocab=64, latency_iters=2000, latency_threads=[1, 2],
    models=["LightGBM", "MLP", "CNN1D", "FTTransformer", "PALT-Teacher", "PALT", "PALT-KD"],
)
STAGE = "robust_a"   # "robust_a", "robust_b", "robust_c", "robust_d" or "robust_e"  <-- set before running
S5, S3 = [0, 1, 2, 3, 4], [0, 1, 2]
BASE4 = ["LightGBM", "MLP", "FTTransformer", "PALT"]
if STAGE == "robust_a":
    CFG["runs"] = [dict(mode="session", fset="strict", seeds=S5, models=BASE4),
                   dict(mode="random", fset="strict", seeds=S3, models=["LightGBM", "PALT"], weighted=True),
                   dict(mode="random", fset="strict-noseq", seeds=S3, models=BASE4)]
elif STAGE == "robust_c":
    PLRM = ["PALT-PLR", "PALT-PLR-RandGroup"]
    CFG["runs"] = [dict(mode="random", fset="strict", seeds=S5, models=PLRM),
                   dict(mode="temporal", fset="strict", seeds=S5, models=PLRM)]
elif STAGE == "robust_d":
    CFG["runs"] = [dict(mode="random", fset="strict", seeds=[0], export=True,
                        models=["LightGBM", "XGBoost", "RandomForest", "PALT", "MLP-PLR", "PALT-PLR"]),
                   dict(mode="random", fset="strict-noseq", seeds=S3, models=["XGBoost", "RandomForest", "MLP-PLR", "PALT-PLR"])]
elif STAGE in ("robust_e", "robust_e_cpu", "robust_e_gpu"):
    # robust_e_cpu: tree ensembles and the oracle only (runs without a GPU); robust_e_gpu: the neural detectors
    TREE3 = ["LightGBM", "XGBoost", "RandomForest"]; NN4 = ["MLP", "MLP-PLR", "PALT", "PALT-PLR"]
    SESS = {"robust_e": ["XGBoost", "RandomForest", "MLP-PLR", "PALT-PLR"], "robust_e_cpu": ["XGBoost", "RandomForest"],
            "robust_e_gpu": ["MLP-PLR", "PALT-PLR"]}[STAGE]
    SEQ = {"robust_e": TREE3 + NN4, "robust_e_cpu": TREE3, "robust_e_gpu": NN4}[STAGE]
    CFG["runs"] = [dict(mode="session", fset="strict", seeds=S5, models=SESS),
                   dict(mode="random", fset="strict", seeds=S3, models=SEQ, permute_seq=True, oracle=(STAGE != "robust_e_gpu")),
                   dict(mode="random", fset="strict", seeds=S3, models=SEQ, coarse=16)]
else:
    NEW = ["XGBoost", "CatBoost", "RandomForest", "MLP-PLR", "PALT-RandGroup", "PALT-NoLocal", "PALT-FullAttn", "PALT-ZeroAbsent"]
    CFG["runs"] = [dict(mode="random", fset="strict", seeds=S5, models=NEW),
                   dict(mode="temporal", fset="strict", seeds=S5, models=NEW)]
CFG["seeds"] = S5
CFG["export_seed"] = CFG["seeds"][0]
CFG.update(json.loads(os.environ.get("PALT_CFG", "{}")))  # local override for smoke tests
os.makedirs(CFG["out_dir"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else platform.processor())
print("torch", torch.__version__, "| pandas", pd.__version__, "| python", sys.version.split()[0])

if CFG["data_csv"] is None:
    cands = glob.glob("/kaggle/input/**/DNN-EdgeIIoT-dataset.csv", recursive=True) + \
            glob.glob("**/DNN-EdgeIIoT-dataset.csv", recursive=True) + \
            glob.glob("**/ML-EdgeIIoT-dataset.csv", recursive=True)
    if not cands:
        seen = [os.path.join(r, d) for r, ds, _ in os.walk("/kaggle/input") for d in ds][:30] if os.path.isdir("/kaggle/input") else []
        raise FileNotFoundError("Edge-IIoTset CSV not found. In the right panel: Add Input -> search "
                                "'Edge-IIoTset Cyber Security Dataset of IoT & IIoT' -> Add. /kaggle/input contains: " + str(seen))
    CFG["data_csv"] = cands[0]
print("data:", CFG["data_csv"])

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_json(obj, name):
    with open(os.path.join(CFG["out_dir"], name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=str)

# %% [markdown]
# ## 1. Load and audit

# %%
LABEL, BIN = "Attack_type", "Attack_label"
# drop list published with the dataset (Readme / Kaggle notebook)
DROP = ["frame.time", "ip.src_host", "ip.dst_host", "arp.src.proto_ipv4", "arp.dst.proto_ipv4",
        "http.file_data", "http.request.full_uri", "icmp.transmit_timestamp", "http.request.uri.query",
        "tcp.options", "tcp.payload", "tcp.srcport", "tcp.dstport", "udp.port", "mqtt.msg"]
# protocol groups, ordered by stack layer (L2/L3 -> L4 -> L7)
GROUPS = [("arp", ["arp."]), ("icmp", ["icmp."]), ("tcp", ["tcp."]), ("udp", ["udp."]),
          ("http", ["http."]), ("dns", ["dns."]), ("mqtt", ["mqtt."]), ("modbus", ["mbtcp.", "modbus."])]
# identifier-like fields flagged by the purity screen (raw sequence numbers, checksums, stream indices)
STRICT_EXTRA = ["tcp.ack_raw", "tcp.checksum", "icmp.checksum", "udp.stream"]
LAYER_WINDOWS = [["arp", "icmp"], ["tcp", "udp"], ["http", "dns", "mqtt", "modbus"]]
EMPTY = {"", "0", "0.0", "0.00", "nan", "NaN", "None", "-", "0x00000000"}

t0 = time.time()
raw = pd.read_csv(CFG["data_csv"], dtype=str, keep_default_na=False, low_memory=False)
raw.columns = [c.strip() for c in raw.columns]
for c in raw.columns:
    raw[c] = raw[c].str.strip()
if CFG["max_rows"] and len(raw) > CFG["max_rows"]:
    raw = raw.sample(n=CFG["max_rows"], random_state=0)
raw["_row"] = np.arange(len(raw))
# malformed timestamp field: frame.time should look like "<year> HH:MM:SS.fraction".
# In the ML file every MITM and DDoS_UDP record fails this check (e.g. "6.0" or an IP address),
# so the field format alone identifies those classes. Rows are kept (frame.time is dropped anyway);
# the counts are reported, and the temporal split falls back to file order for such classes.
ok_time = raw["frame.time"].str.match(r"^\d{4} \d{2}:\d{2}:\d{2}(\.\d+)?$") if "frame.time" in raw else pd.Series(True, index=raw.index)
n_bad = int((~ok_time).sum())
bad_counts = raw.loc[~ok_time, LABEL].value_counts().to_dict()
print(f"records with malformed frame.time: {n_bad} {bad_counts}")
print(f"loaded {raw.shape} in {time.time()-t0:.0f}s")

audit = {"file": os.path.basename(CFG["data_csv"]), "rows": len(raw),
         "malformed_frame_time": n_bad, "malformed_frame_time_by_class": bad_counts, "columns": len(raw.columns) - 1}
audit["class_counts"] = raw[LABEL].value_counts().to_dict()
audit["drop_present"] = [c for c in DROP if c in raw.columns]
audit["drop_missing"] = [c for c in DROP if c not in raw.columns]
feat_all = [c for c in raw.columns if c not in (LABEL, BIN, "_row")]
kept = [c for c in feat_all if c not in DROP]
audit["protocol_columns"] = {g: [c for c in kept if any(c.startswith(p) for p in ps)] for g, ps in GROUPS}
audit["unassigned_columns"] = [c for c in kept if not any(c in v for v in audit["protocol_columns"].values())]
audit["coap_columns"] = [c for c in raw.columns if c.lower().startswith("coap")]
def row_hash(frame):  # 64-bit row hash; memory-safe duplicate detection on 2M+ rows
    return pd.util.hash_pandas_object(frame, index=False).values
h_all = row_hash(raw[feat_all + [LABEL]]); h_feat = row_hash(raw[kept]); h_lab = row_hash(raw[kept + [LABEL]])
audit["exact_duplicates_all_columns"] = int(pd.Series(h_all).duplicated().sum())
dup_kept = pd.Series(h_lab, index=raw.index).duplicated()
audit["duplicates_after_drop"] = int(dup_kept.sum())
g = pd.DataFrame({"h": h_feat, "y": raw[LABEL].values}).drop_duplicates().groupby("h")["y"].nunique()
audit["feature_vectors_with_conflicting_labels"] = int((g > 1).sum())
conf = set(g.index[g > 1])
audit["records_in_conflicting_vectors_by_class"] = raw.loc[pd.Series(h_feat).isin(conf).values, LABEL].value_counts().to_dict()
del g, h_all, h_feat, h_lab
audit["class_counts_after_dedup"] = raw.loc[~dup_kept, LABEL].value_counts().to_dict()
empty_all = raw[kept].isin(EMPTY).all(axis=1)
audit["records_with_no_retained_field_by_class"] = raw.loc[empty_all, LABEL].value_counts().to_dict()

# single-feature label purity (leakage screen)
base = raw[LABEL].value_counts(normalize=True).iloc[0]
pur = {}
for c in kept:
    if raw[c].nunique() <= 200_000:
        ct = pd.crosstab(raw[c], raw[LABEL]); pur[c] = float(ct.max(axis=1).sum() / len(raw))
audit["majority_baseline_purity"] = float(base)
audit["top_purity_columns"] = dict(sorted(pur.items(), key=lambda kv: -kv[1])[:15])

# time audit
# frame.time has a year and a time of day but no month/day, so ordering is by time-of-day stamp only
ts = pd.to_datetime(raw["frame.time"].where(ok_time), errors="coerce", format="%Y %H:%M:%S.%f") if "frame.time" in raw else pd.Series(pd.NaT, index=raw.index)
audit["frame_time_parse_ratio"] = float(ts.notna().mean())
audit["frame_time_examples"] = raw["frame.time"].head(5).tolist() if "frame.time" in raw else []
# format shortcut screen: how "empty" is written ("0", "0.0", "0x00000000") per class.
# Different capture files were exported with different writers, so the spelling of an empty field can reveal the class.
fmt = {}
for c in kept:
    m = raw[c].isin(EMPTY)
    ct = pd.crosstab(raw.loc[m, c], raw.loc[m, LABEL])
    if len(ct) > 1:
        fmt[c] = {tok: {k: int(v) for k, v in row.items() if v > 0} for tok, row in ct.iterrows()}
audit["empty_token_spelling_by_class"] = fmt
save_json(audit, "audit.json")
print("columns whose empty-value spelling differs by class:", list(fmt))
print(json.dumps({k: audit[k] for k in ["rows", "malformed_frame_time_by_class", "records_with_no_retained_field_by_class", "class_counts", "duplicates_after_drop",
      "feature_vectors_with_conflicting_labels", "coap_columns", "frame_time_parse_ratio"]}, indent=1, default=str))

# %% [markdown]
# ## 2. Leakage-aware preprocessing
# De-duplicate on the retained columns **before** splitting; fit all encoders on the training split only.

# %%
# session key for the session-grouped split (robust_a): computed from the address/port fields before they are dropped.
# TCP: bidirectional (address:port, address:port) pair; UDP: stream index; otherwise: bidirectional host pair.
# The class label is part of the key, so a session never spans two classes; stream indices restart per capture,
# which can only merge sessions (coarser groups), never split one.
def _clean(s): return s.where(~s.isin(EMPTY), "")
_src, _dst = _clean(raw["ip.src_host"]), _clean(raw["ip.dst_host"])
_sp, _dp = _clean(raw["tcp.srcport"]), _clean(raw["tcp.dstport"])
_us = _clean(raw["udp.stream"]) if "udp.stream" in raw else pd.Series("", index=raw.index)
_a = _src + ":" + _sp; _b = _dst + ":" + _dp
_lo = np.where(_a.values < _b.values, _a.values, _b.values); _hi = np.where(_a.values < _b.values, _b.values, _a.values)
_hl = np.where(_src.values < _dst.values, _src.values, _dst.values); _hh = np.where(_src.values < _dst.values, _dst.values, _src.values)
_is_tcp = ((_sp != "") | (_dp != "")).values
_key = np.where(_is_tcp, "T|" + pd.Series(_lo) + "|" + pd.Series(_hi),
                np.where((_us != "").values, "U|" + _us.values, "H|" + pd.Series(_hl) + "|" + pd.Series(_hh)))
SESSION = pd.util.hash_pandas_object(pd.Series(raw[LABEL].values + "|" + _key), index=False).values
_sk = pd.DataFrame({"s": SESSION, "y": raw[LABEL].values})
audit["sessions_by_class"] = _sk.groupby("y")["s"].nunique().to_dict()
audit["largest_session_share_by_class"] = _sk.groupby("y")["s"].agg(lambda v: float(v.value_counts().iloc[0] / len(v))).to_dict()
save_json(audit, "audit.json")
print("sessions per class:", audit["sessions_by_class"])
del _src, _dst, _sp, _dp, _us, _a, _b, _lo, _hi, _hl, _hh, _key, _sk

AUTHOR_KEPT = list(kept)
BASE_RAW = raw[kept + [LABEL, "_row"]].copy()
BASE_RAW["_t"] = ts.values
del raw; gc.collect()

def canonicalize(col):
    """'0'/'0.0'/'0x00000000'/'' -> 0; hex -> int; numeric strings -> float; other text kept (stripped)."""
    s = col.where(~col.isin(EMPTY), "0")
    hexm = s.str.match(r"^0x[0-9a-fA-F]+$")
    if hexm.any():
        s = s.copy(); s[hexm] = s[hexm].map(lambda v: str(int(v, 16)))
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.99:
        return num.fillna(0).astype(np.float64)          # numeric field
    out = s.copy(); ok = num.notna()
    out[ok] = num[ok].map(lambda v: f"{v:.10g}")            # text field with numeric spellings unified
    return out

BASE_CANON = BASE_RAW[[LABEL, "_row", "_t"]].copy()
for c in AUTHOR_KEPT:
    BASE_CANON[c] = canonicalize(BASE_RAW[c])
classes = sorted(BASE_RAW[LABEL].unique())
cls2id = {c: i for i, c in enumerate(classes)}
NUM_RATIO_RAW = {c: pd.to_numeric(BASE_RAW[c], errors="coerce").notna().mean() for c in AUTHOR_KEPT}
IS_NUM_CANON = {c: pd.api.types.is_float_dtype(BASE_CANON[c]) for c in AUTHOR_KEPT}

def feature_hash(frame):
    return pd.util.hash_pandas_object(frame, index=False).values

def ceiling_stats(B, cols):
    """Per-packet information ceiling: records whose feature vector occurs with several labels cannot all be
    classified correctly; the best any packet-level classifier can do is predict each vector's majority label."""
    h = feature_hash(B[cols]); y = B[LABEL].values
    t = pd.DataFrame({"h": h, "y": y}).value_counts().rename("n").reset_index()
    tot = t.groupby("h")["n"].transform("sum"); mx = t.groupby("h")["n"].transform("max")
    t["is_max"] = t["n"] == mx
    first_max = t[t["is_max"]].drop_duplicates("h")[["h", "y"]].rename(columns={"y": "maj"})
    t = t.merge(first_max, on="h")
    n_lab = t.groupby("h")["y"].transform("nunique")
    out = dict(n_records=int(len(B)), n_unique_vectors=int(t["h"].nunique()),
               unique_vector_label_pairs_by_class=t.groupby("y").size().to_dict(),
               vectors_with_conflict=int((t.groupby("h")["y"].nunique() > 1).sum()),
               records_in_conflicting_vectors_by_class=t[n_lab > 1].groupby("y")["n"].sum().to_dict(),
               ceiling_accuracy=float(t.loc[t["y"] == t["maj"], "n"].sum() / t["n"].sum()),
               ceiling_recall_by_class=(t[t["y"] == t["maj"]].groupby("y")["n"].sum() / t.groupby("y")["n"].sum()).fillna(0).to_dict())
    out = json.loads(json.dumps(out, default=lambda o: int(o) if isinstance(o, (np.integer,)) else float(o)))
    return out

def empty_mask(frame):
    """True where a field is empty; works for raw strings and canonical floats/strings."""
    cols = {}
    for c in frame.columns:
        v = frame[c]
        cols[c] = (v == 0).values if pd.api.types.is_numeric_dtype(v) else v.isin(EMPTY | {"0"}).values
    return pd.DataFrame(cols, index=frame.index)

from sklearn.tree import DecisionTreeClassifier

def prepare(feature_set):
    """Grouped protocol: every distinct canonical feature vector is one group; groups (not records) are split."""
    global df, y_all, kept, cat_cols, num_cols, group_of, kept_groups, H, GRP
    extra = STRICT_EXTRA + (["tcp.seq", "tcp.ack"] if feature_set == "strict-noseq" else [])
    kept = [c for c in AUTHOR_KEPT if not (feature_set.startswith("strict") and c in extra)]
    B = BASE_RAW if feature_set == "raw-author" else BASE_CANON
    df = B[kept + [LABEL, "_row", "_t"]].reset_index(drop=True)
    y_all = df[LABEL].map(cls2id).values.astype(np.int64)
    H = feature_hash(df[kept])
    tnum = pd.to_datetime(df["_t"]).values.astype("datetime64[ns]").astype(np.int64)
    tnum = np.where(df["_t"].notna().values, tnum, np.iinfo(np.int64).max)
    g = pd.DataFrame({"h": H, "y": y_all, "t": tnum, "r": df["_row"].values})
    cnt = g.groupby(["h", "y"]).size().rename("n").reset_index().sort_values(["h", "n"], ascending=[True, False])
    maj = cnt.drop_duplicates("h").set_index("h")["y"]
    GRP = pd.DataFrame({"maj": maj, "tmin": g.groupby("h")["t"].min(), "rmin": g.groupby("h")["r"].min()})
    cat_cols = [c for c in kept if (NUM_RATIO_RAW[c] < 0.99 if feature_set == "raw-author" else not IS_NUM_CANON[c])]
    num_cols = [c for c in kept if c not in cat_cols]
    group_of = {c: g_ for g_, ps in GROUPS for c in kept if any(c.startswith(p) for p in ps)}
    kept_groups = [g_ for g_, _ in GROUPS if any(group_of.get(c) == g_ for c in kept)]
    info = dict(feature_set=feature_set, n_features=len(kept), n_records=len(df), n_groups=len(GRP),
                groups_by_majority_class={classes[k]: int(v) for k, v in GRP["maj"].value_counts().items()},
                categorical=cat_cols, token_groups=kept_groups)
    global SESSION_ID, SESSION_DF
    SESSION_ID = SESSION[df["_row"].values]
    SESSION_DF = pd.DataFrame({"s": SESSION_ID, "y": y_all}).groupby("s").agg(y=("y", "first"), n=("y", "size")).reset_index().set_index("s")
    SESSION_DF.index.name = None; SESSION_DF["s"] = SESSION_DF.index
    SESSION_DF = SESSION_DF.reset_index(drop=True)
    info["sessions"] = int(len(SESSION_DF))
    print({k: v for k, v in info.items() if k != "groups_by_majority_class"}); return info

def make_split(mode, seed):
    """Returns record indices: train/val = one record per (vector, label) in train/val groups;
    test = all records of test groups (traffic multiplicity kept); test_u = one record per (vector, label)."""
    rng = np.random.default_rng(seed); parts = {"tr": [], "va": [], "te": []}; basis = {}
    for k in range(len(classes)):
        gk = GRP[GRP["maj"] == k]
        if len(gk) == 0: continue
        if mode == "random":
            order = gk.index.values[rng.permutation(len(gk))]; basis[classes[k]] = "random"
        else:
            use_time = (gk["tmin"] < np.iinfo(np.int64).max).mean() > 0.9
            order = gk.sort_values("tmin" if use_time else "rmin").index.values
            basis[classes[k]] = "time" if use_time else "file_order"
        n = len(order)
        if n >= 3:
            a = min(max(1, int(round(0.7 * n))), n - 2); b = max(a + 1, min(int(round(0.8 * n)), n - 1))
        elif n == 2: a, b = 1, 1
        else: a, b = 1, 1
        parts["tr"].append(order[:a]); parts["va"].append(order[a:b]); parts["te"].append(order[b:])
    sets = {k: set(np.concatenate(v).tolist()) for k, v in parts.items()}
    first = pd.DataFrame({"h": H, "y": y_all}).drop_duplicates().index.values   # one record per (vector, label)
    in_set = lambda idx, key: idx[np.isin(H[idx], np.fromiter(sets[key], dtype=H.dtype))]
    tr, va, te_u = in_set(first, "tr"), in_set(first, "va"), in_set(first, "te")
    te = in_set(np.arange(len(df)), "te")
    how = ("grouped " + mode + " split by canonical feature vector; per-class basis: " + json.dumps(basis))
    return tr, va, te, te_u, how

def make_session_split(seed):
    """Session-grouped split: sessions (not vectors) are assigned to train/val/test per class, targeting 70/10/20
    of the records. train/val = one record per (vector, label) of their sessions; test = all records of test sessions;
    test_u = one record per (vector, label) of test sessions; test_novel = those test vectors never seen in training."""
    rng = np.random.default_rng(seed); tr_s, va_s, te_s = [], [], []
    S = SESSION_DF
    for k in range(len(classes)):
        sk = S[S["y"] == k]
        if len(sk) == 0: continue
        order = sk.index.values[rng.permutation(len(sk))]
        n = len(order)
        if n >= 3:
            cum = np.cumsum(S.loc[order, "n"].values) / sk["n"].sum()
            a = int(np.searchsorted(cum, 0.7)) + 1; b = int(np.searchsorted(cum, 0.8)) + 1
            a = min(max(a, 1), n - 2); b = min(max(b, a + 1), n - 1)
        else:
            a, b = 1, 1
        tr_s.append(order[:a]); va_s.append(order[a:b]); te_s.append(order[b:])
    sid = SESSION_ID
    in_s = lambda parts: np.isin(sid, S.loc[np.concatenate(parts), "s"].values)
    m_tr, m_va, m_te = in_s(tr_s), in_s(va_s), in_s(te_s)
    def first_of(mask):
        idx = np.where(mask)[0]
        return idx[~pd.DataFrame({"h": H[idx], "y": y_all[idx]}).duplicated().values]
    tr, va, te_u = first_of(m_tr), first_of(m_va), first_of(m_te)
    te = np.where(m_te)[0]
    seen = set(H[m_tr].tolist())
    te_n = te_u[~np.isin(H[te_u], np.fromiter(seen, dtype=H.dtype))]
    how = "session-grouped split (bidirectional address/port pair, UDP stream, or host pair)"
    return tr, va, te, te_u, te_n, how

class Encoder:
    """signed-log + standardization for numeric fields; top-K vocab for categorical fields; presence per group."""
    def fit(self, d):
        X = d[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
        X = np.sign(X) * np.log1p(np.abs(X))
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        self.vocab = {c: {v: i + 1 for i, v in enumerate(d[c].value_counts().index[:CFG["cat_vocab"]])} for c in cat_cols}
        return self
    def transform(self, d):
        X = d[num_cols].apply(pd.to_numeric, errors="coerce").fillna(0).values.astype(np.float64)
        X = np.clip((np.sign(X) * np.log1p(np.abs(X)) - self.mu) / self.sd, -CFG["clip"], CFG["clip"]).astype(np.float32)
        C = np.stack([d[c].map(self.vocab[c]).fillna(0).astype(np.int64).values for c in cat_cols], 1) \
            if cat_cols else np.zeros((len(d), 0), np.int64)
        P = np.stack([(~empty_mask(d[[c for c in kept if group_of.get(c) == g]])).any(axis=1).values
                      for g in kept_groups], 1).astype(np.float32)
        return X, C, P

def presence(d, gmap, groups):
    return np.stack([(~empty_mask(d[[c for c in kept if gmap.get(c) == g]])).any(axis=1).values
                     if any(gmap.get(c) == g for c in kept) else np.zeros(len(d), bool) for g in groups], 1).astype(np.float32)

def random_group_map(seed):
    """Same group sizes as the protocol grouping, fields assigned to groups at random (control for PALT)."""
    cols = [c for c in kept if c in group_of]
    labels = [group_of[c] for c in cols]
    perm = np.random.default_rng(1000 + seed).permutation(len(cols))
    return {cols[i]: labels[j] for i, j in zip(range(len(cols)), perm)}

# %% [markdown]
# ## 3. Models

# %%
def num_idx(g, gm=None): gm = gm or group_of; return [i for i, c in enumerate(num_cols) if gm.get(c) == g]
def cat_idx(g, gm=None): gm = gm or group_of; return [i for i, c in enumerate(cat_cols) if gm.get(c) == g]

class CBFocal(nn.Module):
    """Class-balanced focal loss (Cui et al. 2019 weights, Lin et al. 2017 focusing)."""
    def __init__(self, counts, beta, gamma):
        super().__init__()
        eff = (1 - np.power(beta, counts)) / (1 - beta)
        w = (1 / eff); w = w / w.sum() * len(counts)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32)); self.g = gamma
    def forward(self, logits, y, sw=None):
        logp = F.log_softmax(logits, 1); p = logp.exp()
        lp = logp.gather(1, y[:, None]).squeeze(1); pt = p.gather(1, y[:, None]).squeeze(1)
        l = -(self.w[y]) * (1 - pt) ** self.g * lp
        return (l * sw).sum() / sw.sum() if sw is not None else l.mean()

class CatEmb(nn.Module):
    def __init__(self, n_cat, d):
        super().__init__(); self.e = nn.ModuleList([nn.Embedding(CFG["cat_vocab"] + 1, d) for _ in range(n_cat)])
    def forward(self, C):
        return sum(e(C[:, i]) for i, e in enumerate(self.e)) if len(self.e) else 0

class MLP(nn.Module):
    def __init__(self, n_num, n_cat, n_cls, d=128):
        super().__init__(); self.cat = CatEmb(n_cat, 16)
        self.net = nn.Sequential(nn.Linear(n_num + 16, d), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_cls))
    def forward(self, X, C, P):
        c = self.cat(C); c = c if torch.is_tensor(c) else X.new_zeros(X.size(0), 16)
        return self.net(torch.cat([X, c], 1))

class PLR(nn.Module):
    """Periodic numerical embeddings (Gorishniy et al. 2022): x -> [sin(2*pi*c*x), cos(2*pi*c*x)] -> Linear -> ReLU."""
    def __init__(self, n_num, k=16, d=16, sigma=1.0):
        super().__init__(); self.c = nn.Parameter(torch.randn(n_num, k) * sigma)
        self.lin = nn.Parameter(torch.randn(n_num, 2 * k, d) * (2 * k) ** -0.5); self.b = nn.Parameter(torch.zeros(n_num, d))
    def forward(self, X):
        v = 2 * math.pi * X[..., None] * self.c
        v = torch.cat([torch.sin(v), torch.cos(v)], -1)
        return F.relu(torch.einsum("bnk,nkd->bnd", v, self.lin) + self.b)

class MLPPLR(nn.Module):
    def __init__(self, n_num, n_cat, n_cls, d=128, de=16):
        super().__init__(); self.plr = PLR(n_num, d=de); self.cat = CatEmb(n_cat, 16)
        self.net = nn.Sequential(nn.Linear(n_num * de + 16, d), nn.GELU(), nn.Dropout(0.1),
                                 nn.Linear(d, d), nn.GELU(), nn.Linear(d, n_cls))
    def forward(self, X, C, P):
        c = self.cat(C); c = c if torch.is_tensor(c) else X.new_zeros(X.size(0), 16)
        return self.net(torch.cat([self.plr(X).flatten(1), c], 1))

class CNN1D(nn.Module):
    def __init__(self, n_num, n_cat, n_cls, ch=32):
        super().__init__(); self.cat = CatEmb(n_cat, 8)
        self.net = nn.Sequential(nn.Conv1d(1, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
                                 nn.Conv1d(ch, ch, 3, padding=1), nn.BatchNorm1d(ch), nn.GELU(),
                                 nn.AdaptiveAvgPool1d(1))
        self.head = nn.Linear(ch + 8, n_cls)
    def forward(self, X, C, P):
        c = self.cat(C); c = c if torch.is_tensor(c) else X.new_zeros(X.size(0), 8)
        return self.head(torch.cat([self.net(X[:, None, :]).squeeze(-1), c], 1))

class FTTransformer(nn.Module):
    """One token per field (Gorishniy et al. 2021), compact configuration."""
    def __init__(self, n_num, n_cat, n_cls, d=32, heads=4, blocks=2):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) * 0.02); self.b = nn.Parameter(torch.zeros(n_num, d))
        self.cats = nn.ModuleList([nn.Embedding(CFG["cat_vocab"] + 1, d) for _ in range(n_cat)])
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        layer = nn.TransformerEncoderLayer(d, heads, 2 * d, 0.1, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, blocks); self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))
    def forward(self, X, C, P):
        t = X[:, :, None] * self.w + self.b
        if len(self.cats):
            t = torch.cat([t, torch.stack([e(C[:, i]) for i, e in enumerate(self.cats)], 1)], 1)
        t = torch.cat([self.cls.expand(X.size(0), -1, -1), t], 1)
        return self.head(self.enc(t)[:, 0])

class LocalGlobalBlock(nn.Module):
    """MobileViT-style block on protocol tokens: local DS-conv over the layer-ordered token axis,
    attention (windowed within stack layers, or global), then fusion of local and global paths."""
    def __init__(self, d, heads, attn_mask=None, use_local=True):
        super().__init__(); self.use_local = use_local
        self.local = nn.Sequential(nn.Conv1d(d, d, 3, padding=1, groups=d), nn.Conv1d(d, d, 1), nn.GELU())
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.fuse = nn.Linear(2 * d, d)
        self.register_buffer("mask", attn_mask if attn_mask is not None else torch.zeros(0), persistent=False)
    def forward(self, t):
        loc = t + self.local(t.transpose(1, 2)).transpose(1, 2) if self.use_local else t
        h = self.n1(loc)
        m = self.mask if self.mask.numel() else None
        g = loc + self.attn(h, h, h, attn_mask=m, need_weights=False)[0]
        g = g + self.ffn(self.n2(g))
        return self.fuse(torch.cat([loc, g], -1)) + t

class PALT(nn.Module):
    """Protocol-Aware Local–global Transformer.
    tokens = [CLS] + one token per protocol group; absent protocols use a learned embedding."""
    def __init__(self, n_num, n_cat, n_cls, d=32, heads=4, blocks=2, windowed=True, gmap=None, use_local=True, zero_absent=False, plr=False, de=8):
        super().__init__()
        self.groups = kept_groups; G = len(self.groups); self.zero_absent = zero_absent
        self.nidx = [num_idx(g, gmap) for g in self.groups]; self.cidx = [cat_idx(g, gmap) for g in self.groups]
        for gi, ids in enumerate(self.nidx):
            self.register_buffer(f"ni{gi}", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.plr = PLR(n_num, d=de) if plr else None; self.de = de
        self.proj = nn.ModuleList([nn.Linear(max(len(i), 1) * (de if plr and len(i) else 1), d) for i in self.nidx])
        self.cats = nn.ModuleList([nn.Embedding(CFG["cat_vocab"] + 1, d) for _ in range(n_cat)])
        self.absent = nn.Parameter(torch.randn(G, d) * 0.02)
        self.pos = nn.Parameter(torch.randn(G + 1, d) * 0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        # window mask: tokens attend within their stack layer; CLS attends to all
        win = torch.full((G + 1, G + 1), float("-inf"))
        win[0, :] = 0; win[:, 0] = 0
        for layer in LAYER_WINDOWS:
            ids = [1 + self.groups.index(x) for x in layer if x in self.groups]
            for a in ids:
                for b in ids:
                    win[a, b] = 0
        self.blocks = nn.ModuleList([LocalGlobalBlock(d, heads, win if (windowed and i % 2 == 0) else None, use_local)
                                     for i in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))
    def tokens(self, X, C, P):
        B = X.size(0); toks = []
        E = self.plr(X) if self.plr is not None else None          # (B, n_num, de): periodic embedding per numeric field
        for gi in range(len(self.groups)):
            ids = getattr(self, f"ni{gi}")
            if E is not None and ids.numel():
                x = E.index_select(1, ids).flatten(1)
            else:
                x = X.index_select(1, ids) if ids.numel() else X.new_zeros(B, 1)
            t = self.proj[gi](x)
            for ci in self.cidx[gi]:
                t = t + self.cats[ci](C[:, ci])
            toks.append(t)
        T = torch.stack(toks, 1)
        pres = P[:, :, None]
        T = pres * T + ((1 - pres) * self.absent[None] if not self.zero_absent else 0)
        T = torch.cat([self.cls.expand(B, -1, -1), T], 1) + self.pos[None]
        return T
    def forward(self, X, C, P):
        t = self.tokens(X, C, P)
        for b in self.blocks:
            t = b(t)
        return self.head(t[:, 0])

def build(name, n_num, n_cat, n_cls):
    if name == "MLP": return MLP(n_num, n_cat, n_cls)
    if name == "CNN1D": return CNN1D(n_num, n_cat, n_cls)
    if name == "FTTransformer": return FTTransformer(n_num, n_cat, n_cls, d=CFG["d_student"], heads=CFG["heads"])
    if name == "PALT-Teacher": return PALT(n_num, n_cat, n_cls, d=CFG["d_teacher"], heads=CFG["heads"], blocks=CFG["blocks_teacher"])
    if name in ("PALT", "PALT-KD"): return PALT(n_num, n_cat, n_cls, d=CFG["d_student"], heads=CFG["heads"], blocks=CFG["blocks_student"])
    kw = dict(d=CFG["d_student"], heads=CFG["heads"], blocks=CFG["blocks_student"])
    if name == "PALT-RandGroup": return PALT(n_num, n_cat, n_cls, gmap=RAND_GMAP, **kw)
    if name == "PALT-NoLocal": return PALT(n_num, n_cat, n_cls, use_local=False, **kw)
    if name == "PALT-FullAttn": return PALT(n_num, n_cat, n_cls, windowed=False, **kw)
    if name == "PALT-ZeroAbsent": return PALT(n_num, n_cat, n_cls, zero_absent=True, **kw)
    if name == "MLP-PLR": return MLPPLR(n_num, n_cat, n_cls)
    if name == "PALT-PLR": return PALT(n_num, n_cat, n_cls, plr=True, **kw)
    if name == "PALT-PLR-RandGroup": return PALT(n_num, n_cat, n_cls, plr=True, gmap=RAND_GMAP, **kw)
    raise ValueError(name)

def n_params(m): return sum(p.numel() for p in m.parameters())

def flops_per_sample(m, X, C, P):
    try:
        from torch.utils.flop_counter import FlopCounterMode
        m = m.eval().cpu()
        fast = getattr(torch.backends, "mha", None)
        if fast is not None: fast.set_fastpath_enabled(False)   # fused MHA kernels are invisible to the counter
        with FlopCounterMode(display=False) as fc, torch.no_grad():
            m(X[:1].cpu(), C[:1].cpu(), P[:1].cpu())
        if fast is not None: fast.set_fastpath_enabled(True)
        return int(fc.get_total_flops())
    except Exception as e:
        return f"n/a ({e.__class__.__name__})"

# %% [markdown]
# ## 4. Training and evaluation

# %%
def loaders(X, C, P, y, bs, shuffle, w=None):
    w = np.ones(len(y), np.float32) if w is None else w.astype(np.float32)
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(C), torch.from_numpy(P), torch.from_numpy(y), torch.from_numpy(w))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, num_workers=2, pin_memory=(DEV == "cuda"))

@torch.no_grad()
def predict(m, dl):
    m.eval(); out = []
    for X, C, P, _, _w in dl:
        with torch.autocast(DEV, enabled=(DEV == "cuda")):
            out.append(m(X.to(DEV), C.to(DEV), P.to(DEV)).float().cpu())
    return torch.cat(out)

def metrics(y, pred):
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=range(len(classes)), zero_division=0)
    return dict(macro_f1=f1_score(y, pred, average="macro"), weighted_f1=f1_score(y, pred, average="weighted"),
                balanced_acc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred),
                accuracy=float((y == pred).mean()),
                normal_fpr=float(((pred != cls2id.get("Normal", -1)) & (y == cls2id.get("Normal", -1))).sum() /
                                 max((y == cls2id.get("Normal", -1)).sum(), 1)),
                per_class_f1={classes[i]: float(f[i]) for i in range(len(classes))},
                per_class_support={classes[i]: int(s[i]) for i in range(len(classes))})

def train_nn(name, data, seed, teacher=None, sw=None):
    set_seed(seed)
    (Xtr, Ctr, Ptr, ytr), (Xva, Cva, Pva, yva) = data["train"], data["val"]
    m = build(name, Xtr.shape[1], Ctr.shape[1], len(classes)).to(DEV)
    crit = CBFocal(np.bincount(ytr, minlength=len(classes)).clip(1), CFG["cb_beta"], CFG["focal_gamma"]).to(DEV)
    lr = CFG["lr_teacher"] if name == "PALT-Teacher" else CFG["lr"]
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=CFG["weight_decay"])
    dtr = loaders(Xtr, Ctr, Ptr, ytr, CFG["batch_size"], True, sw); dva = loaders(Xva, Cva, Pva, yva, 8192, False)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=CFG["epochs"] * len(dtr))
    scaler = torch.amp.GradScaler(enabled=(DEV == "cuda"))
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(CFG["epochs"]):
        m.train(); t0 = time.time(); tot = 0
        for X, C, P, y, w in dtr:
            X, C, P, y, w = X.to(DEV, non_blocking=True), C.to(DEV), P.to(DEV), y.to(DEV), w.to(DEV)
            with torch.autocast(DEV, enabled=(DEV == "cuda")):
                logits = m(X, C, P); loss = crit(logits.float(), y, w if sw is not None else None)
                if teacher is not None:
                    with torch.no_grad():
                        tl = teacher(X, C, P).float()
                    T = CFG["kd_T"]
                    kd = F.kl_div(F.log_softmax(logits.float() / T, 1), F.softmax(tl / T, 1), reduction="batchmean") * T * T
                    loss = CFG["kd_alpha"] * kd + (1 - CFG["kd_alpha"]) * loss
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            tot += loss.item() * len(y)
        vf = f1_score(yva, predict(m, dva).argmax(1).numpy(), average="macro")
        hist.append(dict(epoch=ep, loss=tot / len(ytr), val_macro_f1=vf, sec=time.time() - t0))
        print(f"  {name} ep{ep:02d} loss={tot/len(ytr):.4f} valF1={vf:.4f} ({time.time()-t0:.0f}s)")
        if vf > best + 1e-4:
            best, bad = vf, 0; best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        else:
            bad += 1
            if bad >= CFG["patience"]: break
    m.load_state_dict(best_state)
    return m, hist

def train_lgbm(data, seed):
    import lightgbm as lgb
    (Xtr, Ctr, _, ytr), (Xva, Cva, _, yva) = data["train"], data["val"]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, class_weight="balanced",
                             random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(np.hstack([Xtr, Ctr]), ytr, eval_set=[(np.hstack([Xva, Cva]), yva)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    return clf

def train_lgbm_w(data, seed, sw):
    import lightgbm as lgb
    (Xtr, Ctr, _, ytr), (Xva, Cva, _, yva) = data["train"], data["val"]
    clf = lgb.LGBMClassifier(n_estimators=300, learning_rate=0.1, num_leaves=63, class_weight="balanced",
                             random_state=seed, n_jobs=-1, verbose=-1)
    clf.fit(np.hstack([Xtr, Ctr]), ytr, sample_weight=sw, eval_set=[(np.hstack([Xva, Cva]), yva)],
            callbacks=[lgb.early_stopping(30, verbose=False)])
    return clf

def balanced_w(y):
    cnt = np.bincount(y, minlength=len(classes)).astype(np.float64)
    return (len(y) / (len(classes) * np.maximum(cnt, 1)))[y]

def train_tree(name, data, seed):
    (Xtr, Ctr, _, ytr), (Xva, Cva, _, yva) = data["train"], data["val"]
    A, Av = np.hstack([Xtr, Ctr]), np.hstack([Xva, Cva])
    if name == "XGBoost":
        import xgboost as xgb
        present = np.unique(ytr); remap = {c: i for i, c in enumerate(present)}
        clf = xgb.XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=8, tree_method="hist",
                                device="cuda" if DEV == "cuda" else "cpu", early_stopping_rounds=30,
                                random_state=seed, eval_metric="mlogloss")
        yv = np.array([remap.get(v, -1) for v in yva]); keep = yv >= 0
        clf.fit(A, np.vectorize(remap.get)(ytr), sample_weight=balanced_w(ytr), eval_set=[(Av[keep], yv[keep])], verbose=False)
        clf._present = present
        return clf
    if name == "CatBoost":
        from catboost import CatBoostClassifier
        clf = CatBoostClassifier(iterations=1000, learning_rate=0.1, depth=8, loss_function="MultiClass",
                                 auto_class_weights="Balanced", random_seed=seed, verbose=False, od_type="Iter", od_wait=50,
                                 task_type="GPU" if DEV == "cuda" else "CPU")
        clf.fit(A, ytr, eval_set=(Av, yva))
        return clf
    if name == "RandomForest":
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(n_estimators=300, max_features="sqrt", class_weight="balanced_subsample",
                                      n_jobs=-1, random_state=seed).fit(A, ytr)
    raise ValueError(name)

def tree_predict(clf, X, C):
    A = np.hstack([X, C]); p = clf.predict(A)
    p = np.asarray(p).reshape(-1)
    if hasattr(clf, "_present"): p = clf._present[p.astype(int)]
    return p.astype(np.int64)

# %% [markdown]
# ## 5. Export, INT8 quantization and CPU latency (proxy for edge profiles)

# %%
def export_and_bench(m, name, sample, tag, calib):
    import onnxruntime as ort
    from onnxruntime.quantization import quantize_dynamic, QuantType
    res = {}
    m = m.eval().cpu()
    X, C, P = [torch.from_numpy(a[:1]) for a in sample]
    fp32 = os.path.join(CFG["out_dir"], f"{tag}_{name}.onnx")
    kw = dict(input_names=["X", "C", "P"], output_names=["logits"], opset_version=17,
              dynamic_axes={"X": {0: "b"}, "C": {0: "b"}, "P": {0: "b"}, "logits": {0: "b"}})
    try:
        torch.onnx.export(m, (X, C, P), fp32, dynamo=False, **kw)
    except TypeError:
        torch.onnx.export(m, (X, C, P), fp32, **kw)
    names = {i.name for i in ort.InferenceSession(fp32, providers=["CPUExecutionProvider"]).get_inputs()}
    full = {"X": X.numpy(), "C": C.numpy(), "P": P.numpy()}
    feeds = {k: v for k, v in full.items() if k in names}
    int8 = fp32.replace(".onnx", "_int8dyn.onnx")
    quantize_dynamic(fp32, int8, weight_type=QuantType.QInt8, per_channel=True)
    paths = [("fp32", fp32), ("int8_dynamic", int8)]
    try:  # static (QDQ) INT8 with 512 calibration records
        from onnxruntime.quantization import quantize_static, CalibrationDataReader, QuantFormat
        cal = {k: v for k, v in zip(["X", "C", "P"], calib) if k in names}
        class Reader(CalibrationDataReader):
            def __init__(self): self.i = 0
            def get_next(self):
                if self.i >= len(cal["X"]): return None
                r = {k: v[self.i:self.i + 1] for k, v in cal.items()}; self.i += 1; return r
        st = fp32.replace(".onnx", "_int8static.onnx")
        quantize_static(fp32, st, Reader(), quant_format=QuantFormat.QDQ, per_channel=True,
                        op_types_to_quantize=["MatMul", "Gemm", "Conv"],
                        activation_type=QuantType.QInt8, weight_type=QuantType.QInt8)
        paths.append(("int8_static", st))
    except Exception as e:
        res["static_error"] = repr(e)
    for prec, path in paths:
        res[prec] = {"size_kb": os.path.getsize(path) / 1024}
        for th in CFG["latency_threads"]:
            so = ort.SessionOptions(); so.intra_op_num_threads = th; so.inter_op_num_threads = 1
            s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
            for _ in range(100): s.run(None, feeds)
            lat = []
            for _ in range(CFG["latency_iters"]):
                t = time.perf_counter(); s.run(None, feeds); lat.append((time.perf_counter() - t) * 1e3)
            lat = np.array(lat)
            res[prec][f"threads{th}"] = dict(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)),
                                             p99=float(np.percentile(lat, 99)), mean=float(lat.mean()))
    res["paths"] = dict(paths)
    return res

def tree_to_onnx(booster, name, tag, A, ref, relabel):
    """Gradient-boosted trees -> ONNX TreeEnsemble (onnxmltools, opset 15), checked against the native predictions."""
    try:
        import onnxruntime as ort
        from onnxmltools import convert_xgboost, convert_lightgbm
        from onnxmltools.convert.common.data_types import FloatTensorType
        it = [("input", FloatTensorType([None, A.shape[1]]))]
        onx = convert_xgboost(booster, initial_types=it, target_opset=15) if name == "XGBoost" else \
              convert_lightgbm(booster, initial_types=it, target_opset=15, zipmap=False)
        path = os.path.join(CFG["out_dir"], f"{tag}_{name}_trees.onnx")
        open(path, "wb").write(onx.SerializeToString())
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        q = np.concatenate([np.asarray(s.run(None, {"input": A[i:i + 8192]})[0]).reshape(-1) for i in range(0, len(A), 8192)])
        return dict(path=os.path.basename(path), size_kb=os.path.getsize(path) / 1024,
                    agree=float((relabel(q.astype(np.int64)) == np.asarray(ref).reshape(-1)).mean()))
    except Exception as e:
        return dict(error=repr(e))

def export_tree(clf, name, tag, data):
    """Model files for the Docker edge benchmark: XGBoost -> native JSON; random forest -> ONNX (skl2onnx).
    Checks that the exported model reproduces the in-memory predictions on the distinct test vectors."""
    Xu, Cu = data["test_u"][0], data["test_u"][1]
    A = np.hstack([Xu, Cu.astype(np.float32)]).astype(np.float32)
    ref = tree_predict(clf, Xu, Cu)
    if name == "XGBoost":
        import xgboost as xgb
        path = os.path.join(CFG["out_dir"], f"{tag}_XGBoost.xgb.json")
        bst = clf.get_booster()
        bi = getattr(clf, "best_iteration", None)
        if bi is not None: bst = bst[: bi + 1]          # keep the trees the fitted classifier predicts with
        bst.save_model(path)
        b = xgb.Booster(); b.load_model(path); b.set_param({"device": "cpu"})
        p = clf._present[np.asarray(b.inplace_predict(A)).argmax(1)]
        out = dict(path=os.path.basename(path), size_kb=os.path.getsize(path) / 1024, agree=float((p == ref).mean()),
                   n_trees=int(b.num_boosted_rounds()) * len(clf._present), xgboost=xgb.__version__)
        out["onnx"] = tree_to_onnx(b, "XGBoost", tag, A, ref, lambda q: clf._present[q])
        return out
    if name == "LightGBM":
        b = clf.booster_
        path = os.path.join(CFG["out_dir"], f"{tag}_LightGBM.lgb.txt"); b.save_model(path)
        return dict(path=os.path.basename(path), size_kb=os.path.getsize(path) / 1024, n_trees=int(b.num_trees()),
                    onnx=tree_to_onnx(b, "LightGBM", tag, A, clf.predict(A), lambda q: q))
    if name == "RandomForest":
        import sklearn
        nodes = int(sum(e.tree_.node_count for e in clf.estimators_))
        clf.set_params(n_jobs=1); lat = []   # native scikit-learn, one record per call (reference only)
        for i in range(300):
            t0_ = time.perf_counter(); clf.predict(A[i:i + 1]); lat.append((time.perf_counter() - t0_) * 1e3)
        out = dict(n_trees=len(clf.estimators_), n_nodes=nodes, sklearn=sklearn.__version__,
                   sklearn_native_1thr_ms=dict(p50=float(np.percentile(lat, 50)), p99=float(np.percentile(lat, 99))))
        try:
            from skl2onnx import to_onnx
            from skl2onnx.common.data_types import FloatTensorType
            import onnxruntime as ort
            path = os.path.join(CFG["out_dir"], f"{tag}_RandomForest.onnx")
            onx = to_onnx(clf, initial_types=[("input", FloatTensorType([None, A.shape[1]]))],
                          options={id(clf): {"zipmap": False}}, target_opset={"": 17, "ai.onnx.ml": 3})
            open(path, "wb").write(onx.SerializeToString()); del onx
            s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
            p = np.concatenate([s.run(None, {"input": A[i:i + 8192]})[0] for i in range(0, len(A), 8192)])
            out.update(path=os.path.basename(path), size_kb=os.path.getsize(path) / 1024, agree=float((p.astype(np.int64) == ref).mean()))
        except Exception as e:
            out["onnx_error"] = repr(e)
        return out
    return {}

@torch.no_grad()
def int8_accuracy(path, te):
    import onnxruntime as ort
    s = ort.InferenceSession(path, providers=["CPUExecutionProvider"]); out = []
    names = {i.name for i in s.get_inputs()}
    X, C, P, y = te
    for i in range(0, len(y), 8192):
        f = {"X": X[i:i+8192], "C": C[i:i+8192], "P": P[i:i+8192]}
        out.append(s.run(None, {k: v for k, v in f.items() if k in names})[0])
    return metrics(y, np.concatenate(out).argmax(1))

def palt_token_attribution(m, data, per_class=300):
    """|gradient x input| summed over the token width, averaged per true class (CLS excluded)."""
    m = m.to(DEV).eval(); X, C, P, y = data["test"]; rows = []
    for k in range(len(classes)):
        idx = np.where(y == k)[0][:per_class]
        if len(idx) == 0: continue
        Xb, Cb, Pb = [torch.from_numpy(a[idx]).to(DEV) for a in (X, C, P)]
        T = m.tokens(Xb, Cb, Pb).detach().requires_grad_(True)
        t = T
        for b in m.blocks: t = b(t)
        logits = m.head(t[:, 0])
        logits.gather(1, logits.argmax(1, keepdim=True)).sum().backward()
        att = (T.grad * T).sum(-1).abs()[:, 1:].mean(0).detach().cpu().numpy()
        att = att / max(att.sum(), 1e-12)
        rows.append({"class": classes[k], "n": int(len(idx)), **{g_: float(v) for g_, v in zip(m.groups, att)}})
    return rows

# %% [markdown]
# ## 6. Run

# %%
ALL = []
PREP = {}
TREES = ("XGBoost", "CatBoost", "RandomForest")
for run in CFG["runs"]:
    mode, fset, weighted = run["mode"], run["fset"], run.get("weighted", False)
    PREP[fset] = prepare(fset)
    for seed in run["seeds"]:
        te_n = None
        if mode == "session":
            tr, va, te, te_u, te_n, how = make_session_split(seed)
        else:
            tr, va, te, te_u, how = make_split(mode, seed)
        enc = Encoder().fit(df.iloc[tr])
        data = {}
        parts = [("train", tr), ("val", va), ("test", te), ("test_u", te_u)] + ([("test_n", te_n)] if te_n is not None else [])
        for part, ids in parts:
            X, C, P = enc.transform(df.iloc[ids]); data[part] = (X, C, P, y_all[ids])
        sw = None
        if weighted:   # log(1 + multiplicity of the (vector, label) pair among all records of the training groups)
            trg = set(GRP.index[np.isin(GRP.index.values, H[tr])].tolist())
            allm = np.isin(H, np.fromiter(trg, dtype=H.dtype))
            mult = pd.DataFrame({"h": H[allm], "y": y_all[allm]}).value_counts()
            sw = np.log1p(np.array([mult.get((h, y), 1) for h, y in zip(H[tr], y_all[tr])], dtype=np.float64)).astype(np.float32)
        gmap_rand = random_group_map(seed)
        split_info = dict(mode=mode, feature_set=fset, weighted=weighted, prep=PREP[fset], how=how, seed=seed,
                          n_train=len(tr), n_val=len(va), n_test=len(te), n_test_unique=len(te_u),
                          n_test_novel=(len(te_n) if te_n is not None else None),
                          test_unique_counts={classes[k]: int((y_all[te_u] == k).sum()) for k in range(len(classes))},
                          test_novel_counts=({classes[k]: int((y_all[te_n] == k).sum()) for k in range(len(classes))} if te_n is not None else None),
                          train_counts={classes[k]: int((y_all[tr] == k).sum()) for k in range(len(classes))})
        coarse = run.get("coarse")
        if coarse:   # quantile-bin tcp.seq and tcp.ack (edges and scaling from the training vectors); split unchanged
            for c in ("tcp.seq", "tcp.ack"):
                j = num_cols.index(c)
                edges = np.unique(np.quantile(data["train"][0][:, j], np.linspace(0, 1, coarse + 1)[1:-1]))
                btr = np.digitize(data["train"][0][:, j], edges).astype(np.float32)
                mu_, sd_ = btr.mean(), btr.std() + 1e-6
                for k_ in list(data):
                    Xk = data[k_][0].copy()
                    Xk[:, j] = (np.digitize(Xk[:, j], edges).astype(np.float32) - mu_) / sd_
                    data[k_] = (Xk,) + tuple(data[k_][1:])
            split_info["coarse_bins"] = int(coarse)
        tag = f"{mode}{'-w' if weighted else ''}{'-coarse' if coarse else ''}-{fset}_s{seed}"
        np.savez_compressed(os.path.join(CFG["out_dir"], f"split_{tag}.npz"),
                            train=df["_row"].values[tr], val=df["_row"].values[va], test=df["_row"].values[te], test_unique=df["_row"].values[te_u],
                            **({"test_novel": df["_row"].values[te_n]} if te_n is not None else {}))
        print(f"\n=== {tag}: {how} | train {len(tr)} val {len(va)} test {len(te)} test_u {len(te_u)}" + (f" test_novel {len(te_n)}" if te_n is not None else ""))
        evals = [("test", "test"), ("test_unique", "test_u")] + ([("test_novel", "test_n")] if te_n is not None else [])
        if run.get("permute_seq"):   # test vectors unchanged except tcp.seq / tcp.ack, each permuted across the test vectors
            Xp = data["test_u"][0].copy(); rp = np.random.default_rng(10_000 + seed)
            for c in ("tcp.seq", "tcp.ack"):
                j = num_cols.index(c); Xp[:, j] = Xp[rp.permutation(len(Xp)), j]
            data["test_u_perm"] = (Xp,) + tuple(data["test_u"][1:])
            evals.append(("test_unique_permseq", "test_u_perm"))
        dl = {k: loaders(*data[k], 8192, False) for _, k in evals}
        do_export = bool(run.get("export")) and seed == run["seeds"][0]
        if do_export:   # 2,000 real test records for the Docker edge benchmark (inputs only, no labels)
            rs = np.random.default_rng(seed).choice(len(data["test"][3]), min(2000, len(data["test"][3])), replace=False)
            np.savez_compressed(os.path.join(CFG["out_dir"], f"bench_inputs_{tag}.npz"),
                                X=data["test"][0][rs], C=data["test"][1][rs], P=data["test"][2][rs])
        for name in run["models"]:
            t0 = time.time(); rec = dict(split=split_info, model=name)
            if name == "LightGBM":
                clf = train_lgbm_w(data, seed, sw) if weighted else train_lgbm(data, seed)
                for key, k in evals:
                    rec[key] = metrics(data[k][3], clf.predict(np.hstack([data[k][0], data[k][1]])))
                rec.update(train_sec=time.time() - t0, n_trees=int(clf.booster_.num_trees()))
                if do_export:
                    try: rec["export"] = export_tree(clf, "LightGBM", tag, data)
                    except Exception as e: rec["export_error"] = repr(e); print("export failed: LightGBM", repr(e))
            elif name in TREES:
                clf = train_tree(name, data, seed)
                for key, k in evals:
                    rec[key] = metrics(data[k][3], tree_predict(clf, data[k][0], data[k][1]))
                rec.update(train_sec=time.time() - t0)
                if do_export:
                    try: rec["export"] = export_tree(clf, name, tag, data)
                    except Exception as e: rec["export_error"] = repr(e); print("export failed:", name, repr(e))
            else:
                d_use = data
                if name.endswith("RandGroup"):
                    RAND_GMAP = gmap_rand
                    d_use = {k: (v[0], v[1], presence(df.iloc[ids], RAND_GMAP, kept_groups), v[3])
                             for (k, v), ids in zip(data.items(), [i for _, i in parts])}
                    dl_use = {k: loaders(*d_use[k], 8192, False) for _, k in evals}
                else:
                    dl_use = dl
                m, hist = train_nn(name, d_use, seed, sw=sw)
                for key, k in evals:
                    rec[key] = metrics(d_use[k][3], predict(m, dl_use[k]).argmax(1).numpy())
                rec.update(history=hist, train_sec=time.time() - t0, params=n_params(m),
                           flops_per_sample=flops_per_sample(m, *[torch.from_numpy(a) for a in d_use["test"][:3]]))
                if name.endswith("RandGroup"):
                    rec["random_group_map"] = RAND_GMAP
                if do_export and not name.endswith("RandGroup"):
                    try:
                        rng = np.random.default_rng(seed); ci = rng.choice(len(data["train"][3]), 512, replace=False)
                        bench = export_and_bench(m, name, data["test"][:3], tag, [a[ci] for a in data["train"][:3]])
                        rec["onnx"] = {k: v for k, v in bench.items() if k != "paths"}
                        rec["fp32_test_unique"] = int8_accuracy(bench["paths"]["fp32"], data["test_u"])
                    except Exception as e:
                        rec["onnx_error"] = repr(e); print("ONNX export failed:", name, repr(e))
                    m.to(DEV)
                del m
            ALL.append(rec)
            t = rec["test_unique"]
            extra = f" novelF1={rec['test_novel']['macro_f1']:.4f}" if "test_novel" in rec else ""
            print(f"[{tag}] {name:16s} uniqF1={t['macro_f1']:.4f}{extra} recFPR={rec['test']['normal_fpr']:.4f} ({rec['train_sec']:.0f}s)")
            save_json(ALL, "results.json")
        if run.get("oracle"):   # majority label of each vector over all records (the information ceiling), per test vector
            maj_ = GRP["maj"]
            rec = dict(split=split_info, model="Oracle-majority", train_sec=0.0,
                       test=metrics(y_all[te], maj_.loc[H[te]].values), test_unique=metrics(y_all[te_u], maj_.loc[H[te_u]].values))
            ALL.append(rec); save_json(ALL, "results.json")
            print(f"[{tag}] Oracle-majority  uniqF1={rec['test_unique']['macro_f1']:.4f}")
        del data; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Summary tables and download bundle

# %%
rows = []
for r in ALL:
    t = r["test"]
    row = dict(split=r["split"]["mode"] + ("-weighted" if r["split"].get("weighted") else "") + ("-coarse" if r["split"].get("coarse_bins") else ""), features=r["split"]["feature_set"], seed=r["split"]["seed"], model=r["model"],
               macro_f1_novel=(round(100 * r["test_novel"]["macro_f1"], 2) if "test_novel" in r else None),
               macro_f1=round(100 * t["macro_f1"], 2), macro_f1_unique=round(100 * r["test_unique"]["macro_f1"], 2), mcc=round(t["mcc"], 4),
               balanced_acc=round(100 * t["balanced_acc"], 2), normal_fpr=round(100 * t["normal_fpr"], 3),
               params=r.get("params"), flops=r.get("flops_per_sample"),
               macro_f1_unique_permseq=(round(100 * r["test_unique_permseq"]["macro_f1"], 2) if "test_unique_permseq" in r else None))
    for prec in ["fp32", "int8_dynamic", "int8_static"]:
        if "onnx" in r and prec in r["onnx"]:
            row[f"{prec}_kb"] = round(r["onnx"][prec]["size_kb"], 1)
            row[f"{prec}_p99_ms_1thr"] = round(r["onnx"][prec]["threads1"]["p99"], 3)
            if f"{prec}_test" in r:
                row[f"{prec}_macro_f1"] = round(100 * r[f"{prec}_test"]["macro_f1"], 2)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(CFG["out_dir"], "summary.csv"), index=False)
pcf = pd.DataFrame([{**dict(split=r["split"]["mode"] + ("-coarse" if r["split"].get("coarse_bins") else ""), weighted=r["split"].get("weighted", False), features=r["split"]["feature_set"], seed=r["split"]["seed"], model=r["model"], eval=key),
                     **{k: round(100 * v, 2) for k, v in r[key]["per_class_f1"].items()}} for r in ALL for key in ("test_unique", "test_novel", "test_unique_permseq") if key in r])
pcf.to_csv(os.path.join(CFG["out_dir"], "per_class_f1.csv"), index=False)
att = [dict(split=r["split"]["mode"], features=r["split"]["feature_set"], seed=r["split"]["seed"], **a)
       for r in ALL for a in r.get("token_attribution", [])]
if att: pd.DataFrame(att).to_csv(os.path.join(CFG["out_dir"], "palt_token_attribution.csv"), index=False)
agg = summary.groupby(["split", "features", "model"])[["macro_f1", "macro_f1_unique", "macro_f1_novel", "mcc", "balanced_acc", "normal_fpr"]].agg(["mean", "std"]).round(2)
agg.to_csv(os.path.join(CFG["out_dir"], "summary_mean_std.csv")); print(agg.to_string())
env = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None, cpu=platform.processor(),
           cpu_count=os.cpu_count(), cfg=CFG)
save_json(env, "environment.json")
zp = os.path.join(os.path.dirname(CFG["out_dir"]), f"palt_v5_{STAGE}_results.zip")
MODEL_EXT = (".onnx", ".lgb.txt", ".xgb.json")
is_model = lambda f: f.endswith(MODEL_EXT) or os.path.basename(f).startswith("bench_inputs_")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(os.path.join(CFG["out_dir"], "*")):
        if not is_model(f): z.write(f, os.path.basename(f))   # logs only
mods = [f for f in glob.glob(os.path.join(CFG["out_dir"], "*")) if is_model(f) and "_int8" not in f]
if mods:   # FP32 models + real test inputs for the Docker edge benchmark (copy into edge_bench/models/)
    zm = os.path.join(os.path.dirname(CFG["out_dir"]), f"palt_v5_{STAGE}_models.zip")
    with zipfile.ZipFile(zm, "w", zipfile.ZIP_DEFLATED) as z:
        for f in mods: z.write(f, os.path.basename(f))
    print("Models for the edge benchmark:", zm, [os.path.basename(f) for f in mods])
print(summary.to_string(index=False))
print("\nDownload:", zp)
