# %% [markdown]
# # PALT main experiments v4 — Edge-IIoTset, leakage-aware protocol (Kaggle GPU)
# Code for "Beyond Shortcuts: A Leakage-Aware Audit and Benchmark of Intrusion Detectors for Smart-Factory IIoT Gateways".
#
# **Set `STAGE` in the first code cell before running:**
# * `"main"` — strict feature set, grouped random + grouped chronological splits, 7 models × 5 seeds (≈3–4 h)
# * `"ablation"` — raw-author / author / strict, grouped random split, 4 models × 3 seeds (≈4–5 h)
#
# v4 adds: grouped splitting by canonical feature vector (no vector in two sets, test keeps traffic
# multiplicity), a shortcut probe, the per-packet information ceiling, exact unique counts per class,
# and protocol-token attributions for PALT.
#
# **How to run:** Add Input → "Edge-IIoTset Cyber Security Dataset of IoT & IIoT"; GPU T4 x2; Internet ON;
# Save Version → Save & Run All. Download `palt_v4_<stage>_results.zip` from the Output tab.

# %%
import subprocess, sys, importlib.util
for pkg in ["onnxruntime", "onnx", "lightgbm"]:
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
STAGE = "main"   # "main" or "ablation"  <-- set before running
if STAGE == "main":
    CFG.update(runs=[("random", "strict"), ("temporal", "strict")], seeds=[0, 1, 2, 3, 4],
               models=["LightGBM", "MLP", "CNN1D", "FTTransformer", "PALT-Teacher", "PALT", "PALT-KD"])
else:
    CFG.update(runs=[("random", "raw-author"), ("random", "author"), ("random", "strict")], seeds=[0, 1, 2],
               models=["LightGBM", "MLP", "FTTransformer", "PALT"])
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

# ---- v4 analyses: shortcut probe, information ceiling, unique counts (written to analysis_v4.json) ----
from sklearn.tree import DecisionTreeClassifier
ANALYSIS = {}
FMT_COLS = [c for c in ["dns.qry.name.len", "mqtt.conack.flags", "mqtt.protoname", "mqtt.topic",
                        "http.request.method", "http.request.version", "http.referer"] if c in BASE_RAW.columns]
tok = {"0": 0, "0.0": 1, "0x00000000": 2}
Zf = np.stack([BASE_RAW[c].map(lambda v: tok.get(v, 3)).values for c in FMT_COLS], 1)
yf = BASE_RAW[LABEL].map(cls2id).values
rule = BASE_RAW["dns.qry.name.len"].eq("0").values
ANALYSIS["one_field_rule_dns_len_eq_0_is_normal"] = dict(
    accuracy=float(((rule) == (yf == cls2id["Normal"])).mean()),
    attacks_called_normal=int((rule & (yf != cls2id["Normal"])).sum()))
rs = np.random.default_rng(0).permutation(len(yf)); cut = int(0.7 * len(yf))
dt = DecisionTreeClassifier(max_depth=12, class_weight="balanced", random_state=0).fit(Zf[rs[:cut]], yf[rs[:cut]])
pz = dt.predict(Zf[rs[cut:]]); yz = yf[rs[cut:]]
ANALYSIS["shortcut_probe_spelling_only"] = dict(
    features=FMT_COLS, note="decision tree on the spelling of empty values only (0 / 0.0 / 0x00000000 / other)",
    binary_accuracy=float(((pz == cls2id["Normal"]) == (yz == cls2id["Normal"])).mean()),
    macro_f1=float(f1_score(yz, pz, average="macro")),
    per_class_f1={classes[i]: float(v) for i, v in enumerate(f1_score(yz, pz, average=None, labels=range(len(classes))))})
del Zf
for fs_name, B, cols in [("raw-author", BASE_RAW, AUTHOR_KEPT), ("author", BASE_CANON, AUTHOR_KEPT),
                         ("strict", BASE_CANON, [c for c in AUTHOR_KEPT if c not in STRICT_EXTRA])]:
    st = ceiling_stats(B, cols)
    for key in ("unique_vector_label_pairs_by_class", "records_in_conflicting_vectors_by_class", "ceiling_recall_by_class"):
        st[key] = {str(k): v for k, v in st[key].items()}
    ANALYSIS[f"ceiling_{fs_name}"] = st
    print(fs_name, {k: st[k] for k in ("n_unique_vectors", "vectors_with_conflict", "ceiling_accuracy")})
save_json(ANALYSIS, "analysis_v4.json")
print(json.dumps({k: v for k, v in ANALYSIS.items() if not k.startswith("ceiling")}, indent=1))

def prepare(feature_set):
    """Grouped protocol: every distinct canonical feature vector is one group; groups (not records) are split."""
    global df, y_all, kept, cat_cols, num_cols, group_of, kept_groups, H, GRP
    kept = [c for c in AUTHOR_KEPT if not (feature_set == "strict" and c in STRICT_EXTRA)]
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

# %% [markdown]
# ## 3. Models

# %%
def num_idx(g): return [i for i, c in enumerate(num_cols) if group_of.get(c) == g]
def cat_idx(g): return [i for i, c in enumerate(cat_cols) if group_of.get(c) == g]

class CBFocal(nn.Module):
    """Class-balanced focal loss (Cui et al. 2019 weights, Lin et al. 2017 focusing)."""
    def __init__(self, counts, beta, gamma):
        super().__init__()
        eff = (1 - np.power(beta, counts)) / (1 - beta)
        w = (1 / eff); w = w / w.sum() * len(counts)
        self.register_buffer("w", torch.tensor(w, dtype=torch.float32)); self.g = gamma
    def forward(self, logits, y):
        logp = F.log_softmax(logits, 1); p = logp.exp()
        lp = logp.gather(1, y[:, None]).squeeze(1); pt = p.gather(1, y[:, None]).squeeze(1)
        return (-(self.w[y]) * (1 - pt) ** self.g * lp).mean()

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
    def __init__(self, d, heads, attn_mask=None):
        super().__init__()
        self.local = nn.Sequential(nn.Conv1d(d, d, 3, padding=1, groups=d), nn.Conv1d(d, d, 1), nn.GELU())
        self.n1 = nn.LayerNorm(d); self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        self.n2 = nn.LayerNorm(d); self.ffn = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.fuse = nn.Linear(2 * d, d)
        self.register_buffer("mask", attn_mask if attn_mask is not None else torch.zeros(0), persistent=False)
    def forward(self, t):
        loc = t + self.local(t.transpose(1, 2)).transpose(1, 2)
        h = self.n1(loc)
        m = self.mask if self.mask.numel() else None
        g = loc + self.attn(h, h, h, attn_mask=m, need_weights=False)[0]
        g = g + self.ffn(self.n2(g))
        return self.fuse(torch.cat([loc, g], -1)) + t

class PALT(nn.Module):
    """Protocol-Aware Local–global Transformer.
    tokens = [CLS] + one token per protocol group; absent protocols use a learned embedding."""
    def __init__(self, n_num, n_cat, n_cls, d=32, heads=4, blocks=2, windowed=True):
        super().__init__()
        self.groups = kept_groups; G = len(self.groups)
        self.nidx = [num_idx(g) for g in self.groups]; self.cidx = [cat_idx(g) for g in self.groups]
        for gi, ids in enumerate(self.nidx):
            self.register_buffer(f"ni{gi}", torch.tensor(ids, dtype=torch.long), persistent=False)
        self.proj = nn.ModuleList([nn.Linear(max(len(i), 1), d) for i in self.nidx])
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
        self.blocks = nn.ModuleList([LocalGlobalBlock(d, heads, win if (windowed and i % 2 == 0) else None)
                                     for i in range(blocks)])
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, n_cls))
    def tokens(self, X, C, P):
        B = X.size(0); toks = []
        for gi in range(len(self.groups)):
            ids = getattr(self, f"ni{gi}")
            x = X.index_select(1, ids) if ids.numel() else X.new_zeros(B, 1)
            t = self.proj[gi](x)
            for ci in self.cidx[gi]:
                t = t + self.cats[ci](C[:, ci])
            toks.append(t)
        T = torch.stack(toks, 1)
        pres = P[:, :, None]
        T = pres * T + (1 - pres) * self.absent[None]
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
def loaders(X, C, P, y, bs, shuffle):
    ds = torch.utils.data.TensorDataset(torch.from_numpy(X), torch.from_numpy(C), torch.from_numpy(P), torch.from_numpy(y))
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=shuffle, drop_last=False, num_workers=2, pin_memory=(DEV == "cuda"))

@torch.no_grad()
def predict(m, dl):
    m.eval(); out = []
    for X, C, P, _ in dl:
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

def train_nn(name, data, seed, teacher=None):
    set_seed(seed)
    (Xtr, Ctr, Ptr, ytr), (Xva, Cva, Pva, yva) = data["train"], data["val"]
    m = build(name, Xtr.shape[1], Ctr.shape[1], len(classes)).to(DEV)
    crit = CBFocal(np.bincount(ytr, minlength=len(classes)).clip(1), CFG["cb_beta"], CFG["focal_gamma"]).to(DEV)
    lr = CFG["lr_teacher"] if name == "PALT-Teacher" else CFG["lr"]
    opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=CFG["weight_decay"])
    dtr = loaders(Xtr, Ctr, Ptr, ytr, CFG["batch_size"], True); dva = loaders(Xva, Cva, Pva, yva, 8192, False)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=CFG["epochs"] * len(dtr))
    scaler = torch.amp.GradScaler(enabled=(DEV == "cuda"))
    best, best_state, bad, hist = -1, None, 0, []
    for ep in range(CFG["epochs"]):
        m.train(); t0 = time.time(); tot = 0
        for X, C, P, y in dtr:
            X, C, P, y = X.to(DEV, non_blocking=True), C.to(DEV), P.to(DEV), y.to(DEV)
            with torch.autocast(DEV, enabled=(DEV == "cuda")):
                logits = m(X, C, P); loss = crit(logits.float(), y)
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
for mode, fset in CFG["runs"]:
    PREP[fset] = PREP.get(fset) or prepare(fset)
    prepare(fset)
    for seed in CFG["seeds"]:
        tr, va, te, te_u, how = make_split(mode, seed)
        enc = Encoder().fit(df.iloc[tr])
        data = {}
        for part, ids in [("train", tr), ("val", va), ("test", te), ("test_u", te_u)]:
            X, C, P = enc.transform(df.iloc[ids]); data[part] = (X, C, P, y_all[ids])
        split_info = dict(mode=mode, feature_set=fset, prep=PREP[fset], how=how, seed=seed, n_train=len(tr), n_val=len(va), n_test=len(te), n_test_unique=len(te_u),
                          test_counts={classes[k]: int((y_all[te] == k).sum()) for k in range(len(classes))},
                          test_unique_counts={classes[k]: int((y_all[te_u] == k).sum()) for k in range(len(classes))},
                          train_counts={classes[k]: int((y_all[tr] == k).sum()) for k in range(len(classes))})
        tag = f"{mode}-{fset}_s{seed}"
        np.savez_compressed(os.path.join(CFG["out_dir"], f"split_{tag}.npz"),
                            train=df["_row"].values[tr], val=df["_row"].values[va], test=df["_row"].values[te], test_unique=df["_row"].values[te_u])
        print(f"\n=== {tag}: {how} | train {len(tr)} val {len(va)} test {len(te)}")
        dte = loaders(*data["test"], 8192, False); dteu = loaders(*data["test_u"], 8192, False)
        # 2,000 real test records for the Docker edge benchmark (inputs only, no labels)
        rs = np.random.default_rng(seed).choice(len(data["test"][3]), min(2000, len(data["test"][3])), replace=False)
        if seed == CFG["export_seed"]: np.savez_compressed(os.path.join(CFG["out_dir"], f"bench_inputs_{tag}.npz"),
                            X=data["test"][0][rs], C=data["test"][1][rs], P=data["test"][2][rs])
        teacher = None
        for name in CFG["models"]:
            t0 = time.time(); rec = dict(split=split_info, model=name)
            if name == "LightGBM":
                clf = train_lgbm(data, seed)
                pred = clf.predict(np.hstack([data["test"][0], data["test"][1]]))
                pu = clf.predict(np.hstack([data["test_u"][0], data["test_u"][1]]))
                rec.update(test=metrics(data["test"][3], pred), test_unique=metrics(data["test_u"][3], pu), train_sec=time.time() - t0,
                           n_trees=int(clf.booster_.num_trees()))
                if seed == CFG["export_seed"]:  # model file for the Docker edge benchmark
                    clf.booster_.save_model(os.path.join(CFG["out_dir"], f"{tag}_LightGBM.lgb.txt"))
            else:
                if name == "PALT-KD":
                    if teacher is None: continue
                    m, hist = train_nn(name, data, seed, teacher=teacher)
                else:
                    m, hist = train_nn(name, data, seed)
                pred = predict(m, dte).argmax(1).numpy(); pu = predict(m, dteu).argmax(1).numpy()
                rec.update(test=metrics(data["test"][3], pred), test_unique=metrics(data["test_u"][3], pu),
                           history=hist, train_sec=time.time() - t0,
                           params=n_params(m),
                           flops_per_sample=flops_per_sample(m, *[torch.from_numpy(a) for a in data["test"][:3]]))
                if name == "PALT-Teacher":
                    teacher = m.to(DEV).eval()
                if name == "PALT":
                    try: rec["token_attribution"] = palt_token_attribution(m, data)
                    except Exception as e: rec["attribution_error"] = repr(e)
                if seed == CFG["export_seed"] and name in ("PALT", "PALT-KD", "PALT-Teacher", "FTTransformer", "MLP", "CNN1D"):
                    try:
                        rng = np.random.default_rng(seed); ci = rng.choice(len(data["train"][3]), 512, replace=False)
                        bench = export_and_bench(m, name, data["test"][:3], tag, [a[ci] for a in data["train"][:3]])
                        rec["onnx"] = {k: v for k, v in bench.items() if k != "paths"}
                        for prec, path in bench["paths"].items():
                            if True:  # includes fp32 ONNX, to separate export error from quantization error
                                rec[f"{prec}_test"] = int8_accuracy(path, data["test"])
                    except Exception as e:
                        rec["onnx_error"] = repr(e)
                    m.to(DEV)
            ALL.append(rec)
            t = rec["test"]
            print(f"[{tag}] {name:14s} macroF1={t['macro_f1']:.4f} uniqF1={rec['test_unique']['macro_f1']:.4f} MCC={t['mcc']:.4f} "
                  f"MITM={t['per_class_f1'].get('MITM', float('nan')):.3f} "
                  f"Fingerprinting={t['per_class_f1'].get('Fingerprinting', float('nan')):.3f} "
                  f"params={rec.get('params', '-')} ({rec['train_sec']:.0f}s)")
            save_json(ALL, "results.json")
        del data; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 7. Summary tables and download bundle

# %%
rows = []
for r in ALL:
    t = r["test"]
    row = dict(split=r["split"]["mode"], features=r["split"]["feature_set"], seed=r["split"]["seed"], model=r["model"],
               macro_f1=round(100 * t["macro_f1"], 2), macro_f1_unique=round(100 * r["test_unique"]["macro_f1"], 2), mcc=round(t["mcc"], 4),
               balanced_acc=round(100 * t["balanced_acc"], 2), normal_fpr=round(100 * t["normal_fpr"], 3),
               params=r.get("params"), flops=r.get("flops_per_sample"))
    for prec in ["fp32", "int8_dynamic", "int8_static"]:
        if "onnx" in r and prec in r["onnx"]:
            row[f"{prec}_kb"] = round(r["onnx"][prec]["size_kb"], 1)
            row[f"{prec}_p99_ms_1thr"] = round(r["onnx"][prec]["threads1"]["p99"], 3)
            if f"{prec}_test" in r:
                row[f"{prec}_macro_f1"] = round(100 * r[f"{prec}_test"]["macro_f1"], 2)
    rows.append(row)
summary = pd.DataFrame(rows)
summary.to_csv(os.path.join(CFG["out_dir"], "summary.csv"), index=False)
pcf = pd.DataFrame([{**dict(split=r["split"]["mode"], features=r["split"]["feature_set"], seed=r["split"]["seed"], model=r["model"]),
                     **{k: round(100 * v, 2) for k, v in r["test"]["per_class_f1"].items()}} for r in ALL])
pcf.to_csv(os.path.join(CFG["out_dir"], "per_class_f1.csv"), index=False)
att = [dict(split=r["split"]["mode"], features=r["split"]["feature_set"], seed=r["split"]["seed"], **a)
       for r in ALL for a in r.get("token_attribution", [])]
if att: pd.DataFrame(att).to_csv(os.path.join(CFG["out_dir"], "palt_token_attribution.csv"), index=False)
agg = summary.groupby(["split", "features", "model"])[["macro_f1", "macro_f1_unique", "mcc", "balanced_acc", "normal_fpr"]].agg(["mean", "std"]).round(2)
agg.to_csv(os.path.join(CFG["out_dir"], "summary_mean_std.csv")); print(agg.to_string())
env = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None, cpu=platform.processor(),
           cpu_count=os.cpu_count(), cfg=CFG)
save_json(env, "environment.json")
zp = os.path.join(os.path.dirname(CFG["out_dir"]), f"palt_v4_{STAGE}_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(os.path.join(CFG["out_dir"], "*")):
        z.write(f, os.path.basename(f))  # includes .onnx / .lgb.txt / bench_inputs for the edge benchmark
print(summary.to_string(index=False))
print("\nDownload:", zp)
