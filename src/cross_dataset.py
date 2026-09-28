# %% [markdown]
# # Cross-dataset validation v1 — X-IIoTID and CICIoT2023 (Kaggle)
# Companion to `src/main_benchmark.py` (Edge-IIoTset).
#
# **Set `DATASET` and `STAGE` in the first code cell before running:**
# * `DATASET = "xiiotid"` or `"ciciot2023"`
# * `STAGE = "audit"`  — CPU is enough (≈10–40 min). Shortcut audit only: spelling probe, single-feature probes,
#   identifier screen, duplicates / conflicting vectors / information ceiling, record-split overlap.
# * `STAGE = "bench"` — GPU T4. Three nested settings (common practice → grouped → strict) ×
#   LightGBM / MLP / FT-Transformer / PALT × 3 seeds (≈2–4 h).
#
# **Inputs:** Add Input → the dataset (X-IIoTID or CICIoT2023, CSV or Parquet). The loader finds the files by
# their columns, so the folder layout does not matter. Download `cross_<dataset>_<stage>_results.zip`.

# %%
import subprocess, sys, importlib.util
for pkg in ["lightgbm", "pyarrow"]:
    if importlib.util.find_spec(pkg) is None:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", pkg], check=False)

import os, re, gc, glob, json, math, time, random, platform, warnings, zipfile
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (f1_score, matthews_corrcoef, balanced_accuracy_score, precision_recall_fscore_support)
from sklearn.tree import DecisionTreeClassifier
warnings.filterwarnings("ignore")

DATASET = "xiiotid"     # "xiiotid" or "ciciot2023"   <-- set before running
STAGE = "bench"         # "audit" (CPU, audit only) or "bench" (GPU, audit + benchmark)  <-- set before running

CFG = dict(
    input_root="/kaggle/input" if os.path.isdir("/kaggle/input") else "data_cross",
    out_dir="/kaggle/working/results" if os.path.isdir("/kaggle/working") else "results_cross",
    label_col=None,             # auto: X-IIoTID -> finest of class1/class2/class3; CICIoT2023 -> label (34 classes)
    per_class_cap=30_000,       # CICIoT2023 modelling sample: at most this many records per class (bottom-k random keys)
    chunk_rows=2_000_000,       # streaming chunk size for large CSVs
    probe_rows=400_000,         # rows used by single-feature / spelling probes
    seeds=[0, 1, 2],
    models=["LightGBM", "MLP", "FTTransformer", "PALT"],
    settings=["common", "grouped", "strict"],   # e.g. ["strict"] to rerun only strict after filling STRICT_EXTRA
    epochs=30, batch_size=2048, lr=1e-3, lr_teacher=5e-4, weight_decay=1e-4, patience=6, clip=5.0,
    d_student=32, d_teacher=128, heads=4, blocks_student=2, blocks_teacher=4,
    kd_T=4.0, kd_alpha=0.5, cb_beta=0.9999, focal_gamma=2.0, cat_vocab=64,
)
CFG.update(json.loads(os.environ.get("CROSS_CFG", "{}")))   # local override for smoke tests
os.makedirs(CFG["out_dir"], exist_ok=True)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print("dataset:", DATASET, "| stage:", STAGE, "| device:", DEV, torch.cuda.get_device_name(0) if DEV == "cuda" else platform.processor())
print("torch", torch.__version__, "| pandas", pd.__version__, "| python", sys.version.split()[0])

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def save_json(obj, name):
    with open(os.path.join(CFG["out_dir"], name), "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, default=lambda o: o.item() if hasattr(o, "item") else str(o))

# %% [markdown]
# ## 0. Dataset definitions
# * **Identifier rule (fixed before looking at results):** a field is an identifier when its name denotes an address,
#   port, timestamp/date, or session/stream index. `common` drops addresses and timestamps (what most papers do);
#   `strict` additionally drops ports and every field flagged by the audit (filled in `STRICT_EXTRA` after the audit).
# * **Token groups for PALT:** protocol layer where fields are protocol-specific, data source / function where
#   fields are aggregates (flow statistics, host resources, logs). Windows group tokens of the same layer or source.

# %%
ID_RE = {
    "address": r"(^|_)(src|dst|scr|des|source|dest|destination)?_?ip$|ip_?addr|(^|_)mac$",
    "port": r"(^|_)(src|dst|scr|des|source|dest|destination)?_?port(_|$)",
    "time": r"^(date|time|timestamp|ts|stime|ltime|frame_time|start_time|end_time)$|timestamp|(^|_)date(_|$)",
    "session": r"(^|_)(stream|session|flow_?id|conn_?id|uid|seq|ack_raw|checksum)(_|$)",
}
def id_kind(c):
    n = c.lower().replace(" ", "_").replace("-", "_").replace(".", "_")
    if n.startswith(("is_", "bad_", "avg_", "std_")):   # indicators / aggregates, not identifiers
        return None
    for k, rx in ID_RE.items():
        if re.search(rx, n): return k
    return None

DS = {
    "xiiotid": dict(
        signature=["class1"],
        label_candidates=["class1", "class2", "class3"],
        # (token, regex on lower-case column name); first match wins; order = layer / source order
        groups=[("conn", r"protocol|service|duration|conn_state|missed|port"),
                ("flags", r"(^|_)(is_syn|syn|ack|fin|rst)(_|$)|fin or rst|with_payload|checksum"),
                ("volume", r"byte|pkt|packet|paket|rate|ratio"),
                ("cpu", r"user_time|nice_time|system_time|iowait|ideal|idle|ldavg|proc|cswch"),
                ("io_mem", r"tps|kbmem|mem"),
                ("logs", r"ossec|alert|login|file_activity|process_activity|read_write|privileged")],
        windows=[["conn", "flags", "volume"], ["cpu", "io_mem"], ["logs"]],
        common_drop_kinds=("address", "time"),
    ),
    "ciciot2023": dict(
        signature=["flow_duration", "label"],
        label_candidates=["label"],
        groups=[("link_net", r"^(arp|icmp|ipv|llc)$"),
                ("ip_hdr", r"^(header_length|protocol type|duration)$"),
                ("transport", r"^(tcp|udp)$"),
                ("tcp_flags", r"_flag_number$"),
                ("tcp_counts", r"^(ack|syn|fin|urg|rst)_count$"),
                ("app", r"^(http|https|dns|telnet|smtp|ssh|irc|dhcp)$"),
                ("rate", r"^(flow_duration|rate|srate|drate|iat)$"),
                ("size", r"^(tot sum|min|max|avg|std|tot size|number|magnitue|radius|covariance|variance|weight)$")],
        windows=[["link_net", "ip_hdr"], ["transport", "tcp_flags", "tcp_counts"], ["app"], ["rate", "size"]],
        common_drop_kinds=("address", "time"),
    ),
}[DATASET]
# Filled in after the audit stage (fields flagged as identifiers or capture artifacts). Ports are always removed in strict.
STRICT_EXTRA = {"xiiotid": [], "ciciot2023": []}[DATASET]
EMPTY = {"", "0", "0.0", "0.00", "nan", "NaN", "None", "-", "0x00000000", "?"}

# %% [markdown]
# ## 1. Find and load the data

# %%
def find_files():
    fs = [f for f in glob.glob(os.path.join(CFG["input_root"], "**", "*"), recursive=True)
          if f.lower().endswith((".csv", ".parquet", ".pq"))]
    hits = []
    for f in sorted(fs):
        try:
            if f.lower().endswith(".csv"):
                cols = pd.read_csv(f, nrows=0).columns
            else:
                import pyarrow.parquet as pq
                cols = pq.ParquetFile(f).schema.names
        except Exception:
            continue
        low = [c.strip().lower() for c in cols]
        if all(s in low for s in DS["signature"]):
            hits.append(f)
    if not hits:
        seen = [os.path.join(r, d) for r, ds, _ in os.walk(CFG["input_root"]) for d in ds][:40]
        raise FileNotFoundError(f"No {DATASET} files found under {CFG['input_root']}. Add Input -> the dataset. Folders: {seen}")
    return hits

FILES = find_files()
print(f"{len(FILES)} file(s):", [os.path.relpath(f, CFG['input_root']) for f in FILES[:5]], "..." if len(FILES) > 5 else "")

def iter_chunks(as_str=True):
    """Yields DataFrames of stripped strings (as_str) with lower-cased-stripped column mapping preserved."""
    for f in FILES:
        if f.lower().endswith(".csv"):
            it = pd.read_csv(f, dtype=str if as_str else None, keep_default_na=False, chunksize=CFG["chunk_rows"], low_memory=False)
        else:
            d = pd.read_parquet(f)
            it = [d.iloc[i:i + CFG["chunk_rows"]] for i in range(0, len(d), CFG["chunk_rows"])]
        for ch in it:
            ch = ch.copy(); ch.columns = [c.strip() for c in ch.columns]
            if as_str:
                for c in ch.columns:
                    ch[c] = ch[c].astype(str).str.strip()
            yield f, ch

def pick_label(cols, sample):
    cands = [c for c in cols if c.lower() in DS["label_candidates"]]
    if CFG["label_col"]: return CFG["label_col"], cands
    nun = {c: sample[c].nunique() for c in cands}
    return max(cands, key=lambda c: (nun[c] if nun[c] <= 60 else -1)), cands

def canonicalize(col):
    """'0'/'0.0'/'0x00000000'/'' -> 0; hex -> int; numeric strings -> float; other text kept (numeric spellings unified)."""
    s = col.where(~col.isin(EMPTY), "0")
    hexm = s.str.match(r"^0x[0-9a-fA-F]+$")
    if hexm.any():
        s = s.copy(); s[hexm] = s[hexm].map(lambda v: str(int(v, 16)))
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().mean() >= 0.99:
        return num.fillna(0).astype(np.float64)
    out = s.copy(); ok = num.notna()
    out[ok] = num[ok].map(lambda v: f"{v:.10g}")
    return out

def fmt_token(v):
    """Spelling class of a raw value; used to detect export-format shortcuts."""
    if v in ("", "nan", "NaN", "None"): return "empty"
    if v in ("-", "?"): return v
    if re.fullmatch(r"-?\d+", v): return "int"
    if re.fullmatch(r"-?\d+\.\d*", v): return "float" if not re.fullmatch(r"-?\d+\.0+", v) else "int_as_float"
    if re.fullmatch(r"-?\d+(\.\d+)?[eE][-+]?\d+", v): return "sci"
    if re.fullmatch(r"0x[0-9a-fA-F]+", v): return "hex"
    return "text"

t0 = time.time()
BIG = DATASET == "ciciot2023"
AUDIT = {"dataset": DATASET, "files": [os.path.relpath(f, CFG["input_root"]) for f in FILES]}
if not BIG:
    RAW = pd.concat([c for _, c in iter_chunks(True)], ignore_index=True)
    LABEL, LABEL_CANDS = pick_label(RAW.columns, RAW)
    AUDIT["label_levels"] = {c: RAW[c].value_counts().to_dict() for c in LABEL_CANDS}
    FULL_STATS = None
else:
    # one streaming pass over the full data: class counts, (vector, label) counts on canonical values,
    # and a per-class capped random sample (bottom-k random keys) kept as raw strings for modelling and probes.
    rng = np.random.default_rng(12345); keep = None; vc = []; n_total = 0; LABEL = None
    for f, ch in iter_chunks(True):
        if LABEL is None:
            LABEL, LABEL_CANDS = pick_label(ch.columns, ch)
            FEATS0 = [c for c in ch.columns if c not in LABEL_CANDS]
        n_total += len(ch)
        canon = pd.DataFrame({c: pd.to_numeric(ch[c], errors="coerce").fillna(0).astype(np.float64) for c in FEATS0})
        h = pd.util.hash_pandas_object(canon, index=False).values
        vc.append(pd.DataFrame({"h": h, "y": ch[LABEL].values}).value_counts().rename("n").reset_index())
        ch["_key"] = rng.random(len(ch))
        keep = ch if keep is None else pd.concat([keep, ch], ignore_index=True)
        keep = keep.sort_values("_key").groupby(LABEL, sort=False).head(CFG["per_class_cap"]).reset_index(drop=True)
        print(f"  streamed {n_total:,} rows ({time.time()-t0:.0f}s)")
        del canon, ch; gc.collect()
    FULL_STATS = pd.concat(vc).groupby(["h", "y"])["n"].sum().reset_index(); del vc; gc.collect()
    RAW = keep.drop(columns="_key").reset_index(drop=True); del keep
    AUDIT["full_rows"] = int(n_total)
    AUDIT["label_levels"] = {LABEL: FULL_STATS.groupby("y")["n"].sum().sort_values(ascending=False).to_dict()}
RAW["_row"] = np.arange(len(RAW))
classes = sorted(RAW[LABEL].unique()); cls2id = {c: i for i, c in enumerate(classes)}
NORMAL = next((c for c in classes if re.search(r"normal|benign", c, re.I)), None)
FEATS = [c for c in RAW.columns if c not in LABEL_CANDS and c != "_row"]
KIND = {c: id_kind(c) for c in FEATS}
COMMON = [c for c in FEATS if KIND[c] not in DS["common_drop_kinds"]]
print(f"label={LABEL} ({len(classes)} classes, normal={NORMAL}); features={len(FEATS)}; common={len(COMMON)}; "
      f"rows in memory={len(RAW):,} ({time.time()-t0:.0f}s)")
print("identifier-like by name:", {c: k for c, k in KIND.items() if k})
AUDIT.update(label=LABEL, normal_class=NORMAL, n_classes=len(classes), rows_in_memory=int(len(RAW)),
             features=FEATS, identifier_by_name={c: k for c, k in KIND.items() if k}, common_features=COMMON,
             class_counts_in_memory=RAW[LABEL].value_counts().to_dict())

# token groups for PALT
def group_map(cols):
    gm = {}
    for c in cols:
        n = c.strip().lower()
        for g, rx in DS["groups"]:
            if re.search(rx, n): gm[c] = g; break
        else:
            gm[c] = "other"
    return gm
AUDIT["palt_groups"] = {}
for c, g in group_map(COMMON).items():
    AUDIT["palt_groups"].setdefault(g, []).append(c)
print("PALT token groups:", AUDIT["palt_groups"])

CANON = RAW[[LABEL, "_row"]].copy()
for c in FEATS:
    CANON[c] = canonicalize(RAW[c])
IS_NUM = {c: pd.api.types.is_float_dtype(CANON[c]) for c in FEATS}
NUM_RATIO_RAW = {c: pd.to_numeric(RAW[c], errors="coerce").notna().mean() for c in FEATS}

def feature_hash(frame):
    return pd.util.hash_pandas_object(frame, index=False).values

def ceiling_from_counts(t):
    """t: DataFrame h, y, n. Information ceiling = accuracy of predicting each vector's majority label."""
    tot = t.groupby("h")["n"].transform("sum"); mx = t.groupby("h")["n"].transform("max")
    t = t.assign(is_max=t["n"] == mx)
    maj = t[t["is_max"]].drop_duplicates("h")[["h", "y"]].rename(columns={"y": "maj"})
    t = t.merge(maj, on="h"); n_lab = t.groupby("h")["y"].transform("nunique")
    out = dict(n_records=int(t["n"].sum()), n_unique_vectors=int(t["h"].nunique()),
               unique_vector_label_pairs_by_class=t.groupby("y").size().to_dict(),
               vectors_with_conflict=int((t.groupby("h")["y"].nunique() > 1).sum()),
               records_in_conflicting_vectors_by_class=t[n_lab > 1].groupby("y")["n"].sum().to_dict(),
               duplicate_share=float(1 - t["h"].nunique() / t["n"].sum()),
               ceiling_accuracy=float(t.loc[t["y"] == t["maj"], "n"].sum() / t["n"].sum()),
               ceiling_recall_by_class=(t[t["y"] == t["maj"]].groupby("y")["n"].sum() / t.groupby("y")["n"].sum()).fillna(0).to_dict())
    return json.loads(json.dumps(out, default=lambda o: o.item() if hasattr(o, "item") else str(o)))

def ceiling_stats(B, cols):
    t = pd.DataFrame({"h": feature_hash(B[cols]), "y": B[LABEL].values}).value_counts().rename("n").reset_index()
    return ceiling_from_counts(t)

# %% [markdown]
# ## 2. Shortcut audit (both stages write audit.json; the bench stage reuses nothing from it at run time)

# %%
nP = min(len(RAW), CFG["probe_rows"])
rsP = np.random.default_rng(0).permutation(len(RAW))[:nP]
P_RAW, P_CAN = RAW.iloc[rsP], CANON.iloc[rsP]
yP = P_RAW[LABEL].map(cls2id).values; cutP = int(0.7 * nP)
NID = cls2id.get(NORMAL, -1)

def probe(Z, name, depth=12):
    """Decision tree on Z (probe rows, 70/30 record split). Returns macro-F1 and binary normal-vs-attack accuracy."""
    dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=5, class_weight="balanced", random_state=0)
    dt.fit(Z[:cutP], yP[:cutP]); p = dt.predict(Z[cutP:]); y = yP[cutP:]
    return dict(probe=name, macro_f1=float(f1_score(y, p, average="macro")),
                binary_acc=float(((p == NID) == (y == NID)).mean()) if NID >= 0 else None,
                per_class_f1={classes[i]: float(v) for i, v in enumerate(f1_score(y, p, average=None, labels=range(len(classes)), zero_division=0))})

def codes(s):
    return s.to_numpy(dtype=np.float64) if pd.api.types.is_numeric_dtype(s) else pd.factorize(s)[0].astype(np.float64)

# 2a. spelling (export-format) probe: spelling tokens vs value classes of the same fields
def value_class(v):
    try:
        x = float(v) if not v.startswith("0x") else float(int(v, 16))
    except Exception:
        return "text" if v not in EMPTY else "zero"
    return "zero" if x == 0 else ("int" if float(x).is_integer() else "nonint")
spell_cols = [c for c in FEATS if P_RAW[c].map(fmt_token).nunique() > 1]
Zs = np.stack([pd.factorize(P_RAW[c].map(fmt_token))[0] for c in spell_cols], 1) if spell_cols else None
Zv = np.stack([pd.factorize(P_RAW[c].map(value_class))[0] for c in spell_cols], 1) if spell_cols else None
if spell_cols:
    ps, pv = probe(Zs, "spelling_tokens"), probe(Zv, "value_classes")
    AUDIT["spelling_probe"] = dict(columns=spell_cols, spelling=ps, value_class=pv,
                                   spelling_gain_macro_f1=ps["macro_f1"] - pv["macro_f1"])
    # zero written in several ways, by class
    zs = {}
    for c in FEATS:
        z = P_RAW[c][P_RAW[c].map(value_class) == "zero"]
        if z.nunique() > 1:
            ct = pd.crosstab(z, P_RAW.loc[z.index, LABEL])
            zs[c] = {str(k): {str(a): int(b) for a, b in r.items() if b > 0} for k, r in ct.iterrows()}
    AUDIT["zero_spelling_by_class"] = zs
    print(f"spelling probe macroF1={ps['macro_f1']:.3f} vs value-class {pv['macro_f1']:.3f}; "
          f"columns with several zero spellings: {list(zs)[:10]}")
if DATASET == "ciciot2023" and FILES[0].lower().endswith((".parquet", ".pq")):
    AUDIT["spelling_probe_note"] = "input is Parquet (typed); raw spellings are not available, spelling probe not meaningful"

# 2b. single-feature probes and identifier screen
sf = []
for c in FEATS:
    z = codes(P_CAN[c])[:, None]
    r = probe(z, c, depth=10); r.pop("per_class_f1")
    r.update(kind=KIND[c], distinct_ratio=float(RAW[c].nunique() / len(RAW)), distinct=int(RAW[c].nunique()))
    sf.append(r)
sf = sorted(sf, key=lambda r: -r["macro_f1"])
AUDIT["single_feature_probes"] = sf
print("top single-feature probes:")
for r in sf[:12]:
    print(f"  {r['probe'][:28]:28s} macroF1={r['macro_f1']:.3f} bin={r['binary_acc'] if r['binary_acc'] is None else round(r['binary_acc'],3)} "
          f"distinct={r['distinct']:>9,} kind={r['kind']}")

# 2c. duplicates, conflicting vectors, information ceiling, record-split overlap
PORT_SESSION = [c for c in COMMON if KIND[c] in ("port", "session")]
SETS = {"all_fields": FEATS, "common": COMMON, "common_minus_ports_sessions": [c for c in COMMON if c not in PORT_SESSION]}
AUDIT["ceiling"] = {}
for nm, cols in SETS.items():
    st = ceiling_stats(CANON, cols); AUDIT["ceiling"][nm] = st
    print(f"ceiling[{nm}] records={st['n_records']:,} unique={st['n_unique_vectors']:,} dup_share={st['duplicate_share']:.3f} "
          f"conflicts={st['vectors_with_conflict']:,} ceiling_acc={st['ceiling_accuracy']:.4f}")
if FULL_STATS is not None:
    st = ceiling_from_counts(FULL_STATS.copy()); AUDIT["ceiling"]["full_data_all_fields"] = st
    print(f"ceiling[full data, all fields] records={st['n_records']:,} unique={st['n_unique_vectors']:,} "
          f"dup_share={st['duplicate_share']:.3f} conflicts={st['vectors_with_conflict']:,} ceiling_acc={st['ceiling_accuracy']:.4f}")

def expected_overlap(t, test_frac=0.2):
    """Expected share of test records whose exact feature vector also occurs in the training part of a random
    record split (train share 1-test_frac, validation ignored): a record with n_h copies in total is 'seen' unless
    all other n_h-1 copies fall outside training."""
    nh = t.groupby("h")["n"].transform("sum")
    p_seen = 1 - np.power(test_frac, nh - 1)
    t = t.assign(seen=t["n"] * p_seen)
    by = (t.groupby("y")["seen"].sum() / t.groupby("y")["n"].sum()).to_dict()
    return dict(overall=float(t["seen"].sum() / t["n"].sum()), by_class={str(k): float(v) for k, v in by.items()})
cnt = lambda cols: pd.DataFrame({"h": feature_hash(CANON[cols]), "y": CANON[LABEL].values}).value_counts().rename("n").reset_index()
AUDIT["record_split_overlap"] = {nm: expected_overlap(cnt(cols)) for nm, cols in SETS.items()}
if FULL_STATS is not None:
    AUDIT["record_split_overlap"]["full_data_all_fields"] = expected_overlap(FULL_STATS.copy())
print("expected test records already seen in training under a random record split:",
      {k: round(v["overall"], 3) for k, v in AUDIT["record_split_overlap"].items()})

em = pd.concat([(CANON[c] == 0) if IS_NUM[c] else CANON[c].isin(EMPTY | {"0"}) for c in COMMON], axis=1).all(axis=1)
AUDIT["records_with_all_common_fields_empty_by_class"] = CANON.loc[em, LABEL].value_counts().to_dict()
save_json(AUDIT, "audit.json")
print(f"audit done ({time.time()-t0:.0f}s)")

# %% [markdown]
# ## 3. Settings for the benchmark
# * `common`  — common practice: addresses and timestamps removed, raw strings, stratified random **record** split
#   (no de-duplication; duplicates may fall into training and test).
# * `grouped` — same fields, canonical values, grouped split by feature vector (no vector in two sets).
# * `strict`  — grouped, and ports / session fields plus `STRICT_EXTRA` (from the audit) removed.

# %%
STRICT = [c for c in COMMON if c not in PORT_SESSION and c not in STRICT_EXTRA]
SETTINGS = [("common", "record", COMMON), ("grouped", "grouped", COMMON), ("strict", "grouped", STRICT)]
ALL_SETTINGS = list(SETTINGS)
if STRICT == COMMON:   # nothing to remove beyond the common fields -> strict would repeat grouped
    print("strict feature set equals common; strict setting skipped until STRICT_EXTRA is filled from the audit")
    SETTINGS = SETTINGS[:2]
SETTINGS = [x for x in SETTINGS if x[0] in CFG["settings"]]
AUDIT["settings"] = {n: dict(split=m, n_features=len(c), features=c) for n, m, c in SETTINGS}
AUDIT["strict_extra"] = STRICT_EXTRA
# Evaluation class set, fixed before training: a class is left out of the macro-average of every setting when it has
# fewer than 10 distinct feature vectors in any setting (it cannot appear in every test split); it is reported per class.
_pairs = {n: cnt(c).groupby("y").size() for n, _, c in ALL_SETTINGS}
EVAL_EXCLUDE = sorted({k for n in _pairs for k in classes if _pairs[n].get(k, 0) < 10})
AUDIT["distinct_vectors_by_setting_and_class"] = {n: {str(k): int(v) for k, v in p_.items()} for n, p_ in _pairs.items()}
AUDIT["eval_exclude"] = EVAL_EXCLUDE; print("classes left out of the macro-average:", EVAL_EXCLUDE)
save_json(AUDIT, "audit.json")
ORDER = [g for g, _ in DS["groups"]] + ["other"]
LAYER_WINDOWS = DS["windows"] + [["other"]]

def empty_mask(frame):
    cols = {}
    for c in frame.columns:
        v = frame[c]
        cols[c] = (v == 0).values if pd.api.types.is_numeric_dtype(v) else v.isin(EMPTY | {"0"}).values
    return pd.DataFrame(cols, index=frame.index)

def prepare(name, cols):
    global df, y_all, kept, cat_cols, num_cols, group_of, kept_groups, H, GRP
    kept = list(cols)
    B = RAW if name == "common" else CANON
    df = B[kept + [LABEL, "_row"]].reset_index(drop=True)
    y_all = df[LABEL].map(cls2id).values.astype(np.int64)
    H = feature_hash(CANON[kept].reset_index(drop=True))     # vectors always identified on canonical values
    g = pd.DataFrame({"h": H, "y": y_all})
    cnt_ = g.groupby(["h", "y"]).size().rename("n").reset_index().sort_values(["h", "n"], ascending=[True, False])
    GRP = pd.DataFrame({"maj": cnt_.drop_duplicates("h").set_index("h")["y"]})
    cat_cols = [c for c in kept if (NUM_RATIO_RAW[c] < 0.99 if name == "common" else not IS_NUM[c])]
    num_cols = [c for c in kept if c not in cat_cols]
    group_of = group_map(kept)
    kept_groups = [g_ for g_ in ORDER if any(group_of.get(c) == g_ for c in kept)]
    info = dict(setting=name, n_features=len(kept), n_records=len(df), n_unique_vectors=len(GRP),
                categorical=cat_cols, token_groups=kept_groups,
                tokens={g_: [c for c in kept if group_of[c] == g_] for g_ in kept_groups})
    print({k: v for k, v in info.items() if k != "tokens"}); return info

def make_split(mode, seed):
    """record: stratified 70/10/20 over records. grouped: groups (distinct canonical vectors) split 70/10/20 per
    majority label; train/val keep one record per (vector, label), test keeps all records.
    test_u: one record per (vector, label) of the test part (primary metric)."""
    rng = np.random.default_rng(seed)
    if mode == "record":
        parts = {"tr": [], "va": [], "te": []}
        for k in range(len(classes)):
            idx = np.where(y_all == k)[0]; idx = idx[rng.permutation(len(idx))]; n = len(idx)
            a, b = int(round(0.7 * n)), int(round(0.8 * n))
            parts["tr"].append(idx[:a]); parts["va"].append(idx[a:b]); parts["te"].append(idx[b:])
        tr, va, te = [np.sort(np.concatenate(parts[k])) for k in ("tr", "va", "te")]
        d = pd.DataFrame({"h": H[te], "y": y_all[te]}); te_u = te[~d.duplicated().values]
        return tr, va, te, te_u, "stratified random record split (no de-duplication)"
    parts = {"tr": [], "va": [], "te": []}
    for k in range(len(classes)):
        gk = GRP[GRP["maj"] == k]
        if len(gk) == 0: continue
        order = gk.index.values[rng.permutation(len(gk))]; n = len(order)
        if n >= 3:
            a = min(max(1, int(round(0.7 * n))), n - 2); b = max(a + 1, min(int(round(0.8 * n)), n - 1))
        else:
            a, b = 1, 1
        parts["tr"].append(order[:a]); parts["va"].append(order[a:b]); parts["te"].append(order[b:])
    sets = {k: np.concatenate(v) for k, v in parts.items()}
    first = pd.DataFrame({"h": H, "y": y_all}).drop_duplicates().index.values
    in_set = lambda idx, key: idx[np.isin(H[idx], sets[key])]
    tr, va, te_u = in_set(first, "tr"), in_set(first, "va"), in_set(first, "te")
    te = in_set(np.arange(len(df)), "te")
    return tr, va, te, te_u, "grouped random split by canonical feature vector"

class Encoder:
    """signed-log + standardization for numeric fields; top-K vocab for categorical fields; presence per token."""
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
# ## 4. Models (identical to the Edge-IIoTset experiments)

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
    ev = [i for i in range(len(classes)) if classes[i] not in EVAL_EXCLUDE]
    return dict(macro_f1=float(np.mean(f[ev])),          # fixed class set (see EVAL_EXCLUDE), not sklearn's present-label average
                macro_f1_sklearn=f1_score(y, pred, average="macro"), weighted_f1=f1_score(y, pred, average="weighted"),
                balanced_acc=balanced_accuracy_score(y, pred), mcc=matthews_corrcoef(y, pred),
                accuracy=float((y == pred).mean()),
                normal_fpr=float(((pred != cls2id.get(NORMAL, -1)) & (y == cls2id.get(NORMAL, -1))).sum() /
                                 max((y == cls2id.get(NORMAL, -1)).sum(), 1)),
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
# ## 5. Run

# %%
ALL = []
if STAGE == "bench":
    for sname, mode, cols in SETTINGS:
        info = prepare(sname, cols)
        for seed in CFG["seeds"]:
            tr, va, te, te_u, how = make_split(mode, seed)
            enc = Encoder().fit(df.iloc[tr]); data = {}
            for part, ids in [("train", tr), ("val", va), ("test", te), ("test_u", te_u)]:
                X, C, P = enc.transform(df.iloc[ids]); data[part] = (X, C, P, y_all[ids])
            # share of test records whose vector also occurs in training (leakage indicator)
            seen = float(np.isin(H[te], H[tr]).mean())
            split_info = dict(setting=sname, mode=mode, how=how, seed=seed, prep=info, n_train=len(tr), n_val=len(va),
                              n_test=len(te), n_test_unique=len(te_u), test_seen_in_train=seen,
                              test_unique_counts={classes[k]: int((y_all[te_u] == k).sum()) for k in range(len(classes))})
            tag = f"{sname}_s{seed}"
            print(f"\n=== {DATASET} {tag}: {how} | train {len(tr):,} val {len(va):,} test {len(te):,} "
                  f"(unique {len(te_u):,}) | test records seen in train {seen:.3f}")
            dte = loaders(*data["test"], 8192, False); dteu = loaders(*data["test_u"], 8192, False)
            for name in CFG["models"]:
                t1 = time.time(); rec = dict(split=split_info, model=name)
                if name == "LightGBM":
                    clf = train_lgbm(data, seed)
                    pred = clf.predict(np.hstack(data["test"][:2])); pu = clf.predict(np.hstack(data["test_u"][:2]))
                    rec.update(n_trees=int(clf.booster_.num_trees()))
                else:
                    m, hist = train_nn(name, data, seed)
                    pred = predict(m, dte).argmax(1).numpy(); pu = predict(m, dteu).argmax(1).numpy()
                    rec.update(history=hist, params=n_params(m),
                               flops_per_sample=flops_per_sample(m, *[torch.from_numpy(a) for a in data["test"][:3]]))
                    if name == "PALT":
                        try: rec["token_attribution"] = palt_token_attribution(m, data)
                        except Exception as e: rec["attribution_error"] = repr(e)
                    del m
                rec.update(test=metrics(data["test"][3], pred), test_unique=metrics(data["test_u"][3], pu), train_sec=time.time() - t1)
                ALL.append(rec); save_json(ALL, "results.json")
                print(f"[{DATASET} {tag}] {name:14s} uniqF1={rec['test_unique']['macro_f1']:.4f} recF1={rec['test']['macro_f1']:.4f} "
                      f"MCC={rec['test']['mcc']:.4f} normalFPR={rec['test']['normal_fpr']:.4f} params={rec.get('params','-')} ({rec['train_sec']:.0f}s)")
            del data; gc.collect(); torch.cuda.empty_cache()

# %% [markdown]
# ## 6. Summary and download bundle

# %%
if ALL:
    rows = [dict(setting=r["split"]["setting"], seed=r["split"]["seed"], model=r["model"],
                 macro_f1_unique=round(100 * r["test_unique"]["macro_f1"], 2), macro_f1_records=round(100 * r["test"]["macro_f1"], 2),
                 mcc=round(r["test"]["mcc"], 4), normal_fpr=round(100 * r["test"]["normal_fpr"], 3),
                 test_seen_in_train=round(r["split"]["test_seen_in_train"], 4), params=r.get("params"), flops=r.get("flops_per_sample"))
            for r in ALL]
    summary = pd.DataFrame(rows); summary.to_csv(os.path.join(CFG["out_dir"], "summary.csv"), index=False)
    agg = summary.groupby(["setting", "model"])[["macro_f1_unique", "macro_f1_records", "mcc", "normal_fpr"]].agg(["mean", "std"]).round(2)
    agg.to_csv(os.path.join(CFG["out_dir"], "summary_mean_std.csv")); print(agg.to_string())
    pcf = pd.DataFrame([{**dict(setting=r["split"]["setting"], seed=r["split"]["seed"], model=r["model"]),
                         **{k: round(100 * v, 2) for k, v in r["test_unique"]["per_class_f1"].items()}} for r in ALL])
    pcf.to_csv(os.path.join(CFG["out_dir"], "per_class_f1_unique.csv"), index=False)
    att = [dict(setting=r["split"]["setting"], seed=r["split"]["seed"], **a) for r in ALL for a in r.get("token_attribution", [])]
    if att: pd.DataFrame(att).to_csv(os.path.join(CFG["out_dir"], "palt_token_attribution.csv"), index=False)
env = dict(python=sys.version, torch=torch.__version__, cuda=torch.version.cuda,
           gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else None, cpu=platform.processor(), cpu_count=os.cpu_count(),
           cfg=CFG, dataset=DATASET, stage=STAGE, strict_extra=STRICT_EXTRA)
save_json(env, "environment.json")
zp = os.path.join(os.path.dirname(os.path.abspath(CFG["out_dir"])), f"cross_{DATASET}_{STAGE}_results.zip")
with zipfile.ZipFile(zp, "w", zipfile.ZIP_DEFLATED) as z:
    for f in glob.glob(os.path.join(CFG["out_dir"], "*")):
        z.write(f, os.path.basename(f))
print("\nDownload:", zp)
