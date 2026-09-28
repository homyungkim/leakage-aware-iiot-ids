# Beyond Shortcuts: A Leakage-Aware Audit and Benchmark of Intrusion Detectors for Smart-Factory IIoT Gateways

Code, data splits, trained models, and result logs for the paper

> H. M. Kim, "Beyond Shortcuts: A Leakage-Aware Audit and Benchmark of Intrusion Detectors for Smart-Factory IIoT Gateways," manuscript under review, 2026.

The repository contains everything needed to reproduce the tables and figures **except the datasets**, which must be obtained from their original distributors (see [DATA.md](DATA.md)).

## What the paper does

1. **Shortcut audit of Edge-IIoTset.** Three artifact families let detectors reach near-perfect scores without learning traffic behavior:
   * the spelling of empty values (`"0"`, `"0.0"`, `"0x00000000"`) differs between capture files, so a single rule on a DNS field separates normal from attack records with 99.92% accuracy;
   * session identifiers (raw acknowledgment numbers, checksums, stream indices) make 77% of apparently distinct records duplicates once removed;
   * the timestamp field is malformed for every DDoS_UDP and MITM record.
2. **Leakage-aware protocol.** Canonicalize values, remove identifiers, split by canonical feature vector (no vector in two sets), fit all preprocessing on the training split, report macro-F1 on distinct test vectors over a fixed class set, per-class F1, the per-packet information ceiling, and paired tests. Checked against a session-grouped split and frequency-weighted training.
3. **Benchmark** of nine detectors (LightGBM, XGBoost, random forest, CatBoost; MLP, MLP-PLR, 1-D CNN, FT-Transformer, PALT) with deployment cost under emulated gateway budgets, and a replication on X-IIoTID and CICIoT2023.
4. **Mechanism of the ranking.** Removing the relative TCP sequence/acknowledgment numbers nearly erases the tree lead; periodic numerical embeddings (PLR) close most of the gap for neural detectors.
5. **PALT case study**, a protocol-aware lightweight Transformer (one token per protocol, 35 K parameters, 0.41 M FLOPs). Controls show that random field-to-token assignment is as accurate as protocol grouping; protocol structure gives attribution, not accuracy.

Main result (Edge-IIoTset, strict protocol, macro-F1 on distinct test vectors, 14 classes, mean ± std over 5 seeds):

| Model | Grouped random | Grouped chronological |
|---|---|---|
| Random forest | 87.88 ± 0.78 | 80.77 ± 0.16 |
| XGBoost | 85.36 ± 1.00 | 84.40 ± 0.00 |
| LightGBM | 82.64 ± 0.40 | 83.45 ± 0.30 |
| CatBoost | 75.49 ± 0.68 | 77.08 ± 0.66 |
| MLP-PLR | 80.06 ± 0.51 | 78.72 ± 4.39 |
| PALT-PLR | 78.68 ± 3.42 | 80.13 ± 2.27 |
| PALT-Teacher | 73.26 ± 1.87 | 73.58 ± 3.68 |
| PALT (student) | 73.24 ± 3.06 | 71.50 ± 5.73 |
| PALT-KD | 72.81 ± 2.01 | 72.18 ± 2.21 |
| FT-Transformer | 73.63 ± 3.06 | 66.96 ± 3.81 |
| MLP | 71.90 ± 1.52 | 64.25 ± 5.57 |
| 1-D CNN | 65.25 ± 3.32 | 64.95 ± 0.57 |

Robustness checks (macro-F1, %, mean ± std):

| Model | Session-grouped split, all test vectors (5 seeds) | Session-grouped, unseen vectors | tcp.seq / tcp.ack removed (3 seeds) |
|---|---|---|---|
| LightGBM | 72.17 ± 4.65 | 63.61 ± 3.78 | 48.10 ± 1.45 |
| PALT | 64.59 ± 5.30 | 55.13 ± 3.50 | 46.93 ± 1.15 |
| FT-Transformer | 62.12 ± 6.80 | 54.19 ± 6.79 | 47.39 ± 1.28 |
| MLP | 62.02 ± 2.72 | 53.25 ± 4.72 | 46.61 ± 1.18 |
| Random forest | — | — | 48.45 ± 0.32 |
| XGBoost | — | — | 47.93 ± 1.01 |
| MLP-PLR | — | — | 47.84 ± 1.46 |
| PALT-PLR | — | — | 48.13 ± 1.37 |

PALT controls (pooled difference to PALT over 10 paired runs, Wilcoxon p): random grouping −0.6 (0.85), no local path +0.1 (1.00), full attention −0.7 (0.85), zero absent token +1.2 (0.63), PALT-PLR +7.0 (0.010); PALT-PLR vs. PALT-PLR with random grouping +0.8 (1.00).

Replication on two further datasets (macro-F1 on distinct test vectors, %, mean over 3 seeds; common / grouped / strict):

| Model | X-IIoTID (strict: ports removed) | CICIoT2023 (strict: IAT removed) |
|---|---|---|
| LightGBM | 97.6 / 98.3 / 97.7 | 85.8 / 85.0 / 71.1 |
| PALT | 93.5 / 95.1 / 95.4 | 69.8 / 69.7 / 69.0 |
| FT-Transformer | 92.8 / 91.7 / 91.0 | 65.5 / 65.7 / 66.2 |
| MLP | 91.5 / 92.7 / 91.9 | 67.8 / 67.6 / 67.8 |

On CICIoT2023 the audit flags the inter-arrival-time feature (IAT alone: 59.2% macro-F1 over 34 classes). Its values form narrow class-specific bands (182 of 190 interquartile ranges of the flood/scan classes are disjoint) and separate same-protocol DoS from DDoS floods with AUC 0.993–0.999; removing it costs LightGBM 13.8 points and the neural detectors at most 0.7.

## Repository layout

```
notebooks/   Kaggle notebooks exactly as run for the paper (outputs cleared)
  01_edgeiiotset_main.ipynb              7 models x 5 seeds, grouped random + chronological splits (Table VI, Figs. 2-3)
  02_edgeiiotset_ablation.ipynb          raw / canonical / strict feature settings, 4 models x 3 seeds (Table V)
  03_edgeiiotset_chrono_sensitivity.ipynb  alternative orderings of the chronological split (Sec. VI-C)
  04_cross_xiiotid.ipynb                 audit + benchmark on X-IIoTID (Table X)
  05_cross_ciciot2023.ipynb              audit + benchmark on CICIoT2023 (Table X)
  06_cross_ciciot2023_strict_iat.ipynb   CICIoT2023 strict setting with IAT removed (Table X)
  07_edgeiiotset_robust_a.ipynb          session-grouped split, weighted training, tcp.seq/ack removed (Table VII)
  08_edgeiiotset_robust_b.ipynb          XGBoost, CatBoost, random forest, MLP-PLR, PALT controls (Tables VI, VIII)
  09_edgeiiotset_robust_c.ipynb          PALT-PLR and PALT-PLR with random grouping (Tables VI, VIII)
  10_ciciot2023_iat_evidence.ipynb       IAT value bands and separability on CICIoT2023 (Sec. VI-I)
  11_edgeiiotset_robust_d.ipynb          model export for the second edge session; tcp.seq/ack removed for XGBoost, RF, MLP-PLR, PALT-PLR (Tables VII, IX)
src/         the code of the notebooks as plain Python scripts
  audit_edgeiiotset.py                   stand-alone data audit of Edge-IIoTset
  main_benchmark.py                      notebooks 01-02 (set STAGE = "main" or "ablation")
  chrono_sensitivity.py                  notebook 03
  cross_dataset.py                       notebooks 04-06 (set DATASET and settings)
  robustness.py                          notebooks 07-09, 11 (set STAGE = "robust_a", "robust_b", "robust_c" or "robust_d")
  ciciot_iat_evidence.py                 notebook 10
analysis/    recompute_macro_f1.py (fixed 14-class macro-F1), paired_tests.py (Wilcoxon signed-rank, Holm)
figures/     fig1_arch.py (Fig. 1), fig_results.py (Figs. 2-3, read from results/)
edge_bench/  Docker benchmark under emulated gateway budgets (Table IX)
results/     result logs of every run (JSON/CSV), environment records, audits
splits/      train/validation/test row indices for every Edge-IIoTset run
models/      trained models of seed 0 (ONNX FP32 and dynamic INT8, LightGBM text model)
tools/       verify_data.py (checks that your dataset copy matches ours)
```

## Reproducing

### 1. Get the data
Download the datasets as described in [DATA.md](DATA.md) and check your copy:
```bash
pip install -r requirements.txt
python tools/verify_data.py edge-iiotset "path/to/DNN-EdgeIIoT-dataset.csv"
```

### 2. Run the experiments (Kaggle, GPU T4)
All training was done on Kaggle Notebooks (NVIDIA Tesla T4, PyTorch 2.10 + CUDA 12.8, Python 3.12).
1. On Kaggle, open the dataset page → **Code → New Notebook**, then **File → Import Notebook** and choose a notebook from `notebooks/`.
2. Settings: **Accelerator GPU T4 x2**, **Internet on**.
3. **Save Version → Save & Run All (Commit)**. The run continues when the browser is closed.
4. Download the results zip from the **Output** tab.

Approximate run times: 01 ≈ 3–4 h, 02 ≈ 4–5 h, 03 ≈ 2–3 h, 04 ≈ 2–3 h, 05/06 ≈ 2.5–3.5 h (including a full pass over 46.7 M rows), 07 ≈ 3–4 h, 08 ≈ 4–5 h, 09 ≈ 2 h, 10 ≈ 30–45 min (CPU), 11 ≈ 2–3 h.
The notebooks find the CSV files by their columns, so the folder layout of the Kaggle input does not matter.
Outside Kaggle, run the scripts in `src/` with `CFG["data_csv"]` pointing to the file.

### 3. Recompute the reported numbers
```bash
python analysis/recompute_macro_f1.py results/edge-iiotset/{main,ablation,robust_a,robust_b,robust_c,robust_d}/results.json
python analysis/paired_tests.py analysis/macro_f1_fixed_classes.csv
python figures/fig_results.py
```
`recompute_macro_f1.py` averages the stored per-class F1 over a fixed set of 14 classes. scikit-learn's `average="macro"` includes a class only when it occurs in the true or predicted labels, which silently mixes 14- and 15-class averages when Fingerprinting (six distinct vectors) has no test vector.

### 4. Use the published splits
Notebooks 08 and 09 reuse the split indices of notebook 01 (identical files, not duplicated). Each `splits/edge-iiotset/<run>/split_<mode>-<features>_s<seed>.npz` holds `train`, `val`, `test`, and `test_unique`: row positions (0-based, header excluded) in `DNN-EdgeIIoT-dataset.csv` as read by `pandas.read_csv`. `test` keeps the traffic multiplicity; `test_unique` has one record per distinct canonical feature vector. The session-grouped runs (`robust_a/split_session-strict_s*.npz`) also hold `test_novel`, the test vectors never seen in training.
```python
import numpy as np, pandas as pd
df = pd.read_csv("DNN-EdgeIIoT-dataset.csv", dtype=str, keep_default_na=False, low_memory=False)
s = np.load("splits/edge-iiotset/main/split_random-strict_s0.npz")
train, test_unique = df.iloc[s["train"]], df.iloc[s["test_unique"]]
```

### 5. Edge benchmark
See [edge_bench/README.md](edge_bench/README.md). `results/edge-iiotset/edge_bench/` holds the first session (models of notebook 01), `results/edge-iiotset/edge_bench_session2/` the second (models of notebook 11: XGBoost and LightGBM as native and ONNX trees, random forest as ONNX, MLP-PLR, PALT-PLR, with LightGBM and PALT as anchors). Latency is measured on an x86-64 host under Docker CPU and memory limits (OrinNano-class 6 cores/4 GB, RPi4-class 4 cores/2 GB, Gateway-class 2 cores/2 GB). These are emulated budgets, not measurements on ARM hardware.

## Notes
* The Edge-IIoTset shortcut audit is specific to how that dataset was exported; audit any new dataset before training on it.
* `results/cross/ciciot2023_iat_evidence/` holds summary statistics and plots only; the per-class record sample written by notebook 10 is not included.
* `results/*/audit.json` contain dataset statistics (class counts, duplicate shares, probe scores), not records.
* No dataset records are included in this repository. The edge benchmark falls back to synthetic inputs of the right shape when the real test-record file is absent; the paper's numbers used 2,000 real test records exported by notebook 01.

## License
Code: MIT License (see [LICENSE](LICENSE)). The datasets remain under the terms of their original distributors.

## Citation
Repository: https://github.com/homyungkim/leakage-aware-iiot-ids

See [CITATION.cff](CITATION.cff). The entry will be updated with the publication details.
