# Beyond Shortcuts: A Leakage-Aware Audit and Benchmark of Intrusion Detectors for Smart-Factory IIoT Gateways

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23017865.svg)](https://doi.org/10.5281/zenodo.23017865)

Code, data splits, trained models, and result logs for the paper

> H. M. Kim, "Beyond Shortcuts: A Leakage-Aware Audit and Benchmark of Intrusion Detectors for Smart-Factory IIoT Gateways," manuscript under review, 2026.

The repository contains everything needed to reproduce the tables and figures **except the datasets**, which must be obtained from their original distributors (see [DATA.md](DATA.md)).

## What the paper does

1. **Shortcut audit of Edge-IIoTset.** Three artifact families let detectors reach near-perfect scores without learning traffic behavior:
   * the spelling of empty values (`"0"`, `"0.0"`, `"0x00000000"`) differs between capture files, so a single rule on a DNS field separates normal from attack records with 99.92% accuracy;
   * session identifiers (raw acknowledgment numbers, checksums, stream indices) are the only difference for 77% of apparently distinct records;
   * the timestamp field is malformed for every DDoS_UDP and MITM record.
2. **Leakage-aware protocol.** Canonicalize values, remove identifiers, split by canonical feature vector (no vector in two sets), fit all preprocessing on the training split, report macro-F1 on distinct test vectors over a fixed class set, per-class F1, the per-packet information ceiling, and paired tests. Checked against a session-grouped split and frequency-weighted training.
3. **Benchmark** of ten detectors (four tree ensembles: LightGBM, XGBoost, random forest, CatBoost; six neural detectors: MLP, MLP-PLR, 1-D CNN, FT-Transformer, PALT, PALT-PLR) with deployment cost under emulated gateway budgets, and a replication on X-IIoTID and CICIoT2023.
4. **Mechanism of the ranking.** With the split held fixed, coarsening the relative TCP sequence/acknowledgment numbers to 16 bins cuts the lead of the tree ensembles over the MLP from 10–15 to 4–5 points; periodic numerical embeddings (PLR) close most of the gap for neural detectors; a vector-level oracle (97.7%) shows how far all detectors remain below the single-packet ceiling.
5. **PALT case study**, a protocol-aware lightweight Transformer (one token per protocol, 35 K parameters, 0.41 M FLOPs). Controls show that random field-to-token assignment is as accurate as protocol grouping; protocol structure gives attribution, not accuracy.

Main result (Table III of the paper; Edge-IIoTset, strict protocol, macro-F1 on distinct test vectors, 14 classes, mean ± std over 5 seeds):

| Model | Grouped random | Grouped chronological |
|---|---|---|
| Random forest | 87.88 ± 0.78 | 80.77 ± 0.16 |
| XGBoost | 85.36 ± 1.00 | 84.40 ± 0.00 |
| LightGBM | 82.64 ± 0.40 | 83.45 ± 0.30 |
| CatBoost | 75.49 ± 0.68 | 77.08 ± 0.66 |
| MLP-PLR | 80.06 ± 0.51 | 78.72 ± 4.39 |
| PALT-PLR | 78.68 ± 3.42 | 80.13 ± 2.27 |
| PALT | 73.24 ± 3.06 | 71.50 ± 5.73 |
| FT-Transformer | 73.63 ± 3.06 | 66.96 ± 3.81 |
| MLP | 71.90 ± 1.52 | 64.25 ± 5.57 |
| 1-D CNN | 65.25 ± 3.32 | 64.95 ± 0.57 |

Notebook 01 also trains a PALT teacher and a distilled PALT student (PALT-Teacher 73.26 / 73.58, PALT-KD 72.81 / 72.18); these auxiliary runs are logged in `results/` but are not part of the paper.

Session-grouped split (Table IV of the paper; macro-F1, %, 5 seeds):

| Model | All distinct test vectors | Test vectors unseen in training |
|---|---|---|
| Random forest | 77.98 | 66.65 |
| XGBoost | 74.29 | 65.28 |
| LightGBM | 72.17 | 63.61 |
| PALT-PLR | 69.11 | 58.45 |
| MLP-PLR | 67.40 | 56.73 |
| PALT | 64.59 | 55.13 |
| FT-Transformer | 62.12 | 54.19 |
| MLP | 62.02 | 53.25 |

Interventions on tcp.seq and tcp.ack with the split held fixed (Table V of the paper; grouped random split, macro-F1, %, 3 seeds; —: not run):

| Model | Intact | Permuted at test | Coarsened (16 bins) | Removed |
|---|---|---|---|---|
| Random forest | 87.96 | 44.48 | 73.46 | 48.45 |
| XGBoost | 85.18 | 34.18 | 73.08 | 47.93 |
| LightGBM | 82.53 | 34.57 | 72.72 | 48.10 |
| MLP-PLR | 80.27 | 29.55 | 72.12 | 47.84 |
| PALT-PLR | 77.70 | 43.18 | 70.99 | 48.13 |
| PALT | 73.68 | 34.77 | 69.51 | 46.93 |
| FT-Transformer | 72.78 | — | — | 47.39 |
| MLP | 72.66 | 21.01 | 68.59 | 46.61 |

Vector-level oracle (majority label of each test vector, same metric and splits): 97.7%.

PALT controls (Table VI of the paper; pooled difference to PALT over 10 paired runs, Wilcoxon p): random grouping −0.6 (0.85), no local path +0.1 (1.00), full attention −0.7 (0.85), zero absent token +1.2 (0.63), PALT-PLR +7.0 (0.010); PALT-PLR vs. PALT-PLR with random grouping +0.8 (1.00).

Replication on two further datasets (Table VIII of the paper; macro-F1 on distinct test vectors, %, mean over 3 seeds; common / grouped / strict):

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
  01_edgeiiotset_main.ipynb              7 models x 5 seeds, grouped random + chronological splits (Table III, Fig. 2)
  02_edgeiiotset_ablation.ipynb          raw / canonical / strict feature settings, 4 models x 3 seeds (Table II)
  03_edgeiiotset_chrono_sensitivity.ipynb  alternative orderings of the chronological split (Sec. V-C)
  04_cross_xiiotid.ipynb                 audit + benchmark on X-IIoTID (Table VIII)
  05_cross_ciciot2023.ipynb              audit + benchmark on CICIoT2023 (Table VIII)
  06_cross_ciciot2023_strict_iat.ipynb   CICIoT2023 strict setting with IAT removed (Table VIII)
  07_edgeiiotset_robust_a.ipynb          session-grouped split, weighted training, tcp.seq/ack removed (Tables IV, V)
  08_edgeiiotset_robust_b.ipynb          XGBoost, CatBoost, random forest, MLP-PLR, PALT controls (Tables III, VI)
  09_edgeiiotset_robust_c.ipynb          PALT-PLR and PALT-PLR with random grouping (Tables III, VI)
  10_ciciot2023_iat_evidence.ipynb       IAT value bands and separability on CICIoT2023 (Sec. V-H)
  11_edgeiiotset_robust_d.ipynb          model export for the second edge session; tcp.seq/ack removed for XGBoost, RF, MLP-PLR, PALT-PLR (Tables V, VII)
  12_edgeiiotset_robust_e_cpu.ipynb      session split for XGBoost/RF; tcp.seq/ack permuted at test and coarsened to 16 bins; vector-level oracle (tree ensembles, CPU; Tables IV, V)
  13_edgeiiotset_robust_e_gpu.ipynb      the same interventions for MLP, MLP-PLR, PALT, PALT-PLR (GPU; Tables IV, V)
src/         the code of the notebooks as plain Python scripts
  audit_edgeiiotset.py                   stand-alone data audit of Edge-IIoTset
  main_benchmark.py                      notebooks 01-02 (set STAGE = "main" or "ablation")
  chrono_sensitivity.py                  notebook 03
  cross_dataset.py                       notebooks 04-06 (set DATASET and settings)
  robustness.py                          notebooks 07-09, 11-13 (set STAGE = "robust_a" ... "robust_d", "robust_e_cpu" or "robust_e_gpu")
  ciciot_iat_evidence.py                 notebook 10
analysis/    recompute_macro_f1.py (fixed 14-class macro-F1), paired_tests.py (Wilcoxon signed-rank, Holm)
figures/     fig1_arch.py (Fig. 1), fig_results.py (Fig. 2, read from results/)
edge_bench/  Docker benchmark under emulated gateway budgets (Table VII)
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

Approximate run times: 01 ≈ 3–4 h, 02 ≈ 4–5 h, 03 ≈ 2–3 h, 04 ≈ 2–3 h, 05/06 ≈ 2.5–3.5 h (including a full pass over 46.7 M rows), 07 ≈ 3–4 h, 08 ≈ 4–5 h, 09 ≈ 2 h, 10 ≈ 30–45 min (CPU), 11 ≈ 2–3 h, 12 ≈ 4 h (CPU), 13 ≈ 2–3 h.
The notebooks find the CSV files by their columns, so the folder layout of the Kaggle input does not matter.
Outside Kaggle, run the scripts in `src/` with `CFG["data_csv"]` pointing to the file.

### 3. Recompute the reported numbers
```bash
python analysis/recompute_macro_f1.py results/edge-iiotset/{main,ablation,robust_a,robust_b,robust_c,robust_d,robust_e_cpu,robust_e_gpu}/results.json
python analysis/paired_tests.py analysis/macro_f1_fixed_classes.csv
python figures/fig_results.py
```
`recompute_macro_f1.py` averages the stored per-class F1 over a fixed set of 14 classes. scikit-learn's `average="macro"` includes a class only when it occurs in the true or predicted labels, which silently mixes 14- and 15-class averages when Fingerprinting (six distinct vectors) has no test vector.

### 4. Use the published splits
Notebooks 08 and 09 reuse the split indices of notebook 01, and notebook 13 those of notebook 12 (identical files, not duplicated). Each `splits/edge-iiotset/<run>/split_<mode>-<features>_s<seed>.npz` holds `train`, `val`, `test`, and `test_unique`: row positions (0-based, header excluded) in `DNN-EdgeIIoT-dataset.csv` as read by `pandas.read_csv`. `test` keeps the traffic multiplicity; `test_unique` has one record per distinct canonical feature vector. The session-grouped runs (`robust_a/split_session-strict_s*.npz`) also hold `test_novel`, the test vectors never seen in training.
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
Archived version (v1.0-submission): https://doi.org/10.5281/zenodo.23017865

See [CITATION.cff](CITATION.cff). The entry will be updated with the publication details.
