# Result logs

| Folder | Produced by | Paper |
|---|---|---|
| `edge-iiotset/main/` | notebook 01 (strict features; grouped random and chronological splits; 7 models x 5 seeds) | Table III, Fig. 2, Secs. V-C, V-F |
| `edge-iiotset/ablation/` | notebook 02 (raw / canonical / strict features; grouped random split; 4 models x 3 seeds) | Table II |
| `edge-iiotset/chrono_v4b/` | notebook 03 (file-order and random-for-two-classes variants of the chronological split) | Sec. V-C |
| `edge-iiotset/robust_a/` | notebook 07 (session-grouped split, frequency-weighted training, tcp.seq/ack removed) | Tables IV, V |
| `edge-iiotset/robust_b/`, `edge-iiotset/robust_c/` | notebooks 08-09 (XGBoost, CatBoost, random forest, MLP-PLR, PALT-PLR, PALT controls) | Tables III, VI |
| `edge-iiotset/robust_d/` | notebook 11 (model export for the second edge session; tcp.seq/ack removed for four models) | Tables V, VII |
| `edge-iiotset/robust_e_cpu/`, `edge-iiotset/robust_e_gpu/` | notebooks 12-13 (session split, tcp.seq/ack permuted and coarsened, vector-level oracle) | Tables IV, V, Sec. V-F |
| `cross/xiiotid/`, `cross/ciciot2023/` | notebooks 04-05 (common, grouped, strict settings) | Table VIII |
| `cross/ciciot2023_strict_iat/` | notebook 06 (CICIoT2023 strict setting, IAT removed) | Table VIII |
| `cross/ciciot2023_iat_evidence/` | notebook 10 (IAT value bands; summary statistics only) | Sec. V-H |
| `edge-iiotset/edge_bench/`, `edge-iiotset/edge_bench_session2/` | `edge_bench/run_profiles.ps1` (Intel Core Ultra 7 258V, Windows 11, Docker 29.8); `edge_summary.csv` is a flat table of p50/p99, size, load time, and memory | Table VII |

Files: `results.json` (one record per run and model: split sizes, per-class counts, metrics on all test records and on distinct test vectors, training time), `summary*.csv` (flat tables), `per_class_f1*.csv`, `palt_token_attribution.csv` (per-protocol attributions, Sec. V-E), `audit.json` (dataset statistics, shortcut probes, information ceiling), `analysis_v4.json` (shortcut probe and ceiling details), `environment.json` (library versions and configuration).
The macro-F1 columns in `summary*.csv` of the Edge-IIoTset runs use scikit-learn's `average="macro"`; the paper uses the fixed 14-class average from `analysis/recompute_macro_f1.py`.
