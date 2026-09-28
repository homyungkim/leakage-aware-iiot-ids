# Result logs

| Folder | Produced by | Paper |
|---|---|---|
| `edge-iiotset/main/` | notebook 01 (strict features; grouped random and chronological splits; 7 models x 5 seeds) | Table VI, Figs. 2-3, Sec. VI-D-F |
| `edge-iiotset/ablation/` | notebook 02 (raw / canonical / strict features; grouped random split; 4 models x 3 seeds) | Table V |
| `edge-iiotset/chrono_v4b/` | notebook 03 (file-order and random-for-two-classes variants of the chronological split) | Sec. VI-C |
| `cross/xiiotid/`, `cross/ciciot2023/` | notebooks 04-05 (common, grouped, strict settings) | Table VIII |
| `edge-iiotset/edge_bench/` | `edge_bench/run_profiles.ps1` on the seed-0 random-split models (Intel Core Ultra 7 258V, Windows 11, Docker 29.8); `edge_summary.csv` is a flat table of p50/p99, size, load time, and memory | Table VII |
| `cross/ciciot2023_strict_iat/` | notebook 06 (CICIoT2023 strict setting, IAT removed) | Table VIII |

Files: `results.json` (one record per run and model: split sizes, per-class counts, metrics on all test records and on distinct test vectors, training time), `summary*.csv` (flat tables), `per_class_f1*.csv`, `palt_token_attribution.csv` (Fig. 3), `audit.json` (dataset statistics, shortcut probes, information ceiling), `analysis_v4.json` (shortcut probe and ceiling details), `environment.json` (library versions and configuration).
The macro-F1 columns in `summary*.csv` of the Edge-IIoTset runs use scikit-learn's `average="macro"`; the paper uses the fixed 14-class average from `analysis/recompute_macro_f1.py`.
