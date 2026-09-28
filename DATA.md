# Datasets

No dataset records are included in this repository. Obtain the datasets from their distributors and follow their terms of use.

| Dataset | Used file | Source | Rows / classes |
|---|---|---|---|
| Edge-IIoTset | `Edge-IIoTset dataset/Selected dataset for ML and DL/DNN-EdgeIIoT-dataset.csv` | IEEE DataPort, doi:10.21227/mbc1-1h68; Kaggle mirror `mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot` (used in the paper) | 2,219,201 / 15 (label `Attack_type`) |
| X-IIoTID | `X-IIoTID dataset.csv` | Kaggle `munaalhawawreh/xiiotid-iiot-intrusion-dataset` (used in the paper) | 820,834 / 19 (label `class1`) |
| CICIoT2023 | all 169 `part-*.csv` files merged into one CSV | Canadian Institute for Cybersecurity, University of New Brunswick; the paper used the merged Kaggle copy `akashdogra/ciciot23csv` | 46,686,579 / 34 (label `label`) |

Check your copy with
```bash
python tools/verify_data.py edge-iiotset "path/to/DNN-EdgeIIoT-dataset.csv"
python tools/verify_data.py xiiotid      "path/to/X-IIoTID dataset.csv"
python tools/verify_data.py ciciot2023   path/to/merged.csv      # or the folder with part-*.csv
```
The script compares the row count and the count of every class with `tools/expected_counts.json` and prints the SHA-256 of each file.

## Notes on use
* **Edge-IIoTset.** The published split indices refer to row positions of `DNN-EdgeIIoT-dataset.csv` read with `pandas.read_csv(..., dtype=str, keep_default_na=False)`. A different row order (for example a re-exported copy) invalidates them; `verify_data.py` detects copies with different class counts, and the notebooks regenerate the splits from the seed in any case.
* **X-IIoTID and CICIoT2023.** Splits are regenerated deterministically from the seed by `src/cross_dataset.py`. CICIoT2023 is audited in full and benchmarked on a class-stratified sample of at most 30,000 records per class (seeded).
* The edge benchmark reads 2,000 real test records (`bench_inputs_*.npz`) exported by notebook 01. They are derived from Edge-IIoTset and are therefore not distributed; without them `edge_bench/bench.py` uses synthetic inputs of the same shape.
