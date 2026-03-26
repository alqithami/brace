# BRACE: Budgeted Rumour-Risk Containment with Calibrated Signals and No-Leak Simulation

This repository accompanies **BRACE**, a leakage-aware benchmark for **budgeted rumour-risk containment under partial observability**. The project combines calibrated binary risk estimation, a root-text panic/urgency proxy, a no-leak step-wise diffusion simulator, and budget-constrained containment policies, including transparent greedy baselines and PPO.

The repository is intended to support **targeted reruns and extension of the experimental pipeline**.

---

## What BRACE covers

BRACE is organized around four components:

1. **Risk estimation and calibration**
   - Binary rumour-risk baselines (TF-IDF, structure-only, hybrid, Transformer)
   - Post-hoc calibration, with temperature scaling used as the default downstream operational signal in the paper

2. **Urgency / panic features**
   - Root-text urgency proxy derived from tweet-domain affect modeling
   - Precomputed panic feature files are stored under `features/`

3. **No-leak simulation**
   - Step-wise diffusion simulator trained only on prefix-observable quantities
   - Used for offline counterfactual containment evaluation

4. **Budgeted containment**
   - Greedy reference policies and PPO-based containment policy
   - Hard budgets and per-step caps for operational-style evaluation

---

## Repository contents

```text
brace/
├── data_processed/
│   ├── twitter15/
│   │   ├── cascades.parquet
│   │   ├── edges.parquet
│   │   └── nodes.parquet
│   ├── twitter16/
│   │   ├── cascades.parquet
│   │   ├── edges.parquet
│   │   └── nodes.parquet
│   └── pheme/
│       ├── cascades.parquet
│       ├── edges.parquet
│       └── nodes.parquet
├── features/
│   ├── panic_pheme.parquet
│   ├── panic_twitter15.parquet
│   └── panic_twitter16.parquet
├── logs/
├── rdic_pipeline/
│   ├── configs/
│   ├── rdic/
│   ├── scripts/
│   ├── README.md
│   └── requirements-mac.txt
├── splits/
│   ├── twitter15/within/
│   ├── twitter16/within/
│   ├── twitter_cross/
│   └── pheme/loeo/
├── 14_train_rumor_transformer.py
├── collect_transformer_binary_metrics.py
├── metrics_files.txt
└── requirements-m4max.txt
```

### Main assets already included

- **Processed datasets** for Twitter15, Twitter16, and PHEME in Parquet format
- **Within-dataset, cross-dataset, and PHEME LOEO split files**
- **Precomputed panic features** under `features/`
- **Training and preprocessing scripts** under `rdic_pipeline/scripts/`
- **Experiment logs** under `logs/`
- **Transformer metric collection utility**

---

## Datasets used in the paper

| Dataset | Evaluation units | Graph nodes | Text availability | Primary use |
|---|---:|---:|---|---|
| Twitter15 | 1,490 cascades | 56,069 | root text only | Detection + containment |
| Twitter16 | 818 cascades | 27,688 | root text only | Detection + containment |
| PHEME | 6,425 threads | 104,582 | root + replies | Detection only (LOEO) |

### Notes

- In the public Twitter15/16 release used here, **non-root reply text is not available**, so text-based components operate on the root post only.
- PHEME is used for **event-level detection generalization** and not for the containment policy results.

---

## Split protocol

The repository includes split files for:

- **Twitter15 within-dataset** evaluation
- **Twitter16 within-dataset** evaluation
- **Twitter15 → Twitter16** and **Twitter16 → Twitter15** transfer evaluation
- **PHEME leave-one-event-out (LOEO)** evaluation

The paper removes overlapping Twitter15/16 cascade identifiers before cross-dataset transfer evaluation.

---

## Environment setup

### Recommended Python version

- **Python 3.11** is recommended.

### Option A: Apple Silicon / M-series setup

The repository currently includes `requirements-m4max.txt` for the Apple M4 Max environment used in the experiments.

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-m4max.txt
pip install gymnasium stable-baselines3
```

### Option B: Generic environment template

A generic starting point is provided in `environment.yml.template` (see companion files supplied with this README draft).

### Suggested runtime flags

```bash
export PYTHONPATH="$(pwd)/rdic_pipeline:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

---

## Quick start

### 1) Use the processed release already included

If you only want to inspect or rerun parts of the pipeline, the processed artifacts are already present:

- `data_processed/`
- `features/`
- `splits/`

You can begin directly with training / evaluation scripts.

### 2) Train a TF-IDF risk model

```bash
python rdic_pipeline/scripts/10_train_rumor_tfidf.py --dataset twitter15 --split within
```

### 3) Calibrate the risk scores

```bash
python rdic_pipeline/scripts/11_calibrate_temperature.py --dataset twitter15 --split within
```

### 4) Train the emotion model and infer panic features

```bash
python rdic_pipeline/scripts/20_train_emotion.py --stage tweeteval
python rdic_pipeline/scripts/20_train_emotion.py --stage semeval --init-from models/emotion/tweeteval
python rdic_pipeline/scripts/21_infer_panic.py --dataset twitter15
```

### 5) Train the step simulator

```bash
python rdic_pipeline/scripts/30_train_simulator_step.py --dataset twitter15 --split within
```

### 6) Train the PPO containment policy

```bash
python rdic_pipeline/scripts/40_train_policy_ppo.py --dataset twitter15 --split within --max-episodes 200
```

### 7) Run the Transformer baseline

```bash
python rdic_pipeline/scripts/14_train_rumor_transformer_binary.py --task within --dataset twitter15
```

### 8) Collect binary Transformer metrics

```bash
python collect_transformer_binary_metrics.py
```

---

## Rebuilding from raw data

For a raw-data rebuild, place the expected ZIP files under `data_raw/` and use the auto-discovery pipeline documented in `rdic_pipeline/README.md`.

Typical sequence:

```bash
python rdic_pipeline/scripts/00_prepare_data.py --data-raw data_raw
python rdic_pipeline/scripts/00_validate_paths.py --data-raw data_raw
python rdic_pipeline/scripts/01_preprocess_pheme.py --data-raw data_raw
python rdic_pipeline/scripts/02_preprocess_twitter1516.py --data-raw data_raw
python rdic_pipeline/scripts/03_preprocess_emotion.py --data-raw data_raw
python rdic_pipeline/scripts/04_make_splits.py --data-raw data_raw
```

Outputs are written to:

- `data_processed/`
- `splits/`

---

## Mapping to the paper

The repository is meant to support the experiments reported in:

> **Budgeted Rumour-Risk Containment with Calibrated Signals and No-Leak Simulation**

A practical reading guide for reviewers:

- **Detection and transfer**: see the TF-IDF / structure / hybrid / Transformer scripts and the split files
- **Calibration**: see temperature-scaling scripts and panic feature generation
- **Simulation**: see `30_train_simulator_step.py`
- **Containment**: see `40_train_policy_ppo.py` and the greedy reference logic used in evaluation

### Representative headline results from the paper

- TF-IDF SVM is the strongest no-leak within-dataset detector in the reported root-only setting.
- Temperature scaling improves reliability and is the default downstream risk signal in the main containment tables.
- On Twitter16, matched-budget PPO is competitive with strong greedy baselines; under tighter unseen budgets, transfer degrades and matched-budget retraining restores substantial performance.

---

## Release status and reproducibility note

This public snapshot is **useful and non-trivial**.

### Present in the current snapshot

- processed Parquet data
- split files
- panic feature files
- preprocessing / training scripts
- logs
- Transformer utilities

---

## Citation

If you use this repository, please cite the associated paper. 

---

## Contact

For questions about the code release, experiment organization, or artifact mapping, please contact the repository maintainer.
