# RDIC: Preprocessing + Training Pipeline (Auto-Path Discovery)

This repo provides **end-to-end scripts** for:
- Preparing raw ZIPs (optional) and auto-discovering dataset paths
- Preprocessing: PHEME (all-rnr-annotated-threads), Twitter15/16 (trees + root text), TweetEval Emotion, SemEval-2018 Affect
- Creating splits (within + cross with overlap removal + PHEME leave-one-event-out)
- Training baselines (TF-IDF LR/SVM), calibration (temperature scaling)
- Training emotion model (TweetEval -> SemEval multi-label) and extracting panic features
- Training a step-level simulator and a PPO policy on CPU (sanity runs)

> **No manual path edits.** Put your ZIPs (or extracted folders) under `data_raw/`.  
> The scripts will discover them automatically.

## 0) Create venv (Python 3.11 recommended)
```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
pip install -r requirements-mac.txt
```

## 1) Put data
Create:
```
data_raw/
  pheme.zip
  Twitter15_16_dataset-main.zip
  Twitter15 and Twitter16.zip
  tweeteval.zip
  SemEval 2018.zip
```
(Or extracted equivalents.)

## 2) Prepare (optional extraction) + validate discovery
```bash
python scripts/00_prepare_data.py --data-raw data_raw
python scripts/00_validate_paths.py --data-raw data_raw
```

## 3) Preprocess
```bash
python scripts/01_preprocess_pheme.py --data-raw data_raw
python scripts/02_preprocess_twitter1516.py --data-raw data_raw
python scripts/03_preprocess_emotion.py --data-raw data_raw
python scripts/04_make_splits.py --data-raw data_raw
```

Outputs go to:
- `data_processed/`
- `splits/`

## 4) Train baselines + calibration
Example (Twitter15 within):
```bash
python scripts/10_train_rumor_tfidf.py --dataset twitter15 --split within
python scripts/11_calibrate_temperature.py --dataset twitter15 --split within
```

## 5) Train emotion model + infer panic
```bash
python scripts/20_train_emotion.py --stage tweeteval
python scripts/20_train_emotion.py --stage semeval --init-from models/emotion/tweeteval
python scripts/21_infer_panic.py --dataset pheme
```

## 6) Simulator + PPO (CPU sanity)
```bash
python scripts/30_train_simulator_step.py --dataset twitter15 --split within
python scripts/40_train_policy_ppo.py --dataset twitter15 --split within --max-episodes 200
```

## Notes
- Twitter15/16 public releases usually do **not** contain texts for non-root nodes. We use root text + structure/time features for all nodes.
- PHEME provides full JSON texts (source + reactions), so emotion extraction runs on all nodes.
