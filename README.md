# Binary Transformer Baseline


## Purpose
Run **binary** (rumour vs non-rumour) Transformer experiments so the comparison is **fair** vs:
- TF-IDF (binary)
- Structure/Hybrid (binary)
- RL (binary risk)

---

# 1) Where to place the script (inside your existing repo)
Put the training script here:
- `~/RDIC/rdic_pipeline/scripts/14_train_rumor_transformer_binary.py`

Put the collector here (optional but recommended):
- `~/RDIC/collect_transformer_binary_metrics.py`

---

# 2) Terminal session setup (run once per terminal)
```bash
cd ~/RDIC
source .venv/bin/activate
export PYTHONPATH="$(pwd)/rdic_pipeline:$PYTHONPATH"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ENABLE_MPS_FALLBACK=1
mkdir -p logs models/rumor_transformer_binary
```

---

# 3) Create the training script (copy/paste)
Run:
```bash
cat > rdic_pipeline/scripts/14_train_rumor_transformer_binary.py <<'PY'
#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

def load_ids(path: Path) -> list[str]:
    return pd.read_csv(path)["cascade_id"].astype(str).tolist()

def metrics_binary(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_hat = (p >= 0.5).astype(int)
    out = {
        "macro_f1": float(f1_score(y_true, y_hat, average="macro")),
        "acc": float(accuracy_score(y_true, y_hat)),
    }
    out["auc"] = float(roc_auc_score(y_true, p)) if len(np.unique(y_true)) > 1 else float("nan")
    return out

def to_hf_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    ds = Dataset.from_dict({"text": df["text"].tolist(), "label": df["label"].tolist()})

    def tok_fn(ex):
        tok = tokenizer(ex["text"], truncation=True, max_length=max_length)
        tok["labels"] = ex["label"]
        return tok

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    return ds

def get_within(processed: Path, splits: Path, dataset: str):
    c = pd.read_parquet(processed / dataset / "cascades.parquet")
    c["cascade_id"] = c["cascade_id"].astype(str)
    c["label"] = c["label_binary"].astype(int)
    c["text"] = c["source_text"].fillna("").astype(str)

    d = splits / dataset / "within"
    tr = set(load_ids(d / "train.csv"))
    dv = set(load_ids(d / "dev.csv"))
    te = set(load_ids(d / "test.csv"))

    train = c[c["cascade_id"].isin(tr)][["text", "label"]]
    dev   = c[c["cascade_id"].isin(dv)][["text", "label"]]
    test  = c[c["cascade_id"].isin(te)][["text", "label"]]
    return train, dev, test

def get_cross(processed: Path, splits: Path, direction: str):
    d = splits / "twitter_cross"
    if direction == "15to16":
        train_ds, test_ds = "twitter15", "twitter16"
        tr = set(load_ids(d / "15to16_train.csv"))
        dv = set(load_ids(d / "15to16_dev.csv"))
        te = set(load_ids(d / "15to16_test.csv"))
    elif direction == "16to15":
        train_ds, test_ds = "twitter16", "twitter15"
        tr = set(load_ids(d / "16to15_train.csv"))
        dv = set(load_ids(d / "16to15_dev.csv"))
        te = set(load_ids(d / "16to15_test.csv"))
    else:
        raise ValueError("direction must be 15to16 or 16to15")

    c_tr = pd.read_parquet(processed / train_ds / "cascades.parquet")
    c_te = pd.read_parquet(processed / test_ds / "cascades.parquet")

    for c in (c_tr, c_te):
        c["cascade_id"] = c["cascade_id"].astype(str)
        c["label"] = c["label_binary"].astype(int)
        c["text"] = c["source_text"].fillna("").astype(str)

    train = c_tr[c_tr["cascade_id"].isin(tr)][["text", "label"]]
    dev   = c_tr[c_tr["cascade_id"].isin(dv)][["text", "label"]]
    test  = c_te[c_te["cascade_id"].isin(te)][["text", "label"]]
    return train, dev, test, train_ds, test_ds

def get_pheme_loeo(processed: Path, splits: Path, event: str):
    c = pd.read_parquet(processed / "pheme" / "cascades.parquet")
    c["cascade_id"] = c["cascade_id"].astype(str)
    c["label"] = c["label_binary"].astype(int)
    c["text"] = c["source_text"].fillna("").astype(str)

    fold = splits / "pheme" / "loeo" / f"fold_{event}"
    tr = set(load_ids(fold / "train.csv"))
    dv = set(load_ids(fold / "dev.csv"))
    te = set(load_ids(fold / "test.csv"))

    train = c[c["cascade_id"].isin(tr)][["text", "label"]]
    dev   = c[c["cascade_id"].isin(dv)][["text", "label"]]
    test  = c[c["cascade_id"].isin(te)][["text", "label"]]
    return train, dev, test

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["within", "cross", "pheme_loeo"])
    ap.add_argument("--dataset", default="twitter15")
    ap.add_argument("--direction", default="15to16")
    ap.add_argument("--event", default=None)
    ap.add_argument("--processed", default="data_processed")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--model", default="cardiffnlp/twitter-roberta-base")
    ap.add_argument("--outdir", default="models/rumor_transformer_binary")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", type=str, default=None, help="Optional run tag to avoid overwriting. Default: timestamp.")
    args = ap.parse_args()

    processed = Path(args.processed)
    splits = Path(args.splits)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.task == "within":
        train, dev, test = get_within(processed, splits, args.dataset)
        base_id = f"binary_within_{args.dataset}"
    elif args.task == "cross":
        train, dev, test, tr_ds, te_ds = get_cross(processed, splits, args.direction)
        base_id = f"binary_cross_{args.direction}_{tr_ds}_to_{te_ds}"
    else:
        if not args.event:
            raise SystemExit("--event is required for pheme_loeo")
        train, dev, test = get_pheme_loeo(processed, splits, args.event)
        base_id = f"binary_pheme_loeo_{args.event}"

    if len(train) == 0 or len(dev) == 0 or len(test) == 0:
        raise SystemExit(f"Empty split: train={len(train)} dev={len(dev)} test={len(test)}")

    tag = args.tag or datetime.now().strftime("%Y%m%d-%H%M%S")
    model_slug = args.model.replace("/", "-")
    run_id = f"{base_id}_{model_slug}_seed{args.seed}_{tag}"

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_ds = to_hf_dataset(train, tokenizer, args.max_length)
    dev_ds   = to_hf_dataset(dev, tokenizer, args.max_length)
    test_ds  = to_hf_dataset(test, tokenizer, args.max_length)

    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
        seed=args.seed,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=dev_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(str(run_dir / "best"))
    tokenizer.save_pretrained(str(run_dir / "best"))

    dev_pred = trainer.predict(dev_ds).predictions
    test_pred = trainer.predict(test_ds).predictions

    dev_p = torch.softmax(torch.tensor(dev_pred), dim=1).numpy()[:, 1]
    test_p = torch.softmax(torch.tensor(test_pred), dim=1).numpy()[:, 1]

    dev_m = metrics_binary(np.array(dev["label"].values), dev_p)
    test_m = metrics_binary(np.array(test["label"].values), test_p)

    metrics = {"task": args.task, "run_id": run_id, "model": args.model, "dev": dev_m, "test": test_m}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame({"y": dev["label"].values, "p": dev_p}).to_parquet(run_dir / "dev_preds.parquet", index=False)
    pd.DataFrame({"y": test["label"].values, "p": test_p}).to_parquet(run_dir / "test_preds.parquet", index=False)

    print("Saved:", run_dir)
    print("Dev:", dev_m)
    print("Test:", test_m)

if __name__ == "__main__":
    main()

PY

chmod +x rdic_pipeline/scripts/14_train_rumor_transformer_binary.py
```

---

# 4) Run the binary experiments (sequential)

## 4.1 Within — Twitter15 (binary)
```bash
python rdic_pipeline/scripts/14_train_rumor_transformer_binary.py --task within --dataset twitter15 \
  --processed data_processed --splits splits \
  --model cardiffnlp/twitter-roberta-base \
  --epochs 3 --batch 8 --grad_accum 4 --max_length 128 --seed 42 \
  2>&1 | tee logs/bin_within_twitter15.log
```

## 4.2 Within — Twitter16 (binary)
```bash
python rdic_pipeline/scripts/14_train_rumor_transformer_binary.py --task within --dataset twitter16 \
  --processed data_processed --splits splits \
  --model cardiffnlp/twitter-roberta-base \
  --epochs 3 --batch 8 --grad_accum 4 --max_length 128 --seed 42 \
  2>&1 | tee logs/bin_within_twitter16.log
```

## 4.3 Cross — 15→16 (binary)
```bash
python rdic_pipeline/scripts/14_train_rumor_transformer_binary.py --task cross --direction 15to16 \
  --processed data_processed --splits splits \
  --model cardiffnlp/twitter-roberta-base \
  --epochs 3 --batch 8 --grad_accum 4 --max_length 128 --seed 42 \
  2>&1 | tee logs/bin_cross_15to16.log
```

## 4.4 Cross — 16→15 (binary)
```bash
python rdic_pipeline/scripts/14_train_rumor_transformer_binary.py --task cross --direction 16to15 \
  --processed data_processed --splits splits \
  --model cardiffnlp/twitter-roberta-base \
  --epochs 3 --batch 8 --grad_accum 4 --max_length 128 --seed 42 \
  2>&1 | tee logs/bin_cross_16to15.log
```

## 4.5 PHEME LOEO — all events (binary) loop
```bash
for d in splits/pheme/loeo/fold_*; do
  ev=$(basename "$d" | sed 's/fold_//')
  python rdic_pipeline/scripts/14_train_rumor_transformer_binary.py --task pheme_loeo --event "$ev" \
    --processed data_processed --splits splits \
    --model roberta-base \
    --epochs 3 --batch 8 --grad_accum 4 --max_length 128 --seed 42 \
    2>&1 | tee "logs/bin_pheme_loeo_${ev}.log"
done
```

### Memory tip (if needed)
If you hit memory pressure on MPS:
- use `--batch 4 --grad_accum 8`

---

# 5) Collect results into ONE small file (easy to send)

## 5.1 Create the collector script
```bash
cat > collect_transformer_binary_metrics.py <<'PY'
import json, glob, os
import pandas as pd
import subprocess

rows=[]
for path in glob.glob("models/rumor_transformer_binary/**/metrics.json", recursive=True):
    with open(path,"r",encoding="utf-8") as f:
        m=json.load(f)
    rows.append({
        "run_dir": os.path.dirname(path),
        "model": m.get("model"),
        "task": m.get("task"),
        "run_id": m.get("run_id"),
        "dev_macro_f1": m["dev"].get("macro_f1"),
        "dev_auc": m["dev"].get("auc"),
        "dev_acc": m["dev"].get("acc"),
        "test_macro_f1": m["test"].get("macro_f1"),
        "test_auc": m["test"].get("auc"),
        "test_acc": m["test"].get("acc"),
    })

df=pd.DataFrame(rows).sort_values("run_dir")
df.to_csv("transformer_binary_metrics_summary.csv", index=False)
print("Wrote transformer_binary_metrics_summary.csv with", len(df), "rows")

with open("metrics_files.txt","w",encoding="utf-8") as f:
    for p in glob.glob("models/rumor_transformer_binary/**/metrics.json", recursive=True):
        f.write(p + "\n")
subprocess.run(["tar","-czf","transformer_binary_metrics_only.tgz","-T","metrics_files.txt"], check=True)
print("Wrote transformer_binary_metrics_only.tgz")

PY
```

## 5.2 Run collector
```bash
python3 collect_transformer_binary_metrics.py
ls -lh transformer_binary_metrics_summary.csv transformer_binary_metrics_only.tgz
```

**Recommended file to send for analysis:**
- `transformer_binary_metrics_summary.csv`

(Optional):
- `transformer_binary_metrics_only.tgz`

---

# 6) Send results back (example)
```bash
scp transformer_binary_metrics_summary.csv USER@INTEL_HOST:./RDIC/
```

---

## After you upload the CSV here
I will:
- Produce a fair comparison vs TF-IDF/Hybrid and RL (binary)
- Generate LaTeX-ready tables and results text for the paper
