#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import json
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

def _load_ids(csv_path: Path) -> list[str]:
    return pd.read_csv(csv_path)["cascade_id"].astype(str).tolist()

def _metrics_from_probs(y_true: np.ndarray, p: np.ndarray) -> dict:
    y_true = y_true.astype(int)
    y_hat = (p >= 0.5).astype(int)
    out = {
        "macro_f1": float(f1_score(y_true, y_hat, average="macro")),
        "acc": float(accuracy_score(y_true, y_hat)),
    }
    if len(np.unique(y_true)) > 1:
        out["auc"] = float(roc_auc_score(y_true, p))
    else:
        out["auc"] = float("nan")
    return out

def _prepare_dataset(df: pd.DataFrame, tokenizer, max_length: int) -> Dataset:
    df = df[["text", "label"]].copy()
    ds = Dataset.from_pandas(df, preserve_index=False)

    def tok_fn(ex):
        tok = tokenizer(ex["text"], truncation=True, max_length=max_length)
        tok["labels"] = ex["label"]
        return tok

    ds = ds.map(tok_fn, batched=True, remove_columns=ds.column_names)
    return ds

def _get_splits_within(processed: Path, splits: Path, dataset: str):
    c = pd.read_parquet(processed / dataset / "cascades.parquet")
    c["cascade_id"] = c["cascade_id"].astype(str)
    c["label"] = c["label_binary"].astype(int)
    c["text"] = c["source_text"].fillna("").astype(str)

    split_dir = splits / dataset / "within"
    train_ids = set(_load_ids(split_dir / "train.csv"))
    dev_ids   = set(_load_ids(split_dir / "dev.csv"))
    test_ids  = set(_load_ids(split_dir / "test.csv"))

    train = c[c["cascade_id"].isin(train_ids)][["text", "label"]]
    dev   = c[c["cascade_id"].isin(dev_ids)][["text", "label"]]
    test  = c[c["cascade_id"].isin(test_ids)][["text", "label"]]
    return train, dev, test

def _get_splits_pheme_loeo(processed: Path, splits: Path, event: str):
    dataset = "pheme"
    c = pd.read_parquet(processed / dataset / "cascades.parquet")
    c["cascade_id"] = c["cascade_id"].astype(str)
    c["label"] = c["label_binary"].astype(int)
    c["text"] = c["source_text"].fillna("").astype(str)

    fold_dir = splits / "pheme" / "loeo" / f"fold_{event}"
    train_ids = set(_load_ids(fold_dir / "train.csv"))
    dev_ids   = set(_load_ids(fold_dir / "dev.csv"))
    test_ids  = set(_load_ids(fold_dir / "test.csv"))

    train = c[c["cascade_id"].isin(train_ids)][["text", "label"]]
    dev   = c[c["cascade_id"].isin(dev_ids)][["text", "label"]]
    test  = c[c["cascade_id"].isin(test_ids)][["text", "label"]]
    return train, dev, test

def _get_splits_cross(processed: Path, splits: Path, direction: str):
    cross_dir = splits / "twitter_cross"
    if direction == "15to16":
        train_ds, test_ds = "twitter15", "twitter16"
        train_ids = set(_load_ids(cross_dir / "15to16_train.csv"))
        dev_ids   = set(_load_ids(cross_dir / "15to16_dev.csv"))
        test_ids  = set(_load_ids(cross_dir / "15to16_test.csv"))
    elif direction == "16to15":
        train_ds, test_ds = "twitter16", "twitter15"
        train_ids = set(_load_ids(cross_dir / "16to15_train.csv"))
        dev_ids   = set(_load_ids(cross_dir / "16to15_dev.csv"))
        test_ids  = set(_load_ids(cross_dir / "16to15_test.csv"))
    else:
        raise ValueError("direction must be 15to16 or 16to15")

    c_train = pd.read_parquet(processed / train_ds / "cascades.parquet")
    c_test  = pd.read_parquet(processed / test_ds / "cascades.parquet")

    for c in (c_train, c_test):
        c["cascade_id"] = c["cascade_id"].astype(str)
        c["label"] = c["label_binary"].astype(int)
        c["text"] = c["source_text"].fillna("").astype(str)

    train = c_train[c_train["cascade_id"].isin(train_ids)][["text", "label"]]
    dev   = c_train[c_train["cascade_id"].isin(dev_ids)][["text", "label"]]
    test  = c_test[c_test["cascade_id"].isin(test_ids)][["text", "label"]]
    return train, dev, test, train_ds, test_ds

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["within", "cross", "pheme_loeo"])
    ap.add_argument("--dataset", default="twitter15", help="twitter15|twitter16|pheme (for within)")
    ap.add_argument("--direction", default="15to16", help="15to16|16to15 (for cross)")
    ap.add_argument("--event", default=None, help="PHEME event name (for pheme_loeo)")
    ap.add_argument("--processed", default="data_processed")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--model", default="distilroberta-base")
    ap.add_argument("--outdir", default="models/rumor_transformer")
    ap.add_argument("--run_name", default=None)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=4)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    args = ap.parse_args()

    processed = Path(args.processed)
    splits = Path(args.splits)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.task == "within":
        train, dev, test = _get_splits_within(processed, splits, args.dataset)
        run_id = args.run_name or f"{args.dataset}_within"
    elif args.task == "pheme_loeo":
        if not args.event:
            raise SystemExit("--event is required for pheme_loeo")
        train, dev, test = _get_splits_pheme_loeo(processed, splits, args.event)
        run_id = args.run_name or f"pheme_loeo_{args.event}"
    else:
        train, dev, test, train_ds, test_ds = _get_splits_cross(processed, splits, args.direction)
        run_id = args.run_name or f"cross_{args.direction}_{train_ds}_to_{test_ds}"

    if len(train) == 0 or len(dev) == 0 or len(test) == 0:
        raise SystemExit(f"Empty split detected. Sizes: train={len(train)}, dev={len(dev)}, test={len(test)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2)

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    train_ds = _prepare_dataset(train, tokenizer, args.max_length)
    dev_ds   = _prepare_dataset(dev, tokenizer, args.max_length)
    test_ds  = _prepare_dataset(test, tokenizer, args.max_length)

    run_dir = outdir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(run_dir),
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

    dev_pred = trainer.predict(dev_ds)
    test_pred = trainer.predict(test_ds)

    dev_probs = torch.softmax(torch.tensor(dev_pred.predictions), dim=1).numpy()[:, 1]
    test_probs = torch.softmax(torch.tensor(test_pred.predictions), dim=1).numpy()[:, 1]

    dev_metrics = _metrics_from_probs(np.array(dev["label"].values), dev_probs)
    test_metrics = _metrics_from_probs(np.array(test["label"].values), test_probs)

    metrics = {"dev": dev_metrics, "test": test_metrics, "run_id": run_id, "task": args.task, "model": args.model}
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pd.DataFrame({"y": dev["label"].values, "p": dev_probs}).to_parquet(run_dir / "dev_preds.parquet", index=False)
    pd.DataFrame({"y": test["label"].values, "p": test_probs}).to_parquet(run_dir / "test_preds.parquet", index=False)

    print("Saved run to:", run_dir)
    print("Dev:", dev_metrics)
    print("Test:", test_metrics)

if __name__ == "__main__":
    main()
