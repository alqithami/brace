#!/usr/bin/env python3
"""
RDIC - Rumor Transformer Trainer (macOS / MPS-friendly)

Writes (per run):
  models/rumor_transformer/<run_id>/metrics.json
  models/rumor_transformer/<run_id>/dev_preds.parquet
  models/rumor_transformer/<run_id>/test_preds.parquet
  models/rumor_transformer/<run_id>/best/

This script is designed to match the folder layout used in the M4 Max runbook:
  data_processed/<dataset>/cascades.parquet
  splits/<dataset>/within/{train,dev,test}.csv
  splits/twitter_cross/<direction>_{train,dev,test}.csv
  splits/pheme/loeo/fold_<event>/{train,dev,test}.csv

It tries to be robust to different column names by auto-detecting ID/TEXT/LABEL columns.
If your parquet/csv schema differs, it will print helpful diagnostics.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

ID_CANDIDATES = [
    "cascade_id", "cid", "thread_id", "conversation_id", "conv_id",
    "tweet_id", "post_id", "id", "root_id", "source_id", "mid"
]
TEXT_CANDIDATES = [
    "text", "tweet_text", "source_text", "root_text", "content",
    "body", "post_text", "clean_text", "text_clean"
]
LABEL_CANDIDATES = [
    "label", "veracity", "target", "y", "class", "rumor_label", "stance"
]

DEV_FILENAMES = ["dev.csv", "val.csv", "valid.csv", "validation.csv"]
TEST_FILENAMES = ["test.csv", "eval.csv", "evaluation.csv"]


def _slug(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^A-Za-z0-9]+", "-", str(s)).strip("-")
    return s[:max_len] if len(s) > max_len else s


def _first_existing(paths: Sequence[Path]) -> Optional[Path]:
    for p in paths:
        if p.exists():
            return p
    return None


def _pick_col(df: pd.DataFrame, candidates: Sequence[str], what: str) -> Optional[str]:
    cols_lower = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand.lower() in cols_lower:
            return cols_lower[cand.lower()]
    # fallback: for text, pick a likely object/string column
    if what == "text":
        obj_cols = [c for c in df.columns if df[c].dtype == object]
        # heuristic: prefer columns containing 'text'
        for c in obj_cols:
            if "text" in c.lower():
                return c
        if obj_cols:
            return obj_cols[0]
    return None


def _read_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing parquet: {path}")
    return pd.read_parquet(path)


def _read_split_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split CSV: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"Split CSV is empty: {path}")

    # normalize colnames (strip only)
    df.columns = [str(c).strip() for c in df.columns]

    # detect ID column
    id_col = None
    for c in df.columns:
        if c.lower() in {x.lower() for x in ID_CANDIDATES}:
            id_col = c
            break
    if id_col is None:
        id_col = df.columns[0]

    # detect label column (optional)
    label_col = None
    for c in df.columns:
        if c.lower() in {x.lower() for x in LABEL_CANDIDATES}:
            label_col = c
            break
    # fallback: if 2+ columns, assume second column might be label
    if label_col is None and len(df.columns) >= 2:
        if df.columns[1] != id_col:
            label_col = df.columns[1]

    out = pd.DataFrame({"_id": df[id_col].astype(str)})
    if label_col is not None and label_col in df.columns:
        out["_label"] = df[label_col].astype(str)
    return out


def _merge_cascades_split(
    cascades: pd.DataFrame,
    split_df: pd.DataFrame,
    id_col: str,
    text_col: str,
    casc_label_col: Optional[str],
) -> pd.DataFrame:
    c = cascades.copy()
    c["_id"] = c[id_col].astype(str)

    merged = c.merge(split_df, on="_id", how="inner", validate="many_to_one")
    if merged.empty:
        raise ValueError(
            "After merging cascades with split IDs, got 0 rows.\n"
            f"- cascades id_col='{id_col}' example ids: {c['_id'].head(5).tolist()}\n"
            f"- split example ids: {split_df['_id'].head(5).tolist()}\n"
            "This usually means the split CSV IDs don't match the parquet ID column."
        )

    # text
    merged["text"] = merged[text_col].fillna("").astype(str)

    # labels: prefer split-provided label if present; else use parquet label
    if "_label" in merged.columns:
        merged["label_raw"] = merged["_label"].astype(str)
    elif casc_label_col is not None and casc_label_col in merged.columns:
        merged["label_raw"] = merged[casc_label_col].astype(str)
    else:
        raise ValueError(
            "No label column found in split CSV or cascades parquet.\n"
            f"- cascades columns: {list(cascades.columns)}\n"
            f"- split columns: {list(split_df.columns)}\n"
            "Expected a label-like column in either."
        )

    return merged[["_id", "text", "label_raw"]].reset_index(drop=True)


def _make_label_mapping(dfs: Sequence[pd.DataFrame]) -> Tuple[Dict[str, int], Dict[int, str]]:
    all_labels = []
    for df in dfs:
        if df is None or df.empty:
            continue
        all_labels.extend(df["label_raw"].astype(str).tolist())
    uniq = sorted(set(all_labels))
    label2id = {lab: i for i, lab in enumerate(uniq)}
    id2label = {i: lab for lab, i in label2id.items()}
    return label2id, id2label


def _add_numeric_labels(df: pd.DataFrame, label2id: Dict[str, int]) -> pd.DataFrame:
    df = df.copy()
    df["labels"] = df["label_raw"].astype(str).map(label2id).astype(int)
    return df


def _split_train_dev_if_needed(train_df: pd.DataFrame, dev_df: Optional[pd.DataFrame], seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if dev_df is not None:
        return train_df, dev_df

    # default: 10% dev split from train
    y = train_df["label_raw"].astype(str)
    stratify = y if len(set(y.tolist())) > 1 else None
    tr, dv = train_test_split(
        train_df,
        test_size=0.10,
        random_state=seed,
        stratify=stratify,
    )
    return tr.reset_index(drop=True), dv.reset_index(drop=True)


def _compute_metrics_fn():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(labels, preds)
        f1_macro = f1_score(labels, preds, average="macro")
        f1_weighted = f1_score(labels, preds, average="weighted")
        p_macro, r_macro, _, _ = precision_recall_fscore_support(labels, preds, average="macro", zero_division=0)
        return {
            "acc": float(acc),
            "f1_macro": float(f1_macro),
            "f1_weighted": float(f1_weighted),
            "p_macro": float(p_macro),
            "r_macro": float(r_macro),
        }
    return compute_metrics


@dataclass
class SplitPaths:
    train: Path
    dev: Optional[Path]
    test: Path
    train_dataset: str
    test_dataset: str


def _resolve_paths(args) -> SplitPaths:
    splits_root = Path(args.splits)

    if args.task == "within":
        ds = args.dataset
        base = splits_root / ds / "within"
        train = base / "train.csv"
        dev = _first_existing([base / n for n in DEV_FILENAMES])
        test = _first_existing([base / n for n in TEST_FILENAMES]) or (base / "test.csv")
        return SplitPaths(train=train, dev=dev, test=test, train_dataset=ds, test_dataset=ds)

    if args.task == "cross":
        if not args.direction:
            raise ValueError("--direction is required for task=cross (15to16 or 16to15).")
        direction = args.direction.strip()
        if direction not in ("15to16", "16to15"):
            raise ValueError("--direction must be 15to16 or 16to15")

        train_ds = "twitter15" if direction == "15to16" else "twitter16"
        test_ds = "twitter16" if direction == "15to16" else "twitter15"

        base = splits_root / "twitter_cross"
        train = base / f"{direction}_train.csv"
        dev = _first_existing([base / f"{direction}_{n}" for n in DEV_FILENAMES])
        test = _first_existing([base / f"{direction}_{n}" for n in TEST_FILENAMES]) or (base / f"{direction}_test.csv")

        return SplitPaths(train=train, dev=dev, test=test, train_dataset=train_ds, test_dataset=test_ds)

    if args.task == "pheme_loeo":
        if not args.event:
            raise ValueError("--event is required for task=pheme_loeo")
        ev = args.event.strip()
        ev = ev.replace("fold_", "")
        base = splits_root / "pheme" / "loeo" / f"fold_{ev}"
        train = base / "train.csv"
        dev = _first_existing([base / n for n in DEV_FILENAMES])
        test = _first_existing([base / n for n in TEST_FILENAMES]) or (base / "test.csv")
        return SplitPaths(train=train, dev=dev, test=test, train_dataset="pheme", test_dataset="pheme")

    raise ValueError(f"Unknown task: {args.task}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["within", "cross", "pheme_loeo"])
    ap.add_argument("--dataset", default="twitter15", help="Used for task=within (twitter15/twitter16).")
    ap.add_argument("--direction", default=None, help="Used for task=cross: 15to16 or 16to15.")
    ap.add_argument("--event", default=None, help="Used for task=pheme_loeo: e.g. charliehebdo.")
    ap.add_argument("--processed", default="data_processed")
    ap.add_argument("--splits", default="splits")
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.01)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--run_id", default=None, help="Optional override for run directory name.")
    args = ap.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    set_seed(args.seed)

    paths = _resolve_paths(args)

    # Load processed data
    proc_root = Path(args.processed)
    train_parquet = proc_root / paths.train_dataset / "cascades.parquet"
    test_parquet = proc_root / paths.test_dataset / "cascades.parquet"

    train_cascades = _read_parquet(train_parquet)
    test_cascades = train_cascades if paths.test_dataset == paths.train_dataset else _read_parquet(test_parquet)

    # Detect columns (separately, since schemas might differ)
    train_id_col = _pick_col(train_cascades, ID_CANDIDATES, "id") or train_cascades.columns[0]
    train_text_col = _pick_col(train_cascades, TEXT_CANDIDATES, "text")
    train_label_col = _pick_col(train_cascades, LABEL_CANDIDATES, "label")

    test_id_col = _pick_col(test_cascades, ID_CANDIDATES, "id") or test_cascades.columns[0]
    test_text_col = _pick_col(test_cascades, TEXT_CANDIDATES, "text")
    test_label_col = _pick_col(test_cascades, LABEL_CANDIDATES, "label")

    if train_text_col is None:
        raise ValueError(f"Could not infer text column in {train_parquet}. Columns: {list(train_cascades.columns)}")
    if test_text_col is None:
        raise ValueError(f"Could not infer text column in {test_parquet}. Columns: {list(test_cascades.columns)}")

    # Load splits
    split_train = _read_split_csv(paths.train)
    split_dev = _read_split_csv(paths.dev) if paths.dev is not None else None
    split_test = _read_split_csv(paths.test)

    # Merge
    train_df = _merge_cascades_split(train_cascades, split_train, train_id_col, train_text_col, train_label_col)
    dev_df = _merge_cascades_split(train_cascades, split_dev, train_id_col, train_text_col, train_label_col) if split_dev is not None else None
    test_df = _merge_cascades_split(test_cascades, split_test, test_id_col, test_text_col, test_label_col)

    train_df, dev_df = _split_train_dev_if_needed(train_df, dev_df, args.seed)

    # Label mapping across all splits
    label2id, id2label = _make_label_mapping([train_df, dev_df, test_df])
    train_df = _add_numeric_labels(train_df, label2id)
    dev_df = _add_numeric_labels(dev_df, label2id)
    test_df = _add_numeric_labels(test_df, label2id)

    # Prepare HF datasets (only columns needed by model)
    hf_train = Dataset.from_pandas(train_df[["text", "labels"]], preserve_index=False)
    hf_dev = Dataset.from_pandas(dev_df[["text", "labels"]], preserve_index=False)
    hf_test = Dataset.from_pandas(test_df[["text", "labels"]], preserve_index=False)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=args.max_length)

    hf_train = hf_train.map(tok, batched=True, remove_columns=["text"])
    hf_dev = hf_dev.map(tok, batched=True, remove_columns=["text"])
    hf_test = hf_test.map(tok, batched=True, remove_columns=["text"])

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model,
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    # Run directory
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    model_tag = _slug(args.model)
    default_run_id = f"{args.task}_{paths.train_dataset}"
    if args.task == "cross":
        default_run_id += f"_{args.direction}"
    if args.task == "pheme_loeo":
        default_run_id += f"_{_slug(args.event or '')}"
    default_run_id += f"_{model_tag}_seed{args.seed}_{timestamp}"
    run_id = args.run_id or default_run_id

    run_dir = Path("models") / "rumor_transformer" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Training args, with optional use_mps_device if supported by installed transformers
    ta_kwargs = dict(
        output_dir=str(run_dir),
        num_train_epochs=float(args.epochs),
        per_device_train_batch_size=int(args.batch),
        per_device_eval_batch_size=int(args.batch),
        gradient_accumulation_steps=int(args.grad_accum),
        learning_rate=float(args.lr),
        weight_decay=float(args.weight_decay),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        report_to=[],
        dataloader_num_workers=int(args.num_workers),
    )

    sig = inspect.signature(TrainingArguments)
    if "use_mps_device" in sig.parameters:
        ta_kwargs["use_mps_device"] = bool(torch.backends.mps.is_available())

    training_args = TrainingArguments(**ta_kwargs)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=hf_train,
        eval_dataset=hf_dev,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=_compute_metrics_fn(),
    )

    # Train
    trainer.train()

    # Evaluate
    dev_metrics = trainer.evaluate(eval_dataset=hf_dev)
    test_metrics = trainer.evaluate(eval_dataset=hf_test, metric_key_prefix="test")

    # Predict + save preds
    def save_preds(df: pd.DataFrame, hf_ds: Dataset, out_path: Path):
        pred = trainer.predict(hf_ds)
        logits = pred.predictions
        pred_ids = np.argmax(logits, axis=-1).astype(int)
        out = pd.DataFrame({
            "_id": df["_id"].astype(str).values,
            "true_label": df["label_raw"].astype(str).values,
            "true_id": df["labels"].astype(int).values,
            "pred_id": pred_ids,
            "pred_label": [id2label[int(i)] for i in pred_ids],
        })
        out.to_parquet(out_path, index=False)

    save_preds(dev_df, hf_dev, run_dir / "dev_preds.parquet")
    save_preds(test_df, hf_test, run_dir / "test_preds.parquet")

    # Save best model snapshot
    best_dir = run_dir / "best"
    best_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(best_dir))
    tokenizer.save_pretrained(str(best_dir))

    # Save metrics.json
    info = {
        "run_id": run_id,
        "task": args.task,
        "dataset": args.dataset,
        "direction": args.direction,
        "event": args.event,
        "model": args.model,
        "processed": args.processed,
        "splits": args.splits,
        "train_split": str(paths.train),
        "dev_split": str(paths.dev) if paths.dev is not None else None,
        "test_split": str(paths.test),
        "train_rows": int(len(train_df)),
        "dev_rows": int(len(dev_df)),
        "test_rows": int(len(test_df)),
        "label2id": label2id,
        "torch_version": torch.__version__,
        "mps_available": bool(torch.backends.mps.is_available()),
        "mps_built": bool(torch.backends.mps.is_built()),
        "dev_metrics": {k: float(v) for k, v in dev_metrics.items()},
        "test_metrics": {k: float(v) for k, v in test_metrics.items()},
    }
    with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Done. Outputs in: {run_dir}\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("\n❌ Error:", str(e), file=sys.stderr)
        raise
