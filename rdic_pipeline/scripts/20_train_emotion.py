
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, DataCollatorWithPadding

LABELS_TWEETEVAL = ["anger","joy","optimism","sadness"]  # TweetEval Emotion

SEMEVAL_LABELS = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]

class DataCollatorWithPaddingAndFloatLabels(DataCollatorWithPadding):
    """Pads inputs and casts multi-label targets to float32."""
    def __call__(self, features):
        batch = super().__call__(features)
        if "labels" in batch:
            batch["labels"] = batch["labels"].to(torch.float32)
        return batch


def load_tweeteval(processed_root: Path, split: str):
    df = pd.read_parquet(processed_root / f"tweeteval_emotion_{split}.parquet")
    return Dataset.from_pandas(df)

def load_semeval(processed_root: Path, split: str):
    df = pd.read_parquet(processed_root / f"semeval2018_ec_{split}.parquet")
    # rename Tweet->text for consistency
    if "Tweet" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"Tweet":"text"})
    # test may have NONE; keep for inference only
    return df

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed", type=str, default="data_processed/emotion")
    ap.add_argument("--out", type=str, default="models/emotion")
    ap.add_argument("--model", type=str, default="distilroberta-base")
    ap.add_argument("--stage", type=str, required=True, choices=["tweeteval","semeval"])
    ap.add_argument("--init-from", type=str, default=None, help="Path to a previous checkpoint directory (e.g., models/emotion/tweeteval)")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()

    processed = Path(args.processed)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    if args.stage == "tweeteval":
        train_ds = load_tweeteval(processed, "train")
        dev_ds = load_tweeteval(processed, "validation")
        num_labels = 4
        model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)
        def preprocess(ex):
            tok = tokenizer(ex["text"], truncation=True, max_length=128)
            tok["labels"] = ex["label"]
            return tok
        train_ds = train_ds.map(preprocess, batched=True, remove_columns=train_ds.column_names)
        dev_ds = dev_ds.map(preprocess, batched=True, remove_columns=dev_ds.column_names)
        args_out = out_root / "tweeteval"
        training_args = TrainingArguments(
            output_dir=str(args_out),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=2e-5,
            weight_decay=0.01,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
        trainer = Trainer(model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds, tokenizer=tokenizer, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(args_out))
        tokenizer.save_pretrained(str(args_out))
        print("Saved:", args_out)

    else:
        # SemEval multi-label (11 labels)
        train_df = load_semeval(processed, "train")
        dev_df = load_semeval(processed, "dev")

        # build multi-hot labels
        for lab in SEMEVAL_LABELS:
            if lab not in train_df.columns:
                raise SystemExit(f"Missing SemEval label column: {lab}")
        X_train = train_df["text"].astype(str).tolist()
        Y_train = train_df[SEMEVAL_LABELS].astype(int).values
        X_dev = dev_df["text"].astype(str).tolist()
        Y_dev = dev_df[SEMEVAL_LABELS].astype(int).values

        ds_train = Dataset.from_dict({"text": X_train, "labels": [y.tolist() for y in Y_train]})
        ds_dev = Dataset.from_dict({"text": X_dev, "labels": [y.tolist() for y in Y_dev]})

        # init model
        init_path = args.init_from if args.init_from else args.model
        model = AutoModelForSequenceClassification.from_pretrained(init_path, num_labels=len(SEMEVAL_LABELS), problem_type="multi_label_classification", ignore_mismatched_sizes=True)
        def preprocess(ex):
            tok = tokenizer(ex["text"], truncation=True, max_length=128)
            tok["labels"] = [[float(v) for v in row] for row in ex["labels"]]
            return tok
        ds_train = ds_train.map(preprocess, batched=True, remove_columns=ds_train.column_names)
        ds_dev = ds_dev.map(preprocess, batched=True, remove_columns=ds_dev.column_names)

        args_out = out_root / "semeval"
        training_args = TrainingArguments(
            output_dir=str(args_out),
            eval_strategy="epoch",
            save_strategy="epoch",
            logging_steps=50,
            num_train_epochs=args.epochs,
            per_device_train_batch_size=args.batch,
            per_device_eval_batch_size=args.batch,
            learning_rate=2e-5,
            weight_decay=0.01,
            report_to=[],
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        )
        data_collator = DataCollatorWithPaddingAndFloatLabels(tokenizer=tokenizer)
        trainer = Trainer(model=model, args=training_args, train_dataset=ds_train, eval_dataset=ds_dev, tokenizer=tokenizer, data_collator=data_collator)
        trainer.train()
        trainer.save_model(str(args_out))
        tokenizer.save_pretrained(str(args_out))
        print("Saved:", args_out)

if __name__ == "__main__":
    main()
