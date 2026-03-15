
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from rdic.io import entropy_from_probs

SEMEVAL_LABELS = ["anger","anticipation","disgust","fear","joy","love","optimism","pessimism","sadness","surprise","trust"]
IDX_FEAR = SEMEVAL_LABELS.index("fear")
IDX_ANGER = SEMEVAL_LABELS.index("anger")

def sigmoid(x):
    return 1/(1+np.exp(-x))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["pheme","twitter15","twitter16"])
    ap.add_argument("--processed", type=str, default="data_processed")
    ap.add_argument("--emotion-model", type=str, default="models/emotion/semeval", help="Path to SemEval multi-label model dir")
    ap.add_argument("--out", type=str, default="features")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--w_fear", type=float, default=0.45)
    ap.add_argument("--w_anger", type=float, default=0.30)
    ap.add_argument("--w_unc", type=float, default=0.25, help="proxy for confusion: entropy of emotion distribution")
    args = ap.parse_args()

    processed = Path(args.processed) / args.dataset
    nodes = pd.read_parquet(processed / "nodes.parquet")
    nodes["text"] = nodes["text"].fillna("")
    # infer only where text available
    mask = nodes["text"].str.len() > 0
    sub = nodes.loc[mask, ["cascade_id","tweet_id","text"]].copy()

    tokenizer = AutoTokenizer.from_pretrained(args.emotion_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.emotion_model)
    model.eval()

    probs_all = []
    for i in range(0, len(sub), args.batch):
        batch = sub.iloc[i:i+args.batch]["text"].tolist()
        enc = tokenizer(batch, truncation=True, max_length=128, padding=True, return_tensors="pt")
        with torch.no_grad():
            out = model(**enc)
            logits = out.logits.detach().cpu().numpy()
        probs = sigmoid(logits)  # multi-label
        probs_all.append(probs)
    probs_all = np.vstack(probs_all)

    fear = probs_all[:, IDX_FEAR]
    anger = probs_all[:, IDX_ANGER]
    H = entropy_from_probs(probs_all)

    panic = args.w_fear*fear + args.w_anger*anger + args.w_unc*(H / (np.log(probs_all.shape[1]) + 1e-12))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    out_path = out / f"panic_{args.dataset}.parquet"
    out_df = pd.DataFrame({
        "cascade_id": sub["cascade_id"].astype(str).values,
        "tweet_id": sub["tweet_id"].astype(str).values,
        "p_fear": fear,
        "p_anger": anger,
        "emo_entropy": H,
        "panic": panic
    })
    out_df.to_parquet(out_path, index=False)
    print("Saved:", out_path, out_df.shape)

if __name__ == "__main__":
    main()
