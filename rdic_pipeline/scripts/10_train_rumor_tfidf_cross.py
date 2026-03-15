from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
import joblib

def eval_binary(y_true, scores, probs=None):
    y_true = np.asarray(y_true).astype(int)
    if probs is None:
        y_pred = (np.asarray(scores) > 0).astype(int)
        auc = roc_auc_score(y_true, scores) if len(np.unique(y_true)) > 1 else float("nan")
    else:
        probs = np.asarray(probs)
        y_pred = (probs >= 0.5).astype(int)
        auc = roc_auc_score(y_true, probs) if len(np.unique(y_true)) > 1 else float("nan")
    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "acc": accuracy_score(y_true, y_pred),
        "auc": auc
    }

def load_ids(path: Path) -> list[str]:
    return pd.read_csv(path)["cascade_id"].astype(str).tolist()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--direction", required=True, choices=["15to16","16to15"])
    ap.add_argument("--processed", type=str, default="data_processed")
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--out", type=str, default="models/rumor")
    args = ap.parse_args()

    processed = Path(args.processed)
    splits = Path(args.splits) / "twitter_cross"
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.direction == "15to16":
        train_ds, test_ds = "twitter15", "twitter16"
        train_ids = load_ids(splits/"15to16_train.csv")
        dev_ids   = load_ids(splits/"15to16_dev.csv")
        test_ids  = load_ids(splits/"15to16_test.csv")
    else:
        train_ds, test_ds = "twitter16", "twitter15"
        train_ids = load_ids(splits/"16to15_train.csv")
        dev_ids   = load_ids(splits/"16to15_dev.csv")
        test_ids  = load_ids(splits/"16to15_test.csv")

    c_train = pd.read_parquet(processed/train_ds/"cascades.parquet")
    c_test  = pd.read_parquet(processed/test_ds/"cascades.parquet")

    c_train["cascade_id"] = c_train["cascade_id"].astype(str)
    c_test["cascade_id"]  = c_test["cascade_id"].astype(str)

    c_train["y"] = c_train["label_binary"].astype(int)
    c_test["y"]  = c_test["label_binary"].astype(int)

    train_text = c_train.set_index("cascade_id")["source_text"].fillna("").to_dict()
    test_text  = c_test.set_index("cascade_id")["source_text"].fillna("").to_dict()

    X_train = [train_text[i] for i in train_ids if i in train_text]
    y_train = c_train.set_index("cascade_id").loc[[i for i in train_ids if i in train_text], "y"].values
    X_dev   = [train_text[i] for i in dev_ids if i in train_text]
    y_dev   = c_train.set_index("cascade_id").loc[[i for i in dev_ids if i in train_text], "y"].values
    X_test  = [test_text[i] for i in test_ids if i in test_text]
    y_test  = c_test.set_index("cascade_id").loc[[i for i in test_ids if i in test_text], "y"].values

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000)

    # LR
    lr = LogisticRegression(max_iter=5000)
    pipe_lr = Pipeline([("tfidf", vectorizer), ("clf", lr)])
    pipe_lr.fit(X_train, y_train)
    dev_probs = pipe_lr.predict_proba(X_dev)[:,1]
    test_probs = pipe_lr.predict_proba(X_test)[:,1]
    dev = eval_binary(y_dev, scores=np.log(dev_probs/(1-dev_probs+1e-12)), probs=dev_probs)
    test = eval_binary(y_test, scores=np.log(test_probs/(1-test_probs+1e-12)), probs=test_probs)
    print(f"[{args.direction}][LR] dev:", dev, "test:", test)

    out_dir = out_root / "twitter_cross"
    out_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe_lr, out_dir / f"tfidf_lr_{args.direction}.joblib")
    pd.DataFrame({"cascade_id": [i for i in dev_ids if i in train_text], "y": y_dev, "p": dev_probs}).to_csv(out_dir/f"tfidf_lr_{args.direction}_dev_preds.csv", index=False)
    pd.DataFrame({"cascade_id": [i for i in test_ids if i in test_text], "y": y_test, "p": test_probs}).to_csv(out_dir/f"tfidf_lr_{args.direction}_test_preds.csv", index=False)

    # SVM
    svm = LinearSVC()
    pipe_svm = Pipeline([("tfidf", vectorizer), ("clf", svm)])
    pipe_svm.fit(X_train, y_train)
    dev_scores = pipe_svm.decision_function(X_dev)
    test_scores = pipe_svm.decision_function(X_test)
    dev = eval_binary(y_dev, scores=dev_scores, probs=None)
    test = eval_binary(y_test, scores=test_scores, probs=None)
    print(f"[{args.direction}][SVM] dev:", dev, "test:", test)

    joblib.dump(pipe_svm, out_dir / f"tfidf_svm_{args.direction}.joblib")
    pd.DataFrame({"cascade_id": [i for i in dev_ids if i in train_text], "y": y_dev, "score": dev_scores}).to_csv(out_dir/f"tfidf_svm_{args.direction}_dev_scores.csv", index=False)
    pd.DataFrame({"cascade_id": [i for i in test_ids if i in test_text], "y": y_test, "score": test_scores}).to_csv(out_dir/f"tfidf_svm_{args.direction}_test_scores.csv", index=False)

if __name__ == "__main__":
    main()
