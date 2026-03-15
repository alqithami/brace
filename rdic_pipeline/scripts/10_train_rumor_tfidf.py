
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.pipeline import Pipeline
import joblib

def load_split_ids(split_dir: Path):
    train = pd.read_csv(split_dir/"train.csv")["cascade_id"].astype(str).tolist()
    dev = pd.read_csv(split_dir/"dev.csv")["cascade_id"].astype(str).tolist()
    test = pd.read_csv(split_dir/"test.csv")["cascade_id"].astype(str).tolist()
    return train, dev, test

def eval_binary(y_true, scores, probs=None):
    out = {}
    y_pred = (scores > 0).astype(int) if probs is None else (probs >= 0.5).astype(int)
    out["macro_f1"] = f1_score(y_true, y_pred, average="macro")
    out["acc"] = accuracy_score(y_true, y_pred)
    if probs is not None:
        out["auc"] = roc_auc_score(y_true, probs)
    else:
        # use scores for AUC if available
        out["auc"] = roc_auc_score(y_true, scores)
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["twitter15","twitter16","pheme"])
    ap.add_argument("--split", type=str, default="within", help="within | cross15to16 | cross16to15 | pheme_loeo:fold_event")
    ap.add_argument("--processed", type=str, default="data_processed")
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--out", type=str, default="models/rumor")
    args = ap.parse_args()

    processed = Path(args.processed)
    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)

    c = pd.read_parquet(processed / args.dataset / "cascades.parquet")
    c["cascade_id"] = c["cascade_id"].astype(str)
    c["y"] = c["label_binary"].astype(int)
    texts = c.set_index("cascade_id")["source_text"].fillna("").to_dict()

    # locate split directory
    if args.split == "within":
        split_dir = Path(args.splits) / args.dataset / "within"
        train_ids, dev_ids, test_ids = load_split_ids(split_dir)
    elif args.split == "cross15to16":
        split_dir = Path(args.splits) / "twitter_cross"
        # for dataset=twitter15 uses 15to16_train/dev, dataset=twitter16 uses 15to16_test
        if args.dataset == "twitter15":
            train_ids = pd.read_csv(split_dir/"15to16_train.csv")["cascade_id"].astype(str).tolist()
            dev_ids   = pd.read_csv(split_dir/"15to16_dev.csv")["cascade_id"].astype(str).tolist()
            test_ids  = pd.read_csv(split_dir/"15to16_test.csv")["cascade_id"].astype(str).tolist()
        else:
            train_ids = pd.read_csv(split_dir/"15to16_train.csv")["cascade_id"].astype(str).tolist()
            dev_ids   = pd.read_csv(split_dir/"15to16_dev.csv")["cascade_id"].astype(str).tolist()
            test_ids  = pd.read_csv(split_dir/"15to16_test.csv")["cascade_id"].astype(str).tolist()
    elif args.split == "cross16to15":
        split_dir = Path(args.splits) / "twitter_cross"
        train_ids = pd.read_csv(split_dir/"16to15_train.csv")["cascade_id"].astype(str).tolist()
        dev_ids   = pd.read_csv(split_dir/"16to15_dev.csv")["cascade_id"].astype(str).tolist()
        test_ids  = pd.read_csv(split_dir/"16to15_test.csv")["cascade_id"].astype(str).tolist()
    elif args.split.startswith("pheme_loeo:"):
        fold = args.split.split(":",1)[1]
        split_dir = Path(args.splits) / "pheme" / "loeo" / f"fold_{fold}"
        train_ids, dev_ids, test_ids = load_split_ids(split_dir)
    else:
        split_dir = Path(args.splits) / args.dataset / args.split
        train_ids, dev_ids, test_ids = load_split_ids(split_dir)

    X_train = [texts[i] for i in train_ids if i in texts]
    y_train = c.set_index("cascade_id").loc[train_ids, "y"].values
    X_dev   = [texts[i] for i in dev_ids if i in texts]
    y_dev   = c.set_index("cascade_id").loc[dev_ids, "y"].values
    X_test  = [texts[i] for i in test_ids if i in texts]
    y_test  = c.set_index("cascade_id").loc[test_ids, "y"].values

    vectorizer = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_features=200000)

    # Logistic Regression
    lr = LogisticRegression(max_iter=5000, n_jobs=4)
    pipe_lr = Pipeline([("tfidf", vectorizer), ("clf", lr)])
    pipe_lr.fit(X_train, y_train)
    dev_probs = pipe_lr.predict_proba(X_dev)[:,1]
    test_probs = pipe_lr.predict_proba(X_test)[:,1]
    dev = eval_binary(y_dev, scores=np.log(dev_probs/(1-dev_probs+1e-12)), probs=dev_probs)
    test = eval_binary(y_test, scores=np.log(test_probs/(1-test_probs+1e-12)), probs=test_probs)
    print("[LR] dev:", dev, "test:", test)
    joblib.dump(pipe_lr, out / f"tfidf_lr_{args.split}.joblib")

    pd.DataFrame({"cascade_id": dev_ids, "y": y_dev, "p": dev_probs}).to_csv(out / f"tfidf_lr_{args.split}_dev_preds.csv", index=False)
    pd.DataFrame({"cascade_id": test_ids, "y": y_test, "p": test_probs}).to_csv(out / f"tfidf_lr_{args.split}_test_preds.csv", index=False)

    # Linear SVM (scores only)
    svm = LinearSVC()
    pipe_svm = Pipeline([("tfidf", vectorizer), ("clf", svm)])
    pipe_svm.fit(X_train, y_train)
    dev_scores = pipe_svm.decision_function(X_dev)
    test_scores = pipe_svm.decision_function(X_test)
    dev = eval_binary(y_dev, scores=dev_scores, probs=None)
    test = eval_binary(y_test, scores=test_scores, probs=None)
    print("[SVM] dev:", dev, "test:", test)
    joblib.dump(pipe_svm, out / f"tfidf_svm_{args.split}.joblib")
    pd.DataFrame({"cascade_id": dev_ids, "y": y_dev, "score": dev_scores}).to_csv(out / f"tfidf_svm_{args.split}_dev_scores.csv", index=False)
    pd.DataFrame({"cascade_id": test_ids, "y": y_test, "score": test_scores}).to_csv(out / f"tfidf_svm_{args.split}_test_scores.csv", index=False)

if __name__ == "__main__":
    main()
