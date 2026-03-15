
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize

def nll_temperature(T, logits, y):
    T = float(T[0])
    z = logits / max(T, 1e-6)
    p = 1/(1+np.exp(-z))
    eps = 1e-12
    p = np.clip(p, eps, 1-eps)
    return -(y*np.log(p) + (1-y)*np.log(1-p)).mean()

def ece(probs, y, m=15):
    probs = np.asarray(probs)
    y = np.asarray(y)
    bins = np.linspace(0,1,m+1)
    e = 0.0
    n = len(y)
    for i in range(m):
        lo, hi = bins[i], bins[i+1]
        mask = (probs >= lo) & (probs < hi) if i < m-1 else (probs >= lo) & (probs <= hi)
        if mask.sum() == 0:
            continue
        acc = y[mask].mean()
        conf = probs[mask].mean()
        e += (mask.sum()/n)*abs(acc-conf)
    return float(e)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["twitter15","twitter16","pheme"])
    ap.add_argument("--split", type=str, default="within")
    ap.add_argument("--model-dir", type=str, default="models/rumor")
    ap.add_argument("--which", type=str, default="svm", choices=["svm","lr"])
    args = ap.parse_args()

    mdir = Path(args.model_dir) / args.dataset
    if args.which == "svm":
        dev = pd.read_csv(mdir / f"tfidf_svm_{args.split}_dev_scores.csv")
        test = pd.read_csv(mdir / f"tfidf_svm_{args.split}_test_scores.csv")
        dev_logits = dev["score"].values
        test_logits = test["score"].values
    else:
        dev = pd.read_csv(mdir / f"tfidf_lr_{args.split}_dev_preds.csv")
        test = pd.read_csv(mdir / f"tfidf_lr_{args.split}_test_preds.csv")
        # convert probs to logits
        eps = 1e-12
        dev_p = np.clip(dev["p"].values, eps, 1-eps)
        test_p = np.clip(test["p"].values, eps, 1-eps)
        dev_logits = np.log(dev_p/(1-dev_p))
        test_logits = np.log(test_p/(1-test_p))

    y_dev = dev["y"].values.astype(int)
    y_test = test["y"].values.astype(int)

    res = minimize(nll_temperature, x0=np.array([1.0]), args=(dev_logits, y_dev), bounds=[(0.05, 20.0)], method="L-BFGS-B")
    T = float(res.x[0])
    print("Best T:", T, "dev NLL:", res.fun)

    # calibrated probs
    dev_p = 1/(1+np.exp(-(dev_logits/T)))
    test_p = 1/(1+np.exp(-(test_logits/T)))

    dev_ece = ece(dev_p, y_dev)
    test_ece = ece(test_p, y_test)
    dev_brier = float(((dev_p - y_dev)**2).mean())
    test_brier = float(((test_p - y_test)**2).mean())

    print("Calibrated ECE/Brier - dev:", dev_ece, dev_brier, "test:", test_ece, test_brier)

    out = mdir / f"temp_scaling_{args.which}_{args.split}.json"
    out.write_text(pd.Series({"T":T,"dev_ece":dev_ece,"dev_brier":dev_brier,"test_ece":test_ece,"test_brier":test_brier}).to_json(), encoding="utf-8")
    print("Saved:", out)

if __name__ == "__main__":
    main()
