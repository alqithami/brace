
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error
import joblib

def load_ids(split_dir: Path):
    return (
        pd.read_csv(split_dir/"train.csv")["cascade_id"].astype(str).tolist(),
        pd.read_csv(split_dir/"dev.csv")["cascade_id"].astype(str).tolist(),
        pd.read_csv(split_dir/"test.csv")["cascade_id"].astype(str).tolist(),
    )

def build_step_table(nodes: pd.DataFrame, cascades: pd.DataFrame, panic: pd.DataFrame|None):
    # step = depth (already in nodes)
    nodes = nodes.copy()
    nodes["step"] = nodes["step"].fillna(nodes["depth"])
    # compute per cascade per step new activations
    grp = nodes.dropna(subset=["step"]).groupby(["cascade_id","step"]).size().reset_index(name="new_count")
    # cumulative
    grp = grp.sort_values(["cascade_id","step"])
    grp["cum_count"] = grp.groupby("cascade_id")["new_count"].cumsum()

    # merge root-level labels and text-only features (risk/panic come later)
    base = cascades[["cascade_id","label_binary"]].copy()
    # root panic: take mean panic of nodes with is_root if available
    if panic is not None and not panic.empty:
        # if tweet_id==root_tweet_id; approximate by first node where is_root
        pass
    # max depth
    max_step = grp.groupby("cascade_id")["step"].max().reset_index(name="max_step")
    base = base.merge(max_step, on="cascade_id", how="left")
    return grp.merge(base, on="cascade_id", how="left")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True, choices=["twitter15","twitter16","pheme"])
    ap.add_argument("--split", type=str, default="within")
    ap.add_argument("--processed", type=str, default="data_processed")
    ap.add_argument("--splits", type=str, default="splits")
    ap.add_argument("--features", type=str, default="features", help="folder containing panic_*.parquet (optional)")
    ap.add_argument("--out", type=str, default="models/simulator")
    args = ap.parse_args()

    processed = Path(args.processed) / args.dataset
    cascades = pd.read_parquet(processed / "cascades.parquet")
    nodes = pd.read_parquet(processed / "nodes.parquet")
    cascades["cascade_id"] = cascades["cascade_id"].astype(str)
    nodes["cascade_id"] = nodes["cascade_id"].astype(str)

    panic_path = Path(args.features) / f"panic_{args.dataset}.parquet"
    panic = pd.read_parquet(panic_path) if panic_path.exists() else None

    # split dir
    if args.split == "within":
        split_dir = Path(args.splits) / args.dataset / "within"
    else:
        split_dir = Path(args.splits) / args.dataset / args.split
    train_ids, dev_ids, test_ids = load_ids(split_dir)

    step_df = build_step_table(nodes, cascades, panic)

    # build supervised samples: predict next new_count
    X, y = [], []
    for cid, g in step_df.groupby("cascade_id"):
        g = g.sort_values("step")
        steps = g["step"].values.astype(int)
        newc = g["new_count"].values.astype(float)
        cumc = g["cum_count"].values.astype(float)
        max_step = int(g["max_step"].iloc[0]) if not pd.isna(g["max_step"].iloc[0]) else int(steps.max())
        for i in range(len(steps)-1):
            t = steps[i]
            # features: t, cum_count, new_count, remaining steps
            X.append([t, cumc[i], newc[i], max_step - t])
            y.append(newc[i+1])
    X = np.asarray(X)
    y = np.asarray(y)

    # train/dev split by cascade ids
    # build index mask by cascade id membership
    # simplest: rebuild samples with cascade_id list
    X2, y2, cid2 = [], [], []
    for cid, g in step_df.groupby("cascade_id"):
        g = g.sort_values("step")
        steps = g["step"].values.astype(int)
        newc = g["new_count"].values.astype(float)
        cumc = g["cum_count"].values.astype(float)
        max_step = int(g["max_step"].iloc[0]) if not pd.isna(g["max_step"].iloc[0]) else int(steps.max())
        for i in range(len(steps)-1):
            t = steps[i]
            X2.append([t, cumc[i], newc[i], max_step - t])
            y2.append(newc[i+1])
            cid2.append(cid)
    X2 = np.asarray(X2); y2 = np.asarray(y2); cid2 = np.asarray(cid2, dtype=str)

    tr_mask = np.isin(cid2, np.asarray(train_ids, dtype=str))
    dv_mask = np.isin(cid2, np.asarray(dev_ids, dtype=str))

    X_tr, y_tr = X2[tr_mask], y2[tr_mask]
    X_dv, y_dv = X2[dv_mask], y2[dv_mask]

    model = MLPRegressor(hidden_layer_sizes=(64,64), random_state=42, max_iter=200)
    model.fit(X_tr, y_tr)
    pred = model.predict(X_dv)
    rmse = mean_squared_error(y_dv, pred, squared=False) if len(y_dv) else float("nan")
    print("Dev RMSE:", rmse)

    out = Path(args.out) / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out / f"sim_step_{args.split}.joblib")
    print("Saved:", out / f"sim_step_{args.split}.joblib")

if __name__ == "__main__":
    main()
