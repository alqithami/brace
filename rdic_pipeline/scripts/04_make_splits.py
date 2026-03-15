
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import yaml
from rdic.io import label4_to_binary

def stratified_split(df: pd.DataFrame, label_col: str, ratios=(0.7,0.1,0.2), seed=42):
    rng = np.random.default_rng(seed)
    train_ids, dev_ids, test_ids = [], [], []
    for lab, g in df.groupby(label_col):
        ids = g["cascade_id"].tolist()
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(ratios[0]*n))
        n_dev = int(round(ratios[1]*n))
        # remainder test
        train_ids += ids[:n_train]
        dev_ids += ids[n_train:n_train+n_dev]
        test_ids += ids[n_train+n_dev:]
    return train_ids, dev_ids, test_ids

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default="configs/defaults.yaml")
    ap.add_argument("--data-raw", type=str, default="data_raw")
    ap.add_argument("--processed", type=str, default="data_processed")
    ap.add_argument("--out", type=str, default="splits")
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    seed = int(cfg.get("seed", 42))
    within = cfg["splits"]["within"]
    ratios = (within["train"], within["dev"], within["test"])

    processed = Path(args.processed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Twitter within
    for ds in ["twitter15", "twitter16"]:
        c = pd.read_parquet(processed / ds / "cascades.parquet")
        train, dev, test = stratified_split(c, "label_4way", ratios=ratios, seed=seed)
        ddir = out / ds / "within"
        ddir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"cascade_id": train}).to_csv(ddir/"train.csv", index=False)
        pd.DataFrame({"cascade_id": dev}).to_csv(ddir/"dev.csv", index=False)
        pd.DataFrame({"cascade_id": test}).to_csv(ddir/"test.csv", index=False)

    # Twitter cross (remove overlap)
    overlap_path = out / "meta" / "overlap_ids.csv"
    if overlap_path.exists():
        overlap = set(pd.read_csv(overlap_path)["overlap_id"].astype(str))
    else:
        overlap = set()

    c15 = pd.read_parquet(processed / "twitter15" / "cascades.parquet")
    c16 = pd.read_parquet(processed / "twitter16" / "cascades.parquet")
    c15_u = c15[~c15["cascade_id"].isin(overlap)]
    c16_u = c16[~c16["cascade_id"].isin(overlap)]

    # 15->16
    train, dev, _ = stratified_split(c15_u, "label_4way", ratios=(0.9,0.1,0.0), seed=seed)
    cross_dir = out / "twitter_cross"
    cross_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"cascade_id": train}).to_csv(cross_dir/"15to16_train.csv", index=False)
    pd.DataFrame({"cascade_id": dev}).to_csv(cross_dir/"15to16_dev.csv", index=False)
    pd.DataFrame({"cascade_id": c16_u["cascade_id"].tolist()}).to_csv(cross_dir/"15to16_test.csv", index=False)

    # 16->15
    train, dev, _ = stratified_split(c16_u, "label_4way", ratios=(0.9,0.1,0.0), seed=seed)
    pd.DataFrame({"cascade_id": train}).to_csv(cross_dir/"16to15_train.csv", index=False)
    pd.DataFrame({"cascade_id": dev}).to_csv(cross_dir/"16to15_dev.csv", index=False)
    pd.DataFrame({"cascade_id": c15_u["cascade_id"].tolist()}).to_csv(cross_dir/"16to15_test.csv", index=False)

    # PHEME leave-one-event-out (LOEO)
    pheme_c_path = processed / "pheme" / "cascades.parquet"
    if pheme_c_path.exists():
        c = pd.read_parquet(pheme_c_path)
        loeo_dir = out / "pheme" / "loeo"
        loeo_dir.mkdir(parents=True, exist_ok=True)
        events = sorted([e for e in c["event"].dropna().unique()])
        rng = np.random.default_rng(seed)
        dev_frac = float(cfg["splits"].get("pheme_dev_from_train", 0.10))
        for ev in events:
            test_ids = c.loc[c["event"]==ev, "cascade_id"].tolist()
            train_pool = c.loc[c["event"]!=ev].copy()
            # dev from train_pool stratified by label_binary
            train_ids, dev_ids, _ = stratified_split(train_pool, "label_binary", ratios=(1-dev_frac, dev_frac, 0.0), seed=seed)
            fold = loeo_dir / f"fold_{ev}"
            fold.mkdir(parents=True, exist_ok=True)
            pd.DataFrame({"cascade_id": train_ids}).to_csv(fold/"train.csv", index=False)
            pd.DataFrame({"cascade_id": dev_ids}).to_csv(fold/"dev.csv", index=False)
            pd.DataFrame({"cascade_id": test_ids}).to_csv(fold/"test.csv", index=False)

    print("Splits saved to:", out)

if __name__ == "__main__":
    main()
