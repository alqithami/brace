
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rdic.paths import discover_raw_paths

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=str, default="data_raw")
    ap.add_argument("--out", type=str, default="data_processed/emotion")
    args = ap.parse_args()

    rp = discover_raw_paths(Path(args.data_raw))
    if rp.tweeteval_root is None or rp.semeval_root is None:
        raise SystemExit("TweetEval or SemEval not found. Run scripts/00_validate_paths.py")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # TweetEval Emotion (4-class)
    for split in ["train","validation","test"]:
        df = pd.read_csv(rp.tweeteval_root / f"emotion_{split}.csv")
        df.to_parquet(out / f"tweeteval_emotion_{split}.parquet", index=False)

    # SemEval 2018 Affect in Tweets (E-c English) - multi-label
    # Files are tab-separated with header
    for split in ["train","dev","test"]:
        fp = rp.semeval_root / f"2018-E-c-En-{split}.txt"
        df = pd.read_csv(fp, sep="\t")
        df.to_parquet(out / f"semeval2018_ec_{split}.parquet", index=False)

    print("Saved emotion datasets to:", out)

if __name__ == "__main__":
    main()
