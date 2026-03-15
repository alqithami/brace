
from __future__ import annotations
import argparse
from pathlib import Path
from rdic.paths import discover_raw_paths, ensure_extracted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=str, default="data_raw")
    ap.add_argument("--extract", action="store_true", help="If set, extract ZIPs first.")
    args = ap.parse_args()
    data_raw = Path(args.data_raw)
    if args.extract:
        ensure_extracted(data_raw)
    rp = discover_raw_paths(data_raw)
    print("Discovered paths:")
    print("  PHEME root:", rp.pheme_root)
    print("  Twitter tree root:", rp.twitter_tree_root)
    print("  Twitter source root:", rp.twitter_source_root)
    print("  TweetEval root:", rp.tweeteval_root)
    print("  SemEval root:", rp.semeval_root)

    missing = [k for k,v in rp.__dict__.items() if v is None]
    if missing:
        raise SystemExit(f"Missing datasets: {missing}. Put the ZIPs (or extracted folders) under {data_raw} and re-run.")
    print("OK: all required datasets found.")

if __name__ == "__main__":
    main()
