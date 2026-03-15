
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from rdic.paths import discover_raw_paths
from rdic.io import parse_twitter_label_file, parse_source_tweets, parse_tree_line, label4_to_binary, compute_depths

def process_split(dataset: str, tree_root: Path, src_root: Path, out_dir: Path):
    # dataset: twitter15 or twitter16
    label_path = tree_root / dataset / "label.txt"
    tree_dir = tree_root / dataset / "tree"
    src_path = src_root / dataset / "source_tweets.txt"

    labels = parse_twitter_label_file(label_path)
    texts = parse_source_tweets(src_path)

    cascades = []
    nodes = []
    edges = []

    for tf in sorted(tree_dir.glob("*.txt")):
        cascade_id = tf.stem
        label4 = labels.get(cascade_id, None)
        if label4 is None:
            continue
        y_bin = label4_to_binary(label4)
        src_text = texts.get(cascade_id, "")

        # parse edges
        local_edges = []
        local_nodes = {}  # tweet_id -> {uid, delay}
        with open(tf, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                a,b = parse_tree_line(line)
                (p_uid, p_tid, p_delay) = a
                (c_uid, c_tid, c_delay) = b
                # store nodes
                local_nodes[p_tid] = {"uid": p_uid, "time_delay": p_delay}
                local_nodes[c_tid] = {"uid": c_uid, "time_delay": c_delay}
                # store edges, ignore ROOT->ROOT
                if p_tid != "ROOT" and c_tid != "ROOT":
                    local_edges.append((p_tid, c_tid))

        # compute depths from root tweet id
        root_id = cascade_id
        depths = compute_depths(root_id, local_edges) if local_edges else {root_id: 0}

        # cascades row
        cascades.append({
            "dataset": dataset,
            "event": None,
            "cascade_id": cascade_id,
            "root_tweet_id": root_id,
            "label_4way": label4,
            "label_binary": y_bin,
            "veracity": (label4 if y_bin==1 else None),
            "source_text": src_text
        })

        # nodes rows
        for tid, meta in local_nodes.items():
            is_root = (tid == root_id)
            text = src_text if is_root else None
            nodes.append({
                "dataset": dataset,
                "cascade_id": cascade_id,
                "tweet_id": tid,
                "uid": meta.get("uid"),
                "time_delay": meta.get("time_delay"),
                "is_root": is_root,
                "depth": depths.get(tid, None),
                "step": depths.get(tid, None),
                "text": text
            })

        # edges rows
        for p,c in local_edges:
            edges.append({
                "dataset": dataset,
                "cascade_id": cascade_id,
                "parent_id": p,
                "child_id": c
            })

    df_c = pd.DataFrame(cascades).drop_duplicates(subset=["cascade_id"])
    df_n = pd.DataFrame(nodes).drop_duplicates(subset=["cascade_id","tweet_id"])
    df_e = pd.DataFrame(edges).drop_duplicates()

    out_dir.mkdir(parents=True, exist_ok=True)
    df_c.to_parquet(out_dir / "cascades.parquet", index=False)
    df_n.to_parquet(out_dir / "nodes.parquet", index=False)
    df_e.to_parquet(out_dir / "edges.parquet", index=False)

    print(f"[{dataset}] saved {out_dir}")
    print("  cascades:", df_c.shape, "nodes:", df_n.shape, "edges:", df_e.shape)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=str, default="data_raw")
    ap.add_argument("--out-root", type=str, default="data_processed")
    args = ap.parse_args()

    rp = discover_raw_paths(Path(args.data_raw))
    if rp.twitter_tree_root is None or rp.twitter_source_root is None:
        raise SystemExit("Twitter trees or source texts not found. Run scripts/00_validate_paths.py")

    out_root = Path(args.out_root)
    process_split("twitter15", rp.twitter_tree_root, rp.twitter_source_root, out_root / "twitter15")
    process_split("twitter16", rp.twitter_tree_root, rp.twitter_source_root, out_root / "twitter16")

    # compute and save overlap ids (for leakage-free cross)
    c15 = pd.read_parquet(out_root / "twitter15" / "cascades.parquet")
    c16 = pd.read_parquet(out_root / "twitter16" / "cascades.parquet")
    overlap = sorted(set(c15["cascade_id"]).intersection(set(c16["cascade_id"])))
    meta = Path("splits/meta")
    meta.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"overlap_id": overlap}).to_csv(meta / "overlap_ids.csv", index=False)
    print("Saved overlap ids:", meta / "overlap_ids.csv", "n=", len(overlap))

if __name__ == "__main__":
    main()
