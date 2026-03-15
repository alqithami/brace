
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
from rdic.paths import discover_raw_paths
from rdic.io import safe_read_json, pheme_annotation_to_labels, pheme_parse_structure, compute_depths

def iter_thread_dirs(pheme_root: Path):
    # pheme_root contains <event>-all-rnr-threads/
    for event_dir in sorted([p for p in pheme_root.iterdir() if p.is_dir() and p.name.endswith("-all-rnr-threads")]):
        event = event_dir.name.replace("-all-rnr-threads", "")
        for split_dirname in ["rumours", "non-rumours"]:
            d = event_dir / split_dirname
            if not d.exists():
                continue
            for thread_dir in sorted([p for p in d.iterdir() if p.is_dir() and not p.name.startswith("._")]):
                yield event, thread_dir

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=str, default="data_raw")
    ap.add_argument("--out", type=str, default="data_processed/pheme")
    args = ap.parse_args()

    rp = discover_raw_paths(Path(args.data_raw))
    if rp.pheme_root is None:
        raise SystemExit("PHEME not found. Run scripts/00_validate_paths.py.")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cascades = []
    nodes = []
    edges_all = []

    for event, thread_dir in iter_thread_dirs(rp.pheme_root):
        cascade_id = thread_dir.name

        ann_path = thread_dir / "annotation.json"
        struct_path = thread_dir / "structure.json"
        src_dir = thread_dir / "source-tweets"
        react_dir = thread_dir / "reactions"

        if not ann_path.exists() or not struct_path.exists() or not src_dir.exists():
            continue

        ann = safe_read_json(ann_path)
        y_bin, ver = pheme_annotation_to_labels(ann)

        # source tweet json
        src_files = sorted([f for f in src_dir.glob("*.json") if not f.name.startswith("._")])
        src_file = src_files[0] if src_files else None
        if src_file is None:
            continue
        src = safe_read_json(src_file)
        src_text = src.get("text", "")
        src_created = src.get("created_at", None)
        src_lang = src.get("lang", None)
        src_user = None
        if isinstance(src.get("user", None), dict):
            src_user = src["user"].get("id_str", None) or src["user"].get("id", None)
        src_tid = str(src.get("id_str", src.get("id", cascade_id)))

        # structure edges
        struct = safe_read_json(struct_path)
        edges = pheme_parse_structure(struct)
        # sometimes structure root id equals cascade_id
        root_id = next(iter(struct.keys())) if struct else src_tid
        depths = compute_depths(root_id, edges) if edges else {root_id: 0}

        # Add edges
        for p,c in edges:
            edges_all.append({
                "dataset": "pheme",
                "event": event,
                "cascade_id": cascade_id,
                "parent_id": p,
                "child_id": c
            })

        # nodes: root
        nodes.append({
            "dataset": "pheme",
            "event": event,
            "cascade_id": cascade_id,
            "tweet_id": root_id,
            "is_root": True,
            "depth": depths.get(root_id, 0),
            "step": depths.get(root_id, 0),
            "created_at": src_created,
            "lang": src_lang,
            "user_id": src_user,
            "text": src_text
        })

        # nodes: reactions
        if react_dir.exists():
            for rf in react_dir.glob("*.json"):
                if rf.name.startswith("._"):
                    continue
                rj = safe_read_json(rf)
                tid = str(rj.get("id_str", rj.get("id", rf.stem)))
                nodes.append({
                    "dataset": "pheme",
                    "event": event,
                    "cascade_id": cascade_id,
                    "tweet_id": tid,
                    "is_root": False,
                    "depth": depths.get(tid, None),
                    "step": depths.get(tid, None),
                    "created_at": rj.get("created_at", None),
                    "lang": rj.get("lang", None),
                    "user_id": (rj.get("user", {}) or {}).get("id_str", None) if isinstance(rj.get("user", None), dict) else None,
                    "text": rj.get("text", "")
                })

        cascades.append({
            "dataset": "pheme",
            "event": event,
            "cascade_id": cascade_id,
            "root_tweet_id": root_id,
            "label_binary": y_bin,
            "label_4way": ver if ver is not None else ("rumour" if y_bin==1 else "non-rumour"),
            "veracity": ver,
            "source_text": src_text
        })

    df_c = pd.DataFrame(cascades).drop_duplicates(subset=["cascade_id"])
    df_n = pd.DataFrame(nodes).drop_duplicates(subset=["cascade_id","tweet_id"])
    df_e = pd.DataFrame(edges_all).drop_duplicates()

    # save
    df_c.to_parquet(out / "cascades.parquet", index=False)
    df_n.to_parquet(out / "nodes.parquet", index=False)
    df_e.to_parquet(out / "edges.parquet", index=False)

    print("Saved:")
    print(" ", out / "cascades.parquet", df_c.shape)
    print(" ", out / "nodes.parquet", df_n.shape)
    print(" ", out / "edges.parquet", df_e.shape)

if __name__ == "__main__":
    main()
