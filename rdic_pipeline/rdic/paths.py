
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import re
import zipfile
import shutil

@dataclass
class RawPaths:
    pheme_root: Path | None
    twitter_tree_root: Path | None
    twitter_source_root: Path | None
    tweeteval_root: Path | None
    semeval_root: Path | None

def _find_any(root: Path, patterns: list[str]) -> Path | None:
    for pat in patterns:
        hits = list(root.rglob(pat))
        if hits:
            return hits[0]
    return None

def _find_dir_containing(root: Path, required_relpaths: list[str]) -> Path | None:
    # Search candidate directories up to a reasonable depth
    for cand in [root] + [p for p in root.rglob("*") if p.is_dir()]:
        ok = True
        for rp in required_relpaths:
            if not (cand / rp).exists():
                ok = False
                break
        if ok:
            return cand
    return None

def discover_raw_paths(data_raw: Path) -> RawPaths:
    data_raw = Path(data_raw)

    # PHEME: find a directory that contains "<event>-all-rnr-threads"
    pheme_event_dir = _find_any(data_raw, ["*all-rnr-threads*/rumours", "*all-rnr-threads*/non-rumours"])
    pheme_root = None
    if pheme_event_dir:
        # go up: .../<event>-all-rnr-threads/(rumours|non-rumours)
        pheme_root = pheme_event_dir.parent.parent  # parent is event folder; parent.parent is dataset root

        # If we landed inside an event folder, move to the parent that contains all events (e.g., all-rnr-annotated-threads_1)
        # Detect by checking siblings that match "*-all-rnr-threads"
        if pheme_root and not any(p.name.endswith("-all-rnr-threads") for p in pheme_root.iterdir() if p.is_dir()):
            # maybe we are at event folder; move up once
            maybe = pheme_root.parent
            if any(p.name.endswith("-all-rnr-threads") for p in maybe.iterdir() if p.is_dir()):
                pheme_root = maybe

    # Twitter trees: need twitter15/tree and twitter16/tree + label.txt
    twitter_tree_root = _find_dir_containing(data_raw, ["twitter15/tree", "twitter15/label.txt", "twitter16/tree", "twitter16/label.txt"])
    # Sometimes nested under "Twitter15_16_dataset-main/"
    if twitter_tree_root and (twitter_tree_root / "Twitter15_16_dataset-main").exists():
        twitter_tree_root = twitter_tree_root / "Twitter15_16_dataset-main"

    # Twitter source: need source_tweets.txt
    twitter_source_root = _find_dir_containing(data_raw, ["twitter15/source_tweets.txt", "twitter16/source_tweets.txt"])

    # TweetEval: emotion_*.csv
    tweeteval_root = _find_dir_containing(data_raw, ["emotion_train.csv", "emotion_validation.csv", "emotion_test.csv"])

    # SemEval: 2018-E-c-En-*.txt
    semeval_root = _find_dir_containing(data_raw, ["2018-E-c-En-train.txt", "2018-E-c-En-dev.txt", "2018-E-c-En-test.txt"])

    return RawPaths(
        pheme_root=pheme_root,
        twitter_tree_root=twitter_tree_root,
        twitter_source_root=twitter_source_root,
        tweeteval_root=tweeteval_root,
        semeval_root=semeval_root,
    )

def extract_zip(zip_path: Path, out_dir: Path) -> None:
    zip_path = Path(zip_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(out_dir)

def ensure_extracted(data_raw: Path) -> None:
    """If common ZIPs exist in data_raw, extract them into standard subfolders.
    This is optional; scripts also work with already-extracted folders.
    """
    data_raw = Path(data_raw)
    mapping = {
        "pheme.zip": "pheme",
        "Twitter15_16_dataset-main.zip": "twitter_trees",
        "Twitter15 and Twitter16.zip": "twitter_source",
        "tweeteval.zip": "tweeteval",
        "SemEval 2018.zip": "semeval2018",
    }
    for zip_name, sub in mapping.items():
        zp = data_raw / zip_name
        if zp.exists() and zp.is_file():
            out = data_raw / sub
            # If out already has content, skip
            if out.exists() and any(out.iterdir()):
                continue
            extract_zip(zp, out)
