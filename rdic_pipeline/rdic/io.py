
from __future__ import annotations
from pathlib import Path
import json
import ast
import re
import pandas as pd
import numpy as np
import networkx as nx
from datetime import datetime

def safe_read_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def parse_twitter_label_file(path: Path) -> dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # format: label:tweet_id
            if ":" in line:
                lab, tid = line.split(":", 1)
                mapping[tid.strip()] = lab.strip()
    return mapping

def parse_source_tweets(path: Path) -> dict[str, str]:
    mapping = {}
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.rstrip("\n").rstrip("\r")
            if not line:
                continue
            if "\t" in line:
                tid, txt = line.split("\t", 1)
                mapping[tid.strip()] = txt.strip()
    return mapping

def parse_tree_line(line: str):
    # format: ['ROOT','ROOT','0.0']->['uid','tweet_id','delay']
    left, right = line.split("->")
    a = ast.literal_eval(left.strip())
    b = ast.literal_eval(right.strip())
    # returns tuples: (uid, tweet_id, delay)
    return (str(a[0]), str(a[1]), float(a[2])), (str(b[0]), str(b[1]), float(b[2]))

def label4_to_binary(label4: str) -> int:
    # Twitter15/16 mapping: non-rumor => 0, else 1
    return 0 if label4 in {"non-rumor", "non-rumour", "nonrumor", "nonrumour"} else 1

def pheme_annotation_to_labels(ann: dict) -> tuple[int, str|None]:
    # binary: rumour=1, nonrumour=0
    is_r = ann.get("is_rumour", "")
    is_r = str(is_r).lower()
    y = 1 if is_r == "rumour" else 0
    ver = None
    if y == 1:
        # In PHEME, "true" may be '1' and misinformation may be 1/0
        true_v = ann.get("true", None)
        misinfo = ann.get("misinformation", None)
        try:
            true_i = int(true_v) if true_v is not None else 0
        except:
            true_i = 0
        try:
            mis_i = int(misinfo) if misinfo is not None else 0
        except:
            mis_i = 0
        if mis_i == 1:
            ver = "false"
        elif true_i == 1 and mis_i == 0:
            ver = "true"
        else:
            ver = "unverified"
    return y, ver

def pheme_parse_structure(struct: dict) -> list[tuple[str,str]]:
    # struct: {root_id: {child: subtree, ...}}
    edges = []
    if not struct:
        return edges
    root = next(iter(struct.keys()))
    def rec(parent, subtree):
        if subtree == [] or subtree is None:
            return
        if isinstance(subtree, dict):
            for child, sub in subtree.items():
                edges.append((str(parent), str(child)))
                rec(child, sub)
        elif isinstance(subtree, list):
            for item in subtree:
                if isinstance(item, dict):
                    for child, sub in item.items():
                        edges.append((str(parent), str(child)))
                        rec(child, sub)
    rec(root, struct[root])
    return edges

def compute_depths(root_id: str, edges: list[tuple[str,str]]) -> dict[str,int]:
    g = nx.DiGraph()
    g.add_edges_from(edges)
    depths = {root_id: 0}
    # BFS
    queue = [root_id]
    while queue:
        u = queue.pop(0)
        for v in g.successors(u):
            if v not in depths:
                depths[v] = depths[u] + 1
                queue.append(v)
    return depths

def entropy_from_probs(probs: np.ndarray, eps: float=1e-12) -> np.ndarray:
    p = np.clip(probs, eps, 1.0)
    return -(p * np.log(p)).sum(axis=1)
