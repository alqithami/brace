
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import sys

def run(cmd):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, check=True)
    return r.returncode

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=str, default="data_raw")
    args = ap.parse_args()

    run([sys.executable, "scripts/00_validate_paths.py", "--data-raw", args.data_raw])
    run([sys.executable, "scripts/01_preprocess_pheme.py", "--data-raw", args.data_raw])
    run([sys.executable, "scripts/02_preprocess_twitter1516.py", "--data-raw", args.data_raw])
    run([sys.executable, "scripts/03_preprocess_emotion.py", "--data-raw", args.data_raw])
    run([sys.executable, "scripts/04_make_splits.py"])
    print("Smoketest OK.")

if __name__ == "__main__":
    main()
