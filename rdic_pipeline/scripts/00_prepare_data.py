
from __future__ import annotations
import argparse
from pathlib import Path
from rdic.paths import ensure_extracted

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-raw", type=str, default="data_raw", help="Folder containing raw ZIPs or extracted data.")
    args = ap.parse_args()
    ensure_extracted(Path(args.data_raw))
    print("Done. If you provided ZIPs, they were extracted into standard subfolders under data_raw/.")

if __name__ == "__main__":
    main()
