#!/usr/bin/env python3
"""Download only JetClass val (5M) + test (20M), skip the 100M train tarballs."""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
from utils.dataset_utils import extract_archive, get_file

URLS = [
    ("Pythia/", "https://zenodo.org/record/6619768/files/JetClass_Pythia_val_5M.tar", "7235ccb577ed85023ea3ab4d5e6160cf"),
    ("Pythia/", "https://zenodo.org/record/6619768/files/JetClass_Pythia_test_20M.tar", "64e5156d26d101adeb43b8388207d767"),
]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-d", "--basedir", default="datasets")
    p.add_argument("-e", "--envfile", default="env.sh")
    p.add_argument("-f", "--force", action="store_true")
    args = p.parse_args()

    datadir = os.path.join(args.basedir, "JetClass")
    os.makedirs(datadir, exist_ok=True)
    for subdir, url, md5 in URLS:
        print(f"=== {url} ===")
        fpath, downloaded = get_file(url, datadir=datadir, file_hash=md5, force_download=args.force)
        extract_archive(fpath, path=os.path.join(datadir, subdir))

    datapath = f"DATADIR_JetClass={os.path.abspath(datadir)}"
    with open(args.envfile) as f:
        lines = f.readlines()
    with open(args.envfile, "w") as f:
        for l in lines:
            if "DATADIR_JetClass" in l:
                l = f"export {datapath}\n"
            f.write(l)
    print(f'Updated {args.envfile} → export {datapath}')


if __name__ == "__main__":
    main()
