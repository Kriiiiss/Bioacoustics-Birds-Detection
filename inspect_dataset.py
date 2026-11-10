"""Check a prepared split, its train-only normalization, and optional features.

This utility reads existing manifests and data; it does not generate dataset
splits, download audio, recompute normalization, or alter provenance files.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from dataset import BirdsetM2Dataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--norm", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True,
                        help="Original M1 train manifest used to compute normalization")
    parser.add_argument("--path-map", metavar=("OLD_PREFIX", "NEW_PREFIX"),
                        nargs=2, action="append", default=[],
                        help="Map an absolute recorded path prefix to its local location; repeatable")
    parser.add_argument("--check-features", type=int, default=0, metavar="N",
                        help="Read and validate the first N features (0: metadata only, -1: all)")
    args = parser.parse_args()
    if args.check_features < -1:
        parser.error("--check-features must be -1, 0, or a positive integer")

    data = BirdsetM2Dataset(args.manifest, args.split, args.norm,
                            args.train_manifest, path_mappings=args.path_map)
    limit = len(data) if args.check_features == -1 else min(args.check_features, len(data))
    for i in range(limit):
        data[i]  # Validates feature contents, shape, and normalization.
    print(json.dumps({
        "split": args.split,
        "num_samples": len(data),
        "num_outputs": len(data.rows[0]["labels"]),
        "case_counts": dict(sorted(Counter(row.get("case", "") for row in data.rows).items())),
        "normalization_source": "training manifest hash verified",
        "features_checked": limit,
        "feature_shape": [1, 128, 500],
    }, indent=2))


if __name__ == "__main__":
    main()
