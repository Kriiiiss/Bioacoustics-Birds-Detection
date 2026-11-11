"""Evaluate one frozen M4 checkpoint on the already-inspected UHH test set.

Same metrics, thresholds and outputs as evaluate.py (it calls evaluate.evaluate_predictions): thresholds were
chosen on validation data and saved in the checkpoint; this script never chooses, refits, or adjusts one.
The test targets are the last clip of each four-clip sequence; context comes from the same test manifest.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import file_sha256
from evaluate import evaluate_predictions
from m4_dataset import BirdsetM4Dataset
from m4_model import CNNLSTM
from train import choose_device, verify_dataset

VARIANTS = {"M4": True, "M4_noctx": False}      # variant -> uses the earlier clips as context


def _validate_checkpoint(checkpoint, dataset, path_mappings):
    required = ("model_state", "model_name", "n_known", "known_species", "thresholds", "seed", "epoch", "variant",
                "use_context", "hidden", "normalization_manifest_sha256", "norm_stats_sha256",
                "val_manifest_sha256", "train_manifest_sha256")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Missing checkpoint fields: {missing}")
    if checkpoint["model_name"] != "m4_cnn_lstm" or checkpoint["variant"] not in VARIANTS:
        raise ValueError("Only M4 / M4_noctx checkpoints from train_m4.py are supported")
    if checkpoint["use_context"] != VARIANTS[checkpoint["variant"]]:
        raise ValueError("Checkpoint variant and use_context disagree")
    if checkpoint.get("smoke_batches", 0):
        raise ValueError("Refusing to evaluate a limited smoke-run checkpoint on the test set")
    if checkpoint["n_known"] != len(checkpoint["known_species"]):
        raise ValueError("Checkpoint n_known and species names disagree")
    paths = {
        "normalization_manifest_sha256": dataset / "M1/train.jsonl",
        "norm_stats_sha256": dataset / "shared/train_norm_m1.json",
        "val_manifest_sha256": dataset / "shared/val.jsonl",
        "train_manifest_sha256": dataset / "M2/train.jsonl",
    }
    for key, path in paths.items():
        if checkpoint[key] != file_sha256(path):
            raise ValueError(f"Checkpoint provenance does not match {path}")
    known_species, _ = verify_dataset(dataset, path_mappings)
    if checkpoint["known_species"] != known_species:
        raise ValueError("Checkpoint known-species output order does not match the dataset")
    test_hash = file_sha256(dataset / "shared/test.jsonl")
    if checkpoint.get("test_manifest_sha256", test_hash) != test_hash:
        raise ValueError("Test manifest differs from the one recorded during training")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--path-map", action="append", nargs=2, default=[], metavar=("OLD_PREFIX", "NEW_PREFIX"),
                        help="Relocate a prefix in dataset paths; may be repeated")
    parser.add_argument("--output", type=Path, required=True, help="New JSON file for metrics")
    parser.add_argument("--predictions", type=Path, help="New CSV file; defaults to output stem + .predictions.csv")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--cpu-threads", type=int, default=2)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.cpu_threads <= 0:
        parser.error("batch size and cpu threads must be positive")
    dataset = args.dataset_dir.resolve()
    output = args.output.resolve()
    predictions = args.predictions.resolve() if args.predictions else output.with_suffix(".predictions.csv")
    if output == predictions or output.exists() or predictions.exists():
        raise FileExistsError("Output or predictions already exist; choose new paths")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    _validate_checkpoint(checkpoint, dataset, args.path_map)
    device = choose_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    model = CNNLSTM(checkpoint["n_known"], hidden=checkpoint["hidden"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    data = BirdsetM4Dataset(dataset / "shared/test.jsonl", "test", dataset / "shared/train_norm_m1.json",
                            dataset / "M1/train.jsonl", path_mappings=args.path_map,
                            use_context=checkpoint["use_context"], augment=False)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    true_chunks, score_chunks, groups, clip_ids = [], [], [], []
    with torch.inference_mode():
        for x, mask, labels, metadata in loader:
            logits_known, logits_unknown = model(x.to(device), mask.to(device))
            logits = torch.cat((logits_known, logits_unknown), dim=1)
            true_chunks.append(labels.cpu().numpy())
            score_chunks.append(torch.sigmoid(logits).cpu().numpy())
            groups.extend(metadata["case"])
            clip_ids.extend(metadata["clip_id"])
    labels = np.concatenate(true_chunks)
    scores = np.concatenate(score_chunks)
    result = evaluate_predictions(labels, scores, checkpoint["thresholds"], groups, checkpoint["known_species"])
    # padding breakdown: sequences with all four clips real vs. at least one padded context slot
    n_real = np.array([sum(clip is not None for clip in data.sequence_ids(i)) for i in range(len(data))])
    result["sequence_padding"] = {"all_four_real": int((n_real == 4).sum()), "at_least_one_padded": int((n_real < 4).sum())}
    result.update({"dataset": "UHH", "model_name": checkpoint["model_name"], "variant": checkpoint["variant"],
                   "use_context": checkpoint["use_context"],
                   "seed": checkpoint["seed"], "checkpoint_epoch": checkpoint["epoch"],
                   "thresholds_source": "validation; frozen in checkpoint",
                   "test_manifest_sha256": file_sha256(dataset / "shared/test.jsonl"),
                   "note": "Exploratory comparison: teammate's test results were previously inspected; not a pristine blind final test.",
                   "empty_note": "No bird annotated at the preprocessing threshold; quiet or bird absence is not guaranteed."})
    output.parent.mkdir(parents=True, exist_ok=True)
    predictions.parent.mkdir(parents=True, exist_ok=True)
    names = checkpoint["known_species"] + ["unknown"]
    # Exclusive creation avoids silently replacing prior experiments.
    with predictions.open("x", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["clip_id", "case"] + [f"true_{n}" for n in names] + [f"score_{n}" for n in names])
        for clip, case, yrow, prow in zip(clip_ids, groups, labels, scores):
            writer.writerow([clip, case, *yrow.tolist(), *[float(value) for value in prow]])
    with output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(f"Saved frozen-threshold test metrics: {output}\nSaved predictions: {predictions}")


if __name__ == "__main__":
    main()
