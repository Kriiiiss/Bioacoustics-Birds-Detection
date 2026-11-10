"""Evaluate one frozen checkpoint on the already-inspected UHH test set.

All thresholds must have been chosen on validation data and saved in the
checkpoint. This script never chooses, refits, or adjusts a threshold.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from dataset import BirdsetM2Dataset, file_sha256
from metrics import calculate_metrics
from model import ImprovedBirdCNN
from train import choose_device, verify_dataset


def _ratio(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator else None


def evaluate_predictions(labels, scores, thresholds, groups, known_species):
    """Compute all statistics using frozen thresholds and whole-split scores."""
    y = np.asarray(labels, dtype=np.int8)
    p = np.asarray(scores, dtype=np.float64)
    thresholds = np.asarray(thresholds, dtype=np.float64)
    groups = np.asarray(groups)
    result = calculate_metrics(y, p, thresholds, groups, known_species)
    pred = p >= thresholds
    result["unknown_positive_prevalence"] = float(y[:, -1].mean())
    overall_unknown = result["per_class"]["unknown"]
    tp_all, fp_all = overall_unknown["true_positive"], overall_unknown["false_positive"]
    fn_all = overall_unknown["false_negative"]
    result["unknown_precision"] = _ratio(tp_all, tp_all + fp_all)
    result["unknown_recall"] = _ratio(tp_all, tp_all + fn_all)
    result["thresholds"] = {name: float(value) for name, value in zip([*known_species, "unknown"], thresholds)}
    for name in ("E1", "E2", "E3", "E4", "empty"):
        mask = groups == name
        actual, predicted = y[mask].astype(bool), pred[mask]
        known_y, known_pred = actual[:, :-1], predicted[:, :-1]
        known_tp = int((known_y & known_pred).sum())
        known_fp = int((~known_y & known_pred).sum())
        known_fn = int((known_y & ~known_pred).sum())
        unknown_y, unknown_pred = actual[:, -1], predicted[:, -1]
        tp = int((unknown_y & unknown_pred).sum())
        fp = int((~unknown_y & unknown_pred).sum())
        fn = int((unknown_y & ~unknown_pred).sum())
        tn = int((~unknown_y & ~unknown_pred).sum())
        exact = np.all(predicted == actual, axis=1)
        known_exact = np.all(known_pred == known_y, axis=1)
        row = result["groups"][name]
        row.update({
            "known_micro_precision": _ratio(known_tp, known_tp + known_fp),
            "known_micro_recall": _ratio(known_tp, known_tp + known_fn),
            "known_false_positive_labels": known_fp,
            "clips_with_any_false_known": int(np.any(~known_y & known_pred, axis=1).sum()),
            "known_exact_match_rate": _ratio(int(known_exact.sum()), len(actual)),
            "unknown_recall": _ratio(tp, tp + fn),
            "unknown_false_alarm_rate": _ratio(fp, fp + tn),
            "joint_exact_match_rate": _ratio(int(exact.sum()), len(actual)),
        })
        if name == "empty":
            row["any_known_false_alarm_rate"] = _ratio(int(np.any(known_pred, axis=1).sum()), len(actual))
            row["any_alert_rate"] = _ratio(int(np.any(predicted, axis=1).sum()), len(actual))
    return result


def _validate_checkpoint(checkpoint, dataset, path_mappings):
    required = ("model_state", "model_name", "n_known", "known_species", "thresholds", "seed", "epoch", "variant",
                "normalization_manifest_sha256", "norm_stats_sha256", "val_manifest_sha256", "train_manifest_sha256")
    missing = [key for key in required if key not in checkpoint]
    if missing:
        raise ValueError(f"Missing checkpoint fields: {missing}")
    if checkpoint["variant"] not in {"M1", "M2"}:
        raise ValueError("Only M1/M2 checkpoints are supported")
    if checkpoint.get("smoke_batches", 0):
        raise ValueError("Refusing to evaluate a limited smoke-run checkpoint on the test set")
    if checkpoint["n_known"] != len(checkpoint["known_species"]):
        raise ValueError("Checkpoint n_known and species names disagree")
    paths = {
        "normalization_manifest_sha256": dataset / "M1/train.jsonl",
        "norm_stats_sha256": dataset / "shared/train_norm_m1.json",
        "val_manifest_sha256": dataset / "shared/val.jsonl",
        "train_manifest_sha256": dataset / checkpoint["variant"] / "train.jsonl",
    }
    for key, path in paths.items():
        if checkpoint[key] != file_sha256(path):
            raise ValueError(f"Checkpoint provenance does not match {path}")
    if checkpoint["model_name"] != "improved":
        raise ValueError("Only the selected frequency-aware CNN checkpoints are supported")
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
    model = ImprovedBirdCNN(checkpoint["n_known"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device).eval()
    data = BirdsetM2Dataset(dataset / "shared/test.jsonl", "test", dataset / "shared/train_norm_m1.json",
                            dataset / "M1/train.jsonl", path_mappings=args.path_map)
    loader = DataLoader(data, batch_size=args.batch_size, shuffle=False, num_workers=0)
    true_chunks, score_chunks, groups, clip_ids = [], [], [], []
    with torch.inference_mode():
        for features, labels, metadata in loader:
            logits_known, logits_unknown = model(features.to(device))
            logits = torch.cat((logits_known, logits_unknown), dim=1)
            true_chunks.append(labels.cpu().numpy())
            score_chunks.append(torch.sigmoid(logits).cpu().numpy())
            groups.extend(metadata["case"])
            clip_ids.extend(metadata["clip_id"])
    labels = np.concatenate(true_chunks)
    scores = np.concatenate(score_chunks)
    result = evaluate_predictions(labels, scores, checkpoint["thresholds"], groups, checkpoint["known_species"])
    result.update({"dataset": "UHH", "model_name": checkpoint["model_name"], "variant": checkpoint["variant"],
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
