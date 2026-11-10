"""Train the frequency-aware CNN on a prepared UHH M1 or M2 dataset.

The same model and optimization are used for both variants. The held-out test
features are not read during training; checkpoint and thresholds use dev only.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

from dataset import BirdsetM2Dataset, file_sha256, load_norm_stats, read_manifest, resolve_path
from metrics import calculate_metrics, select_validation_thresholds
from model import ImprovedBirdCNN


def choose_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def verify_dataset(dataset: Path, path_mappings=None):
    """Verify preparation hashes and split provenance without reading test features.

    A source audit must be supplied: path mappings relocate files at runtime
    without rewriting manifests, preserving their hashes and existing checkpoints.
    """
    required = {name: dataset / name for name in (
        "M1/train.jsonl", "M2/train.jsonl", "shared/val.jsonl", "shared/test.jsonl",
        "shared/train_norm_m1.json", "report.json", "verification.json")}
    for path in required.values():
        if not path.is_file():
            raise FileNotFoundError(f"Prepared dataset missing {path}")
    report = json.loads(required["report.json"].read_text(encoding="utf-8"))
    verification = json.loads(required["verification.json"].read_text(encoding="utf-8"))
    if report.get("dataset") != "UHH" or verification.get("status") != "passed":
        raise ValueError("A passed UHH dataset verification is required")
    source = Path(resolve_path(report["source"], path_mappings)) / "manifests/audit.json"
    if file_sha256(source) != report["source_audit_sha256"]:
        raise ValueError("Source audit changed since dataset preparation")
    audit = json.loads(source.read_text(encoding="utf-8"))
    known_species = report["known_species"]
    if known_species != audit["known_species"] or len(set(known_species)) != len(known_species):
        raise ValueError("Known species ordering differs between report and source audit")
    for variant in ("M1", "M2"):
        actual = file_sha256(required[f"{variant}/train.jsonl"])
        if actual != report["train_manifest_sha256"][variant]:
            raise ValueError(f"{variant} training manifest differs from preparation report")
    for split in ("val", "test"):
        if file_sha256(required[f"shared/{split}.jsonl"]) != report[f"shared_{split}_sha256"]:
            raise ValueError(f"{split} manifest differs from preparation report")
    rows = {
        "M1": read_manifest(required["M1/train.jsonl"], expected_split="train"),
        "M2": read_manifest(required["M2/train.jsonl"], expected_split="train"),
        "val": read_manifest(required["shared/val.jsonl"], expected_split="val"),
        "test": read_manifest(required["shared/test.jsonl"], expected_split="test"),
    }
    species_groups = report["species_splits"]
    if species_groups != audit["species_splits"]:
        raise ValueError("Unknown-species groups differ between report and source audit")
    known_set = set(known_species)
    unknown_sets = {split: set(species_groups[split]) for split in ("train", "dev", "test")}
    species_sets = [known_set, *unknown_sets.values()]
    if any(species_sets[i] & species_sets[j] for i in range(len(species_sets)) for j in range(i + 1, len(species_sets))):
        raise ValueError("Known and per-split unknown species must be disjoint")
    for name, data in rows.items():
        split = "train" if name in ("M1", "M2") else "dev" if name == "val" else "test"
        allowed_unknown = unknown_sets[split]
        for row in data:
            codes = row.get("species_codes")
            if not isinstance(codes, list) or len(codes) != len(set(codes)):
                raise ValueError(f"{name} row {row['clip_id']} has invalid species_codes")
            present = set(codes)
            known_count = sum(code in present for code in known_species)
            unknown_count = len(present & allowed_unknown)
            expected = [int(code in present) for code in known_species] + [int(unknown_count > 0)]
            case = ("E4" if known_count and unknown_count else "E3" if unknown_count else
                    "E2" if known_count >= 2 else "E1" if known_count == 1 else "empty")
            if (present - known_set - allowed_unknown or row["labels"] != expected or
                row.get("species_count") != len(present) or row.get("case") != case or
                (name == "M1" and len(present) > 1)):
                raise ValueError(f"{name} row {row['clip_id']} has inconsistent species, labels, or case")
    m1_rows = {r["clip_id"]: r for r in rows["M1"]}
    m2_rows = {r["clip_id"]: r for r in rows["M2"]}
    if not m1_rows.keys() <= m2_rows.keys():
        raise ValueError("M1 examples must be contained in M2")
    for clip_id, row in m1_rows.items():
        other = m2_rows[clip_id]
        if row != other:
            raise ValueError(f"M1/M2 manifests disagree for {clip_id}")
    for name in ("M1", "M2"):
        if verification["training_counts"][name] != len(rows[name]):
            raise ValueError(f"{name} count differs from verification")
    for split in ("val", "test"):
        if verification[f"{split if split == 'test' else 'validation'}_count"] != len(rows[split]):
            raise ValueError(f"{split} count differs from verification")
    for field in ("clip_id", "recording_id", "session_id"):
        groups = [{str(r.get(field, "")) for r in rows[split]} for split in ("M2", "val", "test")]
        if any("" in group for group in groups) or any(groups[i] & groups[j] for i in range(3) for j in range(i + 1, 3)):
            raise ValueError(f"Missing or overlapping {field} across splits")
    load_norm_stats(required["shared/train_norm_m1.json"], required["M1/train.jsonl"])
    return known_species, {name: len(data) for name, data in rows.items()}


def validate(model, loader, device, bce):
    model.eval()
    truths, probs, cases = [], [], []
    loss_sum, total = 0.0, 0
    with torch.inference_mode():
        for features, target, meta in loader:
            features, target = features.to(device), target.to(device)
            known, unknown = model(features)
            loss = bce(known, target[:, :-1]) + bce(unknown, target[:, -1:])
            loss_sum += loss.item() * len(target)
            total += len(target)
            truths.append(target.cpu().numpy())
            probs.append(torch.sigmoid(torch.cat((known, unknown), dim=1)).cpu().numpy())
            cases.extend(meta["case"])
    return loss_sum / total, np.concatenate(truths), np.concatenate(probs), cases


def augment_batch(x, band_std, rng):
    """Apply the experiment's gain, cyclic time shift, and time/frequency masks."""
    for sample in x:
        sample += float(rng.uniform(-1.0, 1.0)) / band_std
        sample[:] = torch.roll(sample, int(rng.integers(0, sample.shape[-1])), dims=-1)
        for _ in range(2):
            width = int(rng.integers(0, 41))
            start = int(rng.integers(0, sample.shape[-1] - width + 1))
            sample[:, :, start:start + width] = 0
            height = int(rng.integers(0, 13))
            start = int(rng.integers(0, sample.shape[-2] - height + 1))
            sample[:, start:start + height, :] = 0
    return x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Prepared UHH dataset directory")
    parser.add_argument("--variant", choices=["M1", "M2"], required=True)
    parser.add_argument("--output", type=Path, required=True, help="New or empty run directory")
    parser.add_argument("--path-map", action="append", nargs=2, default=[], metavar=("OLD_PREFIX", "NEW_PREFIX"),
                        help="Relocate a prefix in dataset paths without rewriting manifests; may be repeated")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Limit batches for integration checks; checkpoints cannot be evaluated on test")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.cpu_threads) <= 0 or args.workers < 0 or args.smoke_batches < 0:
        parser.error("epochs, patience, batch-size and cpu-threads must be positive; workers and smoke-batches nonnegative")
    dataset, output = args.dataset_dir.expanduser().resolve(), args.output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty directory {output}")
    known, counts = verify_dataset(dataset, args.path_map)
    output.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.cpu_threads)
    device = choose_device(args.device)
    train_manifest = dataset / args.variant / "train.jsonl"
    m1_manifest = dataset / "M1/train.jsonl"
    val_manifest = dataset / "shared/val.jsonl"
    norm = dataset / "shared/train_norm_m1.json"
    train_ds = BirdsetM2Dataset(train_manifest, "train", norm, m1_manifest, path_mappings=args.path_map)
    val_ds = BirdsetM2Dataset(val_manifest, "val", norm, m1_manifest, path_mappings=args.path_map)
    generator = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = ImprovedBirdCNN(len(known)).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    augment_rng = np.random.default_rng(args.seed)
    band_std = torch.from_numpy(train_ds.std).unsqueeze(0)
    run = {"model_name": "improved", "variant": args.variant, "n_known": len(known), "known_species": known,
           "seed": args.seed, "batch_size": args.batch_size, "learning_rate": args.learning_rate,
           "epochs": args.epochs, "patience": args.patience, "device": str(device), "path_mappings": args.path_map,
           "augmentation": "training only: log-power gain, cyclic time shift, two time and two frequency masks",
           "scheduler": "CosineAnnealingLR(T_max=epochs)", "smoke_batches": args.smoke_batches,
           "train_manifest_sha256": file_sha256(train_manifest), "val_manifest_sha256": file_sha256(val_manifest),
           "norm_stats_sha256": file_sha256(norm), "normalization_manifest_sha256": file_sha256(m1_manifest),
           "dataset_report_sha256": file_sha256(dataset / "report.json"),
           "test_manifest_sha256": file_sha256(dataset / "shared/test.jsonl"),
           "n_train": len(train_ds), "n_val": len(val_ds), "params": sum(p.numel() for p in model.parameters()),
           "note": "Shared training-only M1 normalization; split/test manifests may be inspected for integrity, but test features and labels are never used for training or model selection."}
    (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Verified prepared dataset {counts}; device={device}", flush=True)
    best, bad, steps, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total, loss_sum = 0, 0.0
        for batch_index, (x, y, _) in enumerate(train_dl):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            x = augment_batch(x, band_std, augment_rng)
            x, y = x.to(device), y.to(device)
            known_logits, unknown_logits = model(x)
            loss = bce(known_logits, y[:, :-1]) + bce(unknown_logits, y[:, -1:])
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite loss")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            steps += 1
            total += len(y)
            loss_sum += loss.item() * len(y)
        vd = (DataLoader(Subset(val_ds, range(min(len(val_ds), args.batch_size * args.smoke_batches))),
                         batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
              if args.smoke_batches else val_dl)
        scheduler.step()
        val_loss, y, probabilities, groups = validate(model, vd, device, bce)
        selection = select_validation_thresholds(y, probabilities, known)
        val_metrics = calculate_metrics(y, probabilities, selection["thresholds"], groups, known)
        entry = {"epoch": epoch, "train_loss": loss_sum / total, "validation_loss": val_loss,
                 "validation_known_macro_ap": val_metrics["known_macro_average_precision"],
                 "validation_unknown_ap": val_metrics["unknown_average_precision"],
                 "seconds": time.time() - start, "optimizer_steps_total": steps,
                 "next_learning_rate": opt.param_groups[0]["lr"]}
        history.append(entry)
        if val_loss < best:
            best, bad = val_loss, 0
            checkpoint = {**run, "epoch": epoch, "model_state": model.state_dict(),
                          "thresholds": selection["thresholds"], "threshold_selection": selection,
                          "validation_loss": val_loss}
            torch.save(checkpoint, output / "best.pt")
            (output / "best_validation.json").write_text(json.dumps({"entry": entry, "metrics": val_metrics,
                "threshold_selection": selection}, indent=2, allow_nan=False) + "\n")
        else:
            bad += 1
        (output / "history.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(f"{args.variant} epoch {epoch} train={loss_sum/total:.4f} val={val_loss:.4f} steps={steps} seconds={entry['seconds']:.1f} device={device}", flush=True)
        if bad >= args.patience:
            break
    print(f"Completed {len(history)} epoch(s); test features not loaded. Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
