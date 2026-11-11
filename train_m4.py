"""Train M4 (CNN-LSTM over four consecutive clips of one recording) on a prepared UHH dataset.

Same data, normalisation, loss, optimiser, schedule, early stopping, thresholds and validation metrics as
train.py (M2 training manifest, shared validation manifest, shared training-only M1 normalisation); only the
model and the input differ: the three earlier clips of the same recording are given as context, and only the
last clip is predicted. ``--no-context`` masks the three earlier clips (control with the identical pipeline).
The held-out test features are not read during training; checkpoint and thresholds use validation only.
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

from dataset import file_sha256
from m4_dataset import BirdsetM4Dataset
from m4_model import CNNLSTM
from metrics import calculate_metrics, select_validation_thresholds
from train import choose_device, verify_dataset


def validate(model, loader, device, bce):
    model.eval()
    truths, probs, cases = [], [], []
    loss_sum, total = 0.0, 0
    with torch.inference_mode():
        for x, mask, target, meta in loader:
            x, mask, target = x.to(device), mask.to(device), target.to(device)
            known, unknown = model(x, mask)
            loss = bce(known, target[:, :-1]) + bce(unknown, target[:, -1:])
            loss_sum += loss.item() * len(target)
            total += len(target)
            truths.append(target.cpu().numpy())
            probs.append(torch.sigmoid(torch.cat((known, unknown), dim=1)).cpu().numpy())
            cases.extend(meta["case"])
    return loss_sum / total, np.concatenate(truths), np.concatenate(probs), cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True, help="Prepared UHH dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="New or empty run directory")
    parser.add_argument("--no-context", action="store_true", help="mask the three earlier clips (single-clip control)")
    parser.add_argument("--path-map", action="append", nargs=2, default=[], metavar=("OLD_PREFIX", "NEW_PREFIX"),
                        help="Relocate a prefix in dataset paths without rewriting manifests; may be repeated")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", choices=["auto", "mps", "cuda", "cpu"], default="auto")
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--smoke-batches", type=int, default=0,
                        help="Limit batches for integration checks; checkpoints cannot be evaluated on test")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.cpu_threads, args.hidden) <= 0 or args.workers < 0 or args.smoke_batches < 0:
        parser.error("epochs, patience, batch-size, cpu-threads and hidden must be positive; workers and smoke-batches nonnegative")
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
    use_context = not args.no_context
    variant = "M4" if use_context else "M4_noctx"
    train_manifest = dataset / "M2/train.jsonl"          # M4 trains on all kept training clips
    m1_manifest = dataset / "M1/train.jsonl"             # only used to check the normalisation provenance
    val_manifest = dataset / "shared/val.jsonl"
    norm = dataset / "shared/train_norm_m1.json"
    train_ds = BirdsetM4Dataset(train_manifest, "train", norm, m1_manifest, path_mappings=args.path_map,
                                use_context=use_context, augment=True, seed=args.seed)
    val_ds = BirdsetM4Dataset(val_manifest, "val", norm, m1_manifest, path_mappings=args.path_map,
                              use_context=use_context, augment=False)
    generator = torch.Generator().manual_seed(args.seed)
    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, generator=generator)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    model = CNNLSTM(len(known), hidden=args.hidden).to(device)
    bce = nn.BCEWithLogitsLoss()
    opt = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    run = {"model_name": "m4_cnn_lstm", "variant": variant, "use_context": use_context, "hidden": args.hidden,
           "n_known": len(known), "known_species": known, "seed": args.seed, "batch_size": args.batch_size,
           "learning_rate": args.learning_rate, "epochs": args.epochs, "patience": args.patience,
           "device": str(device), "path_mappings": args.path_map,
           "augmentation": "training only, one draw per sequence applied to every real clip: log-power gain, "
                           "cyclic time shift, two time and two frequency masks",
           "scheduler": "CosineAnnealingLR(T_max=epochs)", "smoke_batches": args.smoke_batches,
           "train_manifest_sha256": file_sha256(train_manifest), "val_manifest_sha256": file_sha256(val_manifest),
           "norm_stats_sha256": file_sha256(norm), "normalization_manifest_sha256": file_sha256(m1_manifest),
           "dataset_report_sha256": file_sha256(dataset / "report.json"),
           "test_manifest_sha256": file_sha256(dataset / "shared/test.jsonl"),
           "n_train": len(train_ds), "n_val": len(val_ds), "params": sum(p.numel() for p in model.parameters()),
           "note": "Shared training-only M1 normalization; split/test manifests may be inspected for integrity, but "
                   "test features and labels are never used for training or model selection."}
    (output / "run_config.json").write_text(json.dumps(run, indent=2) + "\n")
    print(f"Verified prepared dataset {counts}; {variant}; {run['params']:,} parameters; device={device}", flush=True)
    best, bad, steps, history = float("inf"), 0, 0, []
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        model.train()
        total, loss_sum = 0, 0.0
        for batch_index, (x, mask, y, _) in enumerate(train_dl):
            if args.smoke_batches and batch_index >= args.smoke_batches:
                break
            x, mask, y = x.to(device), mask.to(device), y.to(device)
            known_logits, unknown_logits = model(x, mask)
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
        print(f"{variant} epoch {epoch} train={loss_sum/total:.4f} val={val_loss:.4f} steps={steps} seconds={entry['seconds']:.1f} device={device}", flush=True)
        if bad >= args.patience:
            break
    print(f"Completed {len(history)} epoch(s); test features not loaded. Saved to {output}", flush=True)


if __name__ == "__main__":
    main()
