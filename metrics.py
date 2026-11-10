"""Multi-label evaluation for the known-species and unknown-presence heads.

The last column is the independent unknown head. Thresholds are selected on
validation data and passed explicitly to evaluation; evaluating never fits or
changes them. All returned values can be serialized with ``json.dumps``.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np


def _inputs(y_true: np.ndarray, y_prob: np.ndarray):
    labels = np.asarray(y_true)
    probs = np.asarray(y_prob, dtype=np.float64)
    if labels.ndim != 2 or probs.ndim != 2 or labels.shape != probs.shape:
        raise ValueError("y_true and y_prob must have the same [N, K+1] shape")
    if labels.shape[1] < 2:
        raise ValueError("at least one known species and one unknown head are required")
    if not np.all((labels == 0) | (labels == 1)):
        raise ValueError("y_true must contain only 0 and 1")
    if not np.all(np.isfinite(probs)) or np.any((probs < 0) | (probs > 1)):
        raise ValueError("y_prob must contain finite probabilities in [0, 1]")
    return labels.astype(np.int8), probs


def _names(num_outputs: int, known_species: Sequence[str] | None) -> list[str]:
    if known_species is None:
        return [f"known_{i}" for i in range(num_outputs - 1)] + ["unknown"]
    names = list(known_species)
    if len(names) != num_outputs - 1 or any(not isinstance(n, str) or not n for n in names):
        raise ValueError("known_species must contain one nonempty name per known output")
    if len(set(names)) != len(names) or "unknown" in names:
        raise ValueError("known_species must be unique and must not include 'unknown'")
    return names + ["unknown"]


def _binary_counts(labels: np.ndarray, preds: np.ndarray) -> dict:
    return {
        "positive_support": int(np.sum(labels == 1)),
        "negative_support": int(np.sum(labels == 0)),
        "true_positive": int(np.sum((labels == 1) & preds)),
        "false_positive": int(np.sum((labels == 0) & preds)),
        "false_negative": int(np.sum((labels == 1) & ~preds)),
        "true_negative": int(np.sum((labels == 0) & ~preds)),
    }


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float | None:
    """Compute non-interpolated average precision, grouping tied scores.

    AP is deliberately undefined when either class is missing, to prevent
    reporting a misleading discrimination score on an all-positive group.
    """
    positives = int(np.sum(labels == 1))
    if positives == 0 or positives == len(labels):
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_scores = scores[order]
    sorted_labels = labels[order]
    ends = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), len(labels) - 1]
    true_positives = np.cumsum(sorted_labels)[ends]
    precision = true_positives / (ends + 1)
    positive_increments = np.diff(np.r_[0, true_positives])
    return float(np.sum(precision * positive_increments) / positives)


def select_validation_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    known_species: Sequence[str] | None = None,
) -> dict:
    """Select F1-maximizing per-output thresholds using validation data only.

    If a class lacks positive or negative validation examples its threshold is
    0.5 and status records the fallback. Equal F1 chooses the larger threshold.
    The caller must save the returned thresholds and never refit them on test.
    """
    labels, probs = _inputs(y_true, y_prob)
    names = _names(labels.shape[1], known_species)
    thresholds, per_class = [], {}
    for i, name in enumerate(names):
        y, p = labels[:, i], probs[:, i]
        positive_support = int(y.sum())
        negative_support = int(len(y) - positive_support)
        if positive_support == 0 or negative_support == 0:
            threshold, best_f1, status = 0.5, None, "fallback_insufficient_support"
        else:
            order = np.argsort(-p, kind="stable")
            sorted_probs, sorted_labels = p[order], y[order]
            ends = np.r_[np.flatnonzero(np.diff(sorted_probs) != 0), len(y) - 1]
            tp = np.cumsum(sorted_labels)[ends]
            fp = (ends + 1) - tp
            fn = positive_support - tp
            f1 = 2 * tp / (2 * tp + fp + fn)
            # Scores are descending; argmax returns the highest tied threshold.
            best = int(np.argmax(f1))
            threshold, best_f1, status = float(sorted_probs[ends[best]]), float(f1[best]), "selected"
        thresholds.append(threshold)
        per_class[name] = {
            "threshold": threshold,
            "validation_f1": best_f1,
            "positive_support": positive_support,
            "negative_support": negative_support,
            "status": status,
        }
    return {"thresholds": thresholds, "per_class": per_class}


def calculate_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: Sequence[float],
    test_groups: Sequence[str],
    known_species: Sequence[str] | None = None,
) -> dict:
    """Evaluate predictions without modifying thresholds or test assignments.

    ``test_groups`` has E1, E2, E3, E4 or empty per row. Empty denotes no
    annotated species, not a guarantee that there is no audible bird. Unknown
    false-alarm and recall rates are reported by applicable group. E4 joint
    exact match requires every known label and the unknown label to be correct.
    """
    labels, probs = _inputs(y_true, y_prob)
    names = _names(labels.shape[1], known_species)
    cuts = np.asarray(thresholds, dtype=np.float64)
    if cuts.shape != (labels.shape[1],) or not np.all(np.isfinite(cuts)) or np.any((cuts < 0) | (cuts > 1)):
        raise ValueError("thresholds must have K+1 finite values in [0, 1]")
    groups = np.asarray(test_groups)
    if groups.ndim != 1 or len(groups) != len(labels) or not np.all(np.isin(groups, ["E1", "E2", "E3", "E4", "empty"])):
        raise ValueError("test_groups must contain E1, E2, E3, E4 or empty for each row")
    pred = probs >= cuts
    per_class = {}
    for i, name in enumerate(names):
        counts = _binary_counts(labels[:, i], pred[:, i])
        per_class[name] = {"average_precision": _average_precision(labels[:, i], probs[:, i]), **counts}
    known_aps = [per_class[name]["average_precision"] for name in names[:-1]]
    valid_known_aps = [ap for ap in known_aps if ap is not None]
    results = {
        "num_samples": int(len(labels)),
        "per_class": per_class,
        "known_macro_average_precision": float(np.mean(valid_known_aps)) if valid_known_aps else None,
        "known_macro_average_precision_classes": len(valid_known_aps),
        "unknown_average_precision": per_class["unknown"]["average_precision"],
        "groups": {},
    }
    for group in ("E1", "E2", "E3", "E4", "empty"):
        mask = groups == group
        unknown = _binary_counts(labels[mask, -1], pred[mask, -1])
        row = {"num_samples": int(mask.sum()), "unknown": unknown}
        if group in ("E1", "E2", "empty"):
            negatives = unknown["negative_support"]
            row["unknown_false_alarm_rate"] = unknown["false_positive"] / negatives if negatives else None
        else:
            positives = unknown["positive_support"]
            row["unknown_recall"] = unknown["true_positive"] / positives if positives else None
        if group == "E4":
            matches = np.all(pred[mask] == labels[mask].astype(bool), axis=1)
            row["joint_exact_correct"] = int(matches.sum())
            row["joint_exact_count"] = int(mask.sum())
            row["joint_exact_accuracy"] = float(matches.mean()) if len(matches) else None
        results["groups"][group] = row
    return results
