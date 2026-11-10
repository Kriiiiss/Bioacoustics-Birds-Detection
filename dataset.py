"""Load precomputed BirdSet features using an audited M1/M2 manifest.

The first features are stored inside an existing ZIP. Newly generated M2
features live as .npy files. ZIPs are opened lazily per worker process, so
they need not be unpacked or repeatedly re-indexed for every example.

Optional path mappings are applied at read time.  Original manifest bytes stay
unchanged, so normalization and checkpoint provenance hashes remain valid.
"""
from __future__ import annotations

import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


EXPECTED_SHAPE = (128, 500)


def _normalize_path_mappings(path_mappings):
    """Validate explicit prefix pairs, longest prefix first.

    Both paths must be absolute.  Prefixes match path components, not string
    fragments (``/old`` does not match ``/old-other``).  An exact file path
    may also be mapped.  The mapping is purely lexical; no file is modified.
    """
    if path_mappings is None:
        return ()
    result, seen = [], set()
    for pair in path_mappings:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            raise ValueError("each path mapping must have an OLD_PREFIX and a NEW_PREFIX")
        old, new = (os.path.normpath(os.path.expanduser(os.fspath(p))) for p in pair)
        if not os.path.isabs(old) or not os.path.isabs(new):
            raise ValueError("path mapping prefixes must be absolute paths")
        if old in seen:
            raise ValueError(f"duplicate old path prefix: {old}")
        seen.add(old)
        result.append((old, new))
    return tuple(sorted(result, key=lambda pair: len(pair[0]), reverse=True))


def resolve_path(original_path, path_mappings=None):
    """Resolve an original manifest/audit path via explicit prefix mappings.

    The longest matching component prefix wins; an unmatched path is returned
    unchanged (apart from expanding ``~``).  This permits native paths and
    portable mappings together while never modifying provenance metadata.
    """
    return _resolve_with_normalized_mappings(original_path, _normalize_path_mappings(path_mappings))


def _resolve_with_normalized_mappings(original_path, path_mappings):
    """Internal hot path when the caller has already validated mappings."""
    path = os.path.expanduser(os.fspath(original_path))
    if not path_mappings:
        return path
    normalized = os.path.normpath(path)
    for old, new in path_mappings:
        try:
            matches = os.path.commonpath((old, normalized)) == old
        except ValueError:
            matches = False
        if matches:
            suffix = os.path.relpath(normalized, old)
            return new if suffix == os.curdir else os.path.join(new, suffix)
    return path


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path: str | Path, expected_split: str | None = None) -> list[dict]:
    """Load and validate manifest structure, without opening any feature files."""
    path = Path(path).expanduser().resolve()
    rows, seen, label_size = [], set(), None
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if not isinstance(row, dict):
                    raise ValueError("row must be an object")
                clip_id, split, feature, labels = row["clip_id"], row["split"], row["feature"], row["labels"]
                if not isinstance(clip_id, str) or not clip_id.strip() or clip_id in seen:
                    raise ValueError("clip_id must be a unique nonempty string")
                if split not in {"train", "val", "test"}:
                    raise ValueError("split must be train, val, or test")
                if expected_split is not None and split != expected_split:
                    raise ValueError(f"expected split {expected_split!r}, found {split!r}")
                if not isinstance(feature, dict) or feature.get("kind") not in {"zip", "npy"}:
                    raise ValueError("feature.kind must be zip or npy")
                if not isinstance(feature.get("path"), str) or not feature["path"]:
                    raise ValueError("feature.path must be a nonempty string")
                if feature["kind"] == "zip" and (not isinstance(feature.get("member"), str) or not feature["member"]):
                    raise ValueError("zip feature requires a member name")
                if not isinstance(labels, list) or len(labels) < 2 or any(v not in (0, 1) for v in labels):
                    raise ValueError("labels must be a binary list of at least one known class and Unknown")
                if label_size is None:
                    label_size = len(labels)
                elif len(labels) != label_size:
                    raise ValueError("inconsistent number of labels")
                seen.add(clip_id)
                rows.append(row)
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
    if not rows:
        raise ValueError(f"manifest has no samples: {path}")
    return rows


class FeatureReader:
    """Read features without extracting shared ZIPs; handles are process-local."""

    def __init__(self, path_mappings=None) -> None:
        self._owner_pid = os.getpid()
        self._archives: dict[str, zipfile.ZipFile] = {}
        self.path_mappings = _normalize_path_mappings(path_mappings)

    def __getstate__(self):
        return {"_owner_pid": None, "_archives": {}, "path_mappings": self.path_mappings}

    def close(self):
        for archive in self._archives.values():
            archive.close()
        self._archives.clear()

    def _zip(self, path: str) -> zipfile.ZipFile:
        # When a DataLoader uses fork, inherited archive handles must not be
        # shared with the parent or another worker.
        if self._owner_pid != os.getpid():
            self.close()
            self._owner_pid = os.getpid()
        if path not in self._archives:
            self._archives[path] = zipfile.ZipFile(path, "r")
        return self._archives[path]

    def load(self, feature: dict, clip_id: str = "") -> np.ndarray:
        path = _resolve_with_normalized_mappings(feature["path"], self.path_mappings)
        try:
            if feature["kind"] == "zip":
                raw = self._zip(path).read(feature["member"])
                value = np.load(io.BytesIO(raw), allow_pickle=False)
            elif feature["kind"] == "npy":
                value = np.load(path, allow_pickle=False)
            else:
                raise ValueError(f"unknown feature kind: {feature['kind']!r}")
        except (FileNotFoundError, KeyError, OSError, zipfile.BadZipFile, ValueError) as exc:
            raise ValueError(f"Cannot load feature for {clip_id!r}: {feature!r}: {exc}") from exc
        if value.shape == (1, *EXPECTED_SHAPE):
            value = value[0]
        if value.shape != EXPECTED_SHAPE or not np.issubdtype(value.dtype, np.number) or np.iscomplexobj(value):
            raise ValueError(f"Invalid feature for {clip_id!r}: expected real {EXPECTED_SHAPE}, got {value.shape}/{value.dtype}")
        if not np.isfinite(value).all():
            raise ValueError(f"Nonfinite feature for {clip_id!r}")
        return np.asarray(value, dtype=np.float32)

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def load_norm_stats(path: str | Path, train_manifest: str | Path | None = None):
    """Load per-band statistics, optionally checking their train manifest hash."""
    with open(path, encoding="utf-8") as handle:
        stats = json.load(handle)
    if stats.get("source_split") != "train":
        raise ValueError("normalization stats must come from a training-only manifest")
    if train_manifest is not None and stats.get("manifest_sha256") != file_sha256(train_manifest):
        raise ValueError("normalization stats were computed for a different training manifest")
    mean = np.asarray(stats["band_mean"], dtype=np.float32)
    std = np.asarray(stats["band_std"], dtype=np.float32)
    if mean.shape != (128,) or std.shape != (128,) or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("normalization requires 128 finite means and positive standard deviations")
    return mean[:, None], std[:, None]


class BirdsetM2Dataset(Dataset):
    """Return ``(spectrogram, multilabel_target, metadata)`` for M2."""

    def __init__(self, manifest: str | Path, split: str, norm_stats: str | Path,
                 train_manifest: str | Path | None = None, *, path_mappings=None):
        self.rows = read_manifest(manifest, expected_split=split)
        if train_manifest is None:
            if split != "train":
                raise ValueError("val/test loading requires train_manifest to verify normalization provenance")
            train_manifest = manifest
        self.mean, self.std = load_norm_stats(norm_stats, train_manifest)
        self.reader = FeatureReader(path_mappings=path_mappings)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        feature = self.reader.load(row["feature"], row["clip_id"])
        feature = np.asarray((feature - self.mean) / self.std, dtype=np.float32)
        labels = torch.tensor(row["labels"], dtype=torch.float32)
        metadata = {name: str(row.get(name, "")) for name in ("clip_id", "case", "recording_id", "site_id", "session_id")}
        return torch.from_numpy(feature[None, :, :]), labels, metadata
