"""Four-clip sequences for M4 on a prepared UHH manifest.

Every manifest row is one target clip. Its sequence is [t-3, t-2, t-1, t]: the three clips that precede it in
the SAME recording (clip index minus 1, 2, 3) followed by the target. A context clip is looked up in the same
manifest, so the M2 training manifest (all kept training clips) and the validation / test manifests (all kept
clips) contain every context clip that exists. A slot whose clip is not in the manifest (before the start of
the recording, or an excluded clip) is zero-padded and masked.

Features are read with dataset.FeatureReader and normalised with the shared training-only M1 per-band
statistics, exactly as in dataset.BirdsetM2Dataset. Padded slots are zeros after normalisation.

The clip index is read from ``clip_index`` / ``clip_idx`` if a row has it, otherwise from the integer after the
last underscore of ``clip_id`` (e.g. ``UHH_001_S01_20161121_150000_008`` -> 8). The recording is
``recording_id``.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, get_worker_info

from dataset import FeatureReader, load_norm_stats, read_manifest

SEQ_LEN = 4          # three earlier clips + the target clip
N_MELS, N_FRAMES = 128, 500
_TRAILING_INT = re.compile(r"_(\d+)$")


def clip_index(row: dict) -> int:
    for key in ("clip_index", "clip_idx"):
        if key in row:
            return int(row[key])
    match = _TRAILING_INT.search(row["clip_id"])
    if not match:
        raise ValueError(f"cannot read the clip index of {row['clip_id']!r}: no clip_index field and no trailing _<int>")
    return int(match.group(1))


def context_ids(rows: list[dict]) -> list[list[str | None]]:
    """For each row the clip_ids of [t-3, t-2, t-1] (None where there is no such clip in the manifest)."""
    by_key: dict[tuple[str, int], str] = {}
    for row in rows:
        key = (str(row["recording_id"]), clip_index(row))
        if key in by_key:
            raise ValueError(f"two clips share recording {key[0]!r} and index {key[1]}: {by_key[key]!r}, {row['clip_id']!r}")
        by_key[key] = row["clip_id"]
    return [[by_key.get((str(row["recording_id"]), clip_index(row) - back)) for back in (3, 2, 1)] for row in rows]


class BirdsetM4Dataset(Dataset):
    """Return ``(x, mask, labels, metadata)``.

    x        (4, 128, 500) float32, normalised; padded slots are zeros
    mask     (4,) bool, True for a real clip (the last slot is always real)
    labels   (K + 1,) float32 target of the LAST clip: known species followed by Unknown
    metadata dict of strings (clip_id, case, recording_id, site_id, session_id)

    use_context=False masks the three earlier clips (the single-clip control with the identical pipeline).
    augment=True draws one set of random parameters per sequence and applies it to every real clip: a log-power
    gain offset, a circular time shift, two time masks and two frequency masks (as in train.augment_batch).
    """

    def __init__(self, manifest: str | Path, split: str, norm_stats: str | Path, train_manifest: str | Path,
                 *, path_mappings=None, use_context: bool = True, augment: bool = False, seed: int = 0):
        self.rows = read_manifest(manifest, expected_split=split)
        self.mean, self.std = load_norm_stats(norm_stats, train_manifest)
        self.reader = FeatureReader(path_mappings=path_mappings)
        self.use_context, self.augment, self.seed = use_context, augment, seed
        self.by_id = {row["clip_id"]: row for row in self.rows}
        self.context = context_ids(self.rows)
        self._rng = None

    def __len__(self):
        return len(self.rows)

    def _generator(self):
        if self._rng is None:       # one generator per process / worker
            info = get_worker_info()
            self._rng = np.random.default_rng((torch.initial_seed() + self.seed + (info.id if info else 0)) % (2 ** 32))
        return self._rng

    def sequence_ids(self, index: int) -> list[str | None]:
        ids = [*self.context[index], self.rows[index]["clip_id"]]
        if not self.use_context:
            ids[:SEQ_LEN - 1] = [None] * (SEQ_LEN - 1)
        return ids

    def __getitem__(self, index):
        row = self.rows[index]
        ids = self.sequence_ids(index)
        mask = np.array([clip is not None for clip in ids])
        if self.augment:
            rng = self._generator()
            gain = float(rng.uniform(-1.0, 1.0))
            shift = int(rng.integers(0, N_FRAMES))
            holes = []
            for _ in range(2):
                width = int(rng.integers(0, 41))
                time_start = int(rng.integers(0, N_FRAMES - width + 1))
                height = int(rng.integers(0, 13))
                freq_start = int(rng.integers(0, N_MELS - height + 1))
                holes.append((time_start, width, freq_start, height))
        x = np.zeros((SEQ_LEN, N_MELS, N_FRAMES), dtype=np.float32)
        for slot, clip in enumerate(ids):
            if clip is None:
                continue                                        # padded slot stays zero
            source = self.by_id[clip]
            feature = self.reader.load(source["feature"], clip)
            if self.augment:
                feature = np.roll(feature + gain, shift, axis=1)
            feature = (feature - self.mean) / self.std
            if self.augment:
                for time_start, width, freq_start, height in holes:
                    feature[:, time_start:time_start + width] = 0.0
                    feature[freq_start:freq_start + height, :] = 0.0
            x[slot] = feature
        labels = torch.tensor(row["labels"], dtype=torch.float32)
        metadata = {name: str(row.get(name, "")) for name in ("clip_id", "case", "recording_id", "site_id", "session_id")}
        return torch.from_numpy(x), torch.from_numpy(mask), labels, metadata
