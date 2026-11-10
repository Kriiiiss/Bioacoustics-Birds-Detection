# Open-set Bird Detection in Soundscapes

This CS230 project studies multi-label detection of known bird species and the presence of birds outside the known set in real soundscape recordings. A compact CNN receives a five-second log-mel spectrogram and outputs nine independent known-species logits plus an independent Unknown-presence logit. Unknown indicates that at least one species outside the known set is present; it does not identify or count those species.

This repository contains the selected CNN and the training and evaluation code for the UHH M1 and M2 experiments; other team models, including M4, are not included here. M1 and M2 use the **same model** and differ in their training examples:

- **M1:** clips annotated with one known species, one auxiliary unknown species, or no qualifying bird annotation.
- **M2:** the M1 pool plus naturally occurring multi-species clips, including clips with known and auxiliary unknown species.

Clips are real recordings, not synthetically mixed. Several species annotated within one five-second clip do not necessarily vocalize at exactly the same time. Both settings use the same validation and test clips. Because M2 has more examples and more optimizer updates per epoch, an M1–M2 difference cannot by itself isolate the effect of multi-species examples from the effect of sample count or training budget.

## Data and reproducibility

The experiments use the UHH soundscapes from [BirdSet](https://arxiv.org/abs/2403.10380). The dataset and its license are not included in this repository; obtain and use the source data under the terms supplied by its provider. The team's [bird soundscape preprocessing repository](https://github.com/wyy1048576/cs230-birdset) describes the UHH source recordings, species codes, label generation, and initial session split.

**A BirdSet download alone is not enough to run this code.** Training and evaluation require the separately prepared, audited UHH manifests, per-band training normalization, preparation report and source audit, and the corresponding precomputed spectrograms. These large inputs and derived artifacts are not included here. In particular, the teammate's original M1 CSV and normalization are not a replacement for the expanded, cleaned M1/M2 manifests and shared M1-only normalization used by these experiments. The preparation and auditing steps for those expanded artifacts are not currently packaged as a raw-audio-to-training command in this repository.

The selected experiment uses a UHH-specific prepared directory with this layout:

```text
prepared-uhh/
  M1/train.jsonl
  M2/train.jsonl
  shared/val.jsonl
  shared/test.jsonl
  shared/train_norm_m1.json
  report.json
  verification.json
```

The source audit referenced by `report.json` and all spectrogram files or ZIP archives referenced by the manifests must also be available locally. The stored features are unnormalized log-mel arrays of shape `128 x 500`; the loader applies the M1 training-only per-band normalization when reading individual `.npy` files or `.npy` members of a ZIP archive. The manifest labels must preserve the prepared order of the nine known species, followed by Unknown. That species order is read from the report and checked against the source audit. Each record identifies its clip, split, feature location, labels, and relevant recording/session and test-case metadata.

The prepared experiment contains 18,820 M1 training clips, 26,424 M2 training clips, 3,402 validation clips, and 3,188 test clips. The normalization parameters are computed from M1 training clips only and shared across M1 and M2. Recording sessions are disjoint across splits; locations are shared, so this is **not** an unseen-site evaluation. Empty means no bird annotation met the labeling threshold, not verified silence or verified absence of all birds. Annotation omissions, partial audio duplication, and re-encoding cannot be ruled out solely by the supplied file-hash audit.

The prepared manifests and normalization files are provenance-bound: normalization records the SHA-256 of the M1 training manifest, and the reports and checkpoints reference manifest and normalization hashes. **Do not edit a manifest merely to move its feature files.** Instead, preserve the prepared manifest and report bytes and provide explicit path-prefix mappings at runtime. A changed manifest requires regeneration and verification of dependent normalization and audit artifacts; an old checkpoint must not be treated as provenance-compatible merely because its paths were rewritten. Prefix mappings can point to an individual ZIP or to its parent directory; the member names inside the ZIP must match the manifest. These metadata checks do **not** verify that each mapped NPY or ZIP has the original contents: check relocated data files against trusted checksums separately.

## Environment

Python 3.10 or later, NumPy and PyTorch are required. Create an isolated environment and install the requirements:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

For GPU acceleration, install a PyTorch build appropriate for the platform. The reported full experiments used PyTorch with Apple Metal (`mps`) on a Mac mini. `cuda` can be selected where a compatible NVIDIA GPU and PyTorch build are available; `auto` lets the code choose an available device.

## Train and evaluate

Replace the paths below with your own local paths. `--path-map` maps a prefix recorded in the original prepared metadata to its new location **without rewriting the metadata**. Repeat the option when the feature ZIP, feature directories, and source audit do not share a common relocated root. If the paths in the metadata remain valid, omit `--path-map`.

You can check a manifest, its normalization provenance, and two feature files before training:

```bash
python inspect_dataset.py \
  --manifest /path/to/prepared-uhh/M1/train.jsonl \
  --split train \
  --norm /path/to/prepared-uhh/shared/train_norm_m1.json \
  --train-manifest /path/to/prepared-uhh/M1/train.jsonl \
  --path-map /original/project-root /local/project-root \
  --check-features 2
```

This checks the specified manifest and normalization, not the complete train/validation/test audit. Set `--check-features` to `-1` to read every referenced feature in that manifest.

```bash
python train.py \
  --dataset-dir /path/to/prepared-uhh \
  --variant M1 \
  --output /path/to/output/M1 \
  --device mps \
  --path-map /original/project-root /local/project-root

python train.py \
  --dataset-dir /path/to/prepared-uhh \
  --variant M2 \
  --output /path/to/output/M2 \
  --device mps \
  --path-map /original/project-root /local/project-root
```

The selected setup uses Adam, binary cross-entropy for known and Unknown outputs, a 16-clip batch, seed 0, up to 20 epochs, cosine learning-rate scheduling, and validation-loss early stopping. Per-output thresholds are selected on validation data and saved with the best checkpoint. Training does not load test spectrograms or select thresholds on test labels. Use a new or empty output directory for each run.

After the model and thresholds are fixed, run evaluation separately:

```bash
python evaluate.py \
  --dataset-dir /path/to/prepared-uhh \
  --checkpoint /path/to/output/M2/best.pt \
  --output /path/to/output/M2/test_metrics.json \
  --device mps \
  --path-map /original/project-root /local/project-root
```

Evaluation reports known-species macro average precision, Unknown average precision, and overall Unknown precision across the whole test set, as well as case-specific metrics for E1 (one known species), E2 (multiple known species only), E3 (unknown only), E4 (known and unknown), and Empty. The prediction output is intended for error analysis. Unknown recall alone does not establish correct known-species predictions; E4 joint correctness requires all known labels and Unknown to be correct together.

The current test results were inspected during development and should be described as exploratory, not as a previously unseen blind test. Existing reported results use one seed; validation and test unknown-species support is limited and uneven. Do not use the test set to choose models, epochs, thresholds, or hyperparameters. A fresh held-out dataset would be needed for a genuinely independent final evaluation after development.

## Acknowledgments

The project's preprocessing builds on the team's [CS230 bird soundscape repository](https://github.com/wyy1048576/cs230-birdset). The project code organization was informed by the [CS230 code examples](https://github.com/cs230-stanford/cs230-code-examples); the original CS230 examples' MIT notice is retained in `third_party/CS230_LICENSE.txt`. Neither the source dataset nor third-party preprocessing data is included in this repository.
