# Lithofacies Prediction 2D to 3D

Source code for the second stage of the paper "Integrated Rock Physics, Seismic, and Machine
Learning Methodology for Regional 3D Lithofacies Modeling - A Case Study from the Jeju Basin,
Offshore Korea."

A segmentation network (InfoUNet) is trained on synthetic 2D lines cut from a labeled 3D seismic
volume, then applied to real 2D seismic lines to predict lithofacies (basement, igneous, shale,
sand). The network takes the seismic amplitude plus two auxiliary channels, instantaneous phase
and instantaneous frequency. Separate models are trained for 2D trace spacings of 7.5, 12.5 and
25.0 m; for a real line, the model with the nearest trace spacing is used.

## Repository structure

```
config/
  config_lithofacies.yaml       Training / model configuration
module/
  dataset.py                    LithofaciesDataset: random synthetic 2D lines from the 3D volume
  seismic_data.py               SeismicLine / SeismicVolume containers, SEG-Y -> .npy conversion
network/
  coordi_network.py             InfoUNet and DiceLoss
  info_module.py                Concatenation of the auxiliary info channels
utils/                          Paths, DDP helpers, weight init, I/O and plotting helpers
compute_inst_attribute.py       Precompute the instantaneous phase / frequency volumes
train/
  train_lithofacies.py          Training (DistributedDataParallel)
test/
  test_lithofacies_quantity.py  Precision / recall / F1 and confusion matrix for all three models
  evaluate_accuracy.py          Accuracy, per-class metrics and confusion matrix (25.0 m model)
  evaluate_depth_accuracy.py    Accuracy as a function of two-way time (25.0 m model)
  create_validation_figures.py  Seismic / label / prediction figures for random synthetic lines
  create_annotated_figure.py    Annotated example figure
  test_actual_line.py           Inference on real 2D lines
  test_lithofacies.py, test_lithofacies_quality.py   Visual checks on synthetic lines
```

`data/` and `checkpoint/` are not tracked; see **Data and trained models** below.

## Setup

Install PyTorch for your CUDA version (https://pytorch.org/get-started/locally/), then

```bash
pip install -r requirements.txt
```

Tested with Python 3.11 and torch 2.11 (CUDA 12.6). Training uses `torch.distributed` (NCCL) even on
a single GPU.

## Data and trained models

The full field datasets used in the study are not publicly available due to institutional data-sharing
restrictions. A curated subset for testing and the trained models are available at:

> **TODO: add the share link.**

Place them as follows:

```
data/southsea/
  3dn_tbcut.npy                 3D seismic volume (nz, nx, ny), RMS-normalized
  3dn_facies_unc.npy            3D facies labels, same shape
  3dn_tbcut_inst_phase.npy      created by compute_inst_attribute.py
  3dn_tbcut_inst_freq.npy       created by compute_inst_attribute.py
data/ssealine_matched/
  <n>.npy, <n>.nphead           real 2D lines for test/test_actual_line.py
checkpoint/
  lithofacies_prediction_7.5_pat_047
  lithofacies_prediction_12.5_pat_049
  lithofacies_prediction_25.0_pat_044
```

| Trace spacing (TDT) | Checkpoint | Epoch |
|---|---|---|
| 7.5 m | `lithofacies_prediction_7.5_pat_047` | 47 |
| 12.5 m | `lithofacies_prediction_12.5_pat_049` | 49 |
| 25.0 m | `lithofacies_prediction_25.0_pat_044` | 44 |

Each epoch is the one with the lowest validation loss over 50 epochs of training.

Facies classes: 0 unlabeled (ignored), 2 basement, 3 igneous, 4 shale, 5 sand. Class 1 has weight 0
in the loss and is excluded from the reported metrics.

## Usage

Run every script from the repository root as a module. Running a script by path
(`python test/evaluate_accuracy.py`) fails because the repository root is then not on `sys.path`.
Scripts have no command-line arguments: edit the constants at the top of each file (device, tag,
epoch, ...) instead.

1. Precompute the instantaneous attributes (once):
   ```bash
   python compute_inst_attribute.py
   ```
2. Train. Set `DATASET.TDT` and `TAG` in `config/config_lithofacies.yaml` for each trace spacing,
   and `GPUS` for your machine:
   ```bash
   python -m train.train_lithofacies
   ```
   Checkpoints are written to `checkpoint/<TAG>_<epoch>`.
3. Evaluate:
   ```bash
   python -m test.test_lithofacies_quantity
   ```
   ```bash
   python -m test.evaluate_accuracy
   ```
   ```bash
   python -m test.evaluate_depth_accuracy
   ```
   Figures are written under `test/validation_figures/`.

## Citation

> **TODO: add the paper citation.**
