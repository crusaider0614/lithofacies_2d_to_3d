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
process/
  compute_inst_attribute.py     Precompute the instantaneous phase / frequency volumes
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

Checkpoint loading and imports were verified with Python 3.11 and torch 2.11.0+cu126 (Windows);
the released models were trained on a Linux server with 4 GPUs. Training requires CUDA GPUs and
always initializes a `torch.distributed` NCCL process group, even on a single GPU, so it does not
run on Windows or with `DEVICE: "cpu"`.

## Data and trained models

The 3D seismic volume and the 3D lithofacies model used for training cannot be redistributed
because of the data provider's restrictions. The following are available at:

> **TODO: add the share link.**

- the trained models, and the epoch-50 checkpoints that hold the loss histories;
- one real 2D seismic line for each trace spacing (7.5, 12.5 and 25.0 m), as used by
  `test/test_actual_line.py`;
- 2D lines extracted from the validation part of the 3D volume, with the lithofacies labels and
  the info input along them, as examples of the synthetic lines (see below).

Place them as follows:

```
checkpoint/
  lithofacies_prediction_7.5_pat_047
  lithofacies_prediction_12.5_pat_049
  lithofacies_prediction_25.0_pat_044
  lithofacies_prediction_{7.5,12.5,25.0}_pat_050   epoch-50 checkpoints (loss histories)
data/ssealine_matched/
  <n>.npy, <n>.nphead           real 2D lines (one per trace spacing): amplitude (nz, nt) and a
                                text header with synthetic trace coordinates (see below)
data/TODO/                      2D lines extracted from the 3D volume (TODO: file names), each with
                                amplitude (768, 512), lithofacies labels (768, 512; classes 0-5)
                                and the info input (2, 768, 512; inst. phase, inst. frequency)
```

The lines extracted from the 3D volume were drawn with `LithofaciesDataset` (`module/dataset.py`)
from the validation part of the volume (axis-1 indices 2000-2667, `DATASET.VALID_IDX`, not used
for training), with `vdt=12.5`, `tdt` set to the model's trace spacing, `crop_size=(768, 512)`,
the instantaneous phase and frequency as info channels, and no augmentation (no flip, amplitude
scaling or noise). They are therefore larger than the 256 x 256 training crops and are predicted
with overlapping 256 x 256 tiles. Everything else is as in training: the labels are sampled with
nearest-neighbour interpolation, the info channels come from the precomputed 3D attribute
volumes, the trace spacing of each line varies randomly by up to +/-10 % around `tdt`, and the
depth window is drawn so that it contains at least one non-zero label. The amplitude is taken
directly from the 3D volume, which is RMS-normalized to 0.15 as a whole, so it can be fed to the
network as is, as in training. (The evaluation scripts in `test/` additionally rescale each line
or crop to an RMS of 0.15 before inference.)

The survey coordinates of the real lines cannot be published. In the released `.nphead` headers
the trace coordinates are replaced by synthetic ones: trace `i` is placed at `(i * d, 0)`, where
`d` is the line's mean trace spacing in metres. `test/test_actual_line.py` uses the coordinates
only to compute this spacing and pick the model with the nearest TDT, so the predictions are the
same as with the survey coordinates. The coordinates do not give the line's location, and the
SEG-Y export in that script (commented out) would write these synthetic coordinates. Header
format (tab-separated; `amp` is the value the amplitudes are divided by on loading, and `dz` the
sample interval in seconds):

```
shape	<nz>	<nt>
amp	<amp>
dz	<dz>

trace coordi
0	0.0	0.0
1	<d>	0.0
2	<2d>	0.0
...
```

| Trace spacing (TDT) | Checkpoint | Epoch |
|---|---|---|
| 7.5 m | `lithofacies_prediction_7.5_pat_047` | 47 |
| 12.5 m | `lithofacies_prediction_12.5_pat_049` | 49 |
| 25.0 m | `lithofacies_prediction_25.0_pat_044` | 44 |

Each epoch is the one with the lowest validation loss over 50 epochs of training. The epoch-50
checkpoints are only needed by `test/test_lithofacies.py` and `test/test_lithofacies_quality.py`,
which read the loss history from them.

Facies classes: 0 unlabeled, 1 unclassified region above the interpreted interval, 2 basement,
3 igneous, 4 shale, 5 sand. Class 0 is ignored by the loss. Class 1 has zero cross-entropy weight
and is left out of the Dice average, but its pixels still enter the training label-smoothing term
and act as negatives in the Dice terms of classes 2-5. Classes 0 and 1 are excluded from all
reported metrics.

Real 2D lines are rescaled to an RMS amplitude of 0.15 before inference, matching the training
volume.

Training, `process/compute_inst_attribute.py` and every script that draws synthetic lines from
the 3D volume (`test/test_lithofacies*.py`, `test/evaluate_*.py`, `test/create_*.py`) need the
3D data, which are not distributed. They expect this layout (paths set in
`config/config_lithofacies.yaml`):

```
data/southsea/
  3dn_tbcut.npy                 3D seismic volume (nz, nx, ny), RMS-normalized to 0.15
  3dn_facies_unc.npy            3D facies labels, same shape
  3dn_tbcut_inst_phase.npy      created by process/compute_inst_attribute.py
  3dn_tbcut_inst_freq.npy       created by process/compute_inst_attribute.py
```

## Usage

Run every script from the repository root as a module. Running a script by path
(`python test/evaluate_accuracy.py`) fails because the repository root is then not on `sys.path`.
Scripts have no command-line arguments: edit the constants at the top of each file (device, tag,
epoch, ...) instead.

**With the released data**: predict the real 2D lines. List the released lines in `idx_range` in
`test/test_actual_line.py` (the key is the file number minus 1, e.g. `0` for `1.npy`), then
```bash
python -m test.test_actual_line
```
The model is chosen from each line's trace spacing, and the prediction is shown above the
amplitude section.

**With a labeled 3D volume** (not distributed):

1. Precompute the instantaneous attributes (once):
   ```bash
   python -m process.compute_inst_attribute
   ```
2. Train. Set `DATASET.TDT` and `TAG` in `config/config_lithofacies.yaml` for each trace spacing,
   and `GPUS` for your machine (the released models were trained on 4 GPUs; the batch size in the
   config is the total over all GPUs):
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
   `test_lithofacies_quantity` and `evaluate_accuracy` print their results;
   `evaluate_depth_accuracy` writes `test/validation_figures/tdt_25.0/depth_accuracy.png`.
4. Example figures (written to `test/validation_figures/tdt_25.0/`):
   ```bash
   python -m test.create_validation_figures
   ```
   ```bash
   python -m test.create_annotated_figure
   ```

## Citation

> **TODO: add the paper citation.**

## License

BSD 3-Clause; see [LICENSE](LICENSE).
