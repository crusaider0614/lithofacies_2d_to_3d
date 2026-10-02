"""Qualitative check of a trained model on synthetic 2D lines, using tiled inference.

Same idea as test_lithofacies.py, but each 768 x 512 test crop is predicted the way a
full section is predicted at inference time: overlapping 256 x 256 tiles (the training
crop size) blended with a raised-cosine weight window. Not used for a paper table; it is
a visual check that the tiling produces seamless predictions.

What it does:
  1. Reads the loss history from the epoch-``epoch`` (default 50) checkpoint of ``tag``,
     prints/plots it, and reloads the minimum-validation-loss checkpoint.
  2. Draws random synthetic lines (trace spacing 25.0 m, hard-coded in the dataset call)
     from the validation part of the 3D volume via ``LithofaciesDataset``.
  3. Shows amplitude | ground truth | prediction | error mask for each line.

Reads:  config/config_lithofacies.yaml, checkpoint/<tag>_050 and the best-epoch
        checkpoint, data/<DATA_POOL>/ volume, facies, inst_phase and inst_freq .npy files.
Writes: nothing; figures are shown on screen.

Run from the repo root (running by file path fails because the repo root is then not on
sys.path):

    python -m test.test_lithofacies_quality

There is no CLI. Edit the module-level settings (``target_dim``, ``epoch``, ``device``,
``tag``); keep ``tag`` consistent with the ``tdt`` passed to the dataset.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yacs.config
from torch.utils.data import DataLoader

from module.dataset import LithofaciesDataset
from network.coordi_network import get_gen_model
from utils.data import show_2d_array
from utils.project import get_project_root


def add_weighted_avg(average, weight, new_sample, new_weight):
    """Fold one tile prediction into a running weighted average.

    ``average`` (C, H, W) is the current blended logits and ``weight`` (H, W) the weight
    accumulated so far; ``new_sample``/``new_weight`` are the tile's logits and window.
    Returns the updated average and accumulated weight.
    """
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


# Tile size for inference (= training crop size). Each tile's logits are weighted by a 2D
# raised-cosine (Hann-like) window that is ~1 at the tile centre and ~0 at its edges, so
# overlapping tiles blend smoothly and tile-border artefacts are suppressed.
target_dim = 256
weight = np.array([(1 + np.cos((i - (target_dim // 2 - 0.5)) / (target_dim // 2) * np.pi)) / 2 for i in range(target_dim)])
weight = weight[None] * weight[:, None]

epoch = 50

# Define
config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
device = torch.device("cuda:9")
tag = "lithofacies_prediction_25.0_pat"
print("Tag:", tag)

# The last-epoch checkpoint stores the per-epoch loss history; reload the (1-based) epoch
# with minimum validation loss.
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
train_losses = state["train_loss"]
valid_losses = state["valid_loss"]
print(np.argmin(valid_losses) + 1, min(valid_losses))
plt.plot(train_losses)
plt.plot(valid_losses)
plt.show()

state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(np.argmin(valid_losses) + 1).zfill(3)), map_location=lambda storage, loc: storage)

# Load
network = get_gen_model(CF, additional_channel=0).to(device)
network.load_state_dict(state["network"])
network = network.eval()

# Dataset
test_dataset = LithofaciesDataset(
    data_pool=CF.DATASET.DATA_POOL,
    volume_tag=CF.DATASET.VOLUME_TAG,
    facies_tag=CF.DATASET.FACIES_TAG,
    target_idx=CF.DATASET.VALID_IDX,
    vdt=CF.DATASET.VDT,
    tdt=25.0,
    crop_size=(768, 512),
    total_length=10,
    is_coordi=False,
    is_inst_phase=True,
    is_inst_freq=True,
    is_flip=False,
    is_scale=False,
    noise=0.0,
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

# Testing
with torch.no_grad():
    for vt, ft, info in test_loader:
        vt = vt.to(device)
        ft = ft.to(device)
        info = info.to(device)

        nz = vt.shape[2]
        nt = vt.shape[3]

        # Number of tiles along depth (z) and along the line (t): roughly 50 % overlap,
        # with tile starts spread evenly so the first/last tiles touch the section edges.
        nsz = int((nz - target_dim) / (target_dim // 2)) + 2 if nz > target_dim else 1
        nst = int((nt - target_dim) / (target_dim // 2)) + 2 if nt > target_dim else 1

        dp = np.zeros((nz, nt), dtype=np.float32)  # accumulated window weight
        fo = np.zeros((6, nz, nt), dtype=np.float32)  # blended logits of the 6 classes
        # Testing
        for isz in range(nsz):
            sz = int(round((nz - target_dim) * isz / (nsz - 1))) if nsz > 1 else 0
            ez = sz + target_dim
            for ist in range(nst):
                st = int(round((nt - target_dim) * ist / (nst - 1))) if nst > 1 else 0
                et = st + target_dim
                print(sz, ez, st, et)

                vt_crop = vt[:, :, sz:ez, st:et]
                info_crop = info[:, :, sz:ez, st:et]

                fo_crop = network(vt_crop, info_crop).cpu().numpy().squeeze()

                fo[:, sz:ez, st:et], dp[sz:ez, st:et] = add_weighted_avg(
                    fo[:, sz:ez, st:et],
                    dp[sz:ez, st:et],
                    fo_crop,
                    weight
                )

        fo = np.argmax(fo, axis=0).astype(np.int32)
        ft = ft[0].cpu().numpy().squeeze()
        vt = vt[0].cpu().numpy().squeeze()

        hb = -np.ones((nz, 4))
        imgs = np.concatenate((
            5 * vt + 2.5, hb, ft, hb, fo, hb, 5 * (ft != fo)
        ), axis=1)
        show_2d_array(imgs, scale=100, cmap="gray", vmax=5.0, vmin=0.0)
