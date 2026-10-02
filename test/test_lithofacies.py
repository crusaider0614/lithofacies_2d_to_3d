"""Quick visual sanity check of a trained lithofacies model (single forward pass).

Not used for any paper table or figure; it is an interactive debugging aid.

What it does:
  1. Loads the epoch-``epoch`` (default 50, the last epoch) checkpoint of ``tag`` only to
     read the stored train/valid loss history, prints and plots it, and then reloads the
     checkpoint of the epoch with minimum validation loss.
  2. Draws random synthetic 2D lines from the validation part of the 3D volume via
     ``LithofaciesDataset`` (crop 768 x 512 samples, trace spacing CF.DATASET.TDT,
     instantaneous phase/frequency as the 2-channel info input, no augmentation).
  3. Runs the whole crop through InfoUNet in one pass (no tiling) and shows, side by side,
     amplitude | ground-truth facies | predicted facies | error mask (gt != prediction).

Reads:  config/config_lithofacies.yaml,
        checkpoint/<tag>_050 and checkpoint/<tag>_<best epoch>,
        data/<DATA_POOL>/<VOLUME_TAG>.npy, <FACIES_TAG>.npy,
        <VOLUME_TAG>_inst_phase.npy, <VOLUME_TAG>_inst_freq.npy (VALID_IDX slice).
Writes: nothing; figures are shown on screen with matplotlib.

Run from the repo root (test/ is a package; running by file path fails because the repo
root is then not on sys.path):

    python -m test.test_lithofacies

There is no CLI. Edit the module-level settings below (``epoch``, ``device``, ``tag``).
Make sure ``tag`` matches CF.DATASET.TDT, which sets the trace spacing of the test lines.
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


# Epoch whose checkpoint holds the full loss history (the last training epoch).
epoch = 50

# Define
config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
device = torch.device("cuda:5")
tag = CF.TAG
# tag = "lithofacies_prediction_7.5_pat"
# tag = "lithofacies_prediction_12.5_pat"
tag = "lithofacies_prediction_25.0_pat"
# tag = "lithofacies_prediction_25.0_nnorm"
print("Tag:", tag)

# The last-epoch checkpoint stores the per-epoch loss history; argmin(valid_loss) + 1 is the
# (1-based) epoch with minimum validation loss, which is the checkpoint actually evaluated.
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
    tdt=CF.DATASET.TDT,
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

        # ft[ft == 1] = 0
        # ft[ft == 2] = 0

        fo = network(vt, info)
        fo = torch.argmax(fo, dim=1)

        iz = 200
        ix = 32
        iy = 32

        vt = vt[0].cpu().numpy().squeeze()
        ft = ft[0].cpu().numpy().squeeze()
        fo = fo[0].cpu().numpy().squeeze()
        # fo[ft == 0] = 0
        hb = -np.ones((vt.shape[0], 4))

        # imgs = np.concatenate((
        #     hb, vt, hb,
        # ), axis=1)
        # show_2d_array(imgs, scale=100)

        # Panels on a shared 0..5 gray scale: amplitude (scaled/shifted for display),
        # ground-truth classes, predicted classes, and the misclassification mask.
        imgs = np.concatenate((
            5 * vt + 2.5, hb, ft, hb, fo, hb, 5 * (ft != fo)
        ), axis=1)
        show_2d_array(imgs, scale=100, cmap="gray", vmax=5.0, vmin=0.0)

