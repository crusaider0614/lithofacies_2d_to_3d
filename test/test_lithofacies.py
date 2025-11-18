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


epoch = 50

# Define
config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
device = torch.device("cuda:9")
tag = CF.TAG
# tag = "lithofacies_prediction_12.5"
# tag = "lithofacies_prediction_25.0_nnorm"
print("Tag:", tag)

state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
train_losses = state["train_loss"]
valid_losses = state["valid_loss"]
print(np.argmin(valid_losses) + 1, min(valid_losses))
plt.plot(train_losses)
plt.plot(valid_losses)
plt.show()

exit()

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
    is_coordi=True,
    is_flip=CF.DATASET.FLIP,
    noise=CF.DATASET.NOISE,
)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True, drop_last=True)

# Testing
with torch.no_grad():
    for vt, ft, cd in test_loader:
        vt = vt.to(device)
        ft = ft.to(device)
        cd = cd.to(device)

        # ft[ft == 1] = 0
        # ft[ft == 2] = 0

        fo = network(vt, cd)
        fo = torch.argmax(fo, dim=1)

        iz = 200
        ix = 32
        iy = 32

        vt = vt[0].cpu().numpy().squeeze()
        ft = ft[0].cpu().numpy().squeeze()
        fo = fo[0].cpu().numpy().squeeze()
        hb = -np.ones((vt.shape[0], 4))

        # imgs = np.concatenate((
        #     hb, vt, hb,
        # ), axis=1)
        # show_2d_array(imgs, scale=100)

        imgs = np.concatenate((
            5 * vt + 2.5, hb, ft, hb, fo, hb, 5 * (ft != fo)
        ), axis=1)
        show_2d_array(imgs, scale=100, cmap="gray", vmax=5.0, vmin=0.0)

