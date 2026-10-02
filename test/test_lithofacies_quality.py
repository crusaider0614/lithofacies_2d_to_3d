import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import yacs.config
from torch.utils.data import DataLoader

from module.dataset import LithofaciesDataset
from network.coordi_network_2 import get_gen_model
from utils.data import show_2d_array
from utils.project import get_project_root


def add_weighted_avg(average, weight, new_sample, new_weight):
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


target_dim = 256
weight = np.array([(1 + np.cos((i - (target_dim // 2 - 0.5)) / (target_dim // 2) * np.pi)) / 2 for i in range(target_dim)])
weight = weight[None] * weight[:, None]

epoch = 50

# Define
config_file = os.path.join(get_project_root(), "config", "config_lithofacies_2.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
device = torch.device("cuda:9")
tag = "lithofacies_prediction_25.0_pat"
print("Tag:", tag)

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

        nsz = int((nz - target_dim) / (target_dim // 2)) + 2 if nz > target_dim else 1
        nst = int((nt - target_dim) / (target_dim // 2)) + 2 if nt > target_dim else 1

        dp = np.zeros((nz, nt), dtype=np.float32)
        fo = np.zeros((6, nz, nt), dtype=np.float32)
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
