import os
import numpy as np
import torch
import yacs.config
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from module.seismic_data import SeismicVolume, Coordinate
from network.coordi_network_2 import get_gen_model
from utils.project import get_project_root


def add_weighted_avg(average, weight, new_sample, new_weight):
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


target_dim = 256
weight = np.array([(1 + np.cos((i - (target_dim // 2 - 0.5)) / (target_dim // 2) * np.pi)) / 2 for i in range(target_dim)])
weight = weight[None] * weight[:, None]


def get_line_data(volume, ix, direction, nt, mode="bilinear"):
    nz, nx, ny = volume.shape
    if direction == "inline":
        scoordi = Coordinate(ix, 0)
        ecoordi = Coordinate(ix, ny - 1)
    else:
        scoordi = Coordinate(0, ix)
        ecoordi = Coordinate(nx - 1, ix)

    sv = SeismicVolume()
    sv.clean_data(shape=(nz, nx, ny))
    sv.set_coordi(
        Coordinate(0, 0),
        Coordinate(0, ny - 1),
        Coordinate(nx - 1, 0),
        Coordinate(nx - 1, ny - 1)
    )
    sv.data = volume
    line = sv.get_line_coordi(scoordi, ecoordi, nt, mode=mode)
    return line.data


def predict_section(network, vt, info, device):
    nz, nt = vt.shape
    nsz = int((nz - target_dim) / (target_dim // 2)) + 2 if nz > target_dim else 1
    nst = int((nt - target_dim) / (target_dim // 2)) + 2 if nt > target_dim else 1

    dp = np.zeros((nz, nt), dtype=np.float32)
    fo = np.zeros((6, nz, nt), dtype=np.float32)

    with torch.no_grad():
        for isz in range(nsz):
            sz = int(round((nz - target_dim) * isz / (nsz - 1))) if nsz > 1 else 0
            ez = sz + target_dim
            for ist in range(nst):
                st = int(round((nt - target_dim) * ist / (nst - 1))) if nst > 1 else 0
                et = st + target_dim

                vt_crop = torch.tensor(vt[None, None, sz:ez, st:et], dtype=torch.float32, device=device)
                info_crop = torch.tensor(info[None, :, sz:ez, st:et], dtype=torch.float32, device=device)

                fo_crop = network(vt_crop, info_crop).cpu().numpy().squeeze()
                fo[:, sz:ez, st:et], dp[sz:ez, st:et] = add_weighted_avg(
                    fo[:, sz:ez, st:et],
                    dp[sz:ez, st:et],
                    fo_crop,
                    weight
                )

    return np.argmax(fo, axis=0).astype(np.int32)


config_file = os.path.join(get_project_root(), "config", "config_lithofacies_2.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
device = torch.device("cuda:9")

data_pool = CF.DATASET.DATA_POOL
volume_tag = CF.DATASET.VOLUME_TAG
facies_tag = CF.DATASET.FACIES_TAG
valid_idx = CF.DATASET.VALID_IDX
vdt = CF.DATASET.VDT

volume = np.load(os.path.join(get_project_root(), "data", data_pool, volume_tag + ".npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]
facies = np.load(os.path.join(get_project_root(), "data", data_pool, facies_tag + ".npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]
inst_phase = np.load(os.path.join(get_project_root(), "data", data_pool, volume_tag + "_inst_phase.npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]
inst_freq = np.load(os.path.join(get_project_root(), "data", data_pool, volume_tag + "_inst_freq.npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]

nz, nx, ny = volume.shape
print(volume.shape)

for tdt, tag, epoch in [
    (7.5,  "lithofacies_prediction_7.5_pat",  47),
    (12.5, "lithofacies_prediction_12.5_pat", 49),
    (25.0, "lithofacies_prediction_25.0_pat", 44),
]:
    print(f"=== tdt = {tdt} m ===")

    network = get_gen_model(CF, additional_channel=0).to(device)
    state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
    network.load_state_dict(state["network"])
    network = network.eval()

    all_pred = []
    all_true = []

    for direction in ["inline", "crossline"]:
        n_lines = nx if direction == "inline" else ny
        nt = int(round((ny * vdt) / tdt)) if direction == "inline" else int(round((nx * vdt) / tdt))

        for ix in range(n_lines):
            vt = get_line_data(volume, ix, direction, nt, mode="bilinear")
            ft = get_line_data(facies, ix, direction, nt, mode="nearest")
            ip = get_line_data(inst_phase, ix, direction, nt, mode="bilinear")
            if_ = get_line_data(inst_freq, ix, direction, nt, mode="bilinear")

            print(ix, n_lines, vt.shape)

            rms = (vt * vt).mean() ** 0.5
            if rms > 0:
                vt = vt / rms * 0.15

            info = np.concatenate((ip[None], if_[None]), axis=0)
            fo = predict_section(network, vt, info, device)

            mask = (ft != 0) & (ft != 1)
            all_pred.append(fo[mask])
            all_true.append(ft[mask])

            if ix % 10 == 0:
                print(f"{direction} {ix}/{n_lines}")

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    classes = [2, 3, 4, 5]
    class_names = ["Basement", "Igneous", "Shale", "Sand"]

    precision, recall, f1, _ = precision_recall_fscore_support(all_true, all_pred, labels=classes, zero_division=0)
    cm = confusion_matrix(all_true, all_pred, labels=classes)

    print(f"\n{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    print("-" * 45)
    for name, p, r, f in zip(class_names, precision, recall, f1):
        print(f"{name:<12} {p:>10.3f} {r:>10.3f} {f:>10.3f}")

    print("\nConfusion Matrix:")
    print(f"{'':>10}", end="")
    for name in class_names:
        print(f"{name:>10}", end="")
    print()
    for name, row in zip(class_names, cm):
        print(f"{name:>10}", end="")
        for val in row:
            print(f"{val:>10}", end="")
        print()
