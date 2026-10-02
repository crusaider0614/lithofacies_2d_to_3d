"""Per-class precision / recall / F1 and confusion matrix for all three trace-spacing models.

For each model (TDT = 7.5, 12.5, 25.0 m, checkpoints lithofacies_prediction_<TDT>_pat_047 /
_049 / _044, i.e. the minimum-validation-loss epochs), every inline and every crossline of
the validation part of the 3D volume (CF.DATASET.VALID_IDX) is resampled from the 3D trace
spacing VDT (12.5 m) to the model's TDT, predicted with overlapping 256 x 256 tiles, and
compared with the facies labels. Metrics are pooled over all lines and reported for
Basement, Igneous, Shale and Sand; classes 0 (unlabeled) and 1 are excluded.

Reads:  config/config_lithofacies.yaml, the three checkpoints above under checkpoint/,
        data/<DATA_POOL>/<VOLUME_TAG>.npy, <FACIES_TAG>.npy,
        <VOLUME_TAG>_inst_phase.npy, <VOLUME_TAG>_inst_freq.npy.
Writes: nothing; the tables are printed to stdout.

Run from the repo root (running by file path fails because the repo root is then not on
sys.path):

    python -m test.test_lithofacies_quantity

There is no CLI. Edit the module-level settings (``target_dim``, ``device``, and the
(tdt, tag, epoch) list in the main loop).
"""
import os
import numpy as np
import torch
import yacs.config
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix

from module.seismic_data import SeismicVolume, Coordinate
from network.coordi_network import get_gen_model
from utils.project import get_project_root


def add_weighted_avg(average, weight, new_sample, new_weight):
    """Fold one tile prediction (logits ``new_sample``, window ``new_weight``) into the
    running weighted average ``average`` with accumulated weight ``weight``."""
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


# Tile size (= training crop size) and the 2D raised-cosine blending window used for
# overlap-add tiling: ~1 at the tile centre, ~0 at its edges.
target_dim = 256
weight = np.array([(1 + np.cos((i - (target_dim // 2 - 0.5)) / (target_dim // 2) * np.pi)) / 2 for i in range(target_dim)])
weight = weight[None] * weight[:, None]


def get_line_data(volume, ix, direction, nt, mode="bilinear"):
    """Extract inline/crossline ``ix`` of a (nz, nx, ny) array resampled to ``nt`` traces.

    ``direction == "inline"`` takes the line at fixed x running over all y, otherwise the
    line at fixed y over all x. ``mode`` is "bilinear" for continuous data and "nearest"
    for facies labels. Returns an (nz, nt) array.
    """
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
    """Predict facies for a full (nz, nt) section with overlapping target_dim tiles.

    ``vt`` is the (nz, nt) amplitude section and ``info`` the (2, nz, nt) instantaneous
    phase/frequency. Tiles overlap by about 50 %; their logits are blended with the
    raised-cosine ``weight`` window before the argmax. Returns (nz, nt) int32 class labels.
    """
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


config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
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
        # Resample the line from the 3D trace spacing (VDT) to this model's TDT while keeping
        # its physical length, so the model sees the trace spacing it was trained on.
        nt = int(round((ny * vdt) / tdt)) if direction == "inline" else int(round((nx * vdt) / tdt))

        for ix in range(n_lines):
            vt = get_line_data(volume, ix, direction, nt, mode="bilinear")
            ft = get_line_data(facies, ix, direction, nt, mode="nearest")
            ip = get_line_data(inst_phase, ix, direction, nt, mode="bilinear")
            if_ = get_line_data(inst_freq, ix, direction, nt, mode="bilinear")

            print(ix, n_lines, vt.shape)

            # Rescale the amplitude section to RMS 0.15 before inference.
            rms = (vt * vt).mean() ** 0.5
            if rms > 0:
                vt = vt / rms * 0.15

            info = np.concatenate((ip[None], if_[None]), axis=0)
            fo = predict_section(network, vt, info, device)

            # Score only labeled geological classes 2-5 (drop unlabeled 0 and above-first-label 1).
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
