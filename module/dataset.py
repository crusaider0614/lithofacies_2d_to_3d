import os
import random

import numpy as np
from torch.utils.data import Dataset

from module.seismic_data import SeismicVolume, Coordinate, distance
from utils.project import get_project_root


data_shapes = {
    "3dn_process": (1001, 2667, 1200),
    "3dn_facies_newclass": (1001, 2667, 1200)
}


def get_random_boundary_idx(nx, ny, exclude_edge=None):
    edges = {
        1: [(i, 0) for i in range(nx)],
        2: [(nx - 1, i) for i in range(ny)],
        3: [(i, ny - 1) for i in range(nx)],
        4: [(0, i) for i in range(ny)],
    }

    if exclude_edge is None:
        available_edges = edges
    else:
        available_edges = {k: v for k, v in edges.items() if k != exclude_edge}

    edge_lengths = [len(available_edges[k]) for k in available_edges.keys()]
    edge_keys = list(available_edges.keys())

    selected_edge = random.choices(edge_keys, weights=edge_lengths, k=1)[0]
    bx, by = random.choice(available_edges[selected_edge])
    return Coordinate(bx, by), selected_edge


def extract_random_line(volume_data, nt, vdt, tdt, scoordi=None, ecoordi=None, mode="bilinear"):
    target_distance = (nt - 1) * tdt
    crop_distance = (0.9 + 0.2 * random.random()) * target_distance
    nz, nx, ny = volume_data.shape

    is_scoordi = scoordi is not None
    is_ecoordi = ecoordi is not None
    if (not is_scoordi) or (not is_ecoordi):
        while True:
            if not is_scoordi:
                scoordi, s_edge = get_random_boundary_idx(nx, ny)
            if not is_ecoordi:
                ecoordi, e_edge = get_random_boundary_idx(nx, ny, exclude_edge=s_edge)
            if vdt * distance(scoordi, ecoordi) >= crop_distance:
                break
    line_length = vdt * distance(scoordi, ecoordi)

    start_distance = (line_length - crop_distance) * random.random()
    end_distance = start_distance + crop_distance

    new_scoordi = scoordi if is_scoordi else scoordi + (ecoordi - scoordi) * start_distance / line_length
    new_ecoordi = ecoordi if is_ecoordi else scoordi + (ecoordi - scoordi) * end_distance / line_length

    line_data = volume_data.get_line_coordi(new_scoordi, new_ecoordi, nt, mode=mode).data
    return line_data, new_scoordi, new_ecoordi


class LithofaciesDataset(Dataset):
    def __init__(
            self,
            data_pool,
            volume_tag,
            facies_tag=None,
            target_idx=None,
            vdt=1.0,
            tdt=1.0,
            crop_size=None,
            total_length=1,
            is_coordi=False,
            is_flip=False,
            noise=0.0,
    ):
        self.volume_data = np.load(
            os.path.join(get_project_root(), "data", data_pool, volume_tag + ".npy"),
            mmap_mode="r",
        )[:, target_idx[0]: target_idx[1]]
        if facies_tag is not None:
            self.facies_data = np.load(
                os.path.join(get_project_root(), "data", data_pool, facies_tag + ".npy"),
                mmap_mode="r",
            )[:, target_idx[0]: target_idx[1]]
        else:
            self.facies_data = None
        nz, nx, ny = self.volume_data.shape

        self.volume_class = SeismicVolume()
        self.volume_class.clean_data(shape=(nz, nx, ny))
        self.volume_class.set_coordi(
            Coordinate(0, 0),
            Coordinate(0, ny - 1),
            Coordinate(nx - 1, 0),
            Coordinate(nx - 1, ny - 1)
        )
        self.volume_class.data = self.volume_data

        if self.facies_data is not None:
            self.facies_class = SeismicVolume()
            self.facies_class.clean_data(shape=(nz, nx, ny))
            self.facies_class.set_coordi(
                Coordinate(0, 0),
                Coordinate(0, ny - 1),
                Coordinate(nx - 1, 0),
                Coordinate(nx - 1, ny - 1)
            )
            self.facies_class.data = self.facies_data

        self.vdt = vdt
        self.tdt = tdt

        self.total_length = total_length
        self.is_coordi = is_coordi
        self.is_flip = is_flip
        self.noise = noise

        self.crop_size = (crop_size[0], crop_size[1])
        self.is_crop = True

    def __len__(self):
        return self.total_length

    def __getitem__(self, idx):
        if self.facies_data is not None:
            is_withfacies = True
        else:
            is_withfacies = False
        nz, nx, ny = self.volume_data.shape

        vc_datum, scoordi, ecoordi = extract_random_line(
            self.volume_class,
            self.crop_size[1],
            self.vdt,
            self.tdt,
            mode="bilinear",
        )
        if is_withfacies:
            fc_datum, _, _ = extract_random_line(
                self.facies_class,
                self.crop_size[1],
                self.vdt,
                self.tdt,
                scoordi=scoordi,
                ecoordi=ecoordi,
                mode="nearest",
            )
            fc_datum = np.round(fc_datum).astype(np.uint8)

        if self.is_crop:
            mz = nz // 50
            sz = random.randint(-mz, nz - self.crop_size[0] + mz)
            sz = np.clip(sz, 0, nz - self.crop_size[0])
            ez = sz + self.crop_size[0]
            vc_datum = vc_datum[sz: ez]
            if is_withfacies:
                fc_datum = fc_datum[sz: ez]

        if self.is_coordi:
            z_coordi = np.linspace(0.5, nz - 0.5, nz, dtype=np.float32)
            z_coordi = 2.0 * z_coordi / nz - 1.0
            if self.is_crop:
                z_coordi = z_coordi[sz: ez]
            z_coordi = np.repeat(z_coordi[:, None], self.crop_size[1], axis=1)

        if self.is_flip:
            check_value = random.random()
            if check_value < 0.5:
                vc_datum = np.flip(vc_datum, axis=1).copy()
                if is_withfacies:
                    fc_datum = np.flip(fc_datum, axis=1).copy()

        if self.noise != 0.0:
            vc_datum += self.noise * np.random.randn(self.crop_size[0], self.crop_size[1]).astype(np.float32)

        output = (vc_datum[None],)
        if is_withfacies:
            output = output + (fc_datum.astype(np.int64),)
        if self.is_coordi:
            output = output + (z_coordi[None],)
        return output[0] if len(output) == 1 else output


if  __name__ == "__main__":
    from torch.utils.data import DataLoader
    import torch
    from utils.data import show_2d_array

    train_set = LithofaciesDataset(
        data_pool="southsea",
        volume_tag="3dn_tbcut",
        facies_tag="3dn_facies_tbcut",
        target_idx=(2000, 2667),
        vdt=12.5,
        tdt=25.0,
        crop_size=(756, 512),
        total_length=2000,
        is_coordi=True,
        is_flip=True,
        noise=0.01,
    )
    train_loader = DataLoader(train_set, batch_size=1, shuffle=False)

    import time
    with torch.no_grad():
        start_time = time.time()
        for i, (vt, ft, cd) in enumerate(train_loader):
            print(time.time() - start_time)

            vt = vt[0].cpu().numpy().squeeze()
            ft = ft[0].cpu().numpy().squeeze() / 5
            cd = cd[0].cpu().numpy().squeeze()
            hb = np.zeros((vt.shape[0], 4))

            imgs = np.concatenate((
                vt, hb, ft, hb, cd,
            ), axis=1)
            show_2d_array(imgs, scale=100)
