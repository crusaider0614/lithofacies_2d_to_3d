import os
import random
import numpy as np
import torch
import yacs.config

from module.seismic_data import SeismicLine, distance
from network.coordi_network import get_gen_model
from utils.data import show_2d_array
from utils.project import get_project_root
from module.seismic_data import SeismicLine, SeismicVolume
from obspy import read as segy_read


volume_area = SeismicVolume("southsea", "3dn_bedrock_depth")
print(volume_area.shape)
print(volume_area.corner_coordi[0][0])
print(volume_area.corner_coordi[0][1])
print(volume_area.corner_coordi[1][0])
print(volume_area.corner_coordi[1][1])

volume_data = np.load(os.path.join(get_project_root(), "data", "southsea", "3dn_facies_unc.npy"))
print(volume_data.shape)
nz, nx, ny = volume_data.shape
facies_volume = SeismicVolume()
facies_volume.clean_data(volume_data.shape)
facies_volume.set_coordi(
    volume_area.corner_coordi[0][0],
    volume_area.corner_coordi[0][1],
    volume_area.corner_coordi[1][0],
    volume_area.corner_coordi[1][1],
)
facies_volume.data = volume_data

# show_2d_array(volume.data[:, 200], vmin=-5, vmax=5)

idx_range = {
    # 65: (0, -1),
    # 66: (0, -1),
    # 68: (0, -1),
    # 69: (0, 2370),
    # 72: (0, 6920),
    # 73: (0, 3000),
    # 79: (0, 2200),
    # 80: (0, 3300),
    # 81: (0, 3200),
    # 82: (0, 1130),
    # 83: (0, 3050),
    # 84: (0, 2000),
    # 85: (0, 1950),
    # 86: (0, 1940),
    # 87: (0, 1430),
    # 88: (0, 1900),
    # 89: (0, 1940),
    # 90: (0, 1400),
    # 91: (0, 1900),
    # 92: (0, 1400),
    # 93: (0, 2000),
    # 94: (0, 1400),
    # 95: (0, 2000),
    # 96: (0, 2000),
    # 97: (0, 1140),
    # 98: (0, 1300),
    # 99: (0, 2050),
    # 100: (0, -1),
    # 131: (110, 2770),
    # 133: (45, -1),
    # 134: (110, 3300),
    # 137: (45, -1),
    # 138: (110, 3450),
    # 140: (45, -1),
    # 142: (45, -1),
    # 143: (45, -1),
    # 144: (45, -1),
    # 146: (45, -1),
    # 148: (0, -1),
    # 149: (45, -1),
    150: (110, 3300),
    # 152: (45, -1),
    # 153: (45, -1),
    154: (200, 6000),
    # 157: (50, -1),
    # 158: (100, 2400),
    # 161: (0, -1),
    # 162: (0, -1),
    # 163: (95, 1640),
    # 165: (95, 1690),
    # 167: (95, 1100),
    # 169: (75, -1),
    # 170: (170, -1),
    # 180: (180, 4800),
    # 186: (160, 4370),
    # 188: (250, 4720),
    # 191: (50, -1),
    # 132:	(40,	1688),
    # 135:	(110,	3550),
    # 136:	(40,	1888),
    # 139:	(45,	1608),
    # 151:	(40,	1728),
    # 155:	(75,	910),
    # 156:	(0,	1748),
    # 159:	(80,	1360),
    # 160:	(0,	2870),
    # 164:	(0,	1368),
    # 166:	(80,	1321),
    # 168:	(0,	739),
    # 181:	(160,	4950),
    # 182:	(240,	4930),
    # 183:	(190,	5100),
    # 184:	(130,	4200),
    # 185:	(90,	3705),
    # 187:	(75,	2087),
    # 189:	(160,	5280),
    # 190:	(0,	1560),
}

for idx in idx_range.keys():
    print(idx)
    seismic_line = SeismicLine(data_pool="ssealine_matched", tag=str(idx + 1))
    seismic_line.cut_zaxis(0, 1000)
    nt = seismic_line.shape[1]

    stream = segy_read(os.path.join(get_project_root(), "data", "ssealine_matched", "lithofacies", str(idx) + ".segy"))
    facies_data = np.stack([tr.data for tr in stream.traces], axis=0)
    facies_data = np.transpose(facies_data)

    isx = idx_range[idx][0]
    iex = idx_range[idx][1] if idx_range[idx][1] > 0 else nt + idx_range[idx][1]

    facies_line = SeismicLine()
    facies_line.clean_data(facies_data.shape)
    facies_line.set_coordi(seismic_line.idx_to_coordi(isx), seismic_line.idx_to_coordi(iex))
    facies_line.data = facies_data
    facies_line.cut_zaxis(0, 1000)

    actual_line = facies_volume.get_line_coordi(facies_line.scoordi, facies_line.ecoordi, facies_line.shape[1])

    f1 = seismic_line.data[:, isx: iex]
    f1 = f1 / (f1 * f1).mean()**0.5 * 0.15
    f2 = facies_line.data / 5

    # if not valid_mask.any():
    #     print(f"idx {idx}: f2 전부 비어있음, 스킵")
    #     continue
    #
    # first_valid = np.argmax(valid_mask)
    # last_valid = len(valid_mask) - 1 - np.argmax(valid_mask[::-1])
    #
    # f1 = f1[:, first_valid:last_valid + 1]
    # f2 = f2[:, first_valid:last_valid + 1]

    tb = -5 * np.ones((f1.shape[0], 4))
    img = np.concatenate((f1, tb, f2), axis=1)

    show_2d_array(img, vmin=-1, vmax=1)

