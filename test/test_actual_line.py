import os
import random
import numpy as np
import torch
import yacs.config

from module.seismic_data import SeismicLine, distance
from network.coordi_network import get_gen_model
from utils.data import show_2d_array
from utils.project import get_project_root
from obspy.io.segy.segy import SEGYFile, SEGYTraceHeader, SEGYBinaryFileHeader
from obspy.core import Trace, Stream


def add_weighted_avg(average, weight, new_sample, new_weight):
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


def read_trace_coordinates(file_path):
    coords = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        for line in lines[5:]:
            parts = line.split()
            x = float(parts[1])
            y = float(parts[2])
            coords.append((x, y))
    return np.array(coords)


target_dim = 512
gamma_function = lambda r: np.cos(r * np.pi / 2)
crop_size = (768, target_dim)

weight = np.zeros(crop_size[1])
for i in range(crop_size[1]):
    weight[i] = (1 + np.cos((i - (crop_size[1] // 2 - 0.5)) / (crop_size[1] // 2) * np.pi)) / 2
weight = np.repeat(weight[None], crop_size[0], 0)

# Define
config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:5")

# Load
tag = "lithofacies_prediction_7.5"
epoch = 18
network_1 = get_gen_model(CF, additional_channel=0).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
network_1.load_state_dict(state["network"])
network_1 = network_1.eval()

tag = "lithofacies_prediction_12.5"
epoch = 23
network_2 = get_gen_model(CF, additional_channel=0).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
network_2.load_state_dict(state["network"])
network_2 = network_2.eval()

tag = "lithofacies_prediction_25.0"
epoch = 23
network_3 = get_gen_model(CF, additional_channel=0).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage)
network_3.load_state_dict(state["network"])
network_3 = network_3.eval()

sample_interval = 4000
dt = 0.004

# idxs = range(193)
# idx_range = {
#     # 132:	(40,	1688),
#     # 135:	(110,	3550),
#     # 136:	(40,	1888),
#     # 139:	(45,	1608),
#     # 151:	(40,	1728),
#     # 155:	(75,	910),
#     # 156:	(0,	1748),
#     # 159:	(80,	1360),
#     # 160:	(0,	2870),
#     # 164:	(0,	1368),
#     # 166:	(80,	1321),
#     # 168:	(0,	739),
#     # 181:	(160,	4950),
#     # 182:	(240,	4930),
#     # 183:	(190,	5100),
#     # 184:	(130,	4200),
#     # 185:	(90,	3705),
#     # 187:	(75,	2087),
#     189:	(160,	5280),
#     # 190:	(0,	1560),
# }
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
    # 150: (110, 3300),
    # 152: (45, -1),
    # 153: (45, -1),
    # 154: (200, 6000),
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
    189:	(160,	5280),
    # 190:	(0,	1560),
}

# idxs = np.array(idxs) - 1
# idxs = list(range(192))
# random.shuffle(idxs)
for idx in idx_range.keys():
    print(idx)

    line = SeismicLine(data_pool="ssealine", tag=str(idx + 1))
    ldt = distance(line.scoordi, line.ecoordi) / (line.shape[1] - 1)

    nphead_path = os.path.join(get_project_root(), "data", "ssealine", str(idx + 1) + ".nphead")
    coordinate = read_trace_coordinates(nphead_path)

    tdt = min([7.5, 12.5, 25.0], key=lambda x: abs(ldt - x))
    if line.shape[0] <= 1001:
        vt = line.data
    else:
        vt = line.data[:1001]
    vto = vt.copy()
    szp = 64
    ezp = 1001 - szp - crop_size[0]
    vt = vt[szp: szp + crop_size[0]]

    isx = idx_range[idx][0]
    iex = idx_range[idx][1]
    vt = vt[:, isx: iex]

    if vt.shape[1] < target_dim:
        is_pad = True
        wp = target_dim - vt.shape[1]
        lp = wp // 2
        rp = wp - lp
        vt = np.pad(vt, pad_width=((0, 0), (lp, rp)), mode="reflect")
    else:
        is_pad = False
    vt = vt / (vt * vt).mean()**0.5 * 0.15

    if tdt < 10:
        network = network_1
    elif tdt < 15:
        network = network_2
    else:
        network = network_3

    print(vt.shape, tdt, ldt, isx, iex)

    nz, nt = vt.shape

    dp = np.zeros_like(vt, dtype=np.float32)
    fo = np.zeros_like(vt, dtype=np.float32)
    fo = np.repeat(fo[None], 6, axis=0)

    cd = np.linspace(0.5, nz - 0.5, nz, dtype=np.float32)
    cd = 2.0 * cd / nz - 1.0
    cd = np.repeat(cd[:, None], target_dim, axis=1)

    nst = (int((nt - (target_dim + 1)) / (target_dim // 2))) + 1
    # Testing
    with torch.no_grad():
        for isz in range(1):
            sz = 0
            ez = sz + crop_size[0]
            cd_crop = cd[sz: ez]
            cd_crop = torch.tensor(cd_crop[None, None], dtype=torch.float32, device=device)
            for ist in range(nst + 1):
                st = int(round((nt - target_dim) * ist / nst))
                et = st + target_dim
                print(sz, ez, st, et)

                vt_crop = vt[sz: ez, st: et]
                vt_crop = torch.tensor(vt_crop[None, None], dtype=torch.float32, device=device)

                fo_crop = network(vt_crop, cd_crop)
                fo_crop = fo_crop.cpu().numpy().squeeze()

                fo[:, sz: ez, st: et], dp[sz: ez, st: et] = add_weighted_avg(
                    fo[:, sz: ez, st: et],
                    dp[sz: ez, st: et],
                    fo_crop,
                    weight
                )


        fo = np.argmax(fo, axis=0).astype(np.int32)
        foo = fo.copy()
        foo = np.pad(foo, ((szp, 0), (0, 0)), constant_values=1)
        foo = np.pad(foo, ((0, ezp), (0, 0)), constant_values=2)

        print(idx, vto.shape, foo.shape, szp, ezp)

        hb = np.zeros((foo.shape[0], 4))
        imgs = np.concatenate((
            hb, foo, hb,
        ), axis=1)
        show_2d_array(imgs, scale=200, cmap="gray", vmax=5.0, vmin=0.0)

        hb = np.ones((vto.shape[0], 4))
        imgs = np.concatenate((
            hb, vto, hb,
        ), axis=1)
        show_2d_array(imgs, scale=200)

        if is_pad:
            foo = foo[:, lp: -rp]

        num_samples, num_traces = foo.shape
        stream = Stream()
        for it in range(num_traces):
            trace_data = foo[:, it]
            coordi = line.idx_to_coordi(it + isx)
            x = coordi.cx
            y = coordi.cy

            trace = Trace(data=trace_data)

            trace.stats.segy = {}
            trace.stats.delta = dt
            trace.stats.segy.trace_header = SEGYTraceHeader()
            trace.stats.segy.trace_header.trace_sequence_number_within_line = it + 1
            trace.stats.segy.trace_header.source_coordinate_x = int(np.round(x))
            trace.stats.segy.trace_header.source_coordinate_y = int(np.round(y))
            trace.stats.segy.trace_header.number_of_samples_in_this_trace = num_samples
            trace.stats.segy.trace_header.sample_interval_in_microseconds = sample_interval

            stream.append(trace)

        stream.write(os.path.join(get_project_root(), "data", str(idx) + ".segy"), format="SEGY", data_encoding=2)
