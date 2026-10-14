"""Inference on real (field) 2D seismic lines.

Applies the trained models to the real 2D lines stored in data/ssealine_matched/. For each
line listed in ``line_range``:
  1. load the line (<name>.npy amplitude + <name>.nphead header with trace coordinates; see
     module.seismic_data.SeismicLine), keep at most the first 1001 samples and the trace
     range given in ``line_range`` (reflect-padding to 256 traces if the line is narrower);
  2. compute the line's mean trace spacing from the straight-line fit of its trace
     coordinates (whole line, before trimming) and pick the model whose
     training trace spacing TDT (7.5 / 12.5 / 25.0 m) is nearest to it;
  3. rescale the section to RMS 0.15 and compute the 2-channel info input from it
     (sin of the unwrapped Hilbert phase, and instantaneous frequency clipped at its 99th
     percentile and scaled to [-1, 1]), as process/compute_inst_attribute.py does for the 3D volume;
  4. predict with overlapping 256 x 256 tiles blended by a raised-cosine window, and
     display prediction above amplitude.
Writing the prediction to SEG-Y (data/ssealine_matched/lithofacies/new_<name>.segy) is
present but commented out at the end of the loop.

Reads:  config/config_lithofacies.yaml,
        checkpoint/lithofacies_prediction_7.5_pat_047, ..._12.5_pat_049, ..._25.0_pat_044,
        data/ssealine_matched/<name>.npy and <name>.nphead for each name in ``line_range``.
Writes: nothing while the SEG-Y export is commented out; figures are shown on screen.

Run from the repo root (running by file path fails because the repo root is then not on
sys.path):

    python -m test.test_actual_line

There is no CLI. Edit the module-level settings (``device``, checkpoint tags/epochs,
``line_range``).
"""
import os
import numpy as np
import torch
import yacs.config
from scipy.signal import hilbert

from module.seismic_data import SeismicLine, distance
from network.coordi_network import get_gen_model
from utils.data import show_2d_array
from utils.project import get_project_root


def add_weighted_avg(average, weight, new_sample, new_weight):
    """Fold one tile prediction (logits ``new_sample``, window ``new_weight``) into the
    running weighted average ``average`` with accumulated weight ``weight``."""
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


# Tile size (= training crop size) and the 2D raised-cosine blending window for overlap-add
# tiling: ~1 at the tile centre, ~0 at its edges.
target_dim = 256
weight = np.array([(1 + np.cos((i - (target_dim // 2 - 0.5)) / (target_dim // 2) * np.pi)) / 2 for i in range(target_dim)])
weight = weight[None] * weight[:, None]

# Define
config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
with open(config_file, "rt") as f_read:
    CF = yacs.config.load_cfg(f_read)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:0")

# Load one model per training trace spacing (epochs = minimum validation loss).
tag = "lithofacies_prediction_7.5_pat"
epoch = 47
network_1 = get_gen_model(CF, additional_channel=0).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage, weights_only=True)
network_1.load_state_dict(state["network"])
network_1 = network_1.eval()

tag = "lithofacies_prediction_12.5_pat"
epoch = 49
network_2 = get_gen_model(CF, additional_channel=0).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage, weights_only=True)
network_2.load_state_dict(state["network"])
network_2 = network_2.eval()

tag = "lithofacies_prediction_25.0_pat"
epoch = 44
network_3 = get_gen_model(CF, additional_channel=0).to(device)
state = torch.load(os.path.join(get_project_root(), "checkpoint", tag + "_" + str(epoch).zfill(3)), map_location=lambda storage, loc: storage, weights_only=True)
network_3.load_state_dict(state["network"])
network_3 = network_3.eval()

sample_interval = 4000  # microseconds, for the (commented-out) SEG-Y export
dt = 0.004  # sample interval in seconds

# Lines to process: {file name in data/ssealine_matched/ (without extension): (first trace,
# end trace)}; end trace -1 means "to the end". The trace windows trim unusable parts of each
# line. The released lines are named after the model their trace spacing selects.
line_range = {
    "actual_line_7.5": (0, 1130),   # trace spacing ~7.5 m
    "actual_line_12.5": (0, -1),    # trace spacing ~12.3 m
    "actual_line_25.0": (45, -1),   # trace spacing ~24.5 m
}

for name in line_range.keys():
    line = SeismicLine(data_pool="ssealine_matched", tag=name)
    # Mean trace spacing of the line (m), from its fitted start/end coordinates.
    ldt = distance(line.scoordi, line.ecoordi) / (line.shape[1] - 1)

    # Choose the training trace spacing nearest to the line's own spacing.
    tdt = min([7.5, 12.5, 25.0], key=lambda x: abs(ldt - x))
    # Keep at most the first 1001 time samples.
    if line.shape[0] <= 1001:
        vt = line.data
    else:
        vt = line.data[:1001]

    isx = line_range[name][0]
    iex = line_range[name][1] if line_range[name][1] > 0 else vt.shape[1]
    vt = vt[:, isx: iex]

    # Lines narrower than one tile are reflect-padded to target_dim traces.
    if vt.shape[1] < target_dim:
        is_pad = True
        wp = target_dim - vt.shape[1]
        lp = wp // 2
        rp = wp - lp
        vt = np.pad(vt, pad_width=((0, 0), (lp, rp)), mode="reflect")
    else:
        is_pad = False
    # Rescale the amplitude section to RMS 0.15 before inference.
    vt = vt / (vt * vt).mean()**0.5 * 0.15

    # Select the model trained at the chosen TDT (7.5 / 12.5 / 25.0 m).
    if tdt < 10:
        network = network_1
    elif tdt < 15:
        network = network_2
    else:
        network = network_3

    print(name, vt.shape, tdt, ldt, isx, iex)

    nz, nt = vt.shape

    dp = np.zeros_like(vt, dtype=np.float32)
    fo = np.zeros_like(vt, dtype=np.float32)
    fo = np.repeat(fo[None], 6, axis=0)

    # Instantaneous attributes per trace from the analytic (Hilbert) signal; each trace is
    # reflect-padded by 500 samples to suppress edge effects, and dead traces stay zero.
    pad = 500
    inst_phase = np.zeros_like(vt, dtype=np.float32)
    inst_freq = np.zeros_like(vt, dtype=np.float32)

    for iy in range(nt):
        trace = vt[:, iy]
        if np.all(trace == 0):
            continue
        padded = np.pad(trace, (pad, pad), mode="reflect")
        analytic = hilbert(padded)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase) / (2 * np.pi * dt)

        inst_phase[:, iy] = phase[pad:pad + nz].astype(np.float32)
        inst_freq[:, iy] = freq[pad:pad + nz].astype(np.float32)

    # Instantaneous frequency: clip at the 99th percentile of |f| (per line) and scale to [-1, 1].
    freq_clip = np.percentile(np.abs(inst_freq[inst_freq != 0]), 99)
    inst_freq = np.clip(inst_freq, -freq_clip, freq_clip) / freq_clip

    # Instantaneous phase: sin of the unwrapped phase (bounded, no 2*pi jumps).
    inst_phase = np.sin(inst_phase).astype(np.float32)

    info = np.concatenate((inst_phase[None], inst_freq[None]), axis=0)

    # Overlap-add tiling (at least 50 % overlap) over depth and along the line; tile logits are
    # blended with the raised-cosine ``weight`` window.
    nsz = int((nz - target_dim) / (target_dim // 2)) + 2 if nz > target_dim else 1
    nst = int((nt - target_dim) / (target_dim // 2)) + 2 if nt > target_dim else 1
    # Testing
    with torch.no_grad():
        for isz in range(nsz):
            sz = int(round((nz - target_dim) * isz / (nsz - 1))) if nsz > 1 else 0
            ez = sz + target_dim
            for ist in range(nst):
                st = int(round((nt - target_dim) * ist / (nst - 1))) if nst > 1 else 0
                et = st + target_dim

                vt_crop = vt[sz: ez, st: et]
                vt_crop = torch.tensor(vt_crop[None, None], dtype=torch.float32, device=device)

                info_crop = info[:, sz: ez, st: et]
                info_crop = torch.tensor(info_crop[None], dtype=torch.float32, device=device)

                fo_crop = network(vt_crop, info_crop)
                fo_crop = fo_crop.cpu().numpy().squeeze()

                fo[:, sz: ez, st: et], dp[sz: ez, st: et] = add_weighted_avg(
                    fo[:, sz: ez, st: et],
                    dp[sz: ez, st: et],
                    fo_crop,
                    weight
                )

        fo = np.argmax(fo, axis=0).astype(np.int32)
        foo = fo.copy()

        # Display: class map rescaled to [-1, 1] (foo * 2 / 5 - 1) stacked above the amplitude.
        hb = np.zeros((foo.shape[0], 4))
        imgs1 = np.concatenate((
            hb, (foo * 2 / 5 - 1), hb,
        ), axis=1)
        hb = np.zeros((vt.shape[0], 4))
        imgs2 = np.concatenate((
            hb, vt, hb,
        ), axis=1)
        imgs = np.concatenate((
            imgs1, imgs2,
        ), axis=0)
        show_2d_array(imgs, scale=200, cmap="seismic", vmax=1.0, vmin=-1.0)

        # Optional SEG-Y export of the predicted classes (needs obspy); uncomment to use.
        # from obspy.core import Trace, Stream
        # from obspy.io.segy.segy import SEGYTraceHeader
        #
        # if is_pad:
        #     foo = foo[:, lp: -rp]
        #
        # num_samples, num_traces = foo.shape
        # stream = Stream()
        # for it in range(num_traces):
        #     trace_data = foo[:, it]
        #     coordi = line.idx_to_coordi(it + isx)
        #     x = coordi.cx
        #     y = coordi.cy
        #
        #     trace = Trace(data=trace_data)
        #
        #     trace.stats.segy = {}
        #     trace.stats.delta = dt
        #     trace.stats.segy.trace_header = SEGYTraceHeader()
        #     trace.stats.segy.trace_header.trace_sequence_number_within_line = it + 1
        #     trace.stats.segy.trace_header.source_coordinate_x = int(np.round(x))
        #     trace.stats.segy.trace_header.source_coordinate_y = int(np.round(y))
        #     trace.stats.segy.trace_header.number_of_samples_in_this_trace = num_samples
        #     trace.stats.segy.trace_header.sample_interval_in_microseconds = sample_interval
        #
        #     stream.append(trace)
        #
        # stream.write(os.path.join(get_project_root(), "data", "ssealine_matched", "lithofacies", "new_" + name + ".segy"), format="SEGY", data_encoding=2)
