"""Precompute the instantaneous-attribute volumes used as the network's "info" input.

Reads data/southsea/<volume_tag>.npy (shape nz, nx, ny; already RMS-normalized to 0.15) and writes
  <volume_tag>_inst_phase.npy  sin of the unwrapped instantaneous phase, in [-1, 1]
  <volume_tag>_inst_freq.npy   instantaneous frequency, clipped at its 99th percentile and scaled to [-1, 1]
next to it. Run once before training, from the repository root:

    python -m process.compute_inst_attribute
"""
import os
import numpy as np
from scipy.signal import hilbert
from utils.project import get_project_root


data_dir = os.path.join(get_project_root(), "data", "southsea")
volume_tag = "3dn_tbcut"
dt = 0.004  # sample interval in seconds

volume = np.load(os.path.join(data_dir, volume_tag + ".npy"))
nz, nx, ny = volume.shape
print(f"Volume shape: {volume.shape}")

# Reflect-pad each trace before the Hilbert transform to suppress edge effects.
pad = 500
inst_phase = np.zeros_like(volume, dtype=np.float32)
inst_freq = np.zeros_like(volume, dtype=np.float32)

for ix in range(nx):
    for iy in range(ny):
        trace = volume[:, ix, iy]
        if np.all(trace == 0):  # dead / outside-survey trace: leave zeros
            continue
        padded = np.pad(trace, (pad, pad), mode="reflect")
        analytic = hilbert(padded)
        phase = np.unwrap(np.angle(analytic))
        freq = np.gradient(phase) / (2 * np.pi * dt)  # Hz

        inst_phase[:, ix, iy] = phase[pad:pad + nz].astype(np.float32)
        inst_freq[:, ix, iy] = freq[pad:pad + nz].astype(np.float32)

    if (ix + 1) % 50 == 0:
        print(f"  {ix + 1}/{nx} done")

freq_clip = np.percentile(np.abs(inst_freq[inst_freq != 0]), 99)
inst_freq = np.clip(inst_freq, -freq_clip, freq_clip) / freq_clip

# The unwrapped phase grows without bound down the trace; sin() maps it to a bounded,
# continuous value with no 2*pi jumps.
inst_phase = np.sin(inst_phase).astype(np.float32)

np.save(os.path.join(data_dir, volume_tag + "_inst_phase.npy"), inst_phase)
np.save(os.path.join(data_dir, volume_tag + "_inst_freq.npy"), inst_freq)
