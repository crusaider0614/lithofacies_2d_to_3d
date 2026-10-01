import os, sys, random
import numpy as np
import torch
import yacs.config
import matplotlib.pyplot as plt

sys.path.insert(0, '/home/oilpire/project/lithofacies_2d_to_3d')
from module.seismic_data import SeismicVolume, Coordinate
from network.coordi_network_2 import get_gen_model
from utils.project import get_project_root
from create_validation_figures import extract_random_line, predict_section

random.seed(42)
np.random.seed(42)

target_dim = 256
crop_size = (768, 512)
device = torch.device('cuda:9')
n_samples = 100

config_file = os.path.join(get_project_root(), 'config', 'config_lithofacies_2.yaml')
with open(config_file, 'rt') as f:
    CF = yacs.config.load_cfg(f)

data_pool = CF.DATASET.DATA_POOL
volume_tag = CF.DATASET.VOLUME_TAG
facies_tag = CF.DATASET.FACIES_TAG
valid_idx = CF.DATASET.VALID_IDX
vdt = CF.DATASET.VDT
tdt = 25.0

print('Loading data...')
volume = np.load(os.path.join(get_project_root(), 'data', data_pool, volume_tag + '.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]
facies = np.load(os.path.join(get_project_root(), 'data', data_pool, facies_tag + '.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]
inst_phase = np.load(os.path.join(get_project_root(), 'data', data_pool, volume_tag + '_inst_phase.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]
inst_freq = np.load(os.path.join(get_project_root(), 'data', data_pool, volume_tag + '_inst_freq.npy'), mmap_mode='r')[:, valid_idx[0]:valid_idx[1]]

nz, nx, ny = volume.shape

volume_sv = SeismicVolume()
volume_sv.clean_data(shape=(nz, nx, ny))
volume_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
volume_sv.data = volume

facies_sv = SeismicVolume()
facies_sv.clean_data(shape=(nz, nx, ny))
facies_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
facies_sv.data = facies

inst_phase_sv = SeismicVolume()
inst_phase_sv.clean_data(shape=(nz, nx, ny))
inst_phase_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
inst_phase_sv.data = inst_phase

inst_freq_sv = SeismicVolume()
inst_freq_sv.clean_data(shape=(nz, nx, ny))
inst_freq_sv.set_coordi(Coordinate(0, 0), Coordinate(0, ny - 1), Coordinate(nx - 1, 0), Coordinate(nx - 1, ny - 1))
inst_freq_sv.data = inst_freq

network = get_gen_model(CF, additional_channel=0).to(device)
checkpoint_path = os.path.join(get_project_root(), 'checkpoint', 'lithofacies_prediction_25.0_pat_044')
state = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
network.load_state_dict(state['network'])
network.eval()
print('Loaded checkpoint')

nt = crop_size[1]

# Collect per-depth statistics
# depth index 0 = top (1.0s), depth index 767 = bottom (3.0s)
depth_correct = np.zeros(crop_size[0])
depth_total = np.zeros(crop_size[0])

print('')
print('Evaluating', n_samples, 'random sections for depth-wise accuracy...')
with torch.no_grad():
    for idx in range(n_samples):
        if (idx + 1) % 20 == 0:
            print('  Processing', idx + 1, '/', n_samples, '...')
        
        while True:
            vt, scoordi, ecoordi = extract_random_line(volume_sv, nt, vdt, tdt, mode='bilinear')
            ft, _, _ = extract_random_line(facies_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='nearest')
            ft = np.round(ft).astype(np.int32)
            sz = random.randint(0, nz - crop_size[0])
            ez = sz + crop_size[0]
            if np.any(ft[sz:ez] != 0):
                break
        
        vt = vt[sz:ez]
        ft = ft[sz:ez]
        
        ip, _, _ = extract_random_line(inst_phase_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='bilinear')
        ip = ip[sz:ez]
        if_, _, _ = extract_random_line(inst_freq_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='bilinear')
        if_ = if_[sz:ez]
        
        rms = (vt * vt).mean() ** 0.5
        if rms > 0:
            vt = vt / rms * 0.15
        
        vt_tensor = torch.tensor(vt[None, None], dtype=torch.float32, device=device)
        info_tensor = torch.tensor(np.stack([ip, if_])[None], dtype=torch.float32, device=device)
        
        fo = predict_section(network, vt_tensor, info_tensor, device, target_dim)
        fo[ft == 0] = 0
        
        # Per-depth accuracy
        for d in range(crop_size[0]):
            mask = ft[d, :] != 0  # Only classified pixels
            if mask.sum() > 0:
                depth_correct[d] += np.sum(ft[d, mask] == fo[d, mask])
                depth_total[d] += mask.sum()

# Calculate accuracy per depth
depth_accuracy = np.zeros(crop_size[0])
valid_depths = depth_total > 0
depth_accuracy[valid_depths] = depth_correct[valid_depths] / depth_total[valid_depths] * 100

# Convert depth index to two-way time (seconds)
dt_sec = 2.0 / crop_size[0]
t_start = 1.0
twt = t_start + np.arange(crop_size[0]) * dt_sec

# Smooth the curve with moving average
window = 15
depth_accuracy_smooth = np.convolve(depth_accuracy, np.ones(window)/window, mode='same')

# Plot
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

fig, ax = plt.subplots(figsize=(6, 8), dpi=300)

# Plot accuracy vs depth (depth increases downward)
ax.plot(depth_accuracy_smooth, twt, 'b-', linewidth=2, label='Accuracy (smoothed)')
ax.fill_betweenx(twt, depth_accuracy_smooth, 0, alpha=0.3)

# Add reference lines
ax.axvline(x=80, color='green', linestyle='--', linewidth=1.5, label='80% threshold')
ax.axvline(x=90, color='orange', linestyle='--', linewidth=1.5, label='90% threshold')

# Mark target zone (middle section, approximately 1.5s - 2.5s)
target_top = 1.4
target_bottom = 2.4
ax.axhspan(target_top, target_bottom, alpha=0.15, color='green', label='Target zone')

# Add text annotation for target zone accuracy
target_mask = (twt >= target_top) & (twt <= target_bottom)
target_acc = np.mean(depth_accuracy_smooth[target_mask])
ax.text(target_acc + 2, (target_top + target_bottom) / 2, 
        'Target zone\n{:.1f}%'.format(target_acc), 
        fontsize=11, fontweight='bold', va='center', color='darkgreen')

# Mark boundary zone (bottom section, approximately 2.5s - 3.0s)
boundary_top = 2.5
boundary_bottom = 3.0
ax.axhspan(boundary_top, boundary_bottom, alpha=0.15, color='red', label='Boundary zone')

boundary_mask = (twt >= boundary_top) & (twt <= boundary_bottom)
if np.any(boundary_mask) and np.any(depth_accuracy_smooth[boundary_mask] > 0):
    boundary_acc = np.mean(depth_accuracy_smooth[boundary_mask][depth_accuracy_smooth[boundary_mask] > 0])
    ax.text(boundary_acc - 15, (boundary_top + boundary_bottom) / 2, 
            'Boundary\n{:.1f}%'.format(boundary_acc), 
            fontsize=11, fontweight='bold', va='center', color='darkred')

ax.set_xlabel('Accuracy (%)', fontsize=13)
ax.set_ylabel('Two-way time (s)', fontsize=13)
ax.set_title('Prediction Accuracy vs Depth', fontsize=14, fontweight='bold')

ax.set_xlim(0, 105)
ax.set_ylim(3.0, 1.0)  # Inverted y-axis (depth increases downward)

ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)

# Add secondary axis for depth index
ax2 = ax.secondary_yaxis('right')
ax2.set_ylabel('Depth index', fontsize=11)
ax2.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0])
ax2.set_yticklabels(['0', '192', '384', '576', '768'])

plt.tight_layout()
output_path = '/home/oilpire/project/lithofacies_2d_to_3d/test/validation_figures/tdt_25.0/depth_accuracy.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('')
print('Saved:', output_path)

# Print summary statistics
print('')
print('=' * 50)
print('DEPTH-WISE ACCURACY SUMMARY')
print('=' * 50)
print('')
print('Overall accuracy: {:.1f}%'.format(np.mean(depth_accuracy_smooth[valid_depths])))
print('')
print('Target zone (1.4-2.4s):')
print('  Mean accuracy: {:.1f}%'.format(target_acc))
print('  Min accuracy:  {:.1f}%'.format(np.min(depth_accuracy_smooth[target_mask])))
print('')
if np.any(boundary_mask) and np.any(depth_accuracy_smooth[boundary_mask] > 0):
    print('Boundary zone (2.5-3.0s):')
    print('  Mean accuracy: {:.1f}%'.format(boundary_acc))
    print('  Min accuracy:  {:.1f}%'.format(np.min(depth_accuracy_smooth[boundary_mask][depth_accuracy_smooth[boundary_mask] > 0])))
