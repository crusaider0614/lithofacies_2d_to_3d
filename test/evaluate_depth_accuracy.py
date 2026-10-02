import os, random
import numpy as np
import torch
import yacs.config
import matplotlib.pyplot as plt

from module.seismic_data import SeismicVolume, Coordinate
from network.coordi_network import get_gen_model
from utils.project import get_project_root
from test.create_validation_figures import extract_random_line, predict_section

random.seed(42)
np.random.seed(42)

target_dim = 256
crop_size = (768, 512)
device = torch.device('cuda:9')
n_samples = 100

config_file = os.path.join(get_project_root(), 'config', 'config_lithofacies.yaml')
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
depth_correct = np.zeros(crop_size[0])
depth_total = np.zeros(crop_size[0])

# Also track per-class accuracy by depth
classes = [2, 3, 4, 5]  # Basement, Igneous, Shale, Sand
class_names = {2: 'Basement', 3: 'Igneous', 4: 'Shale', 5: 'Sand'}
depth_class_correct = {c: np.zeros(crop_size[0]) for c in classes}
depth_class_total = {c: np.zeros(crop_size[0]) for c in classes}

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
        
        # Per-depth accuracy (overall)
        for d in range(crop_size[0]):
            mask = ft[d, :] != 0
            if mask.sum() > 0:
                depth_correct[d] += np.sum(ft[d, mask] == fo[d, mask])
                depth_total[d] += mask.sum()
                
                # Per-class
                for c in classes:
                    class_mask = ft[d, :] == c
                    if class_mask.sum() > 0:
                        depth_class_correct[c][d] += np.sum(fo[d, class_mask] == c)
                        depth_class_total[c][d] += class_mask.sum()

# Calculate accuracy per depth
depth_accuracy = np.zeros(crop_size[0])
valid_depths = depth_total > 0
depth_accuracy[valid_depths] = depth_correct[valid_depths] / depth_total[valid_depths] * 100

# Per-class accuracy
depth_class_accuracy = {}
for c in classes:
    acc = np.zeros(crop_size[0])
    valid = depth_class_total[c] > 0
    acc[valid] = depth_class_correct[c][valid] / depth_class_total[c][valid] * 100
    depth_class_accuracy[c] = acc

# Convert depth index to two-way time (seconds)
dt_sec = 2.0 / crop_size[0]
t_start = 1.0
twt = t_start + np.arange(crop_size[0]) * dt_sec

# Smooth curves
window = 21
def smooth(arr):
    return np.convolve(arr, np.ones(window)/window, mode='same')

depth_accuracy_smooth = smooth(depth_accuracy)
for c in classes:
    depth_class_accuracy[c] = smooth(depth_class_accuracy[c])

# Plot
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11

fig, axes = plt.subplots(1, 2, figsize=(12, 8), dpi=300)

# Left panel: Overall accuracy
ax1 = axes[0]
ax1.plot(depth_accuracy_smooth, twt, 'b-', linewidth=2.5, label='Overall Accuracy')
ax1.fill_betweenx(twt, depth_accuracy_smooth, 0, alpha=0.2, color='blue')

ax1.axvline(x=80, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
ax1.axvline(x=90, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)

# Target zone (sedimentary section - middle)
target_top = 1.5
target_bottom = 2.3
ax1.axhspan(target_top, target_bottom, alpha=0.12, color='green')
ax1.text(5, (target_top + target_bottom) / 2, 'Target\nzone', fontsize=10, 
         fontweight='bold', va='center', color='darkgreen')

# Boundary zone (basement boundary - bottom)
boundary_top = 2.4
boundary_bottom = 2.9
ax1.axhspan(boundary_top, boundary_bottom, alpha=0.12, color='red')
ax1.text(5, (boundary_top + boundary_bottom) / 2, 'Basement\nboundary', fontsize=10, 
         fontweight='bold', va='center', color='darkred')

# Calculate zone accuracies
target_mask = (twt >= target_top) & (twt <= target_bottom) & valid_depths
boundary_mask = (twt >= boundary_top) & (twt <= boundary_bottom) & valid_depths

target_acc = np.mean(depth_accuracy_smooth[target_mask]) if np.any(target_mask) else 0
boundary_acc = np.mean(depth_accuracy_smooth[boundary_mask]) if np.any(boundary_mask) else 0

ax1.annotate('{:.1f}%'.format(target_acc), xy=(target_acc, (target_top + target_bottom)/2),
             xytext=(target_acc + 8, (target_top + target_bottom)/2 - 0.1),
             fontsize=12, fontweight='bold', color='darkgreen',
             arrowprops=dict(arrowstyle='->', color='darkgreen', lw=1.5))

ax1.annotate('{:.1f}%'.format(boundary_acc), xy=(boundary_acc, (boundary_top + boundary_bottom)/2),
             xytext=(boundary_acc - 25, (boundary_top + boundary_bottom)/2),
             fontsize=12, fontweight='bold', color='darkred',
             arrowprops=dict(arrowstyle='->', color='darkred', lw=1.5))

ax1.set_xlabel('Accuracy (%)', fontsize=13)
ax1.set_ylabel('Two-way time (s)', fontsize=13)
ax1.set_title('(a) Overall Accuracy vs Depth', fontsize=14, fontweight='bold')
ax1.set_xlim(0, 105)
ax1.set_ylim(3.0, 1.0)
ax1.grid(True, alpha=0.3)

# Secondary axis: depth (sample) index within the crop
depth_ticks = [1.0, 1.5, 2.0, 2.5, 3.0]
ax1_depth = ax1.secondary_yaxis('right')
ax1_depth.set_ylabel('Depth index', fontsize=11)
ax1_depth.set_yticks(depth_ticks)
ax1_depth.set_yticklabels([str(int(round((t - t_start) / dt_sec))) for t in depth_ticks])

# Right panel: Per-class accuracy
ax2 = axes[1]
colors = {2: 'blue', 3: 'red', 4: 'orange', 5: 'gold'}
for c in [4, 5]:  # Only Shale and Sand (main classes)
    valid = depth_class_total[c] > 100  # Need enough samples
    if np.any(valid):
        ax2.plot(depth_class_accuracy[c], twt, color=colors[c], linewidth=2, 
                label=class_names[c])

ax2.axhspan(target_top, target_bottom, alpha=0.12, color='green')
ax2.axhspan(boundary_top, boundary_bottom, alpha=0.12, color='red')

ax2.set_xlabel('Accuracy (%)', fontsize=13)
ax2.set_title('(b) Per-Class Accuracy (Shale & Sand)', fontsize=14, fontweight='bold')
ax2.set_xlim(0, 105)
ax2.set_ylim(3.0, 1.0)
ax2.legend(loc='lower left', fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
output_dir = os.path.join(get_project_root(), 'test', 'validation_figures', 'tdt_{:.1f}'.format(tdt))
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, 'depth_accuracy.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print('')
print('Saved:', output_path)

# Print summary
print('')
print('=' * 50)
print('DEPTH-WISE ACCURACY SUMMARY')
print('=' * 50)
print('')
print('Overall accuracy (all depths): {:.1f}%'.format(np.mean(depth_accuracy_smooth[valid_depths])))
print('')
print('Target zone ({:.1f}-{:.1f}s):'.format(target_top, target_bottom))
print('  Overall accuracy: {:.1f}%'.format(target_acc))
if np.any(target_mask):
    print('  Min accuracy:     {:.1f}%'.format(np.min(depth_accuracy_smooth[target_mask])))
for c in [4, 5]:
    zone_mask = target_mask & (depth_class_total[c] > 0)
    if np.any(zone_mask):
        print('  {} accuracy: {:.1f}%'.format(class_names[c], np.mean(depth_class_accuracy[c][zone_mask])))
print('')
print('Boundary zone ({:.1f}-{:.1f}s):'.format(boundary_top, boundary_bottom))
print('  Overall accuracy: {:.1f}%'.format(boundary_acc))
if np.any(boundary_mask):
    print('  Min accuracy:     {:.1f}%'.format(np.min(depth_accuracy_smooth[boundary_mask])))
for c in [4, 5]:
    zone_mask = boundary_mask & (depth_class_total[c] > 0)
    if np.any(zone_mask):
        print('  {} accuracy: {:.1f}%'.format(class_names[c], np.mean(depth_class_accuracy[c][zone_mask])))
