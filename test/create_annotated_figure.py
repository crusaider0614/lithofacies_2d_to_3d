"""Annotated validation figure: one synthetic 2D line with interpretive annotations.

Draws a single random synthetic 2D line (seed 123) from the validation part of the 3D
volume at TDT = 25.0 m, takes its top 768 samples x 512 traces, predicts it with the
TDT = 25.0 m model (checkpoint/lithofacies_prediction_25.0_pat_044) using overlapping
256 x 256 tiles, and saves a 3-panel figure in the style of create_validation_figures.py:
(a) seismic with a "target depth range" bracket and "Fault" / "Chaotic basement" labels,
(b) ground truth with an optional map inset of the line location, (c) prediction with
the two largest misclassified regions circled in (b) and (c).

Note: the annotation positions in panel (a) are fixed fractions of the panel size, not
detected features; they were placed for the specific line drawn with seed 123.

Reads:  config/config_lithofacies.yaml, checkpoint/lithofacies_prediction_25.0_pat_044,
        data/<DATA_POOL>/<VOLUME_TAG>.npy, <FACIES_TAG>.npy,
        <VOLUME_TAG>_inst_phase.npy, <VOLUME_TAG>_inst_freq.npy, and optionally
        test/validation_figures/tdt_25.0/devided_volume.png (map image for the inset).
Writes: test/validation_figures/tdt_25.0/annotated_example.png

Run from the repo root (running by file path fails because the repo root is then not on
sys.path):

    python -m test.create_annotated_figure

There is no CLI. Edit the module-level settings (seeds, ``target_dim``, ``crop_size``,
``device``, ``tdt``, checkpoint path).
"""
import os, random
import numpy as np
import torch
import yacs.config
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle
from PIL import Image

from module.seismic_data import SeismicVolume, Coordinate
from network.coordi_network import get_gen_model
from utils.project import get_project_root
from test.create_validation_figures import extract_random_line, predict_section, facies_to_rgb

plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300

def create_annotated_figure(seismic, ground_truth, prediction, output_path,
                            scoordi=None, ecoordi=None, volume_img_path=None,
                            nx_volume=667, ny_volume=1200, tdt=25.0):
    """Save a 3-panel (seismic / ground truth / prediction) figure with annotations.

    Arguments are as in ``create_validation_figures.create_validation_figure``. In addition,
    panel (a) gets fixed-position interpretive labels, and connected regions where both
    ground truth and prediction are labeled but disagree are found with
    ``scipy.ndimage.label``; the two largest (if > 50 pixels) are circled in (b) and (c).
    """
    img_height, img_width = seismic.shape
    
    panel_aspect = img_height / img_width
    fig_width = 12
    panel_width = fig_width / 3
    fig_height = panel_width * panel_aspect * 1.3

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=300)
    plt.subplots_adjust(wspace=0.08, left=0.08, right=0.98, bottom=0.08, top=0.94)

    distance_km = img_width * tdt / 1000.0
    x_ticks = [0, img_width/2, img_width]
    x_labels = ['0', f'{distance_km/2:.1f}', f'{distance_km:.1f}']

    dt_sec = 2.0 / img_height
    t_start = 1.0
    t_end = t_start + img_height * dt_sec
    y_ticks = [0, img_height/4, img_height/2, 3*img_height/4, img_height]
    y_labels = [f'{t_start + i * (t_end - t_start) / 4:.1f}' for i in range(5)]

    spine_width = 2

    # (a) Seismic Data with annotations
    ax1 = axes[0]
    ax1.set_title('(a) Seismic data', fontsize=14, fontweight='bold', pad=10)
    seismic_normalized = seismic / (np.abs(seismic).max() + 1e-8)
    im1 = ax1.imshow(seismic_normalized, cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax1.set_ylabel('Two-way time (s)', fontsize=11)
    ax1.set_xlabel('Distance (km)', fontsize=11)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.tick_params(axis='both', which='both', length=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_width)

    # Colorbar
    cbar_ax1 = ax1.inset_axes([0.15, 0.05, 0.7, 0.04])
    bbox = FancyBboxPatch((0.12, 0.03), 0.76, 0.08, transform=ax1.transAxes,
                          facecolor='white', edgecolor='black', linewidth=1,
                          boxstyle='round,pad=0.01', zorder=1)
    ax1.add_patch(bbox)
    cbar1 = plt.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_ticks([-1, 0, 1])
    cbar1.set_ticklabels(['-1', '0', '1'])
    cbar1.ax.tick_params(size=0, labelsize=9)
    cbar_ax1.set_zorder(2)

    # === ANNOTATION (l): Selection criteria ===
    # 1. Target depth range - bracket on right side using ax.annotate
    depth_start = int(img_height * 0.20)
    depth_end = int(img_height * 0.85)
    
    # Draw bracket using plot
    bracket_x = img_width + 15
    ax1.plot([bracket_x, bracket_x], [depth_start, depth_end], 
             color='green', lw=2, clip_on=False)
    ax1.plot([bracket_x-8, bracket_x], [depth_start, depth_start], 
             color='green', lw=2, clip_on=False)
    ax1.plot([bracket_x-8, bracket_x], [depth_end, depth_end], 
             color='green', lw=2, clip_on=False)
    ax1.text(bracket_x + 10, (depth_start + depth_end)/2, 'Target\ndepth\nrange', 
             fontsize=8, color='green', ha='left', va='center',
             fontweight='bold', clip_on=False)

    # 2. Fault - arrow pointing to discontinuity
    fault_x, fault_y = int(img_width * 0.25), int(img_height * 0.5)
    ax1.annotate('Fault', xy=(fault_x, fault_y), xytext=(fault_x + 80, fault_y - 100),
                 fontsize=9, color='white', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='lime', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

    # 3. Chaotic basement boundary
    basement_x, basement_y = int(img_width * 0.65), int(img_height * 0.78)
    ax1.annotate('Chaotic\nbasement', xy=(basement_x, basement_y), 
                 xytext=(basement_x - 100, basement_y - 130),
                 fontsize=9, color='white', fontweight='bold',
                 arrowprops=dict(arrowstyle='->', color='cyan', lw=2),
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.8))

    # (b) Ground Truth
    ax2 = axes[1]
    ax2.set_title('(b) Ground Truth', fontsize=14, fontweight='bold', pad=10)
    gt_rgb = facies_to_rgb(ground_truth)
    ax2.imshow(gt_rgb, aspect='auto')
    ax2.set_xlabel('Distance (km)', fontsize=11)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels)
    ax2.set_yticks([])
    ax2.tick_params(axis='both', which='both', length=0)
    for spine in ax2.spines.values():
        spine.set_linewidth(spine_width)

    # Volume inset
    if scoordi is not None and ecoordi is not None and volume_img_path is not None:
        if os.path.isfile(volume_img_path):
            volume_img = np.array(Image.open(volume_img_path))
            inset_ax = ax2.inset_axes([0.68, 0.72, 0.28, 0.26])
            inset_ax.imshow(volume_img)
            vol_img_height, vol_img_width = volume_img.shape[:2]
            scale_cx = vol_img_width / nx_volume
            scale_cy = vol_img_height / ny_volume
            px1, py1 = scoordi.cx * scale_cx, scoordi.cy * scale_cy
            px2, py2 = ecoordi.cx * scale_cx, ecoordi.cy * scale_cy
            inset_ax.plot([px1, px2], [py1, py2], color='black', linewidth=2)
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            for spine in inset_ax.spines.values():
                spine.set_linewidth(1)
                spine.set_color('black')

    # (c) Prediction
    ax3 = axes[2]
    ax3.set_title('(c) Prediction', fontsize=14, fontweight='bold', pad=10)
    pred_rgb = facies_to_rgb(prediction)
    ax3.imshow(pred_rgb, aspect='auto')
    ax3.set_xlabel('Distance (km)', fontsize=11)
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels)
    ax3.set_yticks([])
    ax3.tick_params(axis='both', which='both', length=0)
    for spine in ax3.spines.values():
        spine.set_linewidth(spine_width)

    # === ANNOTATION (m): Highlight differences between b and c ===
    diff_mask = (ground_truth != prediction) & (ground_truth != 0) & (prediction != 0)
    
    from scipy import ndimage
    labeled, num_features = ndimage.label(diff_mask)
    if num_features > 0:
        sizes = ndimage.sum(diff_mask, labeled, range(1, num_features + 1))
        # Get top 2 largest regions
        sorted_idx = np.argsort(sizes)[::-1]
        
        for i, idx in enumerate(sorted_idx[:2]):
            region = labeled == (idx + 1)
            ys, xs = np.where(region)
            if len(xs) > 50:  # Only mark significant differences
                cx, cy = xs.mean(), ys.mean()
                radius = max(30, min(np.std(xs), np.std(ys)) * 1.5)
                
                # Draw dashed circle on both panels
                for ax in [ax2, ax3]:
                    circle = plt.Circle((cx, cy), radius, fill=False, 
                                       edgecolor='red', linewidth=2, linestyle='--')
                    ax.add_patch(circle)
                
                # Label on panel c only
                if i == 0:
                    ax3.annotate('Misclassified', xy=(cx, cy - radius), 
                                xytext=(cx + 60, cy - radius - 80),
                                fontsize=8, color='red', fontweight='bold',
                                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                                         alpha=0.9, edgecolor='red'))

    # Legend
    legend_elements = [
        Patch(facecolor='#9c9c9c', edgecolor='black', label='Unclassified'),
        Patch(facecolor='#0000FF', edgecolor='black', label='Basement'),
        Patch(facecolor='#FF0000', edgecolor='black', label='Igneous'),
        Patch(facecolor='#FFA500', edgecolor='black', label='Shale'),
        Patch(facecolor='#FFFF00', edgecolor='black', label='Sand'),
    ]
    legend = ax3.legend(handles=legend_elements, loc='upper right',
                        fontsize=10, framealpha=0.9, edgecolor='black',
                        handlelength=1.4, handleheight=1.2,
                        borderpad=0.5, labelspacing=0.4)
    legend.get_frame().set_linewidth(1)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()
    print(f'Saved: {output_path}')


# Main
random.seed(123)
np.random.seed(123)

target_dim = 256
crop_size = (768, 512)
device = torch.device('cuda:0')

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
print(f'Volume shape: {volume.shape}')

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
state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
network.load_state_dict(state['network'])
network.eval()
print('Loaded checkpoint')

nt = crop_size[1]

with torch.no_grad():
    while True:
        vt, scoordi, ecoordi = extract_random_line(volume_sv, nt, vdt, tdt, mode='bilinear')
        ft, _, _ = extract_random_line(facies_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='nearest')
        ft = np.round(ft).astype(np.int32)
        # Unlike the other evaluation scripts, the depth window is always the top
        # crop_size[0] samples (no random depth offset).
        if np.any(ft[:crop_size[0]] != 0):
            break

    vt, ft = vt[:crop_size[0]], ft[:crop_size[0]]
    ip, _, _ = extract_random_line(inst_phase_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='bilinear')
    if_, _, _ = extract_random_line(inst_freq_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode='bilinear')
    ip, if_ = ip[:crop_size[0]], if_[:crop_size[0]]
    
    # Rescale the amplitude crop to RMS 0.15 before inference.
    rms = (vt * vt).mean() ** 0.5
    if rms > 0:
        vt = vt / rms * 0.15

    vt_tensor = torch.tensor(vt[None, None], dtype=torch.float32, device=device)
    # 2-channel info input: instantaneous phase, instantaneous frequency.
    info_tensor = torch.tensor(np.stack([ip, if_])[None], dtype=torch.float32, device=device)

    fo = predict_section(network, vt_tensor, info_tensor, device, target_dim)
    # Blank predictions where there is no ground-truth label (class 0).
    fo[ft == 0] = 0
    
    output_dir = os.path.join(get_project_root(), 'test', 'validation_figures', 'tdt_{:.1f}'.format(tdt))
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'annotated_example.png')
    volume_img_path = os.path.join(output_dir, 'devided_volume.png')
    
    create_annotated_figure(
        vt, ft, fo, output_path,
        scoordi=scoordi, ecoordi=ecoordi,
        volume_img_path=volume_img_path,
        nx_volume=nx, ny_volume=ny,
        tdt=tdt
    )
