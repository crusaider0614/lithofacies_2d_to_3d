"""Validation figures: seismic / labeled lithofacies / prediction on synthetic 2D lines.

For each (tdt, tag, epoch) in ``tdt_configs`` (by default only TDT = 25.0 m,
checkpoint/lithofacies_prediction_25.0_pat_044, the minimum-validation-loss epoch), draws
``n_samples`` random synthetic 2D lines from the validation part of the 3D volume
(CF.DATASET.VALID_IDX), crops each to 768 samples x 512 traces at a random labeled depth,
predicts it with overlapping 256 x 256 tiles, and saves a 3-panel paper-style figure:
(a) seismic amplitude, (b) labeled lithofacies with an optional map inset showing the line
location, (c) prediction. Predictions are blanked where the label is 0 (unlabeled).

This module also provides the shared helpers imported by evaluate_accuracy.py,
evaluate_depth_accuracy.py and create_annotated_figure.py: ``predict_section``,
``extract_random_line``, ``facies_to_rgb`` (and ``add_weighted_avg``,
``get_random_boundary_idx``). Importing it has no side effects other than matplotlib
rcParams; ``main()`` only runs when executed as a script.

Reads:  config/config_lithofacies.yaml, the checkpoint(s) in ``tdt_configs``,
        data/<DATA_POOL>/<VOLUME_TAG>.npy, <FACIES_TAG>.npy,
        <VOLUME_TAG>_inst_phase.npy, <VOLUME_TAG>_inst_freq.npy, and optionally
        test/validation_figures/tdt_<TDT>/devided_volume.png (map image for the inset; the
        inset is skipped if the file is missing; it is not produced by this repo).
Writes: test/validation_figures/tdt_<TDT>/validation_sample_<NNN>.png and
        test/validation_figures/tdt_<TDT>/line_coordinates.txt (start/end grid coordinates
        of every sampled line).

Run from the repo root (running by file path fails because the repo root is then not on
sys.path):

    python -m test.create_validation_figures

There is no CLI. Edit the settings at the top of ``main()`` (``target_dim``, ``crop_size``,
``n_samples``, ``device``, ``tdt_configs``). No random seed is set, so each run draws
different lines.
"""
import os
import random
import numpy as np
import torch
import yacs.config
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import ListedColormap
from PIL import Image

from module.seismic_data import SeismicVolume, Coordinate, distance
from network.coordi_network import get_gen_model
from utils.project import get_project_root


# Font settings for paper figures
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 300


def add_weighted_avg(average, weight, new_sample, new_weight):
    """Weighted average for overlapping predictions.

    Folds one tile's logits ``new_sample`` (C, H, W) with window ``new_weight`` (H, W) into
    the running average ``average`` whose accumulated weight is ``weight``. Returns the
    updated (average, weight).
    """
    new_average = (weight[None] * average + new_weight[None] * new_sample) / (weight[None] + new_weight[None])
    return new_average, weight + new_weight


# Lithofacies color mapping (based on 3dn_facies_unc labels)
# Labels: 0 (unclassified), 2 (Basement), 3 (Igneous), 4 (Shale), 5 (Sand)
LITHO_COLORS = {
    0: '#9c9c9c',   # Unclassified (gray)
    2: '#0000FF',   # Basement (blue)
    3: '#FF0000',   # Igneous (red)
    4: '#FFA500',   # Shale (orange)
    5: '#FFFF00',   # Sand (yellow)
}

LITHO_NAMES = {
    0: 'Unclassified',
    2: 'Basement',
    3: 'Igneous',
    4: 'Shale',
    5: 'Sand',
}


def facies_to_rgb(facies_array):
    """Convert facies class indices to RGB image"""
    h, w = facies_array.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    for cls, color in LITHO_COLORS.items():
        mask = facies_array == cls
        # Convert hex to RGB
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        rgb[mask] = [r, g, b]

    return rgb


def predict_section(network, vt, info, device, target_dim=256):
    """Predict lithofacies for a 2D section with sliding window.

    ``vt`` is a (1, 1, nz, nt) amplitude tensor and ``info`` a (1, 2, nz, nt) tensor of
    instantaneous phase/frequency, both already on ``device``. The section is covered by
    target_dim x target_dim tiles (the training crop size) overlapping by about 50 %; each
    tile's 6-class logits are weighted by a 2D raised-cosine window (~1 at the centre, ~0 at
    the edges) and averaged, which removes seams at tile borders. Returns the (nz, nt) int32
    argmax class map. Assumes nz, nt >= target_dim.
    """
    weight = np.array([(1 + np.cos((i - (target_dim // 2 - 0.5)) / (target_dim // 2) * np.pi)) / 2
                       for i in range(target_dim)])
    weight = weight[None] * weight[:, None]

    nz, nt = vt.shape[-2], vt.shape[-1]
    # Tile counts along depth and along the line; tile starts are spread evenly so the
    # first and last tiles align with the section edges.
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

                vt_crop = vt[:, :, sz:ez, st:et]
                info_crop = info[:, :, sz:ez, st:et]

                fo_crop = network(vt_crop, info_crop).cpu().numpy().squeeze()
                fo[:, sz:ez, st:et], dp[sz:ez, st:et] = add_weighted_avg(
                    fo[:, sz:ez, st:et],
                    dp[sz:ez, st:et],
                    fo_crop,
                    weight
                )

    return np.argmax(fo, axis=0).astype(np.int32)


def get_random_boundary_idx(nx, ny, exclude_edge=None):
    """Get random boundary coordinate.

    Picks a random grid point on the boundary of an nx x ny grid. Edges are numbered
    1 (y = 0), 2 (x = nx - 1), 3 (y = ny - 1), 4 (x = 0); the edge is chosen with
    probability proportional to its length, and ``exclude_edge`` can be skipped so that
    start and end points lie on different edges. Returns (Coordinate, edge_id).
    """
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
    """Extract random seismic line from volume.

    Builds a synthetic 2D line of ``nt`` traces from the SeismicVolume ``volume_data``
    (3D trace spacing ``vdt``) whose nominal trace spacing is ``tdt``. A straight line is
    drawn between random points on two different edges of the survey; a segment of length
    (0.9-1.1) x (nt - 1) x tdt is cut from it at a random offset and resampled to ``nt``
    traces, so the actual trace spacing jitters by +/-10 % around ``tdt``.

    If ``scoordi`` and ``ecoordi`` are given (e.g. the coordinates returned by a previous
    call), exactly that segment is extracted; this is how facies and instantaneous
    attributes are sampled along the same line as the amplitude. ``mode`` is "bilinear"
    or "nearest" (use "nearest" for facies labels).

    Returns (line_data of shape (nz, nt), start Coordinate, end Coordinate).
    """
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


def create_validation_figure(seismic, ground_truth, prediction, output_path,
                             scoordi=None, ecoordi=None, volume_img_path=None,
                             nx_volume=667, ny_volume=1200, tdt=25.0):
    """
    Create paper-style validation figure with 3 panels (Julia style):
    (a) Seismic data
    (b) Labeled lithofacies (Ground truth) with volume location inset
    (c) Prediction with legend

    ``seismic``, ``ground_truth`` and ``prediction`` are (nz, nt) arrays. The inset in (b)
    is drawn only if ``scoordi``/``ecoordi`` are given and ``volume_img_path`` exists; the
    line is drawn by scaling grid coordinates by the image size / (nx_volume, ny_volume).
    The x axis is labeled in km using ``tdt``; the y axis assumes the panel spans 1-3 s TWT.
    The figure is saved to ``output_path``.
    """
    img_height, img_width = seismic.shape

    # Calculate aspect ratio to match reference figure (vertically elongated)
    # Reference: panels are ~200 width x ~600 height ratio
    panel_aspect = img_height / img_width  # data aspect ratio
    fig_width = 12  # inches total for 3 panels
    panel_width = fig_width / 3
    fig_height = panel_width * panel_aspect * 1.8  # make it taller (near square overall)

    fig, axes = plt.subplots(1, 3, figsize=(fig_width, fig_height), dpi=300)
    plt.subplots_adjust(wspace=0.08, left=0.08, right=0.98, bottom=0.08, top=0.94)

    # Calculate axis labels
    # X-axis: distance in km (TDT=25m per trace)
    distance_km = img_width * tdt / 1000.0  # convert to km
    x_ticks = [0, img_width/2, img_width]
    x_labels = ["0", f"{distance_km/2:.1f}", f"{distance_km:.1f}"]

    # Y-axis: two-way time in seconds
    # 768 samples = 2 seconds (1s to 3s range), dt = 2.6ms
    # Time increases downward: top=1s, bottom=3s
    dt_sec = 2.0 / img_height  # 2 seconds for full height
    t_start = 1.0  # starting time in seconds
    t_end = t_start + img_height * dt_sec  # ending time

    y_ticks = [0, img_height/4, img_height/2, 3*img_height/4, img_height]
    y_labels = [f"{t_start + i * (t_end - t_start) / 4:.1f}" for i in range(5)]

    # Common spine settings
    spine_width = 2

    # (a) Seismic Data
    ax1 = axes[0]
    ax1.set_title("(a) Seismic data", fontsize=14, fontweight='bold', pad=10)

    # Normalize seismic for display
    seismic_normalized = seismic / (np.abs(seismic).max() + 1e-8)
    im1 = ax1.imshow(seismic_normalized, cmap='seismic', vmin=-1, vmax=1, aspect='auto')
    ax1.set_ylabel("Two-way time (s)", fontsize=11)
    ax1.set_xlabel("Distance (km)", fontsize=11)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels(x_labels)
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels(y_labels)
    ax1.tick_params(axis='both', which='both', length=0)
    for spine in ax1.spines.values():
        spine.set_linewidth(spine_width)

    # Colorbar for seismic (inside the panel, at bottom) with white background box
    cbar_ax1 = ax1.inset_axes([0.15, 0.05, 0.7, 0.04])
    # Add white background box behind colorbar
    from matplotlib.patches import FancyBboxPatch
    bbox = FancyBboxPatch((0.12, 0.03), 0.76, 0.08, transform=ax1.transAxes,
                          facecolor='white', edgecolor='black', linewidth=1,
                          boxstyle='round,pad=0.01', zorder=1)
    ax1.add_patch(bbox)
    cbar1 = plt.colorbar(im1, cax=cbar_ax1, orientation='horizontal')
    cbar1.set_ticks([-1, 0, 1])
    cbar1.set_ticklabels(['-1', '0', '1'])
    cbar1.ax.tick_params(size=0, labelsize=9)
    cbar_ax1.set_zorder(2)

    # (b) Labeled Lithofacies (Ground Truth)
    ax2 = axes[1]
    ax2.set_title("(b) Labeled lithofacies", fontsize=14, fontweight='bold', pad=10)
    gt_rgb = facies_to_rgb(ground_truth)
    ax2.imshow(gt_rgb, aspect='auto')
    ax2.set_xlabel("Distance (km)", fontsize=11)
    ax2.set_xticks(x_ticks)
    ax2.set_xticklabels(x_labels)
    ax2.set_yticks([])
    ax2.tick_params(axis='both', which='both', length=0)
    for spine in ax2.spines.values():
        spine.set_linewidth(spine_width)

    # Add volume location inset (like Julia code)
    if scoordi is not None and ecoordi is not None and volume_img_path is not None:
        if os.path.isfile(volume_img_path):
            volume_img = np.array(Image.open(volume_img_path))

            # Inset axes position (top-right corner, exactly matching legend position)
            # [x, y, width, height] in axes coordinates (0-1)
            inset_ax = ax2.inset_axes([0.64, 0.80, 0.34, 0.18])
            inset_ax.imshow(volume_img)

            # Get image dimensions for scaling
            vol_img_height, vol_img_width = volume_img.shape[:2]

            # Scale coordinates to image size
            # volume_img: width corresponds to nx, height corresponds to ny
            # cx is along nx direction, cy is along ny direction
            scale_cx = vol_img_width / nx_volume   # cx -> image x (width)
            scale_cy = vol_img_height / ny_volume  # cy -> image y (height)

            # Draw line showing where the 2D section was extracted
            # In imshow, x-axis is horizontal (width), y-axis is vertical (height)
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
    ax3.set_title("(c) Prediction", fontsize=14, fontweight='bold', pad=10)
    pred_rgb = facies_to_rgb(prediction)
    ax3.imshow(pred_rgb, aspect='auto')
    ax3.set_xlabel("Distance (km)", fontsize=11)
    ax3.set_xticks(x_ticks)
    ax3.set_xticklabels(x_labels)
    ax3.set_yticks([])
    ax3.tick_params(axis='both', which='both', length=0)
    for spine in ax3.spines.values():
        spine.set_linewidth(spine_width)

    # Legend (top-right, like Julia code)
    legend_elements = [
        Patch(facecolor='#9c9c9c', edgecolor='black', label='Unclassified'),
        Patch(facecolor='#0000FF', edgecolor='black', label='Basement'),
        Patch(facecolor='#FF0000', edgecolor='black', label='Igneous'),
        Patch(facecolor='#FFA500', edgecolor='black', label='Shale'),
        Patch(facecolor='#FFFF00', edgecolor='black', label='Sand'),
    ]

    legend = ax3.legend(handles=legend_elements,
                        bbox_to_anchor=(0.98, 0.98), loc='upper right',
                        fontsize=10, framealpha=0.9, edgecolor='black',
                        handlelength=1.4, handleheight=1.2,
                        borderpad=0.5, labelspacing=0.4)
    legend.get_frame().set_linewidth(1)

    # Save
    plt.savefig(output_path, dpi=300, bbox_inches='tight', pad_inches=0.1, facecolor='white')
    plt.close()

    print(f"Saved: {output_path}")


def main():
    """Generate validation figures for every entry in ``tdt_configs`` (see module docstring)."""
    # Configuration
    target_dim = 256
    crop_size = (768, 512)
    n_samples = 30
    device = torch.device("cuda:9")

    config_file = os.path.join(get_project_root(), "config", "config_lithofacies.yaml")
    with open(config_file, "rt") as f_read:
        CF = yacs.config.load_cfg(f_read)

    # Output directory
    output_dir = os.path.join(get_project_root(), "test", "validation_figures")
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    data_pool = CF.DATASET.DATA_POOL
    volume_tag = CF.DATASET.VOLUME_TAG
    facies_tag = CF.DATASET.FACIES_TAG
    valid_idx = CF.DATASET.VALID_IDX
    vdt = CF.DATASET.VDT

    print("Loading data...")
    volume = np.load(os.path.join(get_project_root(), "data", data_pool, volume_tag + ".npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]
    facies = np.load(os.path.join(get_project_root(), "data", data_pool, facies_tag + ".npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]
    inst_phase = np.load(os.path.join(get_project_root(), "data", data_pool, volume_tag + "_inst_phase.npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]
    inst_freq = np.load(os.path.join(get_project_root(), "data", data_pool, volume_tag + "_inst_freq.npy"), mmap_mode="r")[:, valid_idx[0]:valid_idx[1]]

    nz, nx, ny = volume.shape
    print(f"Volume shape: {volume.shape}")

    # Setup SeismicVolume classes
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

    # TDT configurations with corresponding checkpoints (25m only)
    tdt_configs = [
        (25.0, "lithofacies_prediction_25.0_pat", 44),
    ]

    for tdt, tag, epoch in tdt_configs:
        print(f"\n{'='*60}")
        print(f"Processing TDT = {tdt} m")
        print(f"{'='*60}")

        # Create TDT-specific output directory
        tdt_output_dir = os.path.join(output_dir, f"tdt_{tdt}")
        os.makedirs(tdt_output_dir, exist_ok=True)

        # Volume image path for inset
        volume_img_path = os.path.join(tdt_output_dir, "devided_volume.png")

        # Load network
        network = get_gen_model(CF, additional_channel=0).to(device)
        checkpoint_path = os.path.join(get_project_root(), "checkpoint", f"{tag}_{str(epoch).zfill(3)}")
        state = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        network.load_state_dict(state["network"])
        network = network.eval()
        print(f"Loaded checkpoint: {checkpoint_path}")

        # Coordinate log file
        coord_file = os.path.join(tdt_output_dir, "line_coordinates.txt")
        coord_log = open(coord_file, "w")
        coord_log.write("# Sample coordinates (Start -> End)\n")
        coord_log.write("# Format: sample_idx: Start:(X:x1, Y:y1) -> End:(X:x2, Y:y2)\n\n")

        # Generate figures
        nt = crop_size[1]

        with torch.no_grad():
            for idx in range(n_samples):
                print(f"Processing sample {idx + 1}/{n_samples}...")

                # Extract random line with coordinates
                while True:
                    vt, scoordi, ecoordi = extract_random_line(volume_sv, nt, vdt, tdt, mode="bilinear")
                    ft, _, _ = extract_random_line(facies_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode="nearest")
                    ft = np.round(ft).astype(np.int32)

                    # Crop in z direction
                    mz = nz // 50
                    sz = random.randint(0, nz - crop_size[0])
                    sz = np.clip(sz, 0, nz - crop_size[0])
                    ez = sz + crop_size[0]

                    if np.any(ft[sz:ez] != 0):
                        break

                vt = vt[sz:ez]
                ft = ft[sz:ez]

                # Extract additional info
                ip, _, _ = extract_random_line(inst_phase_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode="bilinear")
                ip = ip[sz:ez]
                if_, _, _ = extract_random_line(inst_freq_sv, nt, vdt, tdt, scoordi=scoordi, ecoordi=ecoordi, mode="bilinear")
                if_ = if_[sz:ez]

                # RMS normalization: rescale the amplitude crop to RMS 0.15 before inference
                rms = (vt * vt).mean() ** 0.5
                if rms > 0:
                    vt = vt / rms * 0.15

                # Prepare tensors
                vt_tensor = torch.tensor(vt[None, None], dtype=torch.float32, device=device)
                info_tensor = torch.tensor(np.stack([ip, if_])[None], dtype=torch.float32, device=device)

                # Predict
                fo = predict_section(network, vt_tensor, info_tensor, device, target_dim)

                # Apply mask: where ground truth is 0 (unclassified), prediction should also be 0
                fo[ft == 0] = 0

                # Log coordinates
                coord_log.write(f"{idx:03d}: Start:(X:{scoordi.cx:.2f}, Y:{scoordi.cy:.2f}) -> End:(X:{ecoordi.cx:.2f}, Y:{ecoordi.cy:.2f})\n")

                # Create figure
                output_path = os.path.join(tdt_output_dir, f"validation_sample_{idx:03d}.png")
                create_validation_figure(
                    vt, ft, fo, output_path,
                    scoordi=scoordi, ecoordi=ecoordi,
                    volume_img_path=volume_img_path,
                    nx_volume=nx, ny_volume=ny,
                    tdt=tdt
                )

        coord_log.close()
        print(f"Coordinates saved to: {coord_file}")

    print(f"\n{'='*60}")
    print("All validation figures generated!")
    print(f"Output directory: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
