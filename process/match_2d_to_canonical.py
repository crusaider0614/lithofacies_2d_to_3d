"""Match the frequency content of the real 2D lines to the 3D data before inference.

The networks are trained on lines cut from the 3D volume, so each real 2D line is first
shaped towards a canonical 3D wavelet (paper, Section 3.4):
  1. Canonical wavelet: sections extracted from the 3D volume (REFERENCE_3D_FILES) are
     resampled to the finest sample interval among them and combined, and a zero-phase
     wavelet is derived from their average amplitude spectrum within T0-T1 s
     (white-reflectivity assumption).
  2. For every 2D line in INPUT_2D_DIR: resample it to the canonical sample interval,
     derive its own zero-phase wavelet in the same window (the whole line if the window is
     empty), design a least-squares shaping filter (FILTER_LENGTH taps, PREWHITENING) that
     maps it to the canonical wavelet, apply the filter to the whole line, and write the
     result with the same file name to OUTPUT_2D_DIR.
The matched lines are then converted to <name>.npy + <name>.nphead for
test/test_actual_line.py with `python -m module.seismic_data` (which reads the *.segy files
in data/ssealine_matched/).

Reads:  data/matching_filter/line1_3d.segy, line2_3d.segy (3D reference sections),
        data/ssealine/*.segy, *.sgy (2D lines before matching).
Writes: data/ssealine_matched/<same file name> (matched SEG-Y; headers copied from the
        input line, sample interval set to the canonical one) and, in
        data/ssealine_matched/diagnostics/: 00_canonical.png, 00_canonical_wavelet.npy,
        line_<name>.png (per line, if SAVE_PER_LINE_FIG), summary.csv (peak, centroid and
        -6 dB band before/after matching and of the target) and 99_aggregate_spectra.png.
These SEG-Y data are not distributed with this repository.

Run from the repo root:

    python -m process.match_2d_to_canonical

There is no CLI. Edit the module-level settings below.
"""
import csv
import os
import re
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from module.matching_filter import (
    read_segy, write_segy_resampled, resample_traces, select_window,
    extract_zero_phase_wavelet, average_amplitude_spectrum, spectral_metrics,
    design_shaping_filter, apply_filter, safe_norm,
)
from utils.project import get_project_root


DATA_ROOT = Path(get_project_root()) / "data"
REFERENCE_3D_FILES = [
    DATA_ROOT / "matching_filter" / "line1_3d.segy",
    DATA_ROOT / "matching_filter" / "line2_3d.segy",
]

INPUT_2D_DIR = DATA_ROOT / "ssealine"
OUTPUT_2D_DIR = DATA_ROOT / "ssealine_matched"

T0 = 0.3  # analysis window for wavelet / spectrum estimation (s)
T1 = 2.0

WAVELET_LENGTH = 129  # samples (odd)
FILTER_LENGTH = 121  # shaping-filter taps (odd)
PREWHITENING = 1e-3

SAVE_PER_LINE_FIG = True
FMAX_PLOT = 125.0  # Hz


def build_canonical_reference(ref_files, t0, t1, wavelet_length):
    """Combine the 3D reference sections and derive the canonical wavelet and spectrum.

    Returns a dict with the combined data, its sample interval dt (the finest among the
    files), the T0-T1 window, the zero-phase wavelet, the average spectrum and its metrics.
    """
    assert len(ref_files) > 0, "REFERENCE_3D_FILES is empty"

    datasets = []
    dts = []
    for p in ref_files:
        if not p.exists():
            raise FileNotFoundError(f"Reference 3D file not found: {p}")
        d, dt = read_segy(p)
        datasets.append(d)
        dts.append(dt)
        print(f"  - {p.name}: shape={d.shape}, dt={dt*1000:.3f} ms")
    canonical_dt = min(dts)

    resampled = []
    for d, dt_i in zip(datasets, dts):
        if abs(dt_i - canonical_dt) / canonical_dt > 1e-3:
            d = resample_traces(d, dt_i, canonical_dt)
        resampled.append(d)

    # Cut all sections to the shortest one so they can be stacked trace-wise.
    min_n_samples = min(d.shape[1] for d in resampled)
    resampled = [d[:, :min_n_samples] for d in resampled]
    combined = np.concatenate(resampled, axis=0)
    print(f"  Combined reference: {combined.shape}, "
          f"dt = {canonical_dt*1000:.3f} ms")

    win = select_window(combined, canonical_dt, t0, t1)
    wavelet = extract_zero_phase_wavelet(win, wavelet_length)
    freqs, spec = average_amplitude_spectrum(win, canonical_dt)
    metrics = spectral_metrics(freqs, spec)
    return dict(data=combined, dt=canonical_dt, window=win,
                wavelet=wavelet, freqs=freqs, spec=spec, metrics=metrics)


def process_one_2d_line(in_path, out_path, canonical, t0, t1,
                        wavelet_length, filter_length, prewhitening,
                        fig_path=None, fmax_plot=125.0):
    """Match one 2D SEG-Y line to the canonical wavelet and write it to out_path.

    Returns the line's size, sample interval, spectral metrics and spectra before and after
    matching (all measured in the T0-T1 window).
    """
    d2d, dt2d = read_segy(in_path)

    if abs(dt2d - canonical['dt']) / canonical['dt'] > 1e-3:
        d2d = resample_traces(d2d, dt2d, canonical['dt'])
        dt2d = canonical['dt']

    try:
        win = select_window(d2d, dt2d, t0, t1)
    except ValueError as e:
        print(f"    Window selection failed ({e}); using the full line")
        win = d2d

    w2d = extract_zero_phase_wavelet(win, wavelet_length)
    f, sp_before = average_amplitude_spectrum(win, dt2d)
    m_before = spectral_metrics(f, sp_before)

    h = design_shaping_filter(w2d, canonical['wavelet'],
                              filter_length=filter_length,
                              prewhitening=prewhitening)
    d2d_matched = apply_filter(d2d, h)

    win_after = select_window(d2d_matched, dt2d, t0, t1) \
                if (t0 is not None or t1 is not None) else d2d_matched
    w2d_after = extract_zero_phase_wavelet(win_after, wavelet_length)
    _, sp_after = average_amplitude_spectrum(win_after, dt2d)
    m_after = spectral_metrics(f, sp_after)

    write_segy_resampled(out_path, in_path, d2d_matched, dt2d)

    if fig_path is not None:
        plot_per_line_diagnostic(canonical, w2d, w2d_after, h,
                                 f, sp_before, sp_after, dt2d,
                                 fig_path, fmax_plot, in_path.stem)

    return dict(line=in_path.name,
                n_traces=d2d.shape[0], n_samples=d2d.shape[1],
                dt_ms=dt2d*1000,
                m_before=m_before, m_after=m_after,
                spec_before=sp_before, spec_after=sp_after,
                freqs=f)


def plot_canonical(canonical, fig_dir, fmax_plot):
    """Plot the canonical wavelet and its average spectrum (00_canonical.png)."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    dt = canonical['dt']

    w = canonical['wavelet']
    t = (np.arange(len(w)) - len(w)//2) * dt * 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, w, 'k-', lw=2)
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set(xlabel='Time (ms)', ylabel='Amplitude',
              title='Canonical 3D wavelet (target)')
    ax[0].grid(alpha=0.3)

    f = canonical['freqs']
    sp = canonical['spec']
    mask = f <= fmax_plot
    ax[1].plot(f[mask], safe_norm(sp)[mask], 'k-', lw=2)
    ax[1].set(xlabel='Frequency (Hz)', ylabel='Normalized amplitude',
              title='Canonical 3D average spectrum')
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_dir / '00_canonical.png', dpi=150)
    plt.close()


def plot_per_line_diagnostic(canonical, w2d, w2d_after, h,
                             freqs, sp_before, sp_after, dt,
                             outfile, fmax_plot, line_name):
    """Per-line figure: wavelets, average spectra (canonical / before / after) and the filter."""
    w_tgt = canonical['wavelet']
    sp_tgt = canonical['spec']
    f_tgt = canonical['freqs']

    fig, ax = plt.subplots(1, 3, figsize=(16, 4))

    t_w = (np.arange(len(w_tgt)) - len(w_tgt)//2) * dt * 1000
    ax[0].plot(t_w, w_tgt, 'k-', lw=2, label='Canonical 3D')
    ax[0].plot(t_w, w2d, 'r--', lw=1.3, label='2D before')
    ax[0].plot(t_w, w2d_after, 'b-', lw=1.3, label='2D after')
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set(xlabel='Time (ms)', ylabel='Amplitude',
              title=f'Wavelets - {line_name}')
    ax[0].legend(fontsize=8)
    ax[0].grid(alpha=0.3)

    mask_t = f_tgt <= fmax_plot
    mask = freqs <= fmax_plot
    ax[1].plot(f_tgt[mask_t], safe_norm(sp_tgt)[mask_t], 'k-', lw=2,
               label='Canonical 3D')
    ax[1].plot(freqs[mask], safe_norm(sp_before)[mask], 'r--', lw=1.3,
               label='2D before')
    ax[1].plot(freqs[mask], safe_norm(sp_after)[mask], 'b-', lw=1.3,
               label='2D after')
    ax[1].set(xlabel='Frequency (Hz)', ylabel='Normalized amplitude',
              title='Average amplitude spectra')
    ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)

    t_h = (np.arange(len(h)) - len(h)//2) * dt * 1000
    ax[2].plot(t_h, h, 'b-')
    ax[2].axhline(0, color='gray', lw=0.5)
    ax[2].set(xlabel='Time (ms)', ylabel='Amplitude',
              title=f'Shaping filter h (length={len(h)})')
    ax[2].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_aggregate_spectra(canonical, results, fig_dir, fmax_plot):
    """Spectra of all lines before and after matching against the canonical spectrum."""
    f_tgt = canonical['freqs']
    sp_tgt = canonical['spec']
    mask_t = f_tgt <= fmax_plot

    fig, ax = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

    for r in results:
        m = r['freqs'] <= fmax_plot
        ax[0].plot(r['freqs'][m], safe_norm(r['spec_before'])[m],
                   color='red', alpha=0.15, lw=0.7)
    ax[0].plot(f_tgt[mask_t], safe_norm(sp_tgt)[mask_t],
               'k-', lw=2.5, label='Canonical 3D')
    ax[0].set(xlabel='Frequency (Hz)', ylabel='Normalized amplitude',
              title=f'All 2D lines BEFORE matching (n={len(results)})')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    for r in results:
        m = r['freqs'] <= fmax_plot
        ax[1].plot(r['freqs'][m], safe_norm(r['spec_after'])[m],
                   color='blue', alpha=0.15, lw=0.7)
    ax[1].plot(f_tgt[mask_t], safe_norm(sp_tgt)[mask_t],
               'k-', lw=2.5, label='Canonical 3D')
    ax[1].set(xlabel='Frequency (Hz)',
              title='All 2D lines AFTER matching')
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / '99_aggregate_spectra.png', dpi=150)
    plt.close()


def save_summary_csv(results, canonical, csv_path):
    """One row per line: size, dt and spectral metrics before/after matching and of the target."""
    m_tgt = canonical['metrics']
    with open(csv_path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow([
            'line', 'n_traces', 'n_samples', 'dt_ms',
            'peak_before_Hz', 'centroid_before_Hz',
            'bw6lo_before_Hz', 'bw6hi_before_Hz',
            'peak_after_Hz', 'centroid_after_Hz',
            'bw6lo_after_Hz', 'bw6hi_after_Hz',
            'peak_target_Hz', 'centroid_target_Hz',
            'bw6lo_target_Hz', 'bw6hi_target_Hz',
        ])
        for r in results:
            mb, ma = r['m_before'], r['m_after']
            w.writerow([
                r['line'], r['n_traces'], r['n_samples'],
                f"{r['dt_ms']:.4f}",
                f"{mb['peak_hz']:.3f}", f"{mb['centroid_hz']:.3f}",
                f"{mb['bw_lo_hz']:.3f}", f"{mb['bw_hi_hz']:.3f}",
                f"{ma['peak_hz']:.3f}", f"{ma['centroid_hz']:.3f}",
                f"{ma['bw_lo_hz']:.3f}", f"{ma['bw_hi_hz']:.3f}",
                f"{m_tgt['peak_hz']:.3f}", f"{m_tgt['centroid_hz']:.3f}",
                f"{m_tgt['bw_lo_hz']:.3f}", f"{m_tgt['bw_hi_hz']:.3f}",
            ])


def numeric_sort_key(p):
    """Sort file names by the first number in them (1, 2, ..., 10), others after."""
    m = re.search(r'(\d+)', p.stem)
    return (0, int(m.group(1))) if m else (1, p.stem)


def main():
    OUTPUT_2D_DIR.mkdir(parents=True, exist_ok=True)
    fig_dir = OUTPUT_2D_DIR / 'diagnostics'
    fig_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("[Step 1] Extracting the canonical 3D wavelet")
    print("=" * 70)
    canonical = build_canonical_reference(
        REFERENCE_3D_FILES, T0, T1, WAVELET_LENGTH)
    m = canonical['metrics']
    print(f"  Canonical metrics: peak={m['peak_hz']:.2f} Hz, "
          f"centroid={m['centroid_hz']:.2f} Hz, "
          f"-6dB={m['bw_lo_hz']:.1f}-{m['bw_hi_hz']:.1f} Hz")
    plot_canonical(canonical, fig_dir, FMAX_PLOT)
    np.save(fig_dir / '00_canonical_wavelet.npy', canonical['wavelet'])

    print("\n" + "=" * 70)
    print(f"[Step 2] Batch matching of 2D lines   (input: {INPUT_2D_DIR})")
    print("=" * 70)
    if not INPUT_2D_DIR.exists():
        print(f"Input directory not found: {INPUT_2D_DIR}")
        return

    segy_files = sorted(
        list(INPUT_2D_DIR.glob('*.segy')) + list(INPUT_2D_DIR.glob('*.sgy')),
        key=numeric_sort_key)
    if not segy_files:
        print(f"No SEG-Y files found in {INPUT_2D_DIR}")
        return
    print(f"  Found {len(segy_files)} SEG-Y file(s).\n")

    results = []
    for i, in_path in enumerate(segy_files, 1):
        out_path = OUTPUT_2D_DIR / in_path.name
        fig_path = (fig_dir / f"line_{in_path.stem}.png"
                    if SAVE_PER_LINE_FIG else None)
        print(f"  [{i:>3d}/{len(segy_files)}] {in_path.name} ...", end=' ')
        try:
            r = process_one_2d_line(
                in_path, out_path, canonical,
                T0, T1, WAVELET_LENGTH, FILTER_LENGTH, PREWHITENING,
                fig_path=fig_path, fmax_plot=FMAX_PLOT)
            results.append(r)
            mb = r['m_before']
            ma = r['m_after']
            print(f"OK  (peak {mb['peak_hz']:5.1f} -> {ma['peak_hz']:5.1f} Hz)")
        except Exception as e:
            print(f"FAIL  ({e})")

    if not results:
        print("No lines were processed.")
        return

    print("\n" + "=" * 70)
    print("[Step 3] Writing the summary")
    print("=" * 70)
    csv_path = fig_dir / 'summary.csv'
    save_summary_csv(results, canonical, csv_path)
    print(f"  CSV : {csv_path}")
    plot_aggregate_spectra(canonical, results, fig_dir, FMAX_PLOT)
    print(f"  Aggregate spectra plot : {fig_dir / '99_aggregate_spectra.png'}")

    print(f"\nMatched {len(results)}/{len(segy_files)} line(s).")
    print(f"  Matched SEG-Y : {OUTPUT_2D_DIR}")
    print(f"  Diagnostics   : {fig_dir}")


if __name__ == "__main__":
    main()
