"""Matching filter on co-located 3D and 2D lines (paper, Fig. 14).

For each pair in PAIRS (a section extracted from the 3D volume along a 2D line, and that
2D line), both sections are brought to the finer of the two sample intervals
(RESAMPLE_TO_FINER_DT), and for each direction in DIRECTIONS a shaping filter is designed
from zero-phase wavelets estimated in the T0-T1 s window and applied to the source section:
  'to_3d': the 2D line is matched to the 3D section (the direction used for inference;
           process/match_2d_to_canonical.py does the same against a canonical 3D wavelet);
  'to_2d': the 3D section is matched to the 2D line (for comparison).
Spectral metrics (peak, centroid, -6 dB band) are printed for the target and for the
source before and after matching.

Reads:  data/matching_filter/line1_3d.segy, line1_2d.segy, line2_3d.segy, line2_2d.segy
        (not distributed with this repository).
Writes: test/validation_figures/matching_filter/results_<line>/<direction>/matched.segy and
        figures/01_wavelets.png, 02_spectra.png, 03_amp_hist.png,
        04_section_before_after.png, 05_filter_response.png, 06_metrics.txt;
        test/validation_figures/matching_filter/combined_lines.png (spectra and sections of
        all pairs for COMBINED_DIRECTION, as in Fig. 14).

Run from the repo root:

    python -m test.seismic_matching_filter

There is no CLI. Edit the module-level settings below.
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from module.matching_filter import (
    read_segy, write_segy_resampled, resample_traces, select_window,
    extract_zero_phase_wavelet, average_amplitude_spectrum, spectral_metrics,
    design_shaping_filter, apply_filter, safe_norm,
)
from utils.project import get_project_root


SEGY_DIR = Path(get_project_root()) / "data" / "matching_filter"
OUTPUT_DIR = Path(get_project_root()) / "test" / "validation_figures" / "matching_filter"
# (3D section, 2D line along the same path, output sub-directory)
PAIRS = [
    ("line1_3d.segy", "line1_2d.segy", "results_line1"),
    ("line2_3d.segy", "line2_2d.segy", "results_line2"),
]

DIRECTIONS = ['to_3d', 'to_2d']

T0 = 0.3  # analysis window for wavelet / spectrum estimation (s)
T1 = 2.0

WAVELET_LENGTH = 129  # samples (odd)
FILTER_LENGTH = 121  # shaping-filter taps (odd)
PREWHITENING = 1e-3

RESAMPLE_TO_FINER_DT = True

FMAX_PLOT = 125.0  # Hz
SECTION_CLIP = 99  # percentile for the colour scale of section plots

SAVE_COMBINED_FIG = True
COMBINED_FIG_PATH = OUTPUT_DIR / "combined_lines.png"
COMBINED_DIRECTION = 'to_3d'
COMBINED_LINE_NAMES = ["Arbitrary line 1", "Arbitrary line 2"]


def plot_wavelets(w_target, w_src_before, w_src_after, dt, outfile,
                  target_label='target', source_label='source', title_suffix=""):
    """Target wavelet against the source wavelet before (left) and after (right) matching."""
    t = (np.arange(len(w_target)) - len(w_target)//2) * dt * 1000
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    ax[0].plot(t, w_target, 'k-', lw=2, label=f'{target_label} target')
    ax[0].plot(t, w_src_before, 'r--', lw=1.5, label=f'{source_label} before')
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set(xlabel='Time (ms)', ylabel='Amplitude',
              title=f'Wavelets - Before matching {title_suffix}')
    ax[0].legend()
    ax[0].grid(alpha=0.3)

    ax[1].plot(t, w_target, 'k-', lw=2, label=f'{target_label} target')
    ax[1].plot(t, w_src_after, 'b-', lw=1.5, label=f'{source_label} after')
    ax[1].axhline(0, color='gray', lw=0.5)
    ax[1].set(xlabel='Time (ms)',
              title=f'Wavelets - After matching {title_suffix}')
    ax[1].legend()
    ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_spectra(freqs, sp_target, sp_src_before, sp_src_after, fmax, outfile,
                 target_label='target', source_label='source', title_suffix=""):
    """Normalised average amplitude spectra of the target and the source before/after."""
    mask = freqs <= fmax
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(freqs[mask], safe_norm(sp_target)[mask], 'k-', lw=2,
            label=f'{target_label} target')
    ax.plot(freqs[mask], safe_norm(sp_src_before)[mask], 'r--', lw=1.5,
            label=f'{source_label} before')
    ax.plot(freqs[mask], safe_norm(sp_src_after)[mask], 'b-', lw=1.5,
            label=f'{source_label} after')
    ax.set(xlabel='Frequency (Hz)', ylabel='Normalized amplitude',
           title=f'Average amplitude spectra {title_suffix}')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_amplitude_histograms(d_target, d_src_before, d_src_after, outfile,
                              target_label='target', source_label='source',
                              title_suffix=""):
    """Amplitude histograms (log density) of the target and the source before/after."""
    fig, ax = plt.subplots(figsize=(8, 5))
    lo = min(d_target.min(), d_src_before.min(), d_src_after.min())
    hi = max(d_target.max(), d_src_before.max(), d_src_after.max())
    bins = np.linspace(lo, hi, 100)
    ax.hist(d_target.flatten(), bins=bins, alpha=0.5,
            label=f'{target_label} target', density=True)
    ax.hist(d_src_before.flatten(), bins=bins, alpha=0.5,
            label=f'{source_label} before', density=True)
    ax.hist(d_src_after.flatten(), bins=bins, alpha=0.5,
            label=f'{source_label} after', density=True)
    ax.set(xlabel='Amplitude', ylabel='Density (log)',
           title=f'Amplitude distributions {title_suffix}')
    ax.set_yscale('log')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_sections(d_target, d_src_before, d_src_after, dt, outfile,
                  clip_pct=99, target_label='target', source_label='source',
                  title_suffix=""):
    """Target section, source before and after matching, and their difference."""
    n_t_t, n_t_s = d_target.shape
    n_s_t, n_s_s = d_src_before.shape

    extent_t = [0, n_t_t, n_t_s * dt, 0]
    extent_s = [0, n_s_t, n_s_s * dt, 0]

    clip_t = max(np.percentile(np.abs(d_target), clip_pct), 1e-12)
    clip_s = max(np.percentile(np.abs(d_src_before), clip_pct),
                 np.percentile(np.abs(d_src_after), clip_pct), 1e-12)

    diff = d_src_after - d_src_before
    clip_d = max(np.percentile(np.abs(diff), clip_pct), 1e-12)

    fig, ax = plt.subplots(1, 4, figsize=(22, 6), sharey=True)

    im0 = ax[0].imshow(d_target.T, cmap='seismic', vmin=-clip_t, vmax=clip_t,
                       aspect='auto', extent=extent_t)
    ax[0].set(title=f'{target_label} original {title_suffix}',
              xlabel='Trace', ylabel='Time (s)')
    plt.colorbar(im0, ax=ax[0], fraction=0.04, pad=0.02)

    im1 = ax[1].imshow(d_src_before.T, cmap='seismic',
                       vmin=-clip_s, vmax=clip_s, aspect='auto', extent=extent_s)
    ax[1].set(title=f'{source_label} before matching {title_suffix}',
              xlabel='Trace')
    plt.colorbar(im1, ax=ax[1], fraction=0.04, pad=0.02)

    im2 = ax[2].imshow(d_src_after.T, cmap='seismic',
                       vmin=-clip_s, vmax=clip_s, aspect='auto', extent=extent_s)
    ax[2].set(title=f'{source_label} after matching {title_suffix}',
              xlabel='Trace')
    plt.colorbar(im2, ax=ax[2], fraction=0.04, pad=0.02)

    im3 = ax[3].imshow(diff.T, cmap='seismic',
                       vmin=-clip_d, vmax=clip_d, aspect='auto', extent=extent_s)
    ax[3].set(title=f'Difference (after - before) {title_suffix}',
              xlabel='Trace')
    plt.colorbar(im3, ax=ax[3], fraction=0.04, pad=0.02)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_filter(h, w_in, w_out, dt, outfile,
                target_label='target', source_label='source', title_suffix=""):
    """The shaping filter, and h * w_source compared with the target wavelet."""
    test = np.convolve(w_in, h, mode='full')
    pad = (len(test) - len(w_out)) // 2
    test_aligned = test[pad : pad + len(w_out)]

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    t_h = (np.arange(len(h)) - len(h)//2) * dt * 1000
    ax[0].plot(t_h, h, 'b-')
    ax[0].axhline(0, color='gray', lw=0.5)
    ax[0].set(xlabel='Time (ms)', ylabel='Amplitude',
              title=f'Shaping filter h (length={len(h)}) {title_suffix}')
    ax[0].grid(alpha=0.3)

    t_w = (np.arange(len(w_out)) - len(w_out)//2) * dt * 1000
    ax[1].plot(t_w, safe_norm(w_out), 'k-', lw=2,
               label=f'Target w_{target_label}')
    ax[1].plot(t_w, safe_norm(test_aligned), 'r--', lw=1.5,
               label=f'h * w_{source_label}')
    ax[1].axhline(0, color='gray', lw=0.5)
    ax[1].set(xlabel='Time (ms)', title=f'Filter validation {title_suffix}')
    ax[1].legend()
    ax[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()


def plot_combined_lines(line_results, outfile,
                        target_label='3D', source_label='2D',
                        fmax=125.0, clip_pct=99):
    """One row per line: spectra, target section, source before and after matching (Fig. 14)."""
    n_lines = len(line_results)
    if n_lines == 0:
        print("  plot_combined_lines: no lines to display")
        return

    fig, axes = plt.subplots(n_lines, 4,
                             figsize=(22, 5 * n_lines),
                             gridspec_kw={'width_ratios': [1.0, 1.3, 1.3, 1.3]})
    if n_lines == 1:
        axes = axes[np.newaxis, :]

    def _show(ax, d, dt, clip, title, ylabel=None, add_cbar=False):
        extent = [0, d.shape[0], d.shape[1] * dt, 0]
        im = ax.imshow(d.T, cmap='seismic', vmin=-clip, vmax=clip,
                       aspect='auto', extent=extent)
        ax.set(title=title, xlabel='Trace')
        if ylabel:
            ax.set_ylabel(ylabel)
        if add_cbar:
            plt.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        return im

    for row, (line_title, r) in enumerate(line_results):
        dt = r['dt']
        freqs = r['freqs']
        d_tgt = r['section_target']
        d_bef = r['section_source_before']
        d_aft = r['section_source_after']

        ax = axes[row, 0]
        mask = freqs <= fmax
        ax.plot(freqs[mask], safe_norm(r['spec_target'])[mask], 'k-', lw=2,
                label=f'{target_label} target')
        ax.plot(freqs[mask], safe_norm(r['spec_source_before'])[mask], 'r--', lw=1.5,
                label=f'{source_label} before')
        ax.plot(freqs[mask], safe_norm(r['spec_source_after'])[mask], 'b-', lw=1.5,
                label=f'{source_label} after')
        ax.set(xlabel='Frequency (Hz)', ylabel='Normalized amplitude',
               title=f'{line_title} - average amplitude spectra')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        clip_tgt = max(np.percentile(np.abs(d_tgt), clip_pct), 1e-12)
        clip_2d = max(np.percentile(np.abs(d_bef), clip_pct),
                      np.percentile(np.abs(d_aft), clip_pct), 1e-12)

        _show(axes[row, 1], d_tgt, dt, clip_tgt,
              title=f'{line_title} - {target_label} target',
              ylabel='Time (s)', add_cbar=True)

        _show(axes[row, 2], d_bef, dt, clip_2d,
              title=f'{line_title} - {source_label} before matching')

        _show(axes[row, 3], d_aft, dt, clip_2d,
              title=f'{line_title} - {source_label} after matching',
              add_cbar=True)

    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()
    print(f"\nCombined figure saved: {outfile}")


def process_one_direction(d_source, d_target, dt,
                          source_template_path,
                          source_label, target_label,
                          outdir, t0, t1,
                          wavelet_length, filter_length, prewhitening,
                          fmax_plot, section_clip, tag=""):
    """Match d_source to d_target, write matched.segy and the diagnostic figures to outdir.

    Wavelets and spectra are estimated in the T0-T1 window; the filter is applied to the
    whole source section. Returns the metrics, spectra and sections used by the summary and
    by plot_combined_lines.
    """
    outdir = Path(outdir)
    figdir = outdir / 'figures'
    outdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(exist_ok=True)

    full_tag = f"{tag} ({source_label}->{target_label})"

    if t0 is not None or t1 is not None:
        _t0 = t0 if t0 is not None else 0.0
        _t1 = t1 if t1 is not None else min(d_source.shape[1], d_target.shape[1]) * dt
        d_src_win = select_window(d_source, dt, _t0, _t1)
        d_tgt_win = select_window(d_target, dt, _t0, _t1)
    else:
        d_src_win, d_tgt_win = d_source, d_target

    print("  [wavelets] extracting ...")
    w_src = extract_zero_phase_wavelet(d_src_win, wavelet_length)
    w_tgt = extract_zero_phase_wavelet(d_tgt_win, wavelet_length)

    f_axis, sp_tgt = average_amplitude_spectrum(d_tgt_win, dt)
    _, sp_src_b = average_amplitude_spectrum(d_src_win, dt)
    m_tgt = spectral_metrics(f_axis, sp_tgt)
    m_src_b = spectral_metrics(f_axis, sp_src_b)

    print(f"  [filter ] designing  (h * w_{source_label} ~= w_{target_label}) ...")
    h = design_shaping_filter(w_src, w_tgt,
                              filter_length=filter_length,
                              prewhitening=prewhitening)

    print("  [apply  ] convolving source with h ...")
    d_src_matched = apply_filter(d_source, h)

    if t0 is not None or t1 is not None:
        _t0 = t0 if t0 is not None else 0.0
        _t1 = t1 if t1 is not None else d_source.shape[1] * dt
        d_src_matched_win = select_window(d_src_matched, dt, _t0, _t1)
    else:
        d_src_matched_win = d_src_matched

    _, sp_src_a = average_amplitude_spectrum(d_src_matched_win, dt)
    w_src_after = extract_zero_phase_wavelet(d_src_matched_win, wavelet_length)
    m_src_a = spectral_metrics(f_axis, sp_src_a)

    out_sgy = outdir / 'matched.segy'
    write_segy_resampled(out_sgy, source_template_path, d_src_matched, dt)
    print(f"  [save   ] {out_sgy}")

    plot_wavelets(w_tgt, w_src, w_src_after, dt,
                  figdir / '01_wavelets.png',
                  target_label, source_label, full_tag)
    plot_spectra(f_axis, sp_tgt, sp_src_b, sp_src_a, fmax_plot,
                 figdir / '02_spectra.png',
                 target_label, source_label, full_tag)
    plot_amplitude_histograms(d_tgt_win, d_src_win, d_src_matched_win,
                              figdir / '03_amp_hist.png',
                              target_label, source_label, full_tag)
    plot_sections(d_target, d_source, d_src_matched, dt,
                  figdir / '04_section_before_after.png',
                  section_clip, target_label, source_label, full_tag)
    plot_filter(h, w_src, w_tgt, dt,
                figdir / '05_filter_response.png',
                target_label, source_label, full_tag)

    metrics_path = figdir / '06_metrics.txt'
    with open(metrics_path, 'w') as f:
        f.write(f"Direction      : {source_label} -> {target_label}\n")
        f.write(f"Pair tag       : {tag}\n")
        f.write(f"Working dt     : {dt*1000:.3f} ms\n")
        f.write(f"Window         : {t0} - {t1} s\n")
        f.write(f"Filter length  : {filter_length}\n")
        f.write(f"Wavelet length : {wavelet_length}\n")
        f.write(f"Prewhitening   : {prewhitening}\n\n")
        f.write("peak / centroid / -6dB band\n")
        f.write("================================================\n")
        for name, m in [(f'{target_label} target           ', m_tgt),
                        (f'{source_label} before matching  ', m_src_b),
                        (f'{source_label} after  matching  ', m_src_a)]:
            f.write(f"{name}: peak={m['peak_hz']:6.2f} Hz, "
                    f"centroid={m['centroid_hz']:6.2f} Hz, "
                    f"-6dB={m['bw_lo_hz']:5.1f}-{m['bw_hi_hz']:5.1f} Hz "
                    f"(BW={m['bw_hz']:5.1f} Hz)\n")
    return dict(metrics_target=m_tgt,
                metrics_source_before=m_src_b,
                metrics_source_after=m_src_a,
                dt=dt,
                freqs=f_axis,
                spec_target=sp_tgt,
                spec_source_before=sp_src_b,
                spec_source_after=sp_src_a,
                section_target=d_target,
                section_source_before=d_source,
                section_source_after=d_src_matched)


def process_pair(target3d_path, input2d_path, outdir_base,
                 t0, t1, wavelet_length, filter_length,
                 prewhitening, fmax_plot, section_clip,
                 directions=('to_3d', 'to_2d'),
                 resample_to_finer=True, tag=""):
    """Read one 3D/2D pair, bring both to the finer sample interval and run each direction."""
    print(f"\n=== {tag} ===")
    print(f"[Read] {target3d_path.name} / {input2d_path.name}")
    d3d, dt3d = read_segy(target3d_path)
    d2d, dt2d = read_segy(input2d_path)
    print(f"  3D : {d3d.shape}, dt = {dt3d*1000:.3f} ms")
    print(f"  2D : {d2d.shape}, dt = {dt2d*1000:.3f} ms")

    dt_mismatch = abs(dt3d - dt2d) / max(dt3d, 1e-12) > 1e-3
    if dt_mismatch and resample_to_finer:
        if dt2d > dt3d:
            print(f"[Resample] 2D : dt={dt2d*1000:.3f} ms -> {dt3d*1000:.3f} ms")
            d2d = resample_traces(d2d, dt2d, dt3d)
            dt2d = dt3d
        else:
            print(f"[Resample] 3D : dt={dt3d*1000:.3f} ms -> {dt2d*1000:.3f} ms")
            d3d = resample_traces(d3d, dt3d, dt2d)
            dt3d = dt2d
    elif dt_mismatch:
        print("  dt mismatch with RESAMPLE_TO_FINER_DT=False; "
              "results may be inaccurate")
    dt = dt3d

    results = {}

    if 'to_3d' in directions:
        print("\n[Direction] to_3d  (2D -> 3D)")
        sub_outdir = Path(outdir_base) / 'to_3d'
        results['to_3d'] = process_one_direction(
            d_source=d2d, d_target=d3d, dt=dt,
            source_template_path=input2d_path,
            source_label='2D', target_label='3D',
            outdir=sub_outdir, t0=t0, t1=t1,
            wavelet_length=wavelet_length, filter_length=filter_length,
            prewhitening=prewhitening, fmax_plot=fmax_plot,
            section_clip=section_clip, tag=tag)

    if 'to_2d' in directions:
        print("\n[Direction] to_2d  (3D -> 2D)")
        sub_outdir = Path(outdir_base) / 'to_2d'
        results['to_2d'] = process_one_direction(
            d_source=d3d, d_target=d2d, dt=dt,
            source_template_path=target3d_path,
            source_label='3D', target_label='2D',
            outdir=sub_outdir, t0=t0, t1=t1,
            wavelet_length=wavelet_length, filter_length=filter_length,
            prewhitening=prewhitening, fmax_plot=fmax_plot,
            section_clip=section_clip, tag=tag)

    return results


def main():
    summary = []
    for target3d_name, input2d_name, outdir_name in PAIRS:
        target3d = SEGY_DIR / target3d_name
        input2d = SEGY_DIR / input2d_name
        if not target3d.exists():
            print(f"File not found: {target3d}; skipped")
            continue
        if not input2d.exists():
            print(f"File not found: {input2d}; skipped")
            continue
        tag = outdir_name.replace("results_", "")
        res = process_pair(target3d, input2d, OUTPUT_DIR / outdir_name,
                           T0, T1, WAVELET_LENGTH, FILTER_LENGTH,
                           PREWHITENING, FMAX_PLOT, SECTION_CLIP,
                           directions=tuple(DIRECTIONS),
                           resample_to_finer=RESAMPLE_TO_FINER_DT,
                           tag=tag)
        summary.append((tag, res))

    if SAVE_COMBINED_FIG and summary:
        line_results = []
        for idx, (tag, res) in enumerate(summary):
            if COMBINED_DIRECTION not in res:
                print(f"  '{tag}' has no '{COMBINED_DIRECTION}' result; "
                      f"excluded from the combined figure")
                continue
            if idx < len(COMBINED_LINE_NAMES):
                line_title = COMBINED_LINE_NAMES[idx]
            else:
                line_title = f"Arbitrary line {idx + 1}"
            line_results.append((line_title, res[COMBINED_DIRECTION]))

        if line_results:
            src_lbl = '2D' if COMBINED_DIRECTION == 'to_3d' else '3D'
            tgt_lbl = '3D' if COMBINED_DIRECTION == 'to_3d' else '2D'
            plot_combined_lines(line_results, COMBINED_FIG_PATH,
                                target_label=tgt_lbl, source_label=src_lbl,
                                fmax=FMAX_PLOT, clip_pct=SECTION_CLIP)

    if summary:
        print("\n" + "=" * 86)
        print("OVERALL SUMMARY")
        print("=" * 86)
        header = f"{'Line':<10}{'Direction':<10}{'Set':<22}" \
                 f"{'Peak(Hz)':>10}{'Centroid':>10}{'BW -6dB':>20}"
        print(header)
        print("-" * 86)
        for tag, r in summary:
            for dirn, dres in r.items():
                src = '2D' if dirn == 'to_3d' else '3D'
                tgt = '3D' if dirn == 'to_3d' else '2D'
                rows = [(f'{tgt} target', dres['metrics_target']),
                        (f'{src} before', dres['metrics_source_before']),
                        (f'{src} after', dres['metrics_source_after'])]
                for name, m in rows:
                    bw = f"{m['bw_lo_hz']:.1f}-{m['bw_hi_hz']:.1f}"
                    print(f"{tag:<10}{dirn:<10}{name:<22}"
                          f"{m['peak_hz']:>10.2f}{m['centroid_hz']:>10.2f}"
                          f"{bw:>20}")
                print("-" * 86)


if __name__ == "__main__":
    main()
