"""Wavelet matching (shaping) filter between seismic data sets.

Shared by process/match_2d_to_canonical.py (matching the real 2D lines to the 3D volume
before inference) and test/seismic_matching_filter.py (comparison on co-located 3D/2D
lines). Sections are 2D arrays of shape (n_traces, n_samples), i.e. one trace per row, as
returned by read_segy; dt is the sample interval in seconds.

The procedure: estimate a zero-phase wavelet from the average amplitude spectrum of a
section (white-reflectivity assumption), design a least-squares shaping filter h with
h * w_source ~= w_target, and convolve every source trace with h.
"""
from fractions import Fraction

import numpy as np
import segyio
from scipy.linalg import toeplitz, lstsq
from scipy.signal import resample_poly


def read_segy(path):
    """Read all traces of a SEG-Y file; returns (data (n_traces, n_samples) float32, dt in s)."""
    with segyio.open(str(path), "r", ignore_geometry=True) as f:
        dt = segyio.tools.dt(f) / 1e6
        data = segyio.tools.collect(f.trace[:])
    return data.astype(np.float32), dt


def write_segy_resampled(out_path, template_path, new_data, new_dt):
    """Write new_data (n_traces, n_samples) as SEG-Y with sample interval new_dt (s).

    The textual header and the trace headers are copied from template_path; the sample
    count and sample interval are updated to match new_data.
    """
    n_traces, n_samples = new_data.shape
    spec = segyio.spec()
    spec.sorting = 2
    spec.format = 1
    spec.samples = np.arange(n_samples) * new_dt * 1000
    spec.tracecount = n_traces

    with segyio.open(str(template_path), "r", ignore_geometry=True) as src, \
         segyio.create(str(out_path), spec) as dst:
        try:
            dst.text[0] = src.text[0]
        except Exception:
            pass
        for i in range(min(src.tracecount, n_traces)):
            dst.header[i] = src.header[i]
            dst.header[i].update({
                segyio.TraceField.TRACE_SAMPLE_COUNT: n_samples,
                segyio.TraceField.TRACE_SAMPLE_INTERVAL: int(round(new_dt * 1e6)),
            })
            dst.trace[i] = new_data[i].astype(np.float32)
        dst.bin[segyio.BinField.Samples] = n_samples
        dst.bin[segyio.BinField.Interval] = int(round(new_dt * 1e6))


def resample_traces(data, dt_in, dt_out):
    """Resample every trace from dt_in to dt_out (polyphase filter, rational ratio).

    The output keeps the same time span: (n_in - 1) * dt_in / dt_out + 1 samples.
    """
    if abs(dt_in - dt_out) / max(dt_in, 1e-12) < 1e-6:
        return data
    n_in = data.shape[1]
    n_out = int(round((n_in - 1) * dt_in / dt_out)) + 1
    f = Fraction(dt_in / dt_out).limit_denominator(1000)
    up, down = f.numerator, f.denominator
    new_data = resample_poly(data, up=up, down=down, axis=1).astype(np.float32)
    if new_data.shape[1] > n_out:
        new_data = new_data[:, :n_out]
    elif new_data.shape[1] < n_out:
        pad = n_out - new_data.shape[1]
        new_data = np.pad(new_data, ((0, 0), (0, pad)))
    return new_data


def select_window(data, dt, t0, t1):
    """Samples between times t0 and t1 (s) of every trace; raises ValueError if empty."""
    i0 = max(0, int(round(t0 / dt)))
    i1 = min(data.shape[1], int(round(t1 / dt)))
    if i1 <= i0:
        raise ValueError(f"Empty window after clipping: t0={t0}, t1={t1}, "
                         f"data range = 0 - {data.shape[1]*dt:.3f} s")
    return data[:, i0:i1]


def extract_zero_phase_wavelet(data, wavelet_length=129, taper=True):
    """Zero-phase wavelet whose amplitude spectrum is the section's average spectrum.

    Each trace is demeaned and Hann-tapered; the root of the trace-averaged power spectrum
    is transformed back with zero phase, centred, cut to wavelet_length (made odd) samples
    and normalised to a peak absolute value of 1.
    """
    if wavelet_length % 2 == 0:
        wavelet_length += 1
    n_traces, n_samples = data.shape
    n_fft = 1 << int(np.ceil(np.log2(max(n_samples, wavelet_length))))
    win = np.hanning(n_samples) if taper else np.ones(n_samples)
    avg_power = np.zeros(n_fft)
    for i in range(n_traces):
        tr = (data[i] - np.mean(data[i])) * win
        F = np.fft.fft(tr, n=n_fft)
        avg_power += np.abs(F) ** 2
    avg_power /= max(n_traces, 1)
    amp_spec = np.sqrt(np.maximum(avg_power, 0.0))
    w_full = np.real(np.fft.ifft(amp_spec))
    w_full = np.fft.fftshift(w_full)
    center = len(w_full) // 2
    half = wavelet_length // 2
    w = w_full[center - half : center + half + 1]
    return w / (np.max(np.abs(w)) + 1e-12)


def average_amplitude_spectrum(data, dt, n_fft=None, taper=True):
    """Trace-averaged amplitude spectrum (demeaned, Hann-tapered traces); returns (freqs in Hz, amplitude)."""
    n_traces, n_samples = data.shape
    if n_fft is None:
        n_fft = 1 << int(np.ceil(np.log2(n_samples)))
    win = np.hanning(n_samples) if taper else np.ones(n_samples)
    avg_amp = np.zeros(n_fft // 2 + 1)
    for i in range(n_traces):
        tr = (data[i] - np.mean(data[i])) * win
        avg_amp += np.abs(np.fft.rfft(tr, n=n_fft))
    avg_amp /= max(n_traces, 1)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    return freqs, avg_amp


def spectral_metrics(freqs, amp):
    """Peak frequency, spectral centroid and the band where amp >= half of its peak (-6 dB), in Hz."""
    pk = int(np.argmax(amp))
    peak_f = freqs[pk]
    centroid = np.sum(freqs * amp) / (np.sum(amp) + 1e-12)
    half_max = amp[pk] / 2.0
    above = amp >= half_max
    if above.any():
        idx = np.where(above)[0]
        bw_lo, bw_hi = freqs[idx[0]], freqs[idx[-1]]
    else:
        bw_lo, bw_hi = peak_f, peak_f
    return dict(peak_hz=peak_f, centroid_hz=centroid,
                bw_lo_hz=bw_lo, bw_hi_hz=bw_hi, bw_hz=bw_hi - bw_lo)


def design_shaping_filter(w_in, w_out, filter_length=101, prewhitening=1e-3):
    """Least-squares shaping filter h (filter_length taps, made odd) with h * w_in ~= w_out.

    Both wavelets are peak-normalised; w_out is centred in the full convolution output, and
    prewhitening adds a ridge (Tikhonov) term sqrt(prewhitening) * h to stabilise the solution.
    """
    if filter_length % 2 == 0:
        filter_length += 1
    w_in = w_in / (np.max(np.abs(w_in)) + 1e-12)
    w_out = w_out / (np.max(np.abs(w_out)) + 1e-12)
    L, M = filter_length, len(w_in)
    out_len = L + M - 1
    col = np.r_[w_in, np.zeros(L - 1)]
    row = np.r_[w_in[0], np.zeros(L - 1)]
    D = toeplitz(col, row)
    w_out_padded = np.zeros(out_len)
    start = (out_len - M) // 2
    w_out_padded[start : start + M] = w_out
    A = np.vstack([D, np.sqrt(prewhitening) * np.eye(L)])
    b = np.concatenate([w_out_padded, np.zeros(L)])
    h, *_ = lstsq(A, b)
    return h


def apply_filter(data, h):
    """Convolve every trace with h, keeping the centred segment of the original length."""
    L = len(h)
    pad = L // 2
    out = np.empty_like(data)
    for i in range(data.shape[0]):
        conv = np.convolve(data[i], h, mode="full")
        out[i] = conv[pad : pad + data.shape[1]]
    return out


def safe_norm(x):
    """x divided by its peak absolute value (unchanged if that is 0); used for plotting."""
    m = np.max(np.abs(x))
    return x / m if m > 0 else x
