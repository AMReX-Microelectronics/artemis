"""Frequency-spectrum helpers shared by Eobs and E-field time scripts."""

from __future__ import annotations

import argparse

import numpy as np


def late_window(
    times: np.ndarray,
    values: np.ndarray,
    late_percent: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the last ``late_percent`` percent of samples (time-sorted).

    ``late_percent=100`` returns the full series; ``late_percent=25`` keeps the
    final quarter of timesteps.
    """
    if times.size == 0:
        return times, values
    if not (0.0 < late_percent <= 100.0):
        raise ValueError("--late-percent must be in (0, 100]")

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    if late_percent >= 100.0:
        return times, values

    n_keep = max(1, int(np.ceil(times.size * late_percent / 100.0)))
    return times[-n_keep:], values[-n_keep:]


def frequency_spectrum(
    times: np.ndarray,
    values: np.ndarray,
    keep_dc: bool,
    fft_pad: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if times.size < 2:
        return np.asarray([]), np.asarray([])

    order = np.argsort(times)
    times = times[order]
    values = values[order]
    dt_values = np.diff(times)
    dt = float(np.median(dt_values))
    if not np.allclose(dt_values, dt, rtol=1.0e-4, atol=max(1.0e-18, abs(dt) * 1.0e-8)):
        uniform_times = np.linspace(float(times[0]), float(times[-1]), times.size)
        values = np.interp(uniform_times, times, values)
        dt = float(uniform_times[1] - uniform_times[0])

    signal = values if keep_dc else values - np.mean(values)
    n_orig = signal.size
    n_fft = max(n_orig, int(np.ceil(n_orig * fft_pad)))
    frequencies = np.fft.rfftfreq(n_fft, d=dt)
    # Normalize by original length so amplitudes stay comparable across pad factors.
    amplitude = np.abs(np.fft.rfft(signal, n=n_fft)) / n_orig
    if n_orig > 2:
        amplitude[1:-1] *= 2.0

    if keep_dc:
        return frequencies, amplitude
    return frequencies[1:], amplitude[1:]


def add_spectrum_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--log-spectrum",
        action="store_true",
        help="Use a logarithmic y-axis for the frequency spectrum.",
    )
    parser.add_argument(
        "--keep-dc",
        action="store_true",
        help="Keep the DC/zero-frequency bin in the spectrum.",
    )
    parser.add_argument(
        "--fft-pad",
        type=float,
        default=5.0,
        help="zero-pad factor before FFT (>=1; 1 = no padding). Default: 5",
    )
    parser.add_argument(
        "--freq-range",
        nargs=2,
        type=float,
        metavar=("FMIN", "FMAX"),
        default=None,
        help="frequency-axis plot limits in plot units "
        "(GHz when the time column is in seconds)",
    )
    parser.add_argument(
        "--late-percent",
        type=float,
        default=100.0,
        metavar="PCT",
        help="use only the last PCT percent of timesteps for time plots and "
        "spectra (default: 100 = full series)",
    )


def validate_spectrum_args(
    fft_pad: float,
    freq_range: list[float] | None,
    late_percent: float = 100.0,
) -> None:
    if fft_pad < 1.0:
        raise ValueError("--fft-pad must be >= 1")
    if freq_range is not None and freq_range[0] >= freq_range[1]:
        raise ValueError("--freq-range requires FMIN < FMAX")
    if not (0.0 < late_percent <= 100.0):
        raise ValueError("--late-percent must be in (0, 100]")


def apply_freq_range(ax, freq_range: list[float] | None) -> None:
    if freq_range is not None:
        ax.set_xlim(float(freq_range[0]), float(freq_range[1]))


def peak_frequency(
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    freq_range: list[float] | None = None,
) -> tuple[float, float] | None:
    """Return ``(f_peak, amp_peak)`` within an optional plot-unit frequency window."""
    if frequencies.size == 0:
        return None
    mask = np.isfinite(frequencies) & np.isfinite(amplitudes)
    if freq_range is not None:
        fmin, fmax = float(freq_range[0]), float(freq_range[1])
        mask &= (frequencies >= fmin) & (frequencies <= fmax)
    if not np.any(mask):
        return None
    freqs = frequencies[mask]
    amps = amplitudes[mask]
    idx = int(np.argmax(amps))
    return float(freqs[idx]), float(amps[idx])


def mark_peak_frequency(
    ax,
    frequencies: np.ndarray,
    amplitudes: np.ndarray,
    *,
    freq_range: list[float] | None = None,
    color=None,
    label: str | None = None,
    freq_unit: str = "GHz",
) -> tuple[float, float] | None:
    """Mark the spectrum peak with a point, dashed line, and frequency annotation."""
    peak = peak_frequency(frequencies, amplitudes, freq_range)
    if peak is None:
        return None
    f_peak, a_peak = peak
    ax.plot(f_peak, a_peak, "o", color=color, markersize=5, zorder=5)
    ax.axvline(f_peak, color=color, linestyle="--", linewidth=0.9, alpha=0.7, zorder=4)
    text = f"{f_peak:.3g} {freq_unit}"
    if label:
        text = f"{label}: {text}"
    ax.annotate(
        text,
        xy=(f_peak, a_peak),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize="x-small",
        color=color,
        ha="left",
        va="bottom",
    )
    return peak


# Backward-compatible alias used by older call sites.
def validate_fft_pad(fft_pad: float) -> None:
    validate_spectrum_args(fft_pad, None)
