#!/usr/bin/env python3
"""Plot E-field component histories and frequency spectra.

Default: average over a mid-plane slice. With --index I J K: sample one cell.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python postprocessing/plot_e_spectrum.py` from the repo root.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from postprocessing.common import (
    add_output_dpi_args,
    add_plane_args,
    add_worker_stride_args,
    configure_matplotlib,
    configure_yt,
    resolve_output_path,
    sorted_plotfiles,
)

configure_matplotlib()
configure_yt()

import matplotlib.pyplot as plt

from postprocessing.fields import preload_field_series
from postprocessing.series import (
    DEFAULT_SERIES,
    add_prefix_series_args,
    resolve_series,
    series_directories,
)
from postprocessing.spectrum import (
    add_spectrum_args,
    apply_freq_range,
    frequency_spectrum,
    late_window,
    mark_peak_frequency,
    validate_spectrum_args,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_prefix_series_args(parser)
    parser.add_argument(
        "--components",
        nargs="+",
        default=("Ex", "Ey", "Ez"),
        choices=("Ex", "Ey", "Ez"),
        help="E-field components to extract.",
    )
    parser.add_argument(
        "--index",
        nargs=3,
        type=int,
        metavar=("I", "J", "K"),
        default=None,
        help="sample one cell at (i,j,k) instead of mid-plane average",
    )
    add_plane_args(
        parser,
        slice_help="index along the normal axis for plane average (default: center)",
    )
    add_worker_stride_args(parser)
    add_output_dpi_args(
        parser,
        default_output=Path("circuit_test_run/E_components_avg_time_spectrum.png"),
    )
    add_spectrum_args(parser)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    validate_spectrum_args(args.fft_pad, args.freq_range, args.late_percent)

    components = tuple(dict.fromkeys(args.components))
    cell_index = tuple(args.index) if args.index is not None else None
    if cell_index is not None and any(idx < 0 for idx in cell_index):
        raise ValueError("--index values must be non-negative")

    series_dirs = resolve_series(
        args.prefix,
        args.auto_series,
        DEFAULT_SERIES,
        selected=args.series,
        require_plotfiles=True,
    )
    print(f"Series ({len(series_dirs)}): {', '.join(label for label, _ in series_dirs)}")
    if args.late_percent < 100.0:
        print(f"Using last {args.late_percent:g}% of timesteps")
    series = series_directories(args.prefix, series_dirs)
    all_paths = [sorted_plotfiles(directory)[:: args.stride] for _, directory in series]
    loaded, location = preload_field_series(
        all_paths,
        components,
        args.plane,
        args.slice_index,
        cell_index,
        args.workers,
    )
    data = [
        (label, times, values)
        for (label, _), (times, values) in zip(series, loaded, strict=True)
    ]

    def field_label(component: str) -> str:
        # Angle brackets denote a spatial average; point samples omit them.
        return component if cell_index is not None else f"<{component}>"

    late_tag = (
        f" (last {args.late_percent:g}%)" if args.late_percent < 100.0 else ""
    )
    fig, axes = plt.subplots(
        len(components),
        2,
        figsize=(13.0, 3.0 * len(components)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, component in enumerate(components):
        ax_time, ax_freq = axes[row]
        for label, times, values in data:
            t_win, y_win = late_window(times, values[component], args.late_percent)
            ax_time.plot(t_win * 1.0e9, y_win, label=label, linewidth=1.6)
            frequencies, amplitudes = frequency_spectrum(
                t_win,
                y_win,
                args.keep_dc,
                args.fft_pad,
            )
            if frequencies.size:
                freq_plot = frequencies * 1.0e-9
                (line,) = ax_freq.plot(
                    freq_plot, amplitudes, label=label, linewidth=1.6
                )
                peak = mark_peak_frequency(
                    ax_freq,
                    freq_plot,
                    amplitudes,
                    freq_range=args.freq_range,
                    color=line.get_color(),
                    label=label,
                    freq_unit="GHz",
                )
                if peak is not None:
                    print(
                        f"{component} / {label}: peak f={peak[0]:.6g} GHz, "
                        f"amp={peak[1]:.3e}"
                    )

        ylabel = field_label(component)
        ax_time.set_title(f"{ylabel} at {location} in time domain{late_tag}")
        ax_time.set_xlabel("time (ns)")
        ax_time.set_ylabel(ylabel)
        ax_time.grid(True, alpha=0.3)

        ax_freq.set_title(f"{ylabel} at {location} frequency spectrum{late_tag}")
        ax_freq.set_xlabel("frequency (GHz)")
        ax_freq.set_ylabel("amplitude")
        ax_freq.grid(True, alpha=0.3)
        if args.log_spectrum:
            ax_freq.set_yscale("log")
        apply_freq_range(ax_freq, args.freq_range)
        if row == 0:
            ax_time.legend(loc="best", fontsize="small")
            if ax_freq.lines:
                ax_freq.legend(loc="best", fontsize="small")

    output = resolve_output_path(args.output, args.prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=args.dpi)
    plt.close(fig)

    print(f"Wrote {output}")
    print(f"Location: {location}")
    print(f"Workers: {args.workers}")
    for label, times, values in data:
        t_win, _ = late_window(times, values[components[0]], args.late_percent)
        print(
            f"{label}: {t_win.size}/{times.size} samples, "
            f"t=[{t_win.min() * 1.0e9:.3e}, {t_win.max() * 1.0e9:.3e}] ns"
        )


if __name__ == "__main__":
    main()
