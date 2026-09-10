#!/usr/bin/env python3
"""Plot Eobs reduced diagnostics: time history and frequency spectrum (2x2)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python postprocessing/plot_eobs.py` from the repo root.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from postprocessing.common import (
    add_output_dpi_args,
    configure_matplotlib,
    resolve_output_path,
)

configure_matplotlib()

import matplotlib.pyplot as plt
import numpy as np

from postprocessing.series import (
    DEFAULT_SERIES,
    add_prefix_series_args,
    resolve_series,
)
from postprocessing.spectrum import (
    add_spectrum_args,
    apply_freq_range,
    frequency_spectrum,
    late_window,
    mark_peak_frequency,
    validate_spectrum_args,
)

DEFAULT_NAMES = ("Eobs0.txt", "Eobs1.txt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_prefix_series_args(parser)
    parser.add_argument(
        "--names",
        nargs="+",
        default=DEFAULT_NAMES,
        help="reduced-diag filenames under each diags_*/reducedfiles/ "
        "(default: Eobs0.txt Eobs1.txt; one row per name)",
    )
    parser.add_argument(
        "--xcol",
        type=int,
        default=1,
        help="0-based column for x-axis (default: 1 = time)",
    )
    parser.add_argument(
        "--ycol",
        type=int,
        default=2,
        help="0-based column for y-axis (default: 2 = Ex)",
    )
    add_output_dpi_args(
        parser, default_output=Path("circuit_test_run/Eobs_time_spectrum.png")
    )
    add_spectrum_args(parser)
    return parser.parse_args()


def load_eobs(path: Path, xcol: int, ycol: int) -> tuple[np.ndarray, np.ndarray]:
    data = np.loadtxt(path, comments="#")
    if data.ndim != 2 or data.shape[1] <= max(xcol, ycol):
        raise ValueError(
            f"{path}: expected >= {max(xcol, ycol) + 1} columns, got {data.shape}"
        )
    return data[:, xcol], data[:, ycol]


def main() -> None:
    args = parse_args()
    validate_spectrum_args(args.fft_pad, args.freq_range, args.late_percent)

    names = tuple(args.names)
    series = resolve_series(
        args.prefix,
        args.auto_series,
        DEFAULT_SERIES,
        selected=args.series,
        require_reducedfiles=True,
    )
    print(f"Series ({len(series)}): {', '.join(label for label, _ in series)}")
    if args.late_percent < 100.0:
        print(f"Using last {args.late_percent:g}% of timesteps")

    ylabel = f"column {args.ycol}" if args.ycol != 2 else "Ex"
    n_rows = len(names)
    fig, axes = plt.subplots(
        n_rows,
        2,
        figsize=(13.0, 3.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )
    late_tag = (
        f" (last {args.late_percent:g}%)" if args.late_percent < 100.0 else ""
    )

    for row, name in enumerate(names):
        ax_time, ax_freq = axes[row]
        stem = Path(name).stem

        for label, dirname in series:
            path = args.prefix / dirname / "reducedfiles" / name
            if not path.is_file():
                print(f"skip missing: {path}")
                continue
            x, y = load_eobs(path, args.xcol, args.ycol)
            x, y = late_window(x, y, args.late_percent)
            if args.xcol == 1:
                ax_time.plot(x * 1.0e9, y, label=label, linewidth=1.4)
            else:
                ax_time.plot(x, y, label=label, linewidth=1.4)

            frequencies, amplitudes = frequency_spectrum(
                x, y, args.keep_dc, args.fft_pad
            )
            if frequencies.size:
                if args.xcol == 1:
                    freq_plot = frequencies * 1.0e-9
                    freq_unit = "GHz"
                else:
                    freq_plot = frequencies
                    freq_unit = "1/x"
                (line,) = ax_freq.plot(
                    freq_plot, amplitudes, label=label, linewidth=1.4
                )
                peak = mark_peak_frequency(
                    ax_freq,
                    freq_plot,
                    amplitudes,
                    freq_range=args.freq_range,
                    color=line.get_color(),
                    label=label,
                    freq_unit=freq_unit,
                )
                if peak is not None:
                    print(
                        f"{stem} / {label}: peak f={peak[0]:.6g} {freq_unit}, "
                        f"amp={peak[1]:.3e}"
                    )
            print(f"{stem} / {label}: {path}  N={x.size}")

        if args.xcol == 1:
            ax_time.set_xlabel("time (ns)")
            ax_freq.set_xlabel("frequency (GHz)")
        else:
            ax_time.set_xlabel(f"column {args.xcol}")
            ax_freq.set_xlabel("frequency (1 / x-units)")
        ax_time.set_ylabel(ylabel)
        ax_time.set_title(f"{args.prefix.name} / {stem} time domain{late_tag}")
        ax_time.grid(True, alpha=0.3)

        ax_freq.set_ylabel("amplitude")
        ax_freq.set_title(
            f"{args.prefix.name} / {stem} frequency spectrum{late_tag}"
        )
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


if __name__ == "__main__":
    main()
