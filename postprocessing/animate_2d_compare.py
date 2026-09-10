#!/usr/bin/env python3
"""Render Artemis diagnostic series side-by-side with a shared color scale."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python postprocessing/animate_2d_compare.py` from the repo root.
if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from postprocessing.common import (
    AXIS_NAME,
    PLANES,
    add_output_dpi_args,
    add_plane_args,
    add_worker_stride_args,
    configure_matplotlib,
    configure_yt,
    panel_layout,
    resolve_output_path,
    shared_clim,
    sorted_plotfiles,
)

configure_matplotlib()
configure_yt()

import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from tqdm import tqdm

from postprocessing.fields import preload_slices
from postprocessing.series import (
    DEFAULT_SERIES,
    add_prefix_series_args,
    resolve_series,
    series_directories,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    add_prefix_series_args(parser)
    parser.add_argument(
        "--variable",
        default="|E|",
        choices=("Ex", "Ey", "Ez", "|E|", "Bx", "By", "Bz", "|B|", "epsilon"),
    )
    add_plane_args(
        parser, slice_help="index along the normal axis (default: center)"
    )
    add_worker_stride_args(parser)
    parser.add_argument("--fps", type=int, default=20)
    add_output_dpi_args(
        parser,
        # pace ffmpeg/7.1 is built without libvpx/libx264; mpeg4+mp4 works.
        default_output=Path("circuit_test_run/E_xy_diag_comparison.mp4"),
        default_dpi=150,
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stride < 1:
        raise ValueError("--stride must be at least 1")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    series_dirs = resolve_series(
        args.prefix,
        args.auto_series,
        DEFAULT_SERIES,
        selected=args.series,
        require_plotfiles=True,
    )
    print(f"Series ({len(series_dirs)}): {', '.join(label for label, _ in series_dirs)}")
    series = series_directories(args.prefix, series_dirs)
    n_series = len(series)
    all_paths = [sorted_plotfiles(directory)[:: args.stride] for _, directory in series]
    frame_counts = [len(paths) for paths in all_paths]
    if len(set(frame_counts)) != 1:
        raise ValueError(
            f"Diagnostic series have different frame counts: {frame_counts}"
        )
    n_frames = frame_counts[0]

    ax0, ax1, normal = PLANES[args.plane]
    signed = not args.variable.startswith("|") and args.variable != "epsilon"
    cmap = "RdBu_r" if signed else "viridis"

    times, all_slices, extents, fixed_indices = preload_slices(
        all_paths,
        args.variable,
        args.plane,
        args.slice_index,
        args.workers,
    )
    shapes = [frames[0].shape for frames in all_slices]
    if len(set(shapes)) != 1:
        raise ValueError(f"Slice shapes differ in the first frame: {shapes}")

    frame0 = [frames[0] for frames in all_slices]
    vmin, vmax = shared_clim(frame0, signed)
    nrows, ncols = panel_layout(n_series)
    fig = plt.figure(figsize=(5.5 * ncols + 1.2, 4.8 * nrows + 0.8))
    # Outer gridspec: plot grid | colorbar column
    outer = fig.add_gridspec(
        1,
        2,
        width_ratios=(ncols, 0.045),
        left=0.055,
        right=0.97,
        bottom=0.08,
        top=0.88,
        wspace=0.08,
    )
    inner = outer[0, 0].subgridspec(nrows, ncols, wspace=0.18, hspace=0.28)
    axes = []
    for index in range(n_series):
        row, col = divmod(index, ncols)
        if index == 0:
            axes.append(fig.add_subplot(inner[row, col]))
        else:
            axes.append(
                fig.add_subplot(inner[row, col], sharex=axes[0], sharey=axes[0])
            )
    colorbar_axis = fig.add_subplot(outer[0, 1])
    images = []
    for ax, (label, _), frame, extent, fixed in zip(
        axes, series, frame0, extents, fixed_indices, strict=True
    ):
        image = ax.imshow(
            frame.T,
            origin="lower",
            extent=extent,
            aspect="auto",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax,
        )
        images.append(image)
        ax.set_title(label)
        ax.set_xlabel(f"{AXIS_NAME[ax0]} (mm)")
        ax.text(
            0.02,
            0.98,
            f"{AXIS_NAME[normal]}-index {fixed}",
            transform=ax.transAxes,
            va="top",
            color="white",
            bbox={"facecolor": "black", "alpha": 0.45, "edgecolor": "none"},
        )
    for row in range(nrows):
        left_index = row * ncols
        if left_index < n_series:
            axes[left_index].set_ylabel(f"{AXIS_NAME[ax1]} (mm)")
    colorbar = fig.colorbar(images[0], cax=colorbar_axis)
    colorbar.set_label(args.variable)
    title = fig.suptitle(
        f"{args.variable} on {args.plane} plane, t = {times[0]:.3e} s"
    )
    output = resolve_output_path(args.output, args.prefix)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(
        fps=args.fps,
        bitrate=7200,
        codec="mpeg4",
        extra_args=(
            "-pix_fmt",
            "yuv420p",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        ),
    )
    with writer.saving(fig, output, dpi=args.dpi):
        for frame_index in tqdm(range(n_frames), desc="Rendering"):
            slices = [frames[frame_index] for frames in all_slices]
            vmin, vmax = shared_clim(slices, signed)
            for image, frame, extent in zip(images, slices, extents, strict=True):
                image.set_data(frame.T)
                image.set_extent(extent)
                image.set_clim(vmin, vmax)
            title.set_text(
                f"{args.variable} on {args.plane} plane, t = {times[frame_index]:.3e} s"
            )
            writer.grab_frame()

    plt.close(fig)
    print(f"Wrote {output}")
    print(f"Series: {n_series}")
    print(f"Frames: {n_frames}")
    print(f"Workers: {args.workers}")


if __name__ == "__main__":
    main()
