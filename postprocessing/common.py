"""Shared setup, constants, and small utilities for postprocessing CLIs."""

from __future__ import annotations

import argparse
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import TypeVar

import numpy as np
from tqdm import tqdm

PLANES = {
    "xy": (0, 1, 2),
    "xz": (0, 2, 1),
    "yz": (1, 2, 0),
}
AXIS_NAME = "xyz"

T = TypeVar("T")
R = TypeVar("R")


def configure_matplotlib() -> None:
    os.environ.setdefault("MPLCONFIGDIR", f"/tmp/mplconfig-{os.getuid()}")
    Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


def configure_yt() -> None:
    import yt

    yt.set_log_level("ERROR")
    logging.getLogger("yt").setLevel(logging.ERROR)


def sorted_plotfiles(directory: Path) -> list[Path]:
    files = sorted(
        directory.glob("plt*"),
        key=lambda path: int(path.name.removeprefix("plt")),
    )
    if not files:
        raise FileNotFoundError(f"No plotfiles found in {directory.resolve()}")
    return files


def map_jobs(
    func: Callable[[T], R],
    jobs: Sequence[T],
    workers: int,
    *,
    desc: str = "Loading",
) -> list[R]:
    """Map ``func`` over ``jobs`` serially or with a process pool."""
    if workers <= 1:
        return [func(job) for job in tqdm(jobs, desc=desc)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(
            tqdm(
                pool.map(func, jobs, chunksize=1),
                total=len(jobs),
                desc=f"{desc} ({workers} workers)",
            )
        )


def panel_layout(n_series: int) -> tuple[int, int]:
    """Choose a compact (nrows, ncols) grid for n_series panels."""
    if n_series <= 0:
        raise ValueError("need at least one series")
    if n_series <= 3:
        return 1, n_series
    if n_series <= 6:
        return 2, (n_series + 1) // 2
    ncols = int(np.ceil(np.sqrt(n_series)))
    nrows = int(np.ceil(n_series / ncols))
    return nrows, ncols


def shared_clim(slices: Iterable[np.ndarray], signed: bool) -> tuple[float, float]:
    """Return color limits spanning all datasets for the current frame."""
    frames = list(slices)
    if signed:
        vmax = max(float(np.max(np.abs(frame))) for frame in frames)
        vmax = max(vmax, np.finfo(float).eps)
        return -vmax, vmax

    vmin = min(float(np.min(frame)) for frame in frames)
    vmax = max(float(np.max(frame)) for frame in frames)
    if vmin == vmax:
        vmax = vmin + np.finfo(float).eps
    return vmin, vmax


def add_worker_stride_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Use every Nth plotfile from each diagnostic series.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="parallel plotfile readers (1 = serial)",
    )


def add_plane_args(parser: argparse.ArgumentParser, *, slice_help: str) -> None:
    parser.add_argument("--plane", default="xy", choices=tuple(PLANES))
    parser.add_argument(
        "--slice-index",
        type=int,
        default=None,
        help=slice_help,
    )


def add_output_dpi_args(
    parser: argparse.ArgumentParser,
    *,
    default_output: Path,
    default_dpi: int = 180,
) -> None:
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="output path; --prefix directory name is appended before the extension",
    )
    parser.add_argument("--dpi", type=int, default=default_dpi)


def resolve_output_path(output: Path, prefix: Path) -> Path:
    """Append ``_{prefix.name}`` before the file extension (unless already present)."""
    suffix_tag = f"_{prefix.name}"
    if output.stem.endswith(suffix_tag):
        return output
    return output.with_name(f"{output.stem}{suffix_tag}{output.suffix}")
