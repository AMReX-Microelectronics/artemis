"""yt/AMReX plotfile field loading for diagnostic postprocessing."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import yt

from .common import AXIS_NAME, PLANES, map_jobs


def load_field(grid, variable: str) -> np.ndarray:
    if variable in ("Ex", "Ey", "Ez", "Bx", "By", "Bz", "epsilon"):
        return grid[("boxlib", variable)].to_ndarray()

    prefix = variable[1]  # E or B from |E| or |B|
    components = [
        grid[("boxlib", f"{prefix}{axis}")].to_ndarray() for axis in "xyz"
    ]
    return np.sqrt(sum(component**2 for component in components))


def load_slice(
    path: Path,
    variable: str,
    plane: str,
    slice_index: int | None,
) -> tuple[float, np.ndarray, list[float], int]:
    """Load one 2D slice from an AMReX plotfile (only the requested field)."""
    ax0, ax1, normal = PLANES[plane]
    ds = yt.load(str(path))
    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    field = load_field(grid, variable)
    dims = field.shape
    fixed = dims[normal] // 2 if slice_index is None else slice_index
    if not 0 <= fixed < dims[normal]:
        raise IndexError(
            f"slice index {fixed} is outside axis {AXIS_NAME[normal]} "
            f"with size {dims[normal]} in {path}"
        )

    index = [slice(None)] * 3
    index[normal] = fixed
    # Copy so worker processes return self-contained arrays.
    slice2d = np.asarray(field[tuple(index)], dtype=np.float64).copy()

    centers = []
    for axis in (ax0, ax1):
        edges = np.linspace(
            float(ds.domain_left_edge[axis]),
            float(ds.domain_right_edge[axis]),
            dims[axis] + 1,
        )
        centers.append(0.5 * (edges[:-1] + edges[1:]))
    extent = [
        float(centers[0][0] * 1e3),
        float(centers[0][-1] * 1e3),
        float(centers[1][0] * 1e3),
        float(centers[1][-1] * 1e3),
    ]
    return float(ds.current_time), slice2d, extent, fixed


def _load_slice_job(
    job: tuple[str, str, str, int | None],
) -> tuple[float, np.ndarray, list[float], int]:
    path, variable, plane, slice_index = job
    return load_slice(Path(path), variable, plane, slice_index)


def preload_slices(
    all_paths: list[list[Path]],
    variable: str,
    plane: str,
    slice_index: int | None,
    workers: int,
) -> tuple[np.ndarray, list[list[np.ndarray]], list[list[float]], list[int]]:
    """Load every series/frame slice, optionally in parallel.

    Returns times[n_frames], slices[n_series][n_frames], extents[n_series],
    and fixed_indices[n_series].
    """
    n_series = len(all_paths)
    n_frames = len(all_paths[0])
    # Jobs ordered as (frame0 series0..N), (frame1 series0..N), ...
    jobs = [
        (str(all_paths[series][frame]), variable, plane, slice_index)
        for frame in range(n_frames)
        for series in range(n_series)
    ]
    loaded = map_jobs(_load_slice_job, jobs, workers)

    times = np.empty(n_frames, dtype=np.float64)
    slices: list[list[np.ndarray]] = [[None] * n_frames for _ in range(n_series)]  # type: ignore[list-item]
    extents: list[list[float]] = [[]] * n_series
    fixed_indices = [0] * n_series

    for frame in range(n_frames):
        frame_times = []
        for series in range(n_series):
            time, slice2d, extent, fixed = loaded[frame * n_series + series]
            frame_times.append(time)
            slices[series][frame] = slice2d
            extents[series] = extent
            fixed_indices[series] = fixed
        frame_times_arr = np.asarray(frame_times)
        tolerance = max(1.0e-15, 1.0e-8 * float(np.max(np.abs(frame_times_arr))))
        if not np.allclose(
            frame_times_arr, frame_times_arr[0], rtol=1.0e-8, atol=tolerance
        ):
            raise ValueError(
                f"Physical times do not match at frame {frame}: {frame_times}"
            )
        times[frame] = frame_times_arr[0]

    return times, slices, extents, fixed_indices


def load_field_samples(
    path: Path,
    components: tuple[str, ...],
    plane: str,
    slice_index: int | None,
    cell_index: tuple[int, int, int] | None,
) -> tuple[float, dict[str, float], str]:
    """Sample E components at one cell, or average over a mid-plane slice."""
    ds = yt.load(str(path))
    dims = tuple(int(n) for n in ds.domain_dimensions)

    if cell_index is not None:
        # Point selection reads only grids that contain the cell, avoiding a
        # full-domain covering_grid materialization.
        i, j, k = cell_index
        for axis, idx in enumerate((i, j, k)):
            if not 0 <= idx < dims[axis]:
                raise IndexError(
                    f"index {cell_index} is outside domain dims {dims} in {path}"
                )
        dds = ds.domain_width / ds.domain_dimensions
        center = ds.domain_left_edge + dds * (
            np.array([i, j, k], dtype=np.float64) + 0.5
        )
        point = ds.point(center)
        values = {
            component: float(np.asarray(point[("boxlib", component)]).ravel()[0])
            for component in components
        }
        return (
            float(ds.current_time),
            values,
            f"cell (i,j,k)=({i},{j},{k})",
        )

    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    _, _, normal = PLANES[plane]
    fixed = dims[normal] // 2 if slice_index is None else int(slice_index)
    if not 0 <= fixed < dims[normal]:
        raise IndexError(
            f"slice index {fixed} is outside axis {AXIS_NAME[normal]} "
            f"with size {dims[normal]} in {path}"
        )
    index = [slice(None)] * 3
    index[normal] = fixed
    values = {
        component: float(
            np.mean(
                np.asarray(grid[("boxlib", component)].to_ndarray(), dtype=np.float64)[
                    tuple(index)
                ]
            )
        )
        for component in components
    }
    return (
        float(ds.current_time),
        values,
        f"{plane} mid-plane ({AXIS_NAME[normal]}-index {fixed})",
    )


def _load_field_samples_job(
    job: tuple[str, tuple[str, ...], str, int | None, tuple[int, int, int] | None],
) -> tuple[float, dict[str, float], str]:
    path, components, plane, slice_index, cell_index = job
    return load_field_samples(Path(path), components, plane, slice_index, cell_index)


def preload_field_series(
    all_paths: list[list[Path]],
    components: tuple[str, ...],
    plane: str,
    slice_index: int | None,
    cell_index: tuple[int, int, int] | None,
    workers: int,
) -> tuple[list[tuple[np.ndarray, dict[str, np.ndarray]]], str]:
    """Load every series/frame sample, optionally in parallel.

    Returns one (times, values) pair per series (frame order preserved),
    plus a location label from the first loaded frame.
    """
    n_series = len(all_paths)
    frame_counts = [len(paths) for paths in all_paths]
    # Jobs ordered as series0 frames..., series1 frames..., ...
    jobs = [
        (str(path), components, plane, slice_index, cell_index)
        for paths in all_paths
        for path in paths
    ]
    loaded = map_jobs(_load_field_samples_job, jobs, workers)

    data: list[tuple[np.ndarray, dict[str, np.ndarray]]] = []
    location = loaded[0][2] if loaded else ""
    offset = 0
    for series in range(n_series):
        n_frames = frame_counts[series]
        series_loaded = loaded[offset : offset + n_frames]
        offset += n_frames

        times = np.empty(n_frames, dtype=np.float64)
        values = {
            component: np.empty(n_frames, dtype=np.float64) for component in components
        }
        for frame, (time, field_values, _) in enumerate(series_loaded):
            times[frame] = time
            for component in components:
                values[component][frame] = field_values[component]
        data.append((times, values))

    return data, location
