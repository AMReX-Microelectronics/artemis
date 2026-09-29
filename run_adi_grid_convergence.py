#!/usr/bin/env python3
"""Run a small periodic plane-wave grid convergence test for the ADI solver."""

from __future__ import annotations

import argparse
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/artemis_matplotlib")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yt


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6


# Plane-wave IC for propagation along +axis with E×B // +axis.
# direction -> (E component, B component, B sign relative to E/c, parser coord)
WAVE_CONFIG = {
    "x": ("Ey", "Bz", "+", "x"),
    "y": ("Ez", "Bx", "+", "y"),
    "z": ("Ey", "Bx", "-", "z"),
}


INPUT_TEMPLATE = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {length:.17e} {length:.17e} {length:.17e}

amr.n_cell = {n} {n} {n}
amr.max_level = 0
amr.max_grid_size = {n}
amr.blocking_factor = 8

boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic

warpx.verbose = 1
warpx.const_dt = {dt:.17e}
warpx.use_filter = 0

algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "0.0"
macroscopic.epsilon_function(x,y,z) = "{eps0:.17e}"
macroscopic.mu_function(x,y,z) = "{mu0:.17e}"

my_constants.pi = 3.141592653589793
my_constants.c = {c0:.17e}
my_constants.L = {length:.17e}
my_constants.E0 = {amplitude:.17e}

warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "{ex_fun}"
warpx.Ey_external_grid_function(x,y,z) = "{ey_fun}"
warpx.Ez_external_grid_function(x,y,z) = "{ez_fun}"

warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "{bx_fun}"
warpx.By_external_grid_function(x,y,z) = "{by_fun}"
warpx.Bz_external_grid_function(x,y,z) = "{bz_fun}"

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {plot_interval}
plt.fields_to_plot = Ex Ey Ez Bx By Bz
plt.file_prefix = {plot_prefix}
"""


def wave_field_functions(direction: str) -> dict[str, str]:
    e_comp, b_comp, b_sign, coord = WAVE_CONFIG[direction]
    sine = f"E0*sin(2*pi*{coord}/L)"
    b_fun = f"{b_sign}{sine}/c" if b_sign == "-" else f"{sine}/c"
    fields = {
        "ex_fun": "0.0",
        "ey_fun": "0.0",
        "ez_fun": "0.0",
        "bx_fun": "0.0",
        "by_fun": "0.0",
        "bz_fun": "0.0",
    }
    fields[f"{e_comp.lower()}_fun"] = sine
    fields[f"{b_comp.lower()}_fun"] = b_fun
    return fields


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Periodic sinusoidal IC grid convergence test for ADI."
    )
    parser.add_argument(
        "--exe",
        default="Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex",
        help="WarpX/Artemis executable.",
    )
    parser.add_argument(
        "--launcher",
        nargs="*",
        default=[],
        help='Optional launcher, e.g. --launcher srun -n 1',
    )
    parser.add_argument("--cells", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--cfl", type=float, default=4.0)
    parser.add_argument("--length", type=float, default=4.0e-6)
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument(
        "--direction",
        nargs="+",
        choices=["x", "y", "z"],
        default=["x", "y", "z"],
        help="Propagation direction(s) to test.",
    )
    parser.add_argument(
        "--time-samples",
        type=int,
        default=4,
        help="Number of aligned output times in one wave period.",
    )
    parser.add_argument("--workdir", default="adi_grid_convergence")
    parser.add_argument("--plot", default="convergence.png")
    parser.add_argument("--keep", action="store_true", help="Keep old workdir contents.")
    return parser.parse_args()


def write_inputs(
    case_dir: Path, n: int, args: argparse.Namespace, direction: str
) -> tuple[Path, int, int]:
    if n % 8 != 0:
        raise ValueError(f"cell count {n} must be divisible by blocking_factor=8")
    nsteps = max(1, int(round(n / args.cfl)))
    if nsteps % args.time_samples != 0:
        raise ValueError(
            f"N={n} gives {nsteps} steps, which is not divisible by "
            f"--time-samples={args.time_samples}. Adjust --cfl or --time-samples."
        )
    plot_interval = nsteps // args.time_samples
    dt = (args.length / C0) / nsteps
    input_file = case_dir / "inputs"
    input_file.write_text(
        INPUT_TEMPLATE.format(
            n=n,
            nsteps=nsteps,
            plot_interval=plot_interval,
            dt=dt,
            length=args.length,
            amplitude=args.amplitude,
            eps0=EPS0,
            mu0=MU0,
            c0=C0,
            plot_prefix="plt",
            **wave_field_functions(direction),
        )
    )
    return input_file, nsteps, plot_interval


def plotfile_step(plotfile: Path) -> int:
    return int(plotfile.name.removeprefix("plt"))


def plotfiles(case_dir: Path) -> list[Path]:
    files = sorted(
        (p for p in case_dir.glob("plt*") if p.is_dir()), key=plotfile_step
    )
    if not files:
        raise FileNotFoundError(f"no plotfile found in {case_dir}")
    return files


def e_error_at_time(
    plotfile: Path, n: int, length: float, amplitude: float, direction: str
) -> tuple[int, float, float, float, float, tuple[int, int, int]]:
    e_comp, _, _, _ = WAVE_CONFIG[direction]
    axis = {"x": 0, "y": 1, "z": 2}[direction]

    ds = yt.load(str(plotfile))
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    field = np.asarray(grid[("mesh", e_comp)].to_ndarray())
    time = float(ds.current_time.to_value())

    dh = length / n
    coord = dh * (0.5 + np.arange(n))
    exact_1d = amplitude * np.sin(2.0 * math.pi * (coord - C0 * time) / length)
    exact = exact_1d.reshape([n if i == axis else 1 for i in range(3)])

    err = field - exact
    abs_err = np.abs(err)
    max_index = tuple(int(i) for i in np.unravel_index(np.argmax(abs_err), abs_err.shape))
    return (
        plotfile_step(plotfile),
        time,
        float(np.mean(err * err)),
        float(np.max(abs_err)),
        float(np.percentile(abs_err, 99.9)),
        max_index,
    )


def space_time_error(
    case_dir: Path,
    n: int,
    length: float,
    amplitude: float,
    expected_samples: int,
    direction: str,
) -> tuple[float, float, float, int, float, tuple[int, int, int]]:
    samples = []
    for plotfile in plotfiles(case_dir):
        samples.append(e_error_at_time(plotfile, n, length, amplitude, direction))

    samples.sort(key=lambda sample: sample[0])
    period = length / C0
    expected_times = period * np.arange(1, expected_samples + 1) / expected_samples

    if len(samples) == expected_samples + 1 and samples[0][0] == 0:
        samples = samples[1:]

    if len(samples) != expected_samples:
        raise RuntimeError(
            f"expected {expected_samples} plotfiles in {case_dir}, "
            f"found {len(samples)} after dropping an optional t=0 plotfile"
        )

    sample_times = np.array([sample[1] for sample in samples])
    if not np.allclose(sample_times, expected_times, rtol=1.0e-10, atol=1.0e-15 * period):
        raise RuntimeError(
            f"plotfile times in {case_dir} are not aligned with the requested intervals"
        )

    mean_square_errors = np.array([sample[2] for sample in samples])
    l2_space_time = math.sqrt(float(np.mean(mean_square_errors)))
    worst = max(samples, key=lambda sample: sample[3])
    p999 = max(sample[4] for sample in samples)
    return (
        l2_space_time / amplitude,
        worst[3] / amplitude,
        p999 / amplitude,
        worst[0],
        worst[1],
        worst[5],
    )


def observed_orders(errors: list[float]) -> list[float]:
    return [
        math.log(errors[i - 1] / errors[i], 2.0)
        for i in range(1, len(errors))
        if errors[i] > 0.0
    ]


def print_table(
    direction: str,
    rows: list[tuple[int, int, int, float, float, float, int, float, tuple[int, int, int]]],
) -> list[float]:
    l2_orders = observed_orders([row[3] for row in rows])
    linf_orders = observed_orders([row[4] for row in rows])
    p999_orders = observed_orders([row[5] for row in rows])
    e_comp = WAVE_CONFIG[direction][0]

    print(
        f"\n=== propagation +{direction} (error in {e_comp}) ===\n"
        "  N  steps plt_int   rel_L2_xt   order  rel_p99.9_xt   order"
        "    rel_Linf_xt   order  worst_step  worst_ijk"
    )
    for i, (
        n,
        nsteps,
        plot_interval,
        l2,
        linf,
        p999,
        worst_step,
        _worst_time,
        worst_index,
    ) in enumerate(rows):
        p2 = "" if i == 0 else f"{l2_orders[i - 1]:7.3f}"
        pinf = "" if i == 0 else f"{linf_orders[i - 1]:7.3f}"
        pp999 = "" if i == 0 else f"{p999_orders[i - 1]:7.3f}"
        print(
            f"{n:3d} {nsteps:6d} {plot_interval:7d} "
            f"{l2:11.4e} {p2:>7} {p999:13.4e} {pp999:>7}"
            f" {linf:13.4e} {pinf:>7} {worst_step:11d}  {worst_index}"
        )
    return l2_orders


def plot_convergence(
    all_rows: dict[
        str, list[tuple[int, int, int, float, float, float, int, float, tuple[int, int, int]]]
    ],
    output: Path,
) -> None:
    n_dirs = len(all_rows)
    fig, axes = plt.subplots(
        1, n_dirs, figsize=(5.2 * n_dirs, 4.2), dpi=160, squeeze=False
    )
    for ax, (direction, rows) in zip(axes[0], all_rows.items()):
        cells = np.array([row[0] for row in rows], dtype=float)
        h = 1.0 / cells
        l2 = np.array([row[3] for row in rows])
        linf = np.array([row[4] for row in rows])
        p999 = np.array([row[5] for row in rows])

        ax.loglog(h, l2, "o-", label="L2")
        ax.loglog(h, p999, "^-", label="p99.9")
        ax.loglog(h, linf, "s-", label="Linf")

        for order, style in [(1, "--"), (2, ":")]:
            ref = l2[0] * (h / h[0]) ** order
            ax.loglog(h, ref, "k" + style, alpha=0.45, label=f"O(h^{order})")

        ax.invert_xaxis()
        ax.set_xlabel("grid spacing h / L")
        ax.set_ylabel("relative error")
        ax.set_title(f"ADI sine-wave +{direction}")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend()

    fig.tight_layout()
    fig.savefig(output)
    plt.close(fig)


def run_direction(
    args: argparse.Namespace, exe: Path, workdir: Path, direction: str
) -> list[tuple[int, int, int, float, float, float, int, float, tuple[int, int, int]]]:
    dir_workdir = workdir / f"prop_{direction}"
    dir_workdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for n in args.cells:
        case_dir = dir_workdir / f"n{n:04d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        input_file, nsteps, plot_interval = write_inputs(case_dir, n, args, direction)

        cmd = [*args.launcher, str(exe), str(input_file)]
        print(
            f"[{direction}] running N={n}, steps={nsteps}, plot_interval={plot_interval}: "
            f"{' '.join(cmd)}"
        )
        subprocess.run(cmd, cwd=case_dir, check=True)

        l2, linf, p999, worst_step, worst_time, worst_index = space_time_error(
            case_dir, n, args.length, args.amplitude, args.time_samples, direction
        )
        rows.append(
            (n, nsteps, plot_interval, l2, linf, p999, worst_step, worst_time, worst_index)
        )
    return rows


def main() -> None:
    args = parse_args()
    yt.funcs.mylog.setLevel(50)

    exe = Path(args.exe).resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    workdir = Path(args.workdir).resolve()
    if workdir.exists() and not args.keep:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    all_rows = {}
    all_l2_orders = {}
    for direction in args.direction:
        rows = run_direction(args, exe, workdir, direction)
        all_rows[direction] = rows
        all_l2_orders[direction] = print_table(direction, rows)

    print("\n=== second-order check (finest L2 observed order) ===")
    ok = True
    for direction, orders in all_l2_orders.items():
        finest = orders[-1] if orders else float("nan")
        passed = finest >= 1.9
        ok = ok and passed
        status = "PASS" if passed else "FAIL"
        print(f"  +{direction}: finest L2 order = {finest:.3f}  [{status}]")
    if not ok:
        raise SystemExit("second-order convergence not verified in all directions")

    plot_path = Path(args.plot)
    if not plot_path.is_absolute():
        plot_path = workdir / plot_path
    plot_convergence(all_rows, plot_path)
    print(f"\nwrote {plot_path}")


if __name__ == "__main__":
    main()
