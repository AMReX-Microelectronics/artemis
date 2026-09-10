#!/usr/bin/env python3
"""CFL=1024 long runs: PEC vs periodic midplane amplitude decay over 128 periods."""

from __future__ import annotations

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


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_dispersion/artemis_mag_decay_cfl1024")
KH_TARGET = 2.15e-3
NZ = 8 * int(round((math.pi / KH_TARGET) / 8.0))  # 1464
N_TRANS = 8
LENGTH_Z = 4.0e-6
CFL = 1024.0
N_PERIODS = 128.0
SAMPLES_PER_PERIOD = 16
MODES = ("pec", "periodic")

COMMON_HEADER = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {lx:.17e} {ly:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {nz}
amr.blocking_factor = 8

{boundary}

warpx.verbose = 0
warpx.const_dt = {dt:.17e}
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0

algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
algo.macroscopic_sigma_method = laxwendroff

macroscopic.sigma_function(x,y,z) = "0.0"
macroscopic.epsilon_function(x,y,z) = "{eps0:.17e}"
macroscopic.mu_function(x,y,z) = "{mu0:.17e}"

my_constants.pi = 3.141592653589793
my_constants.c = {c0:.17e}
my_constants.L = {lz:.17e}
my_constants.E0 = 1.0
my_constants.z0 = {z0:.17e}
my_constants.dz = {dz:.17e}
"""

PEC_BODY = """\
my_constants.freq = {freq:.17e}
my_constants.TP = {tp:.17e}
my_constants.dt = {dt:.17e}
my_constants.flag_none = 0
my_constants.flag_soft = 2

warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Ex_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ey_excitation_flag_function(x,y,z) = "flag_soft"
warpx.Ez_excitation_flag_function(x,y,z) = "flag_none"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ey_excitation_grid_function(x,y,z,t) = "E0*(dt/TP)*sin(pi*z/L)*exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t)"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0.0"
"""

PERIODIC_BODY = """\
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "0.0"
warpx.Ey_external_grid_function(x,y,z) = "E0*sin(2*pi*z/L)"
warpx.Ez_external_grid_function(x,y,z) = "0.0"

warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "-E0*sin(2*pi*z/L)/c"
warpx.By_external_grid_function(x,y,z) = "0.0"
warpx.Bz_external_grid_function(x,y,z) = "0.0"
"""

PROBE = """\
warpx.reduced_diags_names = Eobs0
Eobs0.type = RawEFieldReduction
Eobs0.reduction_type = integral
Eobs0.integration_type = surface
Eobs0.surface_normal = Z
Eobs0.intervals = {diag_interval}
Eobs0.reduced_function(x,y,z) = (z > z0 - dz/2) * (z < z0 + dz/2)
"""


def mode_grid(mode: str) -> tuple[float, int, float, float]:
    if mode == "pec":
        lz, nz = LENGTH_Z, NZ
        k = math.pi / lz
    elif mode == "periodic":
        lz, nz = 2.0 * LENGTH_Z, 2 * NZ
        k = 2.0 * math.pi / lz
    else:
        raise ValueError(mode)
    kh = k * (lz / nz)
    return lz, nz, k, kh


def case_name(mode: str) -> str:
    return f"{mode}_cfl_{CFL:g}_T{int(N_PERIODS)}".replace(".", "p")


def build_inputs(mode: str, **kwargs) -> str:
    if mode == "pec":
        boundary = (
            "boundary.field_lo = periodic periodic pec\n"
            "boundary.field_hi = periodic periodic pec"
        )
        body = PEC_BODY.format(freq=kwargs["freq"], tp=kwargs["tp"], dt=kwargs["dt"])
    else:
        boundary = (
            "boundary.field_lo = periodic periodic periodic\n"
            "boundary.field_hi = periodic periodic periodic"
        )
        body = PERIODIC_BODY

    header = COMMON_HEADER.format(
        nsteps=kwargs["nsteps"],
        lx=kwargs["lx"],
        ly=kwargs["ly"],
        lz=kwargs["lz"],
        nx=kwargs["nx"],
        ny=kwargs["ny"],
        nz=kwargs["nz"],
        boundary=boundary,
        dt=kwargs["dt"],
        eps0=kwargs["eps0"],
        mu0=kwargs["mu0"],
        c0=kwargs["c0"],
        z0=kwargs["z0"],
        dz=kwargs["dz"],
    )
    probe = PROBE.format(diag_interval=kwargs["diag_interval"])
    return header + "\n" + body + "\n" + probe


def read_probe(case_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    return data[:, 1], data[:, 3]


def envelope_peaks(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Local |signal| maxima as a simple oscillation envelope."""
    mag = np.abs(signal)
    if len(mag) < 3:
        return times.copy(), mag
    # Interior peaks: mag[i] >= neighbors (plateaus keep leftmost).
    peaks = (mag[1:-1] >= mag[:-2]) & (mag[1:-1] > mag[2:])
    idx = np.where(peaks)[0] + 1
    if len(idx) == 0:
        return times.copy(), mag
    return times[idx], mag[idx]


def run_case(mode: str, workdir: Path, exe: Path) -> Path:
    nx = ny = N_TRANS
    lz, nz, k, kh = mode_grid(mode)
    dz = lz / nz
    lx = nx * dz
    ly = ny * dz
    f0 = C0 * k / (2.0 * math.pi)
    t0 = 1.0 / f0
    dt = CFL * dz / C0

    steps_per_period = max(1, int(round(t0 / dt)))
    diag_interval = max(1, steps_per_period // SAMPLES_PER_PERIOD)
    nsteps = int(math.ceil(N_PERIODS * t0 / dt))

    case_dir = workdir / case_name(mode)
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    (case_dir / "inputs").write_text(
        build_inputs(
            mode,
            nsteps=nsteps,
            diag_interval=diag_interval,
            dt=dt,
            nx=nx,
            ny=ny,
            nz=nz,
            lx=lx,
            ly=ly,
            lz=lz,
            eps0=EPS0,
            mu0=MU0,
            c0=C0,
            freq=f0,
            tp=t0,
            z0=0.5 * lz,
            dz=dz,
        )
    )

    print(
        f"[Artemis] {mode} CFL={CFL:g}, N={nx}x{ny}x{nz}, L={lz:g}, kh={kh:.6e}, "
        f"f0={f0:.6e} Hz, periods={N_PERIODS:g}, steps={nsteps}, diag={diag_interval}"
    )
    with (case_dir / "run.log").open("w") as log:
        subprocess.run(
            [str(exe), str(case_dir / "inputs")],
            cwd=case_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    return case_dir


def main() -> None:
    exe = EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    workdir = WORKDIR.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    series: dict[str, tuple[np.ndarray, np.ndarray, float]] = {}
    for mode in MODES:
        case_dir = run_case(mode, workdir, exe)
        times, ey = read_probe(case_dir)
        f0 = C0 * mode_grid(mode)[2] / (2.0 * math.pi)
        series[mode] = (times, ey, f0)
        print(f"  {mode}: {len(times)} samples, t_max/T0={times[-1]*f0:.2f}")

    fig, axes = plt.subplots(2, 1, figsize=(10, 7.2), sharex=True)

    styles = {
        "pec": dict(color="C0", label="PEC soft drive"),
        "periodic": dict(color="C1", label="Periodic sine IC"),
    }

    # Raw midplane signal
    for mode in MODES:
        times, ey, f0 = series[mode]
        axes[0].plot(
            times * f0,
            ey,
            color=styles[mode]["color"],
            lw=0.8,
            alpha=0.85,
            label=styles[mode]["label"],
        )
    axes[0].axvline(6.0, color="k", ls=":", lw=1.0, alpha=0.5)
    axes[0].set_ylabel(r"$\int E_y\,\mathrm{d}x\,\mathrm{d}y$ (midplane)")
    axes[0].set_title(rf"CFL$={CFL:g}$: midplane signal ({int(N_PERIODS)} periods)")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=9)

    # Magnitude envelope (local peaks of |Ey|), normalized to post-drive / initial peak
    for mode in MODES:
        times, ey, f0 = series[mode]
        t_pk, a_pk = envelope_peaks(times, ey)
        if mode == "pec":
            # After soft-source pulse (~t > 6 T0)
            mask = t_pk * f0 >= 6.0
            if np.count_nonzero(mask) < 2:
                mask = np.ones(len(t_pk), dtype=bool)
            t_use, a_use = t_pk[mask], a_pk[mask]
            a0 = a_use[0]
        else:
            t_use, a_use = t_pk, a_pk
            a0 = a_use[0] if a_use[0] > 0 else np.max(a_use)
        axes[1].semilogy(
            t_use * f0,
            a_use / (a0 + 1e-300),
            color=styles[mode]["color"],
            lw=1.6,
            marker="o",
            ms=2.5,
            label=styles[mode]["label"],
        )
    axes[1].axvline(6.0, color="k", ls=":", lw=1.0, alpha=0.5)
    axes[1].set_xlabel(r"$t\,f_0$")
    axes[1].set_ylabel(r"$|E|_{\mathrm{env}}\,/\,|E|_{\mathrm{env},0}$")
    axes[1].set_title("Magnitude envelope decay")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=9)
    axes[1].set_xlim(0.0, N_PERIODS)

    kh = mode_grid("pec")[3]
    fig.suptitle(
        rf"1-D ADI amplitude decay at CFL$={CFL:g}$, $k\Delta z={kh:.3g}$",
        fontsize=13,
    )
    fig.tight_layout()

    outdir = Path("adi_dispersion")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "magnitude_decay_cfl1024.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


if __name__ == "__main__":
    main()
