#!/usr/bin/env python3
"""1D ADI / Explicit Yee dispersion vs CFL, with Artemis check at kΔx≈2.15e-3."""

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

import yt


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_dispersion/artemis_kh2p15e-3")
# Crosstalk Si upper-bound scale: kΔz ≈ 2.15e-3 at 10 GHz.
# One wavelength on Nz cells => kh = 2π/Nz; pick Nz divisible by blocking_factor=8.
KH_TARGET = 2.15e-3
NZ = 8 * int(round((2.0 * math.pi / KH_TARGET) / 8.0))  # 2920 → kh ≈ 2.1517e-3
N_TRANS = 16  # thin transverse; OK once PIC CFL guards are off (no species)
LENGTH_Z = 4.0e-6
VERIFY_CFLS = [1.0, 4.0, 16.0, 64.0, 256.0, 1024.0]

INPUT_TEMPLATE = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {lx:.17e} {ly:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {nz}
amr.blocking_factor = 8

boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic

warpx.verbose = 0
warpx.const_dt = {dt:.17e}
warpx.use_filter = 0
# Field-only run: do not grow J/rho guards with CFL (PIC particle-travel padding).
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

warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "0.0"
warpx.Ey_external_grid_function(x,y,z) = "E0*sin(2*pi*z/L)"
warpx.Ez_external_grid_function(x,y,z) = "0.0"

warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "-E0*sin(2*pi*z/L)/c"
warpx.By_external_grid_function(x,y,z) = "0.0"
warpx.Bz_external_grid_function(x,y,z) = "0.0"

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {plot_interval}
plt.fields_to_plot = Ey Bx
plt.file_prefix = plt
"""


def plotfile_step(plotfile: Path) -> int:
    return int(plotfile.name.removeprefix("plt"))


def measure_vp_from_case(case_dir: Path, nz: int, lz: float) -> float:
    """Estimate vp/c from the phase of the fundamental Ey mode."""
    yt.funcs.mylog.setLevel(50)
    files = sorted(
        (p for p in case_dir.glob("plt*") if p.is_dir()), key=plotfile_step
    )
    if len(files) < 3:
        raise RuntimeError(f"need >=3 plotfiles in {case_dir}, found {len(files)}")

    k = 2.0 * math.pi / lz
    dz = lz / nz
    z = dz * (0.5 + np.arange(nz))
    kernel = np.exp(-1j * k * z)

    times = []
    coeffs = []
    for pf in files:
        ds = yt.load(str(pf))
        grid = ds.covering_grid(
            level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
        )
        ey = np.asarray(grid[("mesh", "Ey")].to_ndarray())
        # Average over thin transverse directions; wave varies only in z.
        line = ey.mean(axis=(0, 1))
        times.append(float(ds.current_time.to_value()))
        coeffs.append(np.dot(line, kernel) / nz)

    times = np.asarray(times)
    phases = np.unwrap(np.angle(np.asarray(coeffs)))
    # Skip t=0 if present; fit φ(t) ≈ φ0 - ω t
    if times[0] == 0.0 and len(times) > 3:
        times = times[1:]
        phases = phases[1:]
    omega = -np.polyfit(times, phases, 1)[0]
    return omega / (C0 * k)


def run_artemis_case(cfl: float, workdir: Path, exe: Path) -> tuple[float, float, float]:
    """Run Artemis ADI for one CFL; return (kh, vp/c_num, vp/c_ana)."""
    nx = ny = N_TRANS
    nz = NZ
    dz = LENGTH_Z / nz
    lx = nx * dz
    ly = ny * dz
    lz = LENGTH_Z
    kh = 2.0 * math.pi / nz

    # Phase advance per step; keep dump Δφ < π so np.unwrap stays unambiguous.
    nsteps = 32
    dphi = 2.0 * math.atan(cfl * math.sin(0.5 * kh))
    plot_interval = max(1, min(4, int(0.9 * math.pi / dphi)))
    dt = cfl * dz / C0

    case_dir = workdir / f"cfl_{cfl:g}".replace(".", "p")
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)

    inputs = case_dir / "inputs"
    inputs.write_text(
        INPUT_TEMPLATE.format(
            nsteps=nsteps,
            plot_interval=plot_interval,
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
        )
    )

    cmd = [str(exe), str(inputs)]
    print(
        f"[Artemis] CFL={cfl:g}, N={nx}x{ny}x{nz}, kh={kh:.6e}, steps={nsteps}: "
        f"{' '.join(cmd)}"
    )
    subprocess.run(cmd, cwd=case_dir, check=True)

    vp_num = measure_vp_from_case(case_dir, nz, lz)
    vp_ana = 2.0 * math.atan(cfl * math.sin(0.5 * kh)) / (cfl * kh)
    return kh, vp_num, vp_ana


def main() -> None:
    # Analytical curves
    s_adi = np.logspace(-2, np.log10(1024), 1600)
    s_exp = np.linspace(0.01, 0.999, 1000)
    ks = [2.15e-3, 0.01, 0.05, 0.1, 0.5, 1.0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)

    for kdx in ks:
        vp_adi = 2 * np.arctan(s_adi * np.sin(kdx / 2)) / (s_adi * kdx)
        axes[0].plot(s_adi, vp_adi, lw=2, label=fr"$k\Delta x={kdx:g}$")

    axes[0].axhline(1.0, color="k", ls="--", lw=1.5, label="Exact")
    axes[0].set_xscale("log")
    axes[0].set_xlim(1e-2, 1024)
    axes[0].set_ylim(0, 1.03)
    axes[0].set_xlabel(r"CFL number $S=c\Delta t/\Delta x$")
    axes[0].set_ylabel(r"Normalized phase velocity $v_p/c$")
    axes[0].set_title("ADI-FDTD")
    axes[0].grid(alpha=0.25, which="both")

    for kdx in ks:
        vp_exp = 2 * np.arcsin(s_exp * np.sin(kdx / 2)) / (s_exp * kdx)
        axes[1].plot(s_exp, vp_exp, lw=2, label=fr"$k\Delta x={kdx:g}$")

    axes[1].axhline(1.0, color="k", ls="--", lw=1.5, label="Exact")
    axes[1].axvline(1.0, color="r", ls=":", lw=1.5)
    axes[1].text(
        0.985,
        0.08,
        "CFL limit",
        rotation=90,
        color="r",
        ha="right",
        va="bottom",
        transform=axes[1].get_xaxis_transform(),
    )
    axes[1].set_xlim(0, 1.02)
    axes[1].set_xlabel(r"CFL number $S=c\Delta t/\Delta x$")
    axes[1].set_title("Explicit Yee FDTD")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=9)

    # Artemis numerical experiment at kΔx ≈ 2.15e-3
    exe = EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    workdir = WORKDIR.resolve()
    if workdir.exists():
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True)

    kh_grid = 2.0 * math.pi / NZ
    print(f"Artemis ADI numerical experiment for kΔx ≈ {KH_TARGET:g} (grid kh={kh_grid:.6e})")
    print(f"{'CFL':>8} {'kh':>12} {'vp/c num':>14} {'vp/c ana':>14} {'rel err':>12}")
    cfls = []
    vp_nums = []
    kh_used = None
    for cfl in VERIFY_CFLS:
        kh_used, vp_num, vp_ana = run_artemis_case(cfl, workdir, exe)
        rel = abs(vp_num - vp_ana) / abs(vp_ana)
        print(f"{cfl:8g} {kh_used:12.6e} {vp_num:14.8f} {vp_ana:14.8f} {rel:12.3e}")
        cfls.append(cfl)
        vp_nums.append(vp_num)

    axes[0].plot(
        cfls,
        vp_nums,
        "kx",
        ms=7,
        mew=1.5,
        label=fr"Artemis $k\Delta x={kh_used:.3g}$",
    )
    axes[0].legend(frameon=False, fontsize=9)

    fig.suptitle(r"1-D numerical dispersion versus CFL number", fontsize=14)
    fig.tight_layout()

    outdir = Path("adi_dispersion")
    outdir.mkdir(parents=True, exist_ok=True)
    path = outdir / "fdtd_dispersion_vs_cfl_1024.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


if __name__ == "__main__":
    main()
