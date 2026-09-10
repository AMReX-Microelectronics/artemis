#!/usr/bin/env python3
"""1D ADI resonance vs CFL: PEC standing-wave IC vs periodic traveling-wave IC."""

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
WORKDIR = Path("adi_dispersion/artemis_pec_kh2p15e-3")
# Full-wavelength domain; same L and Nz for PEC and periodic (kΔz = 2π/Nz).
BLOCKING_FACTOR = 8
NZ = 2 * 1464  # 2928 cells on L = 8 µm
LENGTH_Z = 8.0e-6
KH = 2.0 * math.pi / NZ  # = π/1464 ≈ 2.146e-3
KH_LABEL = format(KH, ".3e").replace("e-0", "e-")  # "2.146e-3"
N_TRANS = BLOCKING_FACTOR  # thin transverse
VERIFY_CFLS = [64.0, 128.0, 256.0, 512.0]
N_PERIODS = 16.0
SAMPLES_PER_PERIOD = 16
FFT_PAD_FACTOR = 8  # zero-pad FFT length to pad_factor * n_window
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
# Standing-wave IC: Ey(z,0)=E0 sin(2πz/L), B=0 (Ey vanishes on PEC faces at z=0,L).
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "0.0"
warpx.Ey_external_grid_function(x,y,z) = "E0*sin(2*pi*z/L)"
warpx.Ez_external_grid_function(x,y,z) = "0.0"

warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "0.0"
warpx.By_external_grid_function(x,y,z) = "0.0"
warpx.Bz_external_grid_function(x,y,z) = "0.0"
"""

PERIODIC_BODY = """\
# Traveling-wave IC for one wavelength on the periodic domain.
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
# Surface probe at z=L/4 (Ey antinode for sin(2πz/L)): integral over that face.
warpx.reduced_diags_names = Eobs0
Eobs0.type = RawEFieldReduction
Eobs0.reduction_type = integral
Eobs0.integration_type = surface
Eobs0.surface_normal = Z
Eobs0.intervals = {diag_interval}
Eobs0.reduced_function(x,y,z) = (z > z0 - dz/2) * (z < z0 + dz/2)
"""


def mode_grid(mode: str) -> tuple[float, int, float, float]:
    """Return (lz, nz, k, kh); PEC and periodic share one full-wavelength grid."""
    if mode not in MODES:
        raise ValueError(mode)
    lz, nz = LENGTH_Z, NZ
    k = 2.0 * math.pi / lz
    kh = k * (lz / nz)
    return lz, nz, k, kh


assert abs(mode_grid("pec")[3] - KH) < 1e-15
assert abs(mode_grid("periodic")[3] - KH) < 1e-15


def case_name(mode: str, cfl: float) -> str:
    return f"{mode}_cfl_{cfl:g}".replace(".", "p")


def build_inputs(mode: str, **kwargs) -> str:
    if mode == "pec":
        boundary = (
            "# PEC walls normal to propagation (z); periodic transversely.\n"
            "boundary.field_lo = periodic periodic pec\n"
            "boundary.field_hi = periodic periodic pec"
        )
        body = PEC_BODY
    elif mode == "periodic":
        boundary = (
            "# Periodic in all directions; traveling-wave IC along z.\n"
            "boundary.field_lo = periodic periodic periodic\n"
            "boundary.field_hi = periodic periodic periodic"
        )
        body = PERIODIC_BODY
    else:
        raise ValueError(mode)

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
    """Return (time, Ey) from the RawEFieldReduction surface diagnostic."""
    path = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    if not path.exists():
        raise FileNotFoundError(path)
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # [0]=step, [1]=time, [2]=Ex, [3]=Ey, [4]=Ez
    return data[:, 1], data[:, 3]


def analytical_f_over_f0(cfl: float, kh: float) -> float:
    """ADI numerical frequency normalized by exact f0 = c k / (2π)."""
    return 2.0 * math.atan(cfl * math.sin(0.5 * kh)) / (cfl * kh)


def run_artemis_case(
    mode: str, cfl: float, workdir: Path, exe: Path, *, reuse: bool = False
) -> Path:
    nx = ny = N_TRANS
    lz, nz, k, kh = mode_grid(mode)
    dz = lz / nz
    lx = nx * dz
    ly = ny * dz
    f0 = C0 * k / (2.0 * math.pi)
    t0 = 1.0 / f0
    dt = cfl * dz / C0

    steps_per_period = max(1, int(round(t0 / dt)))
    diag_interval = max(1, steps_per_period // SAMPLES_PER_PERIOD)
    nsteps = int(math.ceil(N_PERIODS * t0 / dt))

    case_dir = workdir / case_name(mode, cfl)
    probe = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    if reuse and probe.exists():
        print(f"[Artemis] reuse {mode} CFL={cfl:g}: {case_dir}")
        return case_dir

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    inputs = case_dir / "inputs"
    inputs.write_text(
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
            z0=0.25 * lz,
            dz=dz,
        )
    )

    cmd = [str(exe), str(inputs)]
    print(
        f"[Artemis] {mode} CFL={cfl:g}, N={nx}x{ny}x{nz}, L={lz:g}, kh={KH_LABEL}, "
        f"f0={f0:.6e} Hz, steps={nsteps}, diag={diag_interval}"
    )
    log_path = case_dir / "run.log"
    with log_path.open("w") as log:
        subprocess.run(cmd, cwd=case_dir, check=True, stdout=log, stderr=subprocess.STDOUT)
    return case_dir


def last_half_window(times: np.ndarray, signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Keep only the last half of the recorded samples for spectral analysis."""
    if len(times) < 4:
        raise RuntimeError("need >=4 probe samples for FFT")
    i0 = len(times) // 2
    return times[i0:], signal[i0:]


def compute_fft(
    times: np.ndarray, signal: np.ndarray, pad_factor: int = FFT_PAD_FACTOR
) -> tuple[np.ndarray, np.ndarray]:
    """FFT of last-half signal with Hanning window and zero padding.

    Magnitude is divided by the sample count N so |FFT| is comparable across
    unequal dump rates (unnormalized rfft peaks scale ~N).
    """
    t, y = last_half_window(times, signal)
    y = y - np.mean(y)
    dt = float(np.median(np.diff(t)))
    n = len(y)
    n_fft = max(n, int(pad_factor) * n)
    windowed = y * np.hanning(n)
    spec = np.fft.rfft(windowed, n=n_fft)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    return freqs, np.abs(spec) / n


def peak_frequency(times: np.ndarray, signal: np.ndarray) -> float:
    """Dominant positive frequency from a padded FFT of the last-half signal."""
    freqs, amp = compute_fft(times, signal)
    peak = 1 + int(np.argmax(amp[1:]))
    return float(freqs[peak])


def plot_resonance_histories(
    series: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]],
    outpath: Path,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex="col")
    cmap = plt.cm.viridis
    colors = {
        cfl: cmap(i / max(1, len(VERIFY_CFLS) - 1)) for i, cfl in enumerate(VERIFY_CFLS)
    }
    titles = {
        "pec": (
            rf"PEC standing-wave IC ($L$, $N_z={NZ}$, "
            rf"$k\Delta z={KH_LABEL}$)"
        ),
        "periodic": (
            rf"Periodic traveling-wave IC ($L$, $N_z={NZ}$, "
            rf"$k\Delta z={KH_LABEL}$)"
        ),
    }

    for row, mode in enumerate(MODES):
        _lz, _nz, k, kh = mode_grid(mode)
        f0 = C0 * k / (2.0 * math.pi)
        ax_t, ax_f = axes[row, 0], axes[row, 1]

        for cfl in VERIFY_CFLS:
            times, ey = series[mode][cfl]
            ax_t.plot(
                times * f0,
                ey,
                color=colors[cfl],
                lw=1.2,
                label=fr"$S={cfl:g}$",
            )
        t_ref = series[mode][VERIFY_CFLS[0]][0]
        ax_t.axvline(t_ref[len(t_ref) // 2] * f0, color="k", ls=":", lw=1.0, alpha=0.6)
        ax_t.set_ylabel(r"$\int E_y\,\mathrm{d}x\,\mathrm{d}y$ ($z=L/4$)")
        ax_t.set_title(titles[mode] + " — time")
        ax_t.grid(alpha=0.25)
        ax_t.legend(frameon=False, fontsize=8, ncol=2)

        for cfl in VERIFY_CFLS:
            freqs, amp = compute_fft(*series[mode][cfl])
            ax_f.plot(
                freqs / f0,
                amp,
                color=colors[cfl],
                lw=1.4,
                label=fr"$S={cfl:g}$",
            )
            ax_f.axvline(
                analytical_f_over_f0(cfl, kh),
                color=colors[cfl],
                ls="--",
                lw=0.9,
                alpha=0.7,
            )
        ax_f.axvline(1.0, color="k", ls=":", lw=1.3, label=r"exact $f_0$")
        ax_f.set_xlim(0.0, 1.5)
        ax_f.set_ylabel(r"$|\mathrm{FFT}|/N$")
        ax_f.set_title(
            titles[mode] + rf" — FFT (last half, pad$\times${FFT_PAD_FACTOR}, $/N$)"
        )
        ax_f.grid(alpha=0.25)
        ax_f.legend(frameon=False, fontsize=8, ncol=2)

    axes[1, 0].set_xlabel(r"$t\,f_0$")
    axes[1, 1].set_xlabel(r"$f / f_0$")
    fig.suptitle(
        rf"1-D ADI: PEC vs periodic at matched $k\Delta z={KH_LABEL}$",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_dispersion_comparison(
    f_over_f0: dict[str, dict[float, float]],
    outpath: Path,
) -> None:
    """ADI f/f0 (= vp/c) vs CFL: analytical curve + Artemis FFT peaks."""
    s_max = max(VERIFY_CFLS)
    s_adi = np.logspace(-2, np.log10(s_max * 1.05), 1600)
    kh = KH  # kΔz = 2π/Nz on the shared full-wavelength grid

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    vp_adi = 2.0 * np.arctan(s_adi * np.sin(kh / 2.0)) / (s_adi * kh)
    ax.plot(
        s_adi,
        vp_adi,
        color="k",
        lw=2,
        label=fr"ADI ana. $k\Delta z={KH_LABEL}$",
    )

    style = {
        "pec": dict(color="C0", marker="x"),
        "periodic": dict(color="C1", marker="o"),
    }
    for mode in MODES:
        cfls = np.asarray(sorted(f_over_f0[mode]))
        nums = np.asarray([f_over_f0[mode][c] for c in cfls])
        ax.plot(
            cfls,
            nums,
            style[mode]["marker"],
            color=style[mode]["color"],
            ms=8,
            mew=1.6,
            linestyle="none",
            label=fr"Artemis {mode}",
        )

    ax.axhline(1.0, color="0.4", ls="--", lw=1.5, label="Exact")
    ax.set_xscale("log")
    ax.set_xlim(min(VERIFY_CFLS) * 0.7, s_max * 1.3)
    ax.set_ylim(0.75, 1.03)
    ax.set_xlabel(r"CFL number $S=c\Delta t/\Delta z$")
    ax.set_ylabel(r"Normalized frequency $f/f_0$ ($=v_p/c$)")
    ax.set_title("ADI numerical dispersion: Artemis vs analytical")
    ax.grid(alpha=0.25, which="both")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    exe = EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    workdir = WORKDIR.resolve()
    workdir.mkdir(parents=True, exist_ok=True)

    series: dict[str, dict[float, tuple[np.ndarray, np.ndarray]]] = {m: {} for m in MODES}
    f_over_f0: dict[str, dict[float, float]] = {m: {} for m in MODES}

    for mode in MODES:
        lz, nz, k, kh = mode_grid(mode)
        f0 = C0 * k / (2.0 * math.pi)
        print(
            f"\n=== {mode}: L={lz:g} m, Nz={nz}, kh={KH_LABEL}, f0={f0:.6e} Hz ==="
        )
        print(
            f"{'CFL':>8} {'f_peak/f0':>12} {'f_adi/f0':>12} {'rel err':>12} {'n samples':>10}"
        )
        for cfl in VERIFY_CFLS:
            case_dir = run_artemis_case(mode, cfl, workdir, exe, reuse=False)
            times, ey = read_probe(case_dir)
            series[mode][cfl] = (times, ey)
            f_peak = peak_frequency(times, ey)
            f_adi = analytical_f_over_f0(cfl, kh) * f0
            rel = abs(f_peak - f_adi) / abs(f_adi)
            f_over_f0[mode][cfl] = f_peak / f0
            print(
                f"{cfl:8g} {f_peak / f0:12.6f} {f_adi / f0:12.6f} "
                f"{rel:12.3e} {len(times):10d}"
            )

    outdir = Path("adi_dispersion")
    outdir.mkdir(parents=True, exist_ok=True)

    path_res = outdir / "pec_standing_resonance_vs_cfl.png"
    plot_resonance_histories(series, path_res)
    print(path_res.resolve())

    path_disp = outdir / "pec_standing_dispersion_vs_cfl.png"
    plot_dispersion_comparison(f_over_f0, path_disp)
    print(path_disp.resolve())


if __name__ == "__main__":
    main()
