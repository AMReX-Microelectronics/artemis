#!/usr/bin/env python3
"""Run and plot the 1D-z CFL=64 in-domain CFS-PML case.

Generates the Artemis input deck, launches the solver, then writes the Ey(z)
centerline video, probe histories, and snapshot panel under adi_dispersion/.
Set ADI_PML_REUSE=1 to plot existing solver output instead.
"""

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

yt.funcs.mylog.setLevel(50)


CASE_DIR = Path("adi_dispersion/artemis_pml_cfl64_gedney")
OUTDIR = Path("adi_dispersion")
CACHE = CASE_DIR / "centerline_ey.npz"
EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")

C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

LENGTH = 8.0e-6
N_TRANS = 8
NZ = 2928
DZ = LENGTH / NZ
FREQ = C0 / LENGTH
N_PERIODS = 16.0
PML_NCELL = 128
CFL = 64.0
PML_KAPPA_MAX = 1.0
PML_ALPHA_MAX = 0.0
PML_M = 3.0
PML_R = 1.0e-8
PLOT_INTERVAL = 2

# (name, z, legend). Probe positions are generated from the case geometry.
PROBES = (
    ("Eobs0", 0.5 * PML_NCELL * DZ, r"inside lo PML (0.022$L$)"),
    ("Eobs1", (PML_NCELL + 4) * DZ, r"just inside (lo) (0.045$L$)"),
    ("Eobs2", 0.25 * LENGTH, r"$z=L/4$ (0.250$L$)"),
    ("Eobs3", 0.50 * LENGTH, r"$z=L/2$ (0.500$L$)"),
    ("Eobs4", 0.75 * LENGTH, r"$z=3L/4$ (0.750$L$)"),
    ("Eobs5", LENGTH - (PML_NCELL + 4) * DZ, r"just inside (hi) (0.955$L$)"),
    ("Eobs6", LENGTH - 0.5 * PML_NCELL * DZ, r"inside hi PML (0.978$L$)"),
)
SNAPSHOT_TF0 = (0.0, 1.5, 3.0, 6.0, 9.0, 12.0, 15.0)


INPUTS = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {ltrans:.17e} {ltrans:.17e} {length:.17e}

amr.n_cell = {ntrans} {ntrans} {nz}
amr.max_level = 0
amr.max_grid_size = {nz}
amr.blocking_factor = 8

# PML along the wave (z); periodic transversely. In-domain: outer pml_ncell
# cells of this domain are absorbing, outer z faces are Dirichlet.
boundary.field_lo = periodic periodic pml
boundary.field_hi = periodic periodic pml
warpx.do_pml_in_domain = 1
warpx.pml_ncell = {pml_ncell}
warpx.pml_kappa_max = {pml_kappa_max:.17e}
warpx.pml_alpha_max = {pml_alpha_max:.17e}
warpx.pml_m = {pml_m:.17e}
warpx.pml_R = {pml_r:.17e}

warpx.verbose = 1
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
my_constants.L = {length:.17e}
my_constants.E0 = 1.0
my_constants.dt = {dt:.17e}
my_constants.TP = {tp:.17e}
my_constants.freq = {freq:.17e}
my_constants.zpml = {pml_width:.17e}
my_constants.dz = {dz:.17e}

# Compactly supported soft source in the middle half of the domain. The
# sin^2 window and its first derivative vanish at z/L=0.25 and z/L=0.75.
warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Ex_excitation_flag_function(x,y,z) = "0.0"
warpx.Ey_excitation_flag_function(x,y,z) = "2.0*(z>0.25*L)*(z<0.75*L)"
warpx.Ez_excitation_flag_function(x,y,z) = "0.0"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ey_excitation_grid_function(x,y,z,t) = "E0*(dt/TP)*sin(2*pi*(z/L-0.25))**2*exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t)"
warpx.Ez_excitation_grid_function(x,y,z,t) = "0.0"

{probes}

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {plot_interval}
plt.fields_to_plot = Ey
plt.file_prefix = diags/plotfiles/plt
"""


def probe_block() -> str:
    names = " ".join(name for name, _z, _label in PROBES)
    lines = [f"warpx.reduced_diags_names = {names}"]
    for name, z, _label in PROBES:
        lines += [
            f"{name}.type = RawEFieldReduction",
            f"{name}.reduction_type = integral",
            f"{name}.integration_type = surface",
            f"{name}.surface_normal = Z",
            f"{name}.intervals = 1",
            f"{name}.reduced_function(x,y,z) = "
            f"(z > {z:.17e} - {DZ:.17e}/2) * "
            f"(z < {z:.17e} + {DZ:.17e}/2)",
        ]
    return "\n".join(lines)


def write_inputs(case_dir: Path) -> Path:
    dt = CFL * DZ / C0
    tp = 1.0 / FREQ
    nsteps = math.ceil(N_PERIODS * tp / dt)
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "inputs"
    path.write_text(
        INPUTS.format(
            nsteps=nsteps,
            ltrans=N_TRANS * DZ,
            length=LENGTH,
            ntrans=N_TRANS,
            nz=NZ,
            pml_ncell=PML_NCELL,
            pml_kappa_max=PML_KAPPA_MAX,
            pml_alpha_max=PML_ALPHA_MAX,
            pml_m=PML_M,
            pml_r=PML_R,
            dt=dt,
            eps0=EPS0,
            mu0=MU0,
            c0=C0,
            tp=tp,
            freq=FREQ,
            pml_width=PML_NCELL * DZ,
            dz=DZ,
            probes=probe_block(),
            plot_interval=PLOT_INTERVAL,
        )
    )
    return path


def case_complete(case_dir: Path) -> bool:
    probe = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    return probe.is_file() and bool(list_plotfiles(case_dir))


def run_case(exe: Path, case_dir: Path, *, reuse: bool) -> None:
    if reuse:
        if not case_complete(case_dir):
            raise FileNotFoundError(
                f"ADI_PML_REUSE=1 but solver output is incomplete in {case_dir}"
            )
        print(f"[Artemis] reuse {case_dir}")
        return

    # Avoid mixing plotfiles and reduced diagnostics from different runs.
    diags = case_dir / "diags"
    if diags.exists():
        shutil.rmtree(diags)
    cache = case_dir / CACHE.name
    if cache.exists():
        cache.unlink()

    inputs = write_inputs(case_dir)
    dt = CFL * DZ / C0
    nsteps = math.ceil(N_PERIODS / (FREQ * dt))
    print(
        f"[Artemis] 1D-z PML CFL={CFL:g}, "
        f"N={N_TRANS}x{N_TRANS}x{NZ}, steps={nsteps}, "
        f"plt.intervals={PLOT_INTERVAL}"
    )
    log_path = case_dir / "run.log"
    with log_path.open("w") as log:
        subprocess.run(
            [str(exe), str(inputs)],
            cwd=case_dir,
            check=True,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    print(f"[Artemis] completed; log: {log_path}")


def list_plotfiles(case_dir: Path) -> list[Path]:
    files = [
        p
        for p in (case_dir / "diags" / "plotfiles").glob("plt*")
        if p.is_dir() and p.name.startswith("plt") and p.name[3:].isdigit()
    ]
    return sorted(files, key=lambda p: int(p.name.removeprefix("plt")))


def extract_centerline(plotfile: Path) -> tuple[float, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    ds.force_periodicity()
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    ey = np.asarray(grid[("mesh", "Ey")].to_ndarray())
    z = np.asarray(grid["z"][0, 0, :].to_ndarray())
    ix = ey.shape[0] // 2
    iy = ey.shape[1] // 2
    return float(ds.current_time.to_value()), z, ey[ix, iy, :]


def load_centerline(case_dir: Path, cache: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reuse = os.environ.get("ADI_PML_REUSE", "0") == "1"
    if reuse and cache.exists():
        data = np.load(cache)
        return data["times"], data["z"], data["ey"]

    files = list_plotfiles(case_dir)
    if not files:
        raise FileNotFoundError(f"no plotfiles in {case_dir / 'diags' / 'plotfiles'}")
    times = []
    lines = []
    z = None
    for i, pf in enumerate(files):
        t, z_pf, ey = extract_centerline(pf)
        times.append(t)
        lines.append(ey)
        z = z_pf if z is None else z
        if (i + 1) % 25 == 0 or i + 1 == len(files):
            print(f"  centerline {i + 1}/{len(files)}")
    times_a = np.asarray(times)
    ey_a = np.stack(lines, axis=0)
    np.savez_compressed(cache, times=times_a, z=z, ey=ey_a)
    return times_a, z, ey_a


def shade_pml(ax) -> None:
    width_um = PML_NCELL * DZ * 1e6
    length_um = LENGTH * 1e6
    ax.axvspan(0.0, width_um, color="0.75", alpha=0.45, lw=0, zorder=0)
    ax.axvspan(length_um - width_um, length_um, color="0.75", alpha=0.45, lw=0, zorder=0)


def read_probe(case_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "diags" / "reducedfiles" / f"{name}.txt"
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # Columns: step, time, integral Ex, integral Ey, integral Ez.
    dx = DZ
    # Transverse box is 8 x 8 cells of size dz (see inputs).
    area = (8 * dx) ** 2
    return data[:, 1], data[:, 3] / area


def plot_probes(case_dir: Path, outpath: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = plt.cm.viridis
    n = len(PROBES)
    for i, (name, _z, label) in enumerate(PROBES):
        t, ey = read_probe(case_dir, name)
        ax.plot(t * FREQ, ey, color=cmap(i / max(1, n - 1)), lw=1.3, label=label)
        print(f"  probe {name}: max|Ey|={float(np.max(np.abs(ey))):.3e}")
    ax.axvline(3.0, color="k", ls=":", lw=1.0, alpha=0.7, label=r"pulse peak $t=3T_p$")
    ax.set_xlabel(r"$t\,f_0$")
    ax.set_ylabel(r"$E_y$ (surface mean at probe)")
    ax.set_title(rf"ADI CFS-PML along $z$, CFL$={CFL:g}$, $N_{{\mathrm{{pml}}}}={PML_NCELL}$")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(outpath.resolve())


def plot_snapshots(
    z: np.ndarray, times: np.ndarray, ey: np.ndarray, outpath: Path
) -> None:
    tf = times * FREQ
    z_um = z * 1e6
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    shade_pml(ax)
    cmap = plt.cm.plasma
    n = len(SNAPSHOT_TF0)
    for i, tgt in enumerate(SNAPSHOT_TF0):
        j = int(np.argmin(np.abs(tf - tgt)))
        ax.plot(
            z_um,
            ey[j],
            color=cmap(i / max(1, n - 1)),
            lw=1.4,
            label=rf"$t f_0={tf[j]:.2f}$",
        )
    ax.set_xlim(z_um[0], z_um[-1])
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$E_y$ (centerline)")
    ax.set_title(r"Centerline $E_y(x_c,y_c,z)$ snapshots")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(outpath.resolve())


def make_centerline_video(
    z: np.ndarray, times: np.ndarray, ey: np.ndarray, outpath: Path
) -> None:
    tf = times * FREQ
    z_um = z * 1e6
    peak = float(np.max(np.abs(ey)))
    ylim = 1.08 * peak if peak > 0.0 else 1.0

    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    shade_pml(ax)
    (line,) = ax.plot(z_um, ey[0], color="#9e2a2b", lw=1.6)
    ax.set_xlim(float(z_um[0]), float(z_um[-1]))
    ax.set_ylim(-ylim, ylim)
    ax.set_xlabel(r"$z$ ($\mu$m)")
    ax.set_ylabel(r"$E_y$ (centerline)")
    title = ax.set_title(rf"ADI CFS-PML along $z$, CFL$={CFL:g}$    $t f_0={tf[0]:.3f}$")
    ax.grid(alpha=0.25)
    fig.tight_layout()

    frame_dir = outpath.parent / (outpath.stem + "_frames")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    n = len(times)
    for i in range(n):
        line.set_ydata(ey[i])
        title.set_text(
            rf"ADI CFS-PML along $z$, CFL$={CFL:g}$    $t f_0={tf[i]:.3f}$"
        )
        fig.savefig(frame_dir / f"frame_{i:04d}.png", dpi=110)
        if (i + 1) % 40 == 0 or i + 1 == n:
            print(f"  video frame {i + 1}/{n}")
    plt.close(fig)

    pattern = str(frame_dir / "frame_%04d.png")
    dest = outpath.with_suffix(".webm")
    cmd = [
        "ffmpeg", "-y", "-framerate", "16", "-i", pattern,
        "-c:v", "libvpx-vp9", "-b:v", "1800k", "-pix_fmt", "yuv420p",
        str(dest),
    ]
    subprocess.run(cmd, check=True, capture_output=True, text=True)
    shutil.rmtree(frame_dir)
    print(dest.resolve())


def main() -> None:
    exe = EXE.resolve()
    if not exe.is_file():
        raise FileNotFoundError(f"executable not found: {exe}")

    case_dir = CASE_DIR.resolve()
    outdir = OUTDIR.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("ADI_PML_REUSE", "0") == "1"
    run_case(exe, case_dir, reuse=reuse)

    print(f"extracting Ey centerline from {case_dir}")
    times, z, ey = load_centerline(case_dir, CACHE.resolve())
    print(
        f"  {len(times)} frames, |Ey| max {float(np.max(np.abs(ey))):.3e}, "
        f"t f0 in [{times[0] * FREQ:.3f}, {times[-1] * FREQ:.3f}]"
    )
    plot_probes(case_dir, outdir / "pml_cfl64_probes.png")
    plot_snapshots(z, times, ey, outdir / "pml_cfl64_snapshots.png")
    make_centerline_video(z, times, ey, outdir / "pml_cfl64_centerline.mp4")


if __name__ == "__main__":
    main()
