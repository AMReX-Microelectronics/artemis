#!/usr/bin/env python3
"""Quasi-2D (xy) ADI: center Gaussian pulse, PEC vs in-domain PML.

ADI is implemented only for 3D Cartesian grids, so this uses a thin periodic
slab in z and evolves Ez(x,y) in the mid-plane.  The drive is a soft
spatially Gaussian, temporally Gaussian-modulated sine at the domain center.

Runs twice (PEC walls in x/y, then an extended computational box with
CFS-PML inside its outer cells) and writes probe plots plus a 2D mid-plane video.
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
from matplotlib.animation import FuncAnimation, PillowWriter
import yt

yt.funcs.mylog.setLevel(50)


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_dispersion/artemis_2d_xy_pulse")
OUTDIR = Path("adi_dispersion")

BLOCKING_FACTOR = 8
NX = NY = 256
NZ = BLOCKING_FACTOR  # thin periodic z
LENGTH = 8.0e-6  # Lx = Ly
# Moderate CFL so several samples fit in a carrier period on this grid.
CFL = 4.0
N_WAVELENGTHS = 8.0  # λ = L / N_WAVELENGTHS
N_PERIODS = 24.0
PLOT_SAMPLES_PER_PERIOD = 8
# CFS-PML inside an extended computational box (do_pml_in_domain = 1).
# 32 cells ≈ λ on each side of the unchanged physical [0,L] box.
PML_NCELL = 32
PML_KAPPA_MAX = 6.0  # Tested: avoids the late bounce seen with kappa_max=48.
PML_ALPHA_MAX = 48.0  # SI conductivity units; much smaller than ω ε0 here.
PML_M = 3.0
PML_R = 1.0e-8
# Artemis Full diag default path is diags/<name><digits>, not diags/plotfiles/.
PLOT_PREFIX = "diags/plt"
CASES = ("pec", "pml")


INPUTS = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = {xlo:.17e} {ylo:.17e} 0.0
geometry.prob_hi = {xhi:.17e} {yhi:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {max_grid_size}
amr.blocking_factor = 8

{boundary}

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
my_constants.Lx = {lx:.17e}
my_constants.Ly = {ly:.17e}
my_constants.E0 = 1.0
my_constants.dt = {dt:.17e}
my_constants.TP = {tp:.17e}
my_constants.freq = {freq:.17e}
my_constants.xc = {xc:.17e}
my_constants.yc = {yc:.17e}
my_constants.sig = {sig:.17e}
my_constants.xpml = {xpml:.17e}
my_constants.ypml = {ypml:.17e}

# Soft Ez drive: spatial Gaussian at domain center, Gaussian-modulated sine
# in time. Soft flag (=2) so ADI adds ~f per step via the RHS.
warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Apply_E_excitation_in_pml_region = 0
warpx.Ex_excitation_flag_function(x,y,z) = "0.0"
warpx.Ey_excitation_flag_function(x,y,z) = "0.0"
warpx.Ez_excitation_flag_function(x,y,z) = "{ez_flag}"
warpx.Ex_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ey_excitation_grid_function(x,y,z,t) = "0.0"
warpx.Ez_excitation_grid_function(x,y,z,t) = \
"E0*(dt/TP)*exp(-((x-xc)**2+(y-yc)**2)/(2*sig**2))*exp(-(t-3*TP)**2/(2*TP**2))*sin(2*pi*freq*t)"

{probes}

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {plot_interval}
plt.fields_to_plot = Ez
plt.file_min_digits = 6
"""


def boundary_block(case: str) -> str:
    if case == "pec":
        return (
            "# PEC in x/y (wave plane); periodic in thin z.\n"
            "boundary.field_lo = pec pec periodic\n"
            "boundary.field_hi = pec pec periodic"
        )
    if case == "pml":
        return (
            "# In-domain CFS-PML in the extended x/y box; "
            "periodic in thin z.\n"
            "boundary.field_lo = pml pml periodic\n"
            "boundary.field_hi = pml pml periodic\n"
            "warpx.do_pml_in_domain = 1\n"
            f"warpx.pml_ncell = {PML_NCELL}\n"
            f"warpx.pml_kappa_max = {PML_KAPPA_MAX:.17e}\n"
            f"warpx.pml_alpha_max = {PML_ALPHA_MAX:.17e}\n"
            f"warpx.pml_m = {PML_M:.17e}\n"
            f"warpx.pml_R = {PML_R:.17e}"
        )
    raise ValueError(case)


def ez_flag(case: str) -> str:
    # Soft source over the physical box only; the added outer cells are PML.
    return "2.0*(x>xpml)*(x<Lx-xpml)*(y>ypml)*(y<Ly-ypml)"



def draw_physical_pml_interface(ax, meta: dict) -> None:
    """Mark the physical-box / PML interface in the extended domain."""
    if meta.get("case") != "pml":
        return
    x0 = meta["phys_lo_x"] * 1e6
    x1 = meta["phys_hi_x"] * 1e6
    y0 = meta["phys_lo_y"] * 1e6
    y1 = meta["phys_hi_y"] * 1e6
    for xv in (x0, x1):
        ax.axvline(xv, color="k", ls="--", lw=0.9, alpha=0.55)
    for yv in (y0, y1):
        ax.axhline(yv, color="k", ls="--", lw=0.9, alpha=0.55)


def probe_plan(
    case: str, lx: float, ly: float, dx: float, dy: float
) -> list[tuple[str, float, float, str]]:
    xc, yc = 0.5 * lx, 0.5 * ly
    # The added PML does not steal physical cells; probe near physical walls.
    margin = 4.0 * dx
    return [
        ("Eobs0", xc, yc, "center"),
        ("Eobs1", xc + 0.20 * lx, yc, r"$0.2L$ east"),
        ("Eobs2", xc, yc + 0.20 * ly, r"$0.2L$ north"),
        ("Eobs3", lx - margin, yc, "near +x wall"),
        ("Eobs4", xc, ly - margin, "near +y wall"),
    ]


def probe_block(
    probes: list[tuple[str, float, float, str]],
    dx: float,
    dy: float,
    dz: float,
    zc: float,
    interval: int,
) -> str:
    # Use a 2-cell window. Ez is cell-centered in x/y, so a strict 1-cell
    # window about a node-aligned center can miss every Ez sample.
    names = " ".join(name for name, _, _, _ in probes)
    lines = [f"warpx.reduced_diags_names = {names}"]
    for name, x0, y0, _label in probes:
        lines += [
            f"{name}.type = RawEFieldReduction",
            f"{name}.reduction_type = integral",
            f"{name}.integration_type = volume",
            f"{name}.intervals = {interval}",
            f"{name}.reduced_function(x,y,z) = "
            f"(x > {x0:.17e} - {dx:.17e}) * (x < {x0:.17e} + {dx:.17e}) * "
            f"(y > {y0:.17e} - {dy:.17e}) * (y < {y0:.17e} + {dy:.17e}) * "
            f"(z > {zc:.17e} - {dz:.17e}) * (z < {zc:.17e} + {dz:.17e})",
        ]
    return "\n".join(lines)


def list_plotfiles(case_dir: Path) -> list[Path]:
    # Artemis Full diag default: diags/<diag_name><digits> (e.g. diags/plt000000).
    # Also accept an explicit plotfiles/ subdir if a file_prefix set one.
    candidates: list[Path] = []
    for pattern in (f"{PLOT_PREFIX}*", "diags/plt*", "diags/plotfiles/plt*"):
        candidates.extend(
            p
            for p in case_dir.glob(pattern)
            if p.is_dir() and p.name.startswith("plt") and p.name[3:].isdigit()
        )
    # De-duplicate while preserving sorted order by step index.
    uniq = {p.resolve(): p for p in candidates}
    return sorted(uniq.values(), key=lambda p: int(p.name.removeprefix("plt")))


def case_complete(case_dir: Path) -> bool:
    probe = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    return probe.exists() and bool(list_plotfiles(case_dir))


def write_inputs(case: str, case_dir: Path) -> tuple[Path, dict]:
    lx = ly = LENGTH
    dx = lx / NX
    dy = ly / NY
    dz = dx
    lz = NZ * dz
    wavelength = lx / N_WAVELENGTHS
    freq = C0 / wavelength
    tp = 1.0 / freq
    dt = CFL * dx / C0
    nsteps = int(math.ceil(N_PERIODS * tp / dt))
    steps_per_period = max(1, int(round(tp / dt)))
    plot_interval = max(1, steps_per_period // PLOT_SAMPLES_PER_PERIOD)
    # The physical box stays [0,Lx]x[0,Ly] in both cases.
    xpml = ypml = 0.0
    pml_width_x = PML_NCELL * dx if case == "pml" else 0.0
    pml_width_y = PML_NCELL * dy if case == "pml" else 0.0
    xlo = -pml_width_x if case == "pml" else 0.0
    ylo = -pml_width_y if case == "pml" else 0.0
    xhi, yhi = lx + pml_width_x, ly + pml_width_y
    nx = NX + (2 * PML_NCELL if case == "pml" else 0)
    ny = NY + (2 * PML_NCELL if case == "pml" else 0)
    xc, yc = 0.5 * lx, 0.5 * ly
    zc = 0.5 * lz
    sig = wavelength / 3.0
    probes = probe_plan(case, lx, ly, dx, dy)

    meta = {
        "case": case,
        "nsteps": nsteps,
        "dt": dt,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "lx": lx,
        "ly": ly,
        "lz": lz,
        "nx": NX,
        "ny": NY,
        "nz": NZ,
        "freq": freq,
        "tp": tp,
        "wavelength": wavelength,
        "xpml": xpml,
        "ypml": ypml,
        # Physical box is unchanged; the PML occupies the added outer cells.
        "phys_lo_x": 0.0,
        "phys_hi_x": lx,
        "phys_lo_y": 0.0,
        "phys_hi_y": ly,
        "pml_ncell": PML_NCELL if case == "pml" else 0,
        "pml_thickness": (PML_NCELL * dx) if case == "pml" else 0.0,
        "grid_nx": nx,
        "grid_ny": ny,
        "xc": xc,
        "yc": yc,
        "sig": sig,
        "plot_interval": plot_interval,
        "probes": probes,
        "cell_volume": dx * dy * dz,
    }
    text = INPUTS.format(
        nsteps=nsteps,
        xlo=xlo,
        xhi=xhi,
        ylo=ylo,
        yhi=yhi,
        lx=lx,
        ly=ly,
        lz=lz,
        nx=nx,
        ny=ny,
        nz=NZ,
        max_grid_size=max(nx, ny, NZ),
        boundary=boundary_block(case),
        dt=dt,
        eps0=EPS0,
        mu0=MU0,
        c0=C0,
        tp=tp,
        freq=freq,
        xc=xc,
        yc=yc,
        sig=sig,
        xpml=xpml,
        ypml=ypml,
        ez_flag=ez_flag(case),
        probes=probe_block(probes, dx, dy, dz, zc, 1),
        plot_interval=plot_interval,
    )
    case_dir.mkdir(parents=True, exist_ok=True)
    path = case_dir / "inputs"
    path.write_text(text)
    return path, meta


def run_case(exe: Path, case: str, case_dir: Path, reuse: bool) -> dict:
    if reuse and case_complete(case_dir):
        if case == "pml":
            previous = (case_dir / "inputs").read_text()
            expected_cells = f"amr.n_cell = {NX + 2 * PML_NCELL} {NY + 2 * PML_NCELL} {NZ}"
            if ("warpx.do_pml_in_domain = 1" not in previous
                    or expected_cells not in previous):
                raise RuntimeError(
                    "Existing PML output uses a different geometry; rerun without "
                    "ADI_PML_REUSE=1 to generate the in-domain case."
                )
        _, meta = write_inputs(case, case_dir)
        print(f"[Artemis] reuse {case_dir}")
        return meta

    if case_dir.exists():
        shutil.rmtree(case_dir)
    inputs, meta = write_inputs(case, case_dir)
    pml_note = ""
    if case == "pml":
        pml_note = (
            f", in-domain PML {meta['pml_ncell']} cells/side "
            f"(+{meta['pml_thickness']*1e6:.3g} µm each side)"
        )
    print(
        f"[Artemis] {case.upper()} 2D-xy CFL={CFL:g}, "
        f"N={meta['nx']}x{meta['ny']}x{meta['nz']} physical, L={meta['lx']:g}, "
        f"grid={meta['grid_nx']}x{meta['grid_ny']}x{meta['nz']}, "
        f"λ={meta['wavelength']:g}, steps={meta['nsteps']}, "
        f"plt.intervals={meta['plot_interval']}{pml_note}"
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
    return meta


def read_probe(case_dir: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    path = case_dir / "diags" / "reducedfiles" / f"{name}.txt"
    data = np.loadtxt(path, comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    # RawEFieldReduction writes Ex, Ey, Ez after (step, time).
    return data[:, 1], data[:, -1]


def sample_probes_from_slices(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    ez: np.ndarray,
    probes: list[tuple[str, float, float, str]],
) -> list[tuple[str, str, np.ndarray, np.ndarray]]:
    """Nearest-cell Ez(t) at each probe from mid-plane slices."""
    out = []
    for name, x0, y0, label in probes:
        ix = int(np.argmin(np.abs(x - x0)))
        iy = int(np.argmin(np.abs(y - y0)))
        out.append((name, label, times, ez[:, ix, iy]))
    return out


def extract_ez_slice(
    plotfile: Path,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    ds = yt.load(str(plotfile))
    # The PML has a non-periodic box; covering_grid can overshoot
    # domain_right_edge by ~1 ULP and yt aborts. Force periodicity for the
    # read only (no MPI wrap; just relaxes the edge check).
    ds.force_periodicity()
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    ez = np.asarray(grid[("mesh", "Ez")].to_ndarray())
    x = np.asarray(grid["x"][:, 0, 0].to_ndarray())
    y = np.asarray(grid["y"][0, :, 0].to_ndarray())
    nz = ez.shape[2]
    return float(ds.current_time.to_value()), x, y, ez[:, :, nz // 2]


def load_slices(
    case_dir: Path, cache: Path
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if cache.exists():
        data = np.load(cache)
        return data["times"], data["x"], data["y"], data["ez"]

    files = list_plotfiles(case_dir)
    if not files:
        raise FileNotFoundError(f"no plotfiles in {case_dir}")
    times = []
    fields = []
    x = y = None
    for i, pf in enumerate(files):
        t, x_pf, y_pf, ez = extract_ez_slice(pf)
        times.append(t)
        fields.append(ez)
        x = x_pf if x is None else x
        y = y_pf if y is None else y
        if (i + 1) % 20 == 0 or i + 1 == len(files):
            print(f"  slice {i + 1}/{len(files)}")
    times_a = np.asarray(times)
    ez_a = np.stack(fields, axis=0)
    np.savez_compressed(cache, times=times_a, x=x, y=y, ez=ez_a)
    return times_a, x, y, ez_a


def plot_probes(
    meta: dict,
    outpath: Path,
    *,
    times: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    ez: np.ndarray,
) -> None:
    freq = meta["freq"]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    cmap = plt.cm.viridis
    series = sample_probes_from_slices(x, y, times, ez, meta["probes"])
    n = len(series)
    for i, (_name, label, t, ez_t) in enumerate(series):
        ax.plot(
            t * freq,
            ez_t,
            color=cmap(i / max(1, n - 1)),
            lw=1.3,
            label=label,
        )
        peak = float(np.max(np.abs(ez_t)))
        print(f"  probe {label}: max|Ez|={peak:.3e}")
    ax.axvline(3.0, color="k", ls=":", lw=1.0, alpha=0.7, label=r"pulse peak $t=3T_p$")
    ax.set_xlabel(r"$t\,f_0$")
    ax.set_ylabel(r"$E_z$ (mid-plane cell at probe)")
    ax.set_title(
        rf"ADI 2D $xy$ {meta['case'].upper()}, CFL$={CFL:g}$, "
        rf"$N={meta['nx']}\times{meta['ny']}$"
    )
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(outpath.resolve())


def plot_snapshots(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    ez: np.ndarray,
    meta: dict,
    outpath: Path,
) -> None:
    freq = meta["freq"]
    targets = np.array([0.0, 3.0, 6.0, 9.0, 12.0, 18.0])
    tf = times * freq
    vmax = float(np.nanpercentile(np.abs(ez), 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    fig, axes = plt.subplots(2, 3, figsize=(12.0, 7.5), constrained_layout=True)
    x_um = x * 1e6
    y_um = y * 1e6
    extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]
    im = None
    for ax, tgt in zip(axes.ravel(), targets, strict=True):
        j = int(np.argmin(np.abs(tf - tgt)))
        im = ax.imshow(
            ez[j].T,
            origin="lower",
            extent=extent,
            cmap="RdBu_r",
            vmin=-vmax,
            vmax=vmax,
            aspect="equal",
        )
        draw_physical_pml_interface(ax, meta)
        ax.set_title(rf"$t f_0={tf[j]:.2f}$")
        ax.set_xlabel(r"$x$ ($\mu$m)")
        ax.set_ylabel(r"$y$ ($\mu$m)")
    fig.colorbar(im, ax=axes, shrink=0.85, label=r"$E_z$")
    fig.suptitle(rf"ADI 2D $E_z(x,y)$ — {meta['case'].upper()}, CFL$={CFL:g}$")
    fig.savefig(outpath, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(outpath.resolve())


def make_video_2d(
    x: np.ndarray,
    y: np.ndarray,
    times: np.ndarray,
    ez: np.ndarray,
    meta: dict,
    outpath: Path,
) -> None:
    freq = meta["freq"]
    vmax = float(np.nanpercentile(np.abs(ez), 99.5))
    if not np.isfinite(vmax) or vmax <= 0.0:
        vmax = 1.0
    x_um = x * 1e6
    y_um = y * 1e6
    extent = [x_um[0], x_um[-1], y_um[0], y_um[-1]]

    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(
        ez[0].T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        animated=True,
    )
    draw_physical_pml_interface(ax, meta)
    cb = fig.colorbar(im, ax=ax, shrink=0.9)
    cb.set_label(r"$E_z$")
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    title = ax.set_title(
        rf"{meta['case'].upper()}  $t f_0 = {times[0] * freq:.3f}$"
    )
    fig.tight_layout()

    frame_dir = outpath.parent / (outpath.stem + "_frames")
    if frame_dir.exists():
        shutil.rmtree(frame_dir)
    frame_dir.mkdir(parents=True, exist_ok=True)
    n = len(times)
    for i in range(n):
        im.set_data(ez[i].T)
        title.set_text(rf"{meta['case'].upper()}  $t f_0 = {times[i] * freq:.3f}$")
        fig.savefig(frame_dir / f"frame_{i:04d}.png", dpi=110)
        if (i + 1) % 25 == 0 or i + 1 == n:
            print(f"  video frame {i + 1}/{n}")
    plt.close(fig)

    fps = 16
    pattern = str(frame_dir / "frame_%04d.png")
    attempts = [
        (
            outpath.with_suffix(".webm"),
            [
                "ffmpeg", "-y", "-framerate", str(fps), "-i", pattern,
                "-c:v", "libvpx-vp9", "-b:v", "1800k", "-pix_fmt", "yuv420p",
            ],
        ),
        (
            outpath.with_suffix(".gif"),
            [
                "ffmpeg", "-y", "-framerate", "12", "-i", pattern,
                "-vf", "scale=640:-1:flags=lanczos",
            ],
        ),
    ]
    last_err: Exception | None = None
    for dest, cmd in attempts:
        try:
            subprocess.run(
                cmd + [str(dest)], check=True, capture_output=True, text=True
            )
            print(dest.resolve())
            return
        except (subprocess.CalledProcessError, FileNotFoundError) as err:
            last_err = err
            print(f"video encoder failed for {dest.name}: {err}")

    gif = outpath.with_suffix(".gif")
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    im = ax.imshow(
        ez[0].T,
        origin="lower",
        extent=extent,
        cmap="RdBu_r",
        vmin=-vmax,
        vmax=vmax,
        aspect="equal",
        animated=True,
    )
    ax.set_xlabel(r"$x$ ($\mu$m)")
    ax.set_ylabel(r"$y$ ($\mu$m)")
    title = ax.set_title(
        rf"{meta['case'].upper()}  $t f_0 = {times[0] * freq:.3f}$"
    )
    fig.tight_layout()

    def update(i: int):
        im.set_data(ez[i].T)
        title.set_text(rf"{meta['case'].upper()}  $t f_0 = {times[i] * freq:.3f}$")
        return (im, title)

    anim = FuncAnimation(fig, update, frames=n, blit=True, interval=60)
    try:
        anim.save(str(gif), writer=PillowWriter(fps=10), dpi=80)
        print(gif.resolve())
        plt.close(fig)
        return
    except Exception as err:
        last_err = err
        plt.close(fig)
    raise RuntimeError(f"could not write video: {last_err}")


def main() -> None:
    exe = EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(f"executable not found: {exe}")

    root = WORKDIR.resolve()
    outdir = OUTDIR.resolve()
    outdir.mkdir(parents=True, exist_ok=True)
    reuse = os.environ.get("ADI_PML_REUSE", "0") == "1"

    for case in CASES:
        case_dir = root / case
        meta = run_case(exe, case, case_dir, reuse=reuse)

        cache = case_dir / "ez_xy_slices.npz"
        print(f"extracting mid-plane Ez(x,y) for {case}")
        times, x, y, ez = load_slices(case_dir, cache)
        plot_probes(
            meta,
            outdir / f"xy2d_{case}_probes.png",
            times=times,
            x=x,
            y=y,
            ez=ez,
        )
        plot_snapshots(
            x, y, times, ez, meta, outdir / f"xy2d_{case}_snapshots.png"
        )
        make_video_2d(
            x, y, times, ez, meta, outdir / f"xy2d_{case}_ez.mp4"
        )


if __name__ == "__main__":
    main()
