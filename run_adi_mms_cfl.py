#!/usr/bin/env python3
"""CFL convergence of 1D ADI MMS (nonzero E or H source) for six TEM polarizations."""

from __future__ import annotations

import argparse
import json
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


C0 = 299792458.0
EPS0 = 8.8541878128e-12
MU0 = 1.25663706212e-6

EXE = Path("Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex")
WORKDIR = Path("adi_mms_cfl")
OUTDIR = Path("adi_mms_cfl")
PLOT_PREFIX = "diags/plotfiles/plt"

L = 8.0e-6
N_LONG = 2928
N_TRANS = 8
N_PERIODS = 4.0
E0 = 1.0
H0 = 1.0
CFLS = [32.0, 64.0, 128.0, 256.0, 512.0]
KINDS = ("e", "h")
# Six TEM cases: both transverse E components on each axis (E ⟂ k).
DIRECTIONS = ("x_Ey", "x_Ez", "y_Ex", "y_Ez", "z_Ex", "z_Ey")

K = 2.0 * math.pi / L
OMEGA = 1.5 * C0 * K
F0 = OMEGA / (2.0 * math.pi)
S_AMP_E = E0 * (C0**2 * K**2 - OMEGA**2) / OMEGA
S_AMP_H = H0 * (C0**2 * K**2 - OMEGA**2) / OMEGA
DX = L / N_LONG
KH = K * DX

def _tem_case(coord: str, e_comp: str, b_comp: str) -> dict:
    axis = "xyz".index(coord)
    bc = ["periodic", "periodic", "periodic"]
    bc[axis] = "pec"
    return {
        "coord": coord,
        "axis": axis,
        "e_comp": e_comp,
        "b_comp": b_comp,
        "normal": coord.upper(),
        "bc": tuple(bc),
        "e_label": rf"${coord},E_{e_comp[1:]}$",
        "h_label": rf"${coord},H_{b_comp[1:]}$",
    }


# 1D PEC cavity along `coord`; E-MMS uses e_comp, H-MMS uses the companion b_comp.
DIR = {
    "x_Ey": _tem_case("x", "Ey", "Bz"),
    "x_Ez": _tem_case("x", "Ez", "By"),
    "y_Ex": _tem_case("y", "Ex", "Bz"),
    "y_Ez": _tem_case("y", "Ez", "Bx"),
    "z_Ex": _tem_case("z", "Ex", "By"),
    "z_Ey": _tem_case("z", "Ey", "Bx"),
}
FIELD_COL = {"Ex": 2, "Ey": 3, "Ez": 4, "Bx": 2, "By": 3, "Bz": 4}
POL_STYLE = (
    ("#1f77b4", "o", "-"),
    ("#d62728", "s", "--"),
)

INPUT_TEMPLATE = """\
max_step = {nsteps}

geometry.dims = 3
geometry.prob_lo = 0.0 0.0 0.0
geometry.prob_hi = {lx:.17e} {ly:.17e} {lz:.17e}

amr.n_cell = {nx} {ny} {nz}
amr.max_level = 0
amr.max_grid_size = {n_long}
amr.blocking_factor = 8

boundary.field_lo = {bc0} {bc1} {bc2}
boundary.field_hi = {bc0} {bc1} {bc2}

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
my_constants.L = {length:.17e}
my_constants.E0 = {e0:.17e}
my_constants.H0 = {h0:.17e}
my_constants.mu0 = {mu0:.17e}
my_constants.omega = {omega:.17e}
my_constants.Samp = {s_amp:.17e}
my_constants.dt = {dt:.17e}
my_constants.xi0 = {xi0:.17e}
my_constants.dxi = {dxi:.17e}
my_constants.flag_none = 0
my_constants.flag_soft = 2

warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.Ex_external_grid_function(x,y,z) = "{ex_init}"
warpx.Ey_external_grid_function(x,y,z) = "{ey_init}"
warpx.Ez_external_grid_function(x,y,z) = "{ez_init}"

warpx.B_ext_grid_init_style = parse_B_ext_grid_function
warpx.Bx_external_grid_function(x,y,z) = "{bx_init}"
warpx.By_external_grid_function(x,y,z) = "{by_init}"
warpx.Bz_external_grid_function(x,y,z) = "{bz_init}"

{excitation}

warpx.reduced_diags_names = obs0
obs0.type = {obs_type}
obs0.reduction_type = integral
obs0.integration_type = surface
obs0.surface_normal = {normal}
obs0.intervals = 1
obs0.reduced_function(x,y,z) = ({coord} > xi0 - dxi/2) * ({coord} < xi0 + dxi/2)

diagnostics.diags_names = plt
plt.diag_type = Full
plt.intervals = {nsteps}
plt.fields_to_plot = {plot_field}
plt.file_prefix = {plot_prefix}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1D MMS CFL convergence for six TEM polarizations (E and H sources)."
    )
    parser.add_argument("--exe", default=str(EXE))
    parser.add_argument("--workdir", default=str(WORKDIR))
    parser.add_argument("--outdir", default=str(OUTDIR))
    parser.add_argument("--cfls", nargs="+", type=float, default=CFLS)
    parser.add_argument("--kinds", nargs="+", choices=list(KINDS), default=list(KINDS))
    parser.add_argument(
        "--directions",
        nargs="+",
        choices=list(DIRECTIONS),
        default=list(DIRECTIONS),
        help="TEM cases: {axis}_{Ecomp}, e.g. z_Ey.",
    )
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument(
        "--plot-only",
        action="store_true",
        help="Rebuild figures from completed cases; do not launch Artemis.",
    )
    return parser.parse_args()


def geometry(direction: str) -> tuple[list[float], list[int], float]:
    cfg = DIR[direction]
    axis = cfg["axis"]
    sizes = [N_TRANS * DX] * 3
    cells = [N_TRANS] * 3
    sizes[axis] = L
    cells[axis] = N_LONG
    area = sizes[(axis + 1) % 3] * sizes[(axis + 2) % 3]
    return sizes, cells, area


def nsteps_for(cfl: float) -> tuple[int, float]:
    dt = cfl * DX / C0
    nsteps = int(math.ceil(N_PERIODS / (F0 * dt)))
    return nsteps, dt


def case_name(kind: str, direction: str, cfl: float) -> str:
    return f"mms_{kind}_prop_{direction}_cfl_{cfl:g}".replace(".", "p")


def field_inits(kind: str, direction: str) -> dict[str, str]:
    cfg = DIR[direction]
    coord = cfg["coord"]
    fields = {
        "ex_init": "0.0",
        "ey_init": "0.0",
        "ez_init": "0.0",
        "bx_init": "0.0",
        "by_init": "0.0",
        "bz_init": "0.0",
    }
    if kind == "e":
        fields[f"{cfg['e_comp'].lower()}_init"] = f"E0*sin(2*pi*{coord}/L)"
    else:
        fields[f"{cfg['b_comp'].lower()}_init"] = f"mu0*H0*cos(2*pi*{coord}/L)"
    return fields


def excitation_block(kind: str, direction: str) -> str:
    cfg = DIR[direction]
    coord = cfg["coord"]
    e_flags = {c: "flag_none" for c in ("Ex", "Ey", "Ez")}
    b_flags = {c: "flag_none" for c in ("Bx", "By", "Bz")}
    e_funs = {c: "0.0" for c in ("Ex", "Ey", "Ez")}
    b_funs = {c: "0.0" for c in ("Bx", "By", "Bz")}
    if kind == "e":
        e_flags[cfg["e_comp"]] = "flag_soft"
        e_funs[cfg["e_comp"]] = f"Samp*dt*sin(2*pi*{coord}/L)*sin(omega*t)"
        return f"""\
warpx.E_excitation_on_grid_style = parse_E_excitation_grid_function
warpx.Ex_excitation_flag_function(x,y,z) = "{e_flags['Ex']}"
warpx.Ey_excitation_flag_function(x,y,z) = "{e_flags['Ey']}"
warpx.Ez_excitation_flag_function(x,y,z) = "{e_flags['Ez']}"
warpx.Ex_excitation_grid_function(x,y,z,t) = "{e_funs['Ex']}"
warpx.Ey_excitation_grid_function(x,y,z,t) = "{e_funs['Ey']}"
warpx.Ez_excitation_grid_function(x,y,z,t) = "{e_funs['Ez']}"
"""
    b_flags[cfg["b_comp"]] = "flag_soft"
    b_funs[cfg["b_comp"]] = f"mu0*Samp*dt*cos(2*pi*{coord}/L)*sin(omega*t)"
    return f"""\
warpx.B_excitation_on_grid_style = parse_b_excitation_grid_function
warpx.Bx_excitation_flag_function(x,y,z) = "{b_flags['Bx']}"
warpx.By_excitation_flag_function(x,y,z) = "{b_flags['By']}"
warpx.Bz_excitation_flag_function(x,y,z) = "{b_flags['Bz']}"
warpx.Bx_excitation_grid_function(x,y,z,t) = "{b_funs['Bx']}"
warpx.By_excitation_grid_function(x,y,z,t) = "{b_funs['By']}"
warpx.Bz_excitation_grid_function(x,y,z,t) = "{b_funs['Bz']}"
"""


def write_inputs(
    case_dir: Path, kind: str, direction: str, nsteps: int, dt: float
) -> Path:
    cfg = DIR[direction]
    sizes, cells, _area = geometry(direction)
    xi0 = 0.25 * L if kind == "e" else (N_LONG / 2 - 0.5) * DX
    plot_field = cfg["e_comp"] if kind == "e" else cfg["b_comp"]
    text = INPUT_TEMPLATE.format(
        nsteps=nsteps,
        lx=sizes[0],
        ly=sizes[1],
        lz=sizes[2],
        nx=cells[0],
        ny=cells[1],
        nz=cells[2],
        n_long=N_LONG,
        bc0=cfg["bc"][0],
        bc1=cfg["bc"][1],
        bc2=cfg["bc"][2],
        dt=dt,
        eps0=EPS0,
        mu0=MU0,
        c0=C0,
        length=L,
        e0=E0,
        h0=H0,
        omega=OMEGA,
        s_amp=S_AMP_E if kind == "e" else S_AMP_H,
        xi0=xi0,
        dxi=DX,
        coord=cfg["coord"],
        normal=cfg["normal"],
        obs_type="RawEFieldReduction" if kind == "e" else "RawBFieldReduction",
        plot_field=plot_field,
        plot_prefix=PLOT_PREFIX,
        excitation=excitation_block(kind, direction),
        **field_inits(kind, direction),
    )
    path = case_dir / "inputs"
    path.write_text(text)
    return path


def probe_path(case_dir: Path) -> Path:
    return case_dir / "diags" / "reducedfiles" / "obs0.txt"


def list_plotfiles(case_dir: Path) -> list[Path]:
    return sorted(p for p in case_dir.glob(f"{PLOT_PREFIX}[0-9]*") if p.is_dir())


def case_complete(case_dir: Path) -> bool:
    return probe_path(case_dir).exists() and any(list_plotfiles(case_dir))


def mms_probe(kind: str, direction: str, t: np.ndarray) -> np.ndarray:
    if kind == "e":
        return E0 * np.sin(K * 0.25 * L) * np.cos(OMEGA * t)
    xi = (N_LONG / 2 - 0.5) * DX
    return H0 * np.cos(K * xi) * np.cos(OMEGA * t)


def mms_line(kind: str, xi: np.ndarray, t: float) -> np.ndarray:
    if kind == "e":
        return E0 * np.sin(K * xi) * np.cos(OMEGA * t)
    return H0 * np.cos(K * xi) * np.cos(OMEGA * t)


def read_probe(case_dir: Path, kind: str, direction: str, area: float) -> tuple[np.ndarray, np.ndarray]:
    cfg = DIR[direction]
    data = np.loadtxt(probe_path(case_dir), comments="#")
    if data.ndim == 1:
        data = data.reshape(1, -1)
    field = cfg["e_comp"] if kind == "e" else cfg["b_comp"]
    raw = data[:, FIELD_COL[field]]
    if kind == "h":
        raw = raw / MU0
    return data[:, 1], raw / area


def centerline(plt_path: Path, field: str, axis: int) -> tuple[float, np.ndarray, np.ndarray]:
    ds = yt.load(str(plt_path))
    dims = tuple(int(n) for n in ds.domain_dimensions)
    grid = ds.covering_grid(
        level=0, left_edge=ds.domain_left_edge, dims=ds.domain_dimensions
    )
    try:
        arr = np.asarray(grid[("mesh", field)]).squeeze()
    except Exception:
        arr = np.asarray(grid[("boxlib", field)]).squeeze()
    lo = np.asarray(ds.domain_left_edge)
    hi = np.asarray(ds.domain_right_edge)
    n = arr.shape[axis]
    xi = np.linspace(float(lo[axis]), float(hi[axis]), n, endpoint=False)
    xi += 0.5 * (float(hi[axis]) - float(lo[axis])) / n
    mid = [dims[i] // 2 for i in range(3)]
    slc = [mid[0], mid[1], mid[2]]
    slc[axis] = slice(None)
    line = np.asarray(arr[tuple(slc)], dtype=np.float64)
    return float(ds.current_time.to_value()), xi, line


def spatial_error(plt_path: Path, kind: str, direction: str) -> float:
    cfg = DIR[direction]
    field = cfg["e_comp"] if kind == "e" else cfg["b_comp"]
    t, xi, line = centerline(plt_path, field, cfg["axis"])
    if kind == "h":
        line = line / MU0
    exact = mms_line(kind, xi, t)
    err = line - exact
    dxi = float(xi[1] - xi[0]) if xi.size > 1 else DX
    return float(np.sqrt(dxi * np.sum(err * err)))


def metrics_path(case_dir: Path) -> Path:
    return case_dir / "metrics.json"


def load_metrics(case_dir: Path) -> dict | None:
    path = metrics_path(case_dir)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_metrics(case_dir: Path, out: dict) -> None:
    metrics_path(case_dir).write_text(json.dumps(out, indent=2))


def run_case(
    exe: Path, workdir: Path, kind: str, direction: str, cfl: float, *, fresh: bool
) -> dict:
    nsteps, dt = nsteps_for(cfl)
    case_dir = workdir / case_name(kind, direction, cfl)
    _sizes, _cells, area = geometry(direction)

    if not fresh:
        cached = load_metrics(case_dir)
        if cached is not None and case_complete(case_dir):
            print(f"[reuse] {case_dir.name}")
            return cached

    if not fresh and case_complete(case_dir):
        print(f"[reuse] {case_dir.name}")
    else:
        if not exe.exists():
            raise FileNotFoundError(exe)
        if case_dir.exists():
            shutil.rmtree(case_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        inputs = write_inputs(case_dir, kind, direction, nsteps, dt)
        print(
            f"[Artemis] {kind.upper()} prop={direction} CFL={cfl:g} "
            f"nsteps={nsteps} dt={dt:.4e}"
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

    times, probe = read_probe(case_dir, kind, direction, area)
    exact = mms_probe(kind, direction, times)
    probe_err = np.abs(probe - exact)
    plts = list_plotfiles(case_dir)
    if not plts:
        raise FileNotFoundError(f"no plotfiles in {case_dir}")
    out = {
        "err_final": spatial_error(plts[-1], kind, direction),
        "err_probe_max": float(np.max(probe_err)),
        "dt": dt,
        "nsteps": nsteps,
    }
    save_metrics(case_dir, out)
    return out


def savefig(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(path.resolve())


def cases_on_axis(axis: str, directions: list[str]) -> list[str]:
    return [d for d in directions if DIR[d]["coord"] == axis]


def observed_order(cfls: list[float], errs: list[float]) -> tuple[np.ndarray, np.ndarray]:
    s = np.asarray(cfls, dtype=float)
    e = np.asarray(errs, dtype=float)
    mid = np.sqrt(s[1:] * s[:-1])
    order = np.log(e[1:] / e[:-1]) / np.log(s[1:] / s[:-1])
    return mid, order


def _format_cfl_axis(ax, cfls: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(cfls)
    ax.set_xticklabels([f"{c:g}" for c in cfls])
    ax.set_xlabel("CFL $S$")
    ax.minorticks_off()


def plot_cfl(results: dict, cfls: list[float], kinds: list[str], directions: list[str], outdir: Path) -> None:
    kh_label = format(KH, ".3e").replace("e-0", "e-")
    x = np.asarray(cfls)
    axes_list = ("x", "y", "z")
    kind_rows = [k for k in ("e", "h") if k in kinds]
    fig, axes = plt.subplots(
        len(kind_rows),
        3,
        figsize=(12.4, 3.9 * len(kind_rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    for i, kind in enumerate(kind_rows):
        for j, axis in enumerate(axes_list):
            ax = axes[i, j]
            cases = cases_on_axis(axis, directions)
            if not cases:
                ax.set_visible(False)
                continue
            ymins = []
            for p, direction in enumerate(cases):
                color, marker, ls = POL_STYLE[p % len(POL_STYLE)]
                y = [results[kind][direction][c]["err_final"] for c in cfls]
                ymins.append(min(y))
                ax.loglog(
                    x,
                    y,
                    ls,
                    color=color,
                    marker=marker,
                    ms=8,
                    mew=1.2,
                    lw=1.8,
                    markerfacecolor="white" if p else color,
                    markeredgecolor=color,
                    label=(
                        rf"$E_{DIR[direction]['e_comp'][1:]}$"
                        if kind == "e"
                        else rf"$H_{DIR[direction]['b_comp'][1:]}$"
                    ),
                )
            e0 = min(ymins)
            ax.loglog(
                x[[0, -1]],
                e0 * (x[[0, -1]] / cfls[0]) ** 2,
                color="0.35",
                ls=":",
                lw=1.4,
                label=r"$\propto S^{2}$",
            )
            ax.set_title(rf"{kind.upper()} source, $k\parallel {axis}$")
            ax.grid(alpha=0.28, which="both")
            if j == 0:
                ax.set_ylabel(r"$\|e\|_{L^2}(t_{\mathrm{end}})$")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            _format_cfl_axis(ax, cfls)

    fig.suptitle(
        rf"ADI MMS CFL convergence ($N={N_LONG}$, $k\Delta\xi={kh_label}$, "
        rf"$\omega=\frac{{3}}{{2}}ck$)",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_cfl_convergence.png")

    fig, axes = plt.subplots(
        len(kind_rows),
        3,
        figsize=(12.4, 3.9 * len(kind_rows)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    for i, kind in enumerate(kind_rows):
        for j, axis in enumerate(axes_list):
            ax = axes[i, j]
            cases = cases_on_axis(axis, directions)
            if not cases:
                ax.set_visible(False)
                continue
            ymins = []
            for p, direction in enumerate(cases):
                color, marker, ls = POL_STYLE[p % len(POL_STYLE)]
                y = [results[kind][direction][c]["err_probe_max"] for c in cfls]
                ymins.append(min(y))
                ax.loglog(
                    x,
                    y,
                    ls,
                    color=color,
                    marker=marker,
                    ms=8,
                    mew=1.2,
                    lw=1.8,
                    markerfacecolor="white" if p else color,
                    markeredgecolor=color,
                    label=(
                        rf"$E_{DIR[direction]['e_comp'][1:]}$"
                        if kind == "e"
                        else rf"$H_{DIR[direction]['b_comp'][1:]}$"
                    ),
                )
            e0 = min(ymins)
            ax.loglog(
                x[[0, -1]],
                e0 * (x[[0, -1]] / cfls[0]) ** 2,
                color="0.35",
                ls=":",
                lw=1.4,
                label=r"$\propto S^{2}$",
            )
            ax.set_title(rf"{kind.upper()} source, $k\parallel {axis}$")
            ax.grid(alpha=0.28, which="both")
            if j == 0:
                ax.set_ylabel(r"$\max_t|e_{\mathrm{probe}}|$")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
            _format_cfl_axis(ax, cfls)
    fig.suptitle(
        rf"ADI MMS probe-max vs CFL ($N={N_LONG}$, $k\Delta\xi={kh_label}$)",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_cfl_probe_max.png")

    fig, axes = plt.subplots(
        len(kind_rows),
        3,
        figsize=(12.4, 3.6 * len(kind_rows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    for i, kind in enumerate(kind_rows):
        for j, axis in enumerate(axes_list):
            ax = axes[i, j]
            cases = cases_on_axis(axis, directions)
            if not cases:
                ax.set_visible(False)
                continue
            for p, direction in enumerate(cases):
                color, marker, ls = POL_STYLE[p % len(POL_STYLE)]
                y = [results[kind][direction][c]["err_final"] for c in cfls]
                mid, order = observed_order(cfls, y)
                ax.semilogx(
                    mid,
                    order,
                    ls,
                    color=color,
                    marker=marker,
                    ms=8,
                    mew=1.2,
                    lw=1.8,
                    markerfacecolor="white" if p else color,
                    markeredgecolor=color,
                    label=(
                        rf"$E_{DIR[direction]['e_comp'][1:]}$"
                        if kind == "e"
                        else rf"$H_{DIR[direction]['b_comp'][1:]}$"
                    ),
                )
            ax.axhline(2.0, color="0.35", ls=":", lw=1.3, label="order 2")
            ax.axhline(1.0, color="0.55", ls="--", lw=1.0, label="order 1")
            ax.set_ylim(-0.2, 2.8)
            ax.set_title(rf"{kind.upper()} source, $k\parallel {axis}$")
            ax.grid(alpha=0.28)
            if j == 0:
                ax.set_ylabel(r"observed order $p$")
            ax.legend(frameon=False, fontsize=8, loc="best")
            _format_cfl_axis(ax, cfls)
    fig.suptitle(
        rf"Observed CFL order $p=\log(e_{{i+1}}/e_i)/\log(S_{{i+1}}/S_i)$ "
        rf"from final $L^2$",
        fontsize=12,
    )
    fig.tight_layout()
    savefig(fig, outdir / "mms_cfl_order.png")


def main() -> None:
    args = parse_args()
    exe = Path(args.exe).resolve()
    if not args.plot_only and not exe.exists():
        raise FileNotFoundError(exe)

    workdir = Path(args.workdir).resolve()
    outdir = Path(args.outdir).resolve()
    if workdir.exists() and args.fresh:
        shutil.rmtree(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    outdir.mkdir(parents=True, exist_ok=True)

    print(
        f"N={N_LONG}, k dxi={KH:.4e}, f0={F0/1e12:.4f} THz, "
        f"omega/(c k)={OMEGA/(C0*K):.3f}"
    )
    print("CFL:", args.cfls)
    print("kinds:", args.kinds, "directions:", args.directions)

    results: dict[str, dict[str, dict[float, dict]]] = {
        kind: {d: {} for d in args.directions} for kind in args.kinds
    }
    print(f"{'kind':>4} {'case':>6} {'CFL':>8} {'L2 final':>12} {'probe max':>12}")
    for kind in args.kinds:
        for direction in args.directions:
            for cfl in args.cfls:
                out = run_case(exe, workdir, kind, direction, cfl, fresh=args.fresh)
                results[kind][direction][cfl] = out
                print(
                    f"{kind:>4} {direction:>6} {cfl:8g} {out['err_final']:12.4e} "
                    f"{out['err_probe_max']:12.4e}"
                )

    print("\n=== CFL L2 ratio (max/min) ===")
    for kind in args.kinds:
        for direction in args.directions:
            errs = [results[kind][direction][c]["err_final"] for c in args.cfls]
            print(f"{kind} {direction}  L2 ratio={max(errs)/min(errs):.3g}")

    plot_cfl(results, list(args.cfls), list(args.kinds), list(args.directions), outdir)


if __name__ == "__main__":
    main()
