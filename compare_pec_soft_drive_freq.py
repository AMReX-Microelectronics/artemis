#!/usr/bin/env python3
"""PEC soft-drive detuning vs standing IC: spectral peak and dispersion vs CFL."""

from __future__ import annotations

import importlib.util
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


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ROOT = Path(__file__).resolve().parent
resonance = load_module(ROOT / "run_adi_pec_resonance.py", "resonance")
standing = load_module(ROOT / "run_adi_pec_standing.py", "standing")

CFLS = resonance.VERIFY_CFLS
FREQ_RATIOS = [round(0.7 + 0.1 * i, 1) for i in range(7)]  # 0.7, ..., 1.3
SOFT_WORKDIR = ROOT / "adi_dispersion/artemis_pec_soft_freq_kh2p15e-3"
STANDING_WORKDIR = ROOT / "adi_dispersion/artemis_pec_standing_kh2p15e-3"
OUT = ROOT / "adi_dispersion/pec_soft_drive_freq_vs_cfl.png"


def soft_case_name(cfl: float, freq_ratio: float) -> str:
    tag = f"f{freq_ratio:g}".replace(".", "p")
    return f"pec_cfl_{cfl:g}_{tag}".replace(".", "p")


def run_soft_case(
    cfl: float, freq_ratio: float, workdir: Path, exe: Path, *, reuse: bool = False
) -> Path:
    """Run PEC soft-drive case with f_drive = freq_ratio * f0."""
    nx = ny = resonance.N_TRANS
    lz, nz, k, _kh = resonance.mode_grid("pec")
    dz = lz / nz
    lx = nx * dz
    ly = ny * dz
    f0 = resonance.C0 * k / (2.0 * math.pi)
    t0 = 1.0 / f0
    dt = cfl * dz / resonance.C0
    drive_freq = freq_ratio * f0

    steps_per_period = max(1, int(round(t0 / dt)))
    diag_interval = max(1, steps_per_period // resonance.SAMPLES_PER_PERIOD)
    nsteps = int(math.ceil(resonance.N_PERIODS * t0 / dt))

    case_dir = workdir / soft_case_name(cfl, freq_ratio)
    probe = case_dir / "diags" / "reducedfiles" / "Eobs0.txt"
    if reuse and probe.exists():
        print(f"[Artemis] reuse soft CFL={cfl:g} f/f0={freq_ratio:g}: {case_dir}")
        return case_dir

    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)

    boundary = (
        "# PEC walls normal to propagation (z); periodic transversely.\n"
        "boundary.field_lo = periodic periodic pec\n"
        "boundary.field_hi = periodic periodic pec"
    )
    header = resonance.COMMON_HEADER.format(
        nsteps=nsteps,
        lx=lx,
        ly=ly,
        lz=lz,
        nx=nx,
        ny=ny,
        nz=nz,
        boundary=boundary,
        dt=dt,
        eps0=resonance.EPS0,
        mu0=resonance.MU0,
        c0=resonance.C0,
        z0=0.25 * lz,
        dz=dz,
    )
    body = resonance.PEC_BODY.format(freq=drive_freq, tp=t0, dt=dt)
    probe_block = resonance.PROBE.format(diag_interval=diag_interval)
    (case_dir / "inputs").write_text(header + "\n" + body + "\n" + probe_block)

    cmd = [str(exe), str(case_dir / "inputs")]
    print(
        f"[Artemis] soft CFL={cfl:g}, f_drive/f0={freq_ratio:g}, "
        f"steps={nsteps}, diag={diag_interval}"
    )
    with (case_dir / "run.log").open("w") as log:
        subprocess.run(cmd, cwd=case_dir, check=True, stdout=log, stderr=subprocess.STDOUT)
    return case_dir


def metrics(times: np.ndarray, ey: np.ndarray) -> dict[str, float]:
    freqs, amp = resonance.compute_fft(times, ey)
    peak_i = 1 + int(np.argmax(amp[1:]))
    _lz, _nz, k, _kh = resonance.mode_grid("pec")
    f0 = resonance.C0 * k / (2.0 * math.pi)
    return {
        "fft_peak": float(amp[peak_i]),
        "f_over_f0": float(freqs[peak_i] / f0),
    }


def run_soft_sweep(exe: Path) -> dict[float, dict[float, dict[str, float]]]:
    SOFT_WORKDIR.mkdir(parents=True, exist_ok=True)
    results: dict[float, dict[float, dict[str, float]]] = {}
    for freq_ratio in FREQ_RATIOS:
        out: dict[float, dict[str, float]] = {}
        print(f"\n=== soft drive f_drive/f0 = {freq_ratio:g} ===")
        print(f"{'CFL':>8} {'FFT peak':>12} {'f/f0':>10}")
        for cfl in CFLS:
            case_dir = run_soft_case(cfl, freq_ratio, SOFT_WORKDIR, exe, reuse=False)
            m = metrics(*resonance.read_probe(case_dir))
            out[cfl] = m
            print(f"{cfl:8g} {m['fft_peak']:12.4e} {m['f_over_f0']:10.6f}")
        results[freq_ratio] = out
    return results


def run_standing_reference(exe: Path) -> dict[float, dict[str, float]]:
    STANDING_WORKDIR.mkdir(parents=True, exist_ok=True)
    out: dict[float, dict[str, float]] = {}
    print("\n=== standing IC (reference) ===")
    print(f"{'CFL':>8} {'FFT peak':>12} {'f/f0':>10}")
    for cfl in CFLS:
        case_dir = standing.run_artemis_case("pec", cfl, STANDING_WORKDIR, exe, reuse=False)
        m = metrics(*standing.read_probe(case_dir))
        out[cfl] = m
        print(f"{cfl:8g} {m['fft_peak']:12.4e} {m['f_over_f0']:10.6f}")
    return out


def plot(
    soft: dict[float, dict[float, dict[str, float]]],
    standing_ref: dict[float, dict[str, float]],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5))
    x = np.asarray(CFLS)
    cmap = plt.cm.viridis
    colors = {
        r: cmap(i / max(1, len(FREQ_RATIOS) - 1)) for i, r in enumerate(FREQ_RATIOS)
    }

    for freq_ratio in FREQ_RATIOS:
        r = soft[freq_ratio]
        color = colors[freq_ratio]
        label = rf"$f_\mathrm{{drive}}/f_0={freq_ratio:g}$"
        axes[0].loglog(
            x, [r[c]["fft_peak"] for c in CFLS], "x-", color=color, ms=7, lw=1.2, label=label
        )
        axes[1].semilogx(
            x, [r[c]["f_over_f0"] for c in CFLS], "x-", color=color, ms=7, lw=1.2, label=label
        )

    axes[0].loglog(
        x,
        [standing_ref[c]["fft_peak"] for c in CFLS],
        "o-",
        color="k",
        ms=7,
        lw=1.5,
        label="Standing IC",
    )
    axes[1].semilogx(
        x,
        [standing_ref[c]["f_over_f0"] for c in CFLS],
        "o-",
        color="k",
        ms=7,
        lw=1.5,
        label="Standing IC",
    )

    kh = resonance.mode_grid("pec")[3]
    s = np.logspace(np.log10(min(CFLS)), np.log10(max(CFLS)), 200)
    f_adi = 2 * np.arctan(s * np.sin(kh / 2)) / (s * kh)
    axes[1].plot(s, f_adi, "k--", lw=1.5, label="ADI analytic")
    axes[1].axhline(1.0, color="0.5", ls=":", lw=1)

    axes[0].set_xlabel("CFL $S$")
    axes[0].set_ylabel(r"$|\mathrm{FFT}|/N$ peak")
    axes[0].set_title("Spectral peak magnitude")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=7.5, ncol=2)

    axes[1].set_xlabel("CFL $S$")
    axes[1].set_ylabel(r"$f_\mathrm{peak}/f_0$")
    axes[1].set_title("Frequency (dispersion)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=7.5, ncol=2)

    fig.suptitle(
        rf"PEC soft-drive detuning + standing IC ($k\Delta z={resonance.KH_LABEL}$)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT.resolve())


def main() -> None:
    exe = resonance.EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(exe)
    plot(run_soft_sweep(exe), run_standing_reference(exe))


if __name__ == "__main__":
    main()
