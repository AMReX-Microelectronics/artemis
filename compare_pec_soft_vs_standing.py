#!/usr/bin/env python3
"""Compare PEC soft-drive vs standing-wave IC: spectral peak and dispersion vs CFL."""

from __future__ import annotations

import importlib.util
import math
import os
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

CASES = {
    "soft_drive": dict(mod=resonance, workdir=ROOT / "adi_dispersion/artemis_pec_soft_kh2p15e-3"),
    "standing_ic": dict(mod=standing, workdir=ROOT / "adi_dispersion/artemis_pec_standing_kh2p15e-3"),
}
CFLS = resonance.VERIFY_CFLS
OUT = ROOT / "adi_dispersion/pec_soft_vs_standing_cfl.png"
STYLES = {
    "soft_drive": ("C0", "x", "Soft drive"),
    "standing_ic": ("C1", "o", "Standing IC"),
}


def metrics(times: np.ndarray, ey: np.ndarray) -> dict[str, float]:
    freqs, amp = resonance.compute_fft(times, ey)
    peak_i = 1 + int(np.argmax(amp[1:]))
    _lz, _nz, k, _kh = resonance.mode_grid("pec")
    f0 = resonance.C0 * k / (2.0 * math.pi)
    return {
        "fft_peak": float(amp[peak_i]),
        "f_over_f0": float(freqs[peak_i] / f0),
    }


def run_case(label: str, mod, workdir: Path, exe: Path) -> dict[float, dict[str, float]]:
    workdir.mkdir(parents=True, exist_ok=True)
    out: dict[float, dict[str, float]] = {}
    print(f"\n=== {label} ({workdir.name}) ===")
    print(f"{'CFL':>8} {'FFT peak':>12} {'f/f0':>10}")
    for cfl in CFLS:
        case_dir = mod.run_artemis_case("pec", cfl, workdir, exe, reuse=False)
        times, ey = mod.read_probe(case_dir)
        m = metrics(times, ey)
        out[cfl] = m
        print(f"{cfl:8g} {m['fft_peak']:12.4e} {m['f_over_f0']:10.6f}")
    return out


def plot(results: dict[str, dict[float, dict[str, float]]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    x = np.asarray(CFLS)

    for key, (color, marker, label) in STYLES.items():
        r = results[key]
        axes[0].loglog(x, [r[c]["fft_peak"] for c in CFLS], marker + "-", color=color, ms=7, label=label)
        axes[1].semilogx(x, [r[c]["f_over_f0"] for c in CFLS], marker + "-", color=color, ms=7, label=label)

    kh = resonance.mode_grid("pec")[3]
    s = np.logspace(np.log10(min(CFLS)), np.log10(max(CFLS)), 200)
    f_adi = 2 * np.arctan(s * np.sin(kh / 2)) / (s * kh)
    axes[1].plot(s, f_adi, "k--", lw=1.5, label="ADI analytic")
    axes[1].axhline(1.0, color="0.5", ls=":", lw=1)

    axes[0].set_xlabel("CFL $S$")
    axes[0].set_ylabel(r"$|\mathrm{FFT}|/N$ peak")
    axes[0].set_title("Spectral peak magnitude")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False)

    axes[1].set_xlabel("CFL $S$")
    axes[1].set_ylabel(r"$f_\mathrm{peak}/f_0$")
    axes[1].set_title("Frequency (dispersion)")
    axes[1].grid(alpha=0.25, which="both")
    axes[1].legend(frameon=False, fontsize=9)

    fig.suptitle(
        rf"PEC: soft drive vs standing IC ($k\Delta z={resonance.KH_LABEL}$)", fontsize=12
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(OUT.resolve())


def cfl_spread(results: dict[str, dict[float, dict[str, float]]]) -> None:
    print("\n=== CFL sensitivity (max/min ratio across CFLs) ===")
    for key in results:
        fft_peaks = [results[key][c]["fft_peak"] for c in CFLS]
        freqs = [results[key][c]["f_over_f0"] for c in CFLS]
        print(
            f"{key:12s}  FFT ratio={max(fft_peaks)/min(fft_peaks):.3g}  "
            f"f/f0 spread={max(freqs)-min(freqs):.4f}"
        )


def main() -> None:
    exe = resonance.EXE.resolve()
    if not exe.exists():
        raise FileNotFoundError(exe)

    results = {}
    for label, cfg in CASES.items():
        results[label] = run_case(label, cfg["mod"], cfg["workdir"], exe)

    cfl_spread(results)
    plot(results)


if __name__ == "__main__":
    main()
