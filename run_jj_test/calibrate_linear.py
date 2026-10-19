#!/usr/bin/env python3
"""Broadband kick + free ring-down of the linear deck to locate the L-mode."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
EXE = (ROOT / "../Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex").resolve()
RESULTS = ROOT / "results" / "calibration"
NSTEPS = 12000
# Gaussian-derivative kick, 20 ps wide (spectral peak ~8 GHz) for the few-GHz L-mode.
KICK = (
    "Edrive * cfl/0.8 * (t-100e-12)/20e-12 * exp(-(t-100e-12)**2/(2*(20e-12)**2)) * "
    "(x >= 297.5e-6) * (x <= 362.5e-6) * (z > h_si - ddz) * (z < h_si + ddz) * "
    "(y > 675.e-6 - ddy) * (y < 675.e-6 + ddy)"
)


def run(label: str, inductance: float, nsteps: int, cfl: float, scheme: str, resistance: float) -> Path:
    folder = RESULTS / label
    probe = folder / "reducedfiles" / "Egap.txt"
    if probe.exists() and "finalized" in (folder / "run.log").read_text():
        return folder
    (folder / "reducedfiles").mkdir(parents=True, exist_ok=True)
    command = [
        str(EXE), str(ROOT / "inputs_artemis_crosstalk_L"),
        f"max_step={nsteps}",
        f"my_constants.cfl={cfl:g}",
        f"my_constants.R_jj={resistance:g}",
        f"algo.time_stepping_scheme={scheme}",
        "amrex.the_arena_init_size=0",
        f"my_constants.L_jj={inductance:.17g}",
        f"warpx.Ex_excitation_grid_function(x,y,z,t)={KICK}",
        f"warpx.used_inputs_file={folder / 'warpx_used_inputs'}",
        *(f"{name}.path={folder / 'reducedfiles'}/" for name in ("Eobs0", "Eobs1", "Egap")),
    ]
    print(f"run {label} L={inductance:.4g}", flush=True)
    with (folder / "run.log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return folder


def spectrum_peaks(folder: Path, start=200e-12, count=4):
    data = np.loadtxt(folder / "reducedfiles" / "Egap.txt")
    time, signal = data[:, 1], data[:, 2]
    mask = time > start
    signal = signal[mask] * np.hanning(mask.sum())
    dt = time[1] - time[0]
    nfft = 16 * len(signal)
    amp = np.abs(np.fft.rfft(signal, nfft))
    freq = np.fft.rfftfreq(nfft, dt)
    band = (freq > 0.5e9) & (freq < 150e9)
    amp, freq = amp[band], freq[band]
    local = np.where((amp[1:-1] > amp[:-2]) & (amp[1:-1] > amp[2:]))[0] + 1
    best = local[np.argsort(amp[local])[::-1][:count]]
    return [(freq[i] / 1e9, amp[i] / amp[best[0]]) for i in sorted(best)]


def zero_crossing_frequency(folder: Path, start=200e-12) -> float:
    data = np.loadtxt(folder / "reducedfiles" / "Egap.txt")
    time, signal = data[:, 1], data[:, 2]
    i = np.where((signal[:-1] < 0) & (signal[1:] >= 0) & (time[:-1] > start))[0]
    crossings = time[i] - signal[i] * (time[i + 1] - time[i]) / (signal[i + 1] - signal[i])
    return (len(crossings) - 1) / (crossings[-1] - crossings[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inductances", nargs="+", type=float, help="per-edge L in H")
    parser.add_argument("--steps", type=int, default=NSTEPS, help="steps at cfl 8; scaled by 8/cfl")
    parser.add_argument("--cfl", type=float, default=8.0)
    parser.add_argument("--scheme", choices=("adi", "fdtd"), default="adi")
    parser.add_argument("--resistance", type=float, default=0.0, help="shunt R per edge in ohms (0 = none)")
    args = parser.parse_args()
    for inductance in args.inductances:
        suffix = ("" if args.cfl == 8.0 else f"_cfl{args.cfl:g}") + ("" if args.scheme == "adi" else "_fdtd") + (f"_R{args.resistance:g}" if args.resistance else "")
        nsteps = round(args.steps * 8.0 / args.cfl)
        folder = run(f"L{inductance:.4e}{suffix}", inductance, nsteps, args.cfl, args.scheme, args.resistance)
        peaks = ", ".join(f"{f:.2f} GHz ({a:.2f})" for f, a in spectrum_peaks(folder))
        print(f"L={inductance:.4e} H/edge ({26 * inductance * 1e12:.1f} pH total), cfl {args.cfl:g}: "
              f"zero-crossing {zero_crossing_frequency(folder) / 1e9:.4f} GHz; peaks {peaks}", flush=True)


if __name__ == "__main__":
    main()
