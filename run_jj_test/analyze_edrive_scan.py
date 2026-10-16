#!/usr/bin/env python3
"""Plot nonlinear Josephson frequency shift versus drive amplitude."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
SCAN = ROOT / "results" / "edrive_scan"
DRIVES = (100, 2000, 4000, 6000, 8000, 10000, 12000)
AREA = 2.5e-6 * 3.0e-6
QE = 1.602176634e-19
HBAR = 1.054571817e-34


def load(kind: str, drive: int):
    data = np.loadtxt(SCAN / kind / f"E{drive:06d}" / "reducedfiles" / "Eobs1.txt")
    voltage = data[:, 2] / AREA
    phase = np.zeros(len(data))
    phase[1:] = np.cumsum(
        0.5 * (voltage[1:] + voltage[:-1]) * np.diff(data[:, 1])
    ) * (2.0 * QE / HBAR)
    return data[:, 1], phase


def first_postpulse_cycle(time, phase, start=80e-12):
    indices = np.where(
        (phase[:-1] < 0.0) & (phase[1:] >= 0.0) & (time[:-1] > start)
    )[0]
    crossings = np.array([
        time[i] - phase[i] * (time[i + 1] - time[i]) / (phase[i + 1] - phase[i])
        for i in indices
    ])
    if len(crossings) < 2:
        return np.nan, np.nan
    mask = (time >= crossings[0]) & (time <= crossings[1])
    return 1.0 / (crossings[1] - crossings[0]), np.max(np.abs(phase[mask]))


def pendulum_ratio(amplitude):
    """Return pi/[2 K(sin(A/2))] using the arithmetic-geometric mean."""
    arithmetic = np.ones_like(amplitude)
    geometric = np.cos(0.5 * amplitude)
    for _ in range(20):
        arithmetic, geometric = (
            0.5 * (arithmetic + geometric),
            np.sqrt(arithmetic * geometric),
        )
    return arithmetic


rows = []
for drive in DRIVES:
    for kind in ("linear", "rcsj"):
        time, phase = load(kind, drive)
        frequency, amplitude = first_postpulse_cycle(time, phase)
        rows.append((drive, kind, frequency / 1e9, amplitude, np.max(np.abs(phase))))

rcsj_rows = [row for row in rows if row[1] == "rcsj"]
f0 = rcsj_rows[0][2]
rcsj_amplitudes = np.asarray([row[3] for row in rcsj_rows])
theory_frequencies = f0 * pendulum_ratio(rcsj_amplitudes)
theory_by_drive = {
    row[0]: theory for row, theory in zip(rcsj_rows, theory_frequencies, strict=True)
}

csv = SCAN / "frequency_vs_edrive.csv"
with csv.open("w") as stream:
    stream.write(
        "Edrive_V_per_m,case,first_cycle_frequency_GHz,"
        "first_cycle_phase_amplitude_rad,max_phase_rad,"
        "pendulum_theory_GHz,theory_error_percent\n"
    )
    for row in rows:
        if row[1] == "rcsj":
            theory = theory_by_drive[row[0]]
            error = 100.0 * (row[2] - theory) / theory
            stream.write(
                f"{row[0]},{row[1]},{row[2]:.10g},{row[3]:.10g},{row[4]:.10g},"
                f"{theory:.10g},{error:.10g}\n"
            )
        else:
            stream.write(f"{row[0]},{row[1]},{row[2]:.10g},{row[3]:.10g},{row[4]:.10g},,\n")

fig, (ax_f, ax_a) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
for kind, marker in (("linear", "o"), ("rcsj", "s")):
    selected = np.asarray([[r[0], r[2], r[3]] for r in rows if r[1] == kind])
    ax_f.plot(selected[:, 0], selected[:, 1], marker=marker, label=kind)
    ax_a.plot(selected[:, 0], selected[:, 2], marker=marker, label=kind)
ax_f.plot(DRIVES, theory_frequencies, "k--", label="pendulum theory")
ax_f.set_ylabel("first post-pulse frequency (GHz)")
ax_f.grid(alpha=0.3)
ax_f.legend()
ax_a.set_xlabel("Edrive (V/m)")
ax_a.set_ylabel("first-cycle phase amplitude (rad)")
ax_a.grid(alpha=0.3)
ax_a.legend()
fig.suptitle("Josephson frequency softening versus drive amplitude")
fig.tight_layout()
fig.savefig(SCAN / "frequency_vs_edrive.png", dpi=180)
print(csv.read_text())
