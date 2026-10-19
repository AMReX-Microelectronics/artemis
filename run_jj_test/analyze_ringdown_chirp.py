#!/usr/bin/env python3
"""Compare free ring-down frequency with exact pendulum softening."""

from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "ringdown_chirp"
AREA = 2.5e-6 * 3.0e-6
N_JJ = 26  # junctions in series across the gap
QE = 1.602176634e-19
HBAR = 1.054571817e-34
DRIVE_OFF = 1.251e-9
# Small-signal frequency: weakest-drive RCSJ row of the Edrive scan.
F0 = 1e9 * next(
    float(row["first_cycle_frequency_GHz"])
    for row in csv.DictReader((ROOT / "results" / "edrive_scan" / "frequency_vs_edrive.csv").open())
    if row["case"] == "rcsj"
)


def pendulum_ratio(amplitude):
    """pi/(2K(k^2)), evaluated by the arithmetic-geometric mean."""
    a = np.ones_like(amplitude)
    b = np.abs(np.cos(0.5 * amplitude))
    for _ in range(30):
        anext = 0.5 * (a + b)
        bnext = np.sqrt(a * b)
        if np.max(np.abs(anext - bnext)) < 1.0e-14:
            a = anext
            break
        a, b = anext, bnext
    return a


data = np.loadtxt(RESULTS / "reducedfiles" / "Egap.txt")
time = data[:, 1]
voltage = data[:, 2] / AREA / N_JJ  # per junction
phase = np.zeros(len(data))
phase[1:] = np.cumsum(
    0.5 * (voltage[1:] + voltage[:-1]) * np.diff(time)
) * (2.0 * QE / HBAR)

indices = np.where((phase[:-1] < 0.0) & (phase[1:] >= 0.0))[0]
crossings = np.asarray([
    time[i] - phase[i] * (time[i + 1] - time[i]) / (phase[i + 1] - phase[i])
    for i in indices
])
rows = []
for i, frequency in enumerate(1.0 / np.diff(crossings)):
    mask = (time >= crossings[i]) & (time <= crossings[i + 1])
    segment = phase[mask]
    amplitude = 0.5 * (np.max(segment) - np.min(segment))
    midpoint = 0.5 * (crossings[i] + crossings[i + 1])
    rows.append((midpoint, frequency, amplitude, midpoint > DRIVE_OFF))
cycles = np.asarray(rows)
theory = F0 * pendulum_ratio(cycles[:, 2])

with (RESULTS / "ringdown_cycles.csv").open("w", newline="") as stream:
    writer = csv.writer(stream)
    writer.writerow(("cycle_mid_time_ns", "measured_frequency_GHz",
                     "phase_amplitude_rad", "pendulum_frequency_GHz",
                     "free_ringdown"))
    for row, expected in zip(cycles, theory):
        writer.writerow((row[0] * 1e9, row[1] / 1e9, row[2],
                         expected / 1e9, bool(row[3])))

fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True,
                         gridspec_kw={"height_ratios": [1, 1.45]})
axes[0].plot(time * 1e9, phase, linewidth=1.4)
axes[0].axvspan(0.0, DRIVE_OFF * 1e9, color="0.85", alpha=0.7)
axes[0].axhline(np.pi / 2, color="0.2", linestyle=":", linewidth=1.0)
axes[0].text(time[-1] * 1e9 * 0.86, np.pi / 2 + 0.08, r"$\pi/2$")
axes[0].text(0.012, 0.94, "driven (excluded from comparison)",
             transform=axes[0].transAxes, va="top", color="0.35")
axes[0].set_ylabel(r"junction phase $\varphi$ (rad)")
axes[0].grid(alpha=0.28)

axes[1].axvspan(0.0, DRIVE_OFF * 1e9, color="0.85", alpha=0.7)
axes[1].plot(cycles[:, 0] * 1e9, cycles[:, 1] / 1e9, "o-", markersize=3.5,
             label="measured response f (zero-crossing)")
axes[1].plot(cycles[:, 0] * 1e9, theory / 1e9, "k--", linewidth=1.5,
             label=r"pendulum $f_0\pi/(2K(\sin\varphi_0/2))$")
axes[1].axhline(F0 / 1e9, color="green", linestyle=":", linewidth=1.3,
               label=f"small-signal limit {F0 / 1e9:.3f} GHz")
ax_amp = axes[1].twinx()
ax_amp.plot(cycles[:, 0] * 1e9, cycles[:, 2], color="red", alpha=0.5,
            linewidth=1.2, label=r"$\varphi_0$ per cycle")
ax_amp.set_ylabel(r"$\varphi_0$ per cycle (rad)", color="red")
ax_amp.tick_params(axis="y", colors="red")
axes[1].set_xlabel("time (ns)")
axes[1].set_ylabel("frequency (GHz)")
axes[1].grid(alpha=0.28)
handles, labels = axes[1].get_legend_handles_labels()
handles2, labels2 = ax_amp.get_legend_handles_labels()
axes[1].legend(handles + handles2, labels + labels2, loc="best", fontsize=8)

fig.suptitle("Finite down-chirp followed by free RCSJ ring-down")
fig.tight_layout()
fig.savefig(RESULTS / "ringdown_chirp.png", dpi=180)

free = cycles[:, 3].astype(bool) & (cycles[:, 2] > 0.05)
relative_error = (cycles[free, 1] - theory[free]) / theory[free]
print(f"max |phase|: {np.max(np.abs(phase)):.6f} rad")
print(f"free cycles compared: {np.count_nonzero(free)}")
print(f"median |frequency error|: {np.median(np.abs(relative_error)) * 100:.3f}%")
print(f"max |frequency error|: {np.max(np.abs(relative_error)) * 100:.3f}%")
