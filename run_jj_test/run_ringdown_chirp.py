#!/usr/bin/env python3
"""Run a finite down-chirp followed by free RCSJ ring-down."""

import argparse
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EXE = (ROOT / "../Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex").resolve()
RESULTS = ROOT / "results" / "ringdown_chirp"
# Down-chirp through the 5.556 GHz mode (+/-0.6 GHz over +/-2 sigma), demo-style: strong
# shunt damping (R = 200 ohm, Q ~ 17) so the free ring-down sweeps the amplitude.
# Timing is the 16.6 GHz chirp scaled by 16.586/5.556.
# Envelope starts at exp(-4.5) ~ 1% of peak; drive off 2 sigma after the peak.
NSTEPS = 60000  # 4.88 ns at cfl 16: drive off at 1.25 ns + ~20 free cycles
EDRIVE = 2.52  # reduced model: peak 1.41 rad, 1.24 rad at drive-off
R_JJ = 200.0
T0 = 0.749e-9
SIGMA = 0.250e-9
FCENTER = 6.0e9
RATE = -2.40e18
DRIVE_OFF = 1.251e-9

SPATIAL = (
    "(x >= 297.5e-6) * (x <= 362.5e-6) * "
    "(z > h_si - ddz) * (z < h_si + ddz) * "
    "(y > 675.e-6 - ddy) * (y < 675.e-6 + ddy)"
)
EXCITATION = (
    "Edrive * cfl/0.8 * (t < drive_off) * "
    "exp(-(t-chirp_t0)**2/(2*chirp_sigma**2)) * "
    "sin(2*pi*(chirp_fcenter*(t-chirp_t0) + "
    "0.5*chirp_rate*(t-chirp_t0)**2)) * " + SPATIAL
)


def complete(results: Path, nsteps: int) -> bool:
    probe = results / "reducedfiles" / "Egap.txt"
    log = results / "run.log"
    used = results / "warpx_used_inputs"
    if not probe.exists() or not log.exists() or not used.exists():
        return False
    rows = sum(
        1 for line in probe.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    return rows == nsteps + 1 and "finalized" in log.read_text()


def run(results: Path, edrive: float, cfl: float) -> None:
    nsteps = round(NSTEPS * 16.0 / cfl)
    if complete(results, nsteps):
        print(f"reuse completed ring-down chirp in {results}", flush=True)
        return
    (results / "reducedfiles").mkdir(parents=True, exist_ok=True)
    command = [
        str(EXE), str(ROOT / "inputs_artemis_crosstalk_rcsj"),
        f"max_step={nsteps}",
        f"my_constants.cfl={cfl:g}",
        "amrex.the_arena_init_size=0",
        f"my_constants.Edrive={edrive:.17g}",
        f"my_constants.R_jj={R_JJ:.17g}",
        f"my_constants.chirp_t0={T0:.17g}",
        f"my_constants.chirp_sigma={SIGMA:.17g}",
        f"my_constants.chirp_fcenter={FCENTER:.17g}",
        f"my_constants.chirp_rate={RATE:.17g}",
        f"my_constants.drive_off={DRIVE_OFF:.17g}",
        f"warpx.Ex_excitation_grid_function(x,y,z,t)={EXCITATION}",
        f"warpx.used_inputs_file={results / 'warpx_used_inputs'}",
        *(f"{name}.path={results / 'reducedfiles'}/" for name in ("Eobs0", "Eobs1", "Egap")),
    ]
    print(f"run finite chirp + ring-down RCSJ: Edrive={edrive:g} cfl={cfl:g}", flush=True)
    with (results / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not complete(results, nsteps):
        raise RuntimeError(f"ring-down run failed; inspect {results / 'run.log'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--edrive", type=float, default=EDRIVE)
    parser.add_argument("--cfl", type=float, default=16.0, help="steps scale as 16/cfl")
    parser.add_argument("--results", type=Path, default=RESULTS)
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    run(args.results.resolve(), args.edrive, args.cfl)
