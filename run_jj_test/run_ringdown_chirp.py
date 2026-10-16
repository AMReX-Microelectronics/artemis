#!/usr/bin/env python3
"""Run a finite down-chirp followed by free RCSJ ring-down."""

from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parent
EXE = (ROOT / "../Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex").resolve()
RESULTS = ROOT / "results" / "ringdown_chirp"
NSTEPS = 15000
EDRIVE = 12000.0
T0 = 110.0e-12
SIGMA = 60.0e-12
FCENTER = 78.0e9
RATE = -1.8e20
DRIVE_OFF = 250.0e-12
R_JJ = 7.5e4

SPATIAL = (
    "(x >= 297.5e-6) * (x <= 362.5e-6) * "
    "(z > h_si - ddz) * (z < h_si + ddz) * "
    "(y > 675.e-6 - ddy) * (y < 675.e-6 + ddy)"
)
EXCITATION = (
    "Edrive * (t < drive_off) * "
    "exp(-(t-chirp_t0)**2/(2*chirp_sigma**2)) * "
    "sin(2*pi*(chirp_fcenter*(t-chirp_t0) + "
    "0.5*chirp_rate*(t-chirp_t0)**2)) * " + SPATIAL
)


def complete() -> bool:
    probe = RESULTS / "reducedfiles" / "Eobs1.txt"
    log = RESULTS / "run.log"
    used = RESULTS / "warpx_used_inputs"
    if not probe.exists() or not log.exists() or not used.exists():
        return False
    rows = sum(
        1 for line in probe.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    )
    return rows == NSTEPS + 1 and "finalized" in log.read_text()


def run() -> None:
    if complete():
        print("reuse completed ring-down chirp", flush=True)
        return
    (RESULTS / "reducedfiles").mkdir(parents=True, exist_ok=True)
    command = [
        str(EXE), str(ROOT / "inputs_artemis_crosstalk_rcsj"),
        f"max_step={NSTEPS}",
        "amrex.the_arena_init_size=0",
        f"my_constants.Edrive={EDRIVE:.17g}",
        f"my_constants.R_jj={R_JJ:.17g}",
        f"my_constants.chirp_t0={T0:.17g}",
        f"my_constants.chirp_sigma={SIGMA:.17g}",
        f"my_constants.chirp_fcenter={FCENTER:.17g}",
        f"my_constants.chirp_rate={RATE:.17g}",
        f"my_constants.drive_off={DRIVE_OFF:.17g}",
        f"warpx.Ex_excitation_grid_function(x,y,z,t)={EXCITATION}",
        f"warpx.used_inputs_file={RESULTS / 'warpx_used_inputs'}",
        f"Eobs0.path={RESULTS / 'reducedfiles'}/",
        f"Eobs1.path={RESULTS / 'reducedfiles'}/",
    ]
    print("run finite chirp + ring-down RCSJ", flush=True)
    with (RESULTS / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not complete():
        raise RuntimeError(f"ring-down run failed; inspect {RESULTS / 'run.log'}")


if __name__ == "__main__":
    RESULTS.mkdir(parents=True, exist_ok=True)
    run()
