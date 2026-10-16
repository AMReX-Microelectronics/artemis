#!/usr/bin/env python3
"""Run explicit calibrated linear and RCSJ Edrive sweeps."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXE = (ROOT / "../Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex").resolve()
RESULTS = ROOT / "results" / "edrive_scan"
DRIVES = (100, 2000, 4000, 6000, 8000, 10000, 12000)
NSTEPS = 7500


def case_dir(kind: str, drive: int) -> Path:
    return RESULTS / kind / f"E{drive:06d}"


def complete(folder: Path) -> bool:
    probe = folder / "reducedfiles" / "Eobs1.txt"
    log = folder / "run.log"
    used_inputs = folder / "warpx_used_inputs"
    if not probe.exists() or not log.exists() or not used_inputs.exists():
        return False
    with probe.open() as stream:
        rows = sum(1 for line in stream if line.strip() and not line.startswith("#"))
    return rows == NSTEPS + 1 and "finalized" in log.read_text()


def run_case(kind: str, drive: int) -> None:
    folder = case_dir(kind, drive)
    if complete(folder):
        print(f"reuse {kind} Edrive={drive}", flush=True)
        return
    (folder / "reducedfiles").mkdir(parents=True, exist_ok=True)
    deck = ROOT / ("inputs_artemis_crosstalk_L" if kind == "linear" else "inputs_artemis_crosstalk_rcsj")
    command = [
        str(EXE), str(deck),
        "amrex.the_arena_init_size=0",
        f"my_constants.Edrive={drive}",
        f"warpx.used_inputs_file={folder / 'warpx_used_inputs'}",
        f"Eobs0.path={folder / 'reducedfiles'}/",
        f"Eobs1.path={folder / 'reducedfiles'}/",
    ]
    print(f"run {kind} Edrive={drive}", flush=True)
    with (folder / "run.log").open("w") as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
    if result.returncode or not complete(folder):
        raise RuntimeError(f"{kind} Edrive={drive} failed; inspect {folder / 'run.log'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--linear-only", action="store_true")
    group.add_argument("--rcsj-only", action="store_true")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    if not args.rcsj_only:
        for drive in DRIVES:
            run_case("linear", drive)
    if not args.linear_only:
        for drive in DRIVES:
            run_case("rcsj", drive)


if __name__ == "__main__":
    main()
