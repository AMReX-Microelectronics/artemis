#!/usr/bin/env python3
"""Run linear and RCSJ Edrive sweeps for the 20 GHz Gaussian drive."""

from __future__ import annotations

import argparse
import os
import queue
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


ROOT = Path(__file__).resolve().parent
EXE = (ROOT / "../Bin/main3d.gnu.TPROF.MTMPI.CUDA.ex").resolve()
RESULTS = ROOT / "results" / "edrive_scan"
# Edrive at cfl 0.8 (deck scales by cfl/0.8). Drive at resonance: ~0.91 rad/(V/m) per
# junction at that normalisation; the reduced model gives A = 0.2 ... 1.3 rad.
DRIVES = (0.01, 0.219, 0.545, 0.864, 1.074, 1.286, 1.395)
NSTEPS = 30000  # 2.44 ns at cfl 16: 5.556 GHz pulse (ends 6*TP = 1.08 ns) + ~7.5 free cycles


def case_dir(kind: str, drive: float) -> Path:
    return RESULTS / kind / f"E{drive:g}"


def complete(folder: Path) -> bool:
    probe = folder / "reducedfiles" / "Egap.txt"
    log = folder / "run.log"
    used_inputs = folder / "warpx_used_inputs"
    if not probe.exists() or not log.exists() or not used_inputs.exists():
        return False
    with probe.open() as stream:
        rows = sum(1 for line in stream if line.strip() and not line.startswith("#"))
    return rows == NSTEPS + 1 and "finalized" in log.read_text()


def run_case(kind: str, drive: int, gpus: queue.Queue) -> None:
    folder = case_dir(kind, drive)
    if complete(folder):
        print(f"reuse {kind} Edrive={drive}", flush=True)
        return
    (folder / "reducedfiles").mkdir(parents=True, exist_ok=True)
    deck = ROOT / ("inputs_artemis_crosstalk_L" if kind == "linear" else "inputs_artemis_crosstalk_rcsj")
    command = [
        str(EXE), str(deck),
        f"max_step={NSTEPS}",
        "amrex.the_arena_init_size=0",
        f"my_constants.Edrive={drive}",
        f"warpx.used_inputs_file={folder / 'warpx_used_inputs'}",
        *(f"{name}.path={folder / 'reducedfiles'}/" for name in ("Eobs0", "Eobs1", "Egap")),
    ]
    gpu = gpus.get()
    try:
        print(f"run {kind} Edrive={drive} on GPU {gpu}", flush=True)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
        with (folder / "run.log").open("w") as log:
            result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, env=env)
    finally:
        gpus.put(gpu)
    if result.returncode or not complete(folder):
        raise RuntimeError(f"{kind} Edrive={drive} failed; inspect {folder / 'run.log'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--linear-only", action="store_true")
    group.add_argument("--rcsj-only", action="store_true")
    parser.add_argument("--gpus", type=int, nargs="+", default=[0], help="GPU ids to spread cases over")
    parser.add_argument("--per-gpu", type=int, default=1, help="concurrent cases per GPU")
    parser.add_argument("--drives", type=float, nargs="+", default=list(DRIVES), help="Edrive values in V/m")
    args = parser.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    kinds = [kind for kind, skip in (("linear", args.rcsj_only), ("rcsj", args.linear_only)) if not skip]
    gpus: queue.Queue = queue.Queue()
    for _ in range(args.per_gpu):
        for gpu in args.gpus:
            gpus.put(gpu)
    with ThreadPoolExecutor(max_workers=len(args.gpus) * args.per_gpu) as pool:
        futures = [pool.submit(run_case, kind, drive, gpus) for kind in kinds for drive in args.drives]
        for future in futures:
            future.result()


if __name__ == "__main__":
    main()
