#!/usr/bin/env python3
"""Run with: python3 analysis.py /absolute/path/to/artemis.ex.

Compare bulk and edge resistance on a non-cubic periodic grid, including
additive conductivity, disabled resistors, and directional isolation.
"""
import math
from pathlib import Path
import subprocess
import sys
import tempfile

EXE = Path(sys.argv[1]).resolve()
BASE = '''
max_step = 8
geometry.dims = 3
geometry.prob_lo = 0 0 0
geometry.prob_hi = 0.008 0.016 0.024
amr.n_cell = 8 8 8
amr.max_level = 0
amr.max_grid_size = 8
amr.blocking_factor = 8
boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic
warpx.verbose = 0
warpx.const_dt = 1.e-12
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0
algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
macroscopic.epsilon = 8.8541878128e-12
macroscopic.mu = 1.25663706212e-6
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.reduced_diags_names = energy
energy.type = FieldEnergy
energy.intervals = 1
energy.path = diags/
particles.nspecies = 0
'''


def run(root, name, sigma, resistance=None, component=None, enabled=True,
        fail=False):
    work = root / name
    work.mkdir()
    text = BASE + f'\nmacroscopic.sigma = {sigma}\n'
    for axis in 'xyz':
        amplitude = 1 if component is None or component == axis else 0
        text += f'warpx.E{axis}_external_grid_function(x,y,z) = "{amplitude}"\n'
    if resistance is not None:
        text += f'warpx.use_lumped_resistor = {int(enabled)}\n'
        for axis, value in zip('xyz', resistance):
            text += f'macroscopic.lumped_resistor_{axis}_function(x,y,z) = "{value:.17g}"\n'
    (work / 'inputs').write_text(text)
    result = subprocess.run([str(EXE), 'inputs'], cwd=work,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    (work / 'run.log').write_text(result.stdout)
    if fail:
        assert result.returncode != 0 and 'finite and nonnegative' in result.stdout
        return
    if result.returncode:
        raise RuntimeError(result.stdout[-6000:])
    return [[float(x) for x in line.split()] for line in
            (work / 'diags/energy.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')]


def compare(a, b):
    assert len(a) == len(b) and len(a) > 1
    for row_a, row_b in zip(a, b):
        for x, y in zip(row_a, row_b):
            assert math.isclose(x, y, rel_tol=2e-12, abs_tol=1e-28), (x, y)


with tempfile.TemporaryDirectory(prefix='artemis-resistor-') as tmp:
    root = Path(tmp)
    # dx=(.001,.002,.003); R=length/(area*sigma), sigma_R=1 S/m.
    resistance = (1e-3 / (2e-3 * 3e-3),
                  2e-3 / (1e-3 * 3e-3),
                  3e-3 / (1e-3 * 2e-3))
    bulk = run(root, 'bulk', 1)
    compare(bulk, run(root, 'resistor', 0, resistance))
    compare(run(root, 'bulk2', 2), run(root, 'combined', 1, resistance))
    compare(bulk, run(root, 'zeros', 1, (0, 0, 0)))
    compare(bulk, run(root, 'disabled', 1, resistance, enabled=False))
    for i, axis in enumerate('xyz'):
        directional = tuple(resistance[j] if j == i else 0 for j in range(3))
        compare(run(root, axis+'bulk', 1, component=axis),
                run(root, axis+'resistor', 0, directional, component=axis))
        other = 'xyz'[(i+1) % 3]
        compare(run(root, axis+'vacuum', 0, component=other),
                run(root, axis+'isolated', 0, directional, component=other))
    run(root, 'negative', 0, (-1, 0, 0), fail=True)
print('PASS: equivalence, additive loss, zero/disabled resistors, directional isolation, negative input')
