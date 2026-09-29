#!/usr/bin/env python3
"""Run with: python3 analysis.py /absolute/path/to/artemis.ex.

Compare bulk and edge capacitance on a non-cubic periodic grid, including
disabled capacitors, directional isolation, wave propagation, and LC coupling.
"""
import math
from pathlib import Path
import subprocess
import sys
import tempfile

EXE = Path(sys.argv[1]).resolve()
EPS0 = 8.8541878128e-12
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
macroscopic.mu = 1.25663706212e-6
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.reduced_diags_names = energy
energy.type = FieldEnergy
energy.intervals = 1
energy.path = diags/
particles.nspecies = 0
'''


def run(root, name, epsilon, capacitance=None, component=None, enabled=True,
        fail=False, wave=False, inductor=False, resistor=False):
    work = root / name
    work.mkdir()
    text = BASE + f'\nmacroscopic.sigma = 1\nmacroscopic.epsilon = {epsilon:.17g}\n'
    if resistor:
        text += 'warpx.use_lumped_resistor = 1\n'
        for axis in 'xyz':
            text += f'macroscopic.lumped_resistor_{axis}_function(x,y,z) = "1000"\n'
    if inductor:
        text += 'algo.use_lumped_inductor = 1\n'
        for axis in 'xyz':
            text += f'inductor.inductor_{axis}_function(x,y,z) = \"1e-8\"\n'
    for axis in 'xyz':
        amplitude = 1 if component is None or component == axis else 0
        field = f'{amplitude}*cos(2*pi*z/0.024)' if wave else str(amplitude)
        text += f'warpx.E{axis}_external_grid_function(x,y,z) = \"{field}\"\n'
    text += 'my_constants.pi = 3.141592653589793\n'
    if capacitance is not None:
        text += f'warpx.use_lumped_capacitor = {int(enabled)}\n'
        for axis, value in zip('xyz', capacitance):
            text += f'macroscopic.lumped_capacitor_{axis}_function(x,y,z) = "{value:.17g}"\n'
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


with tempfile.TemporaryDirectory(prefix='artemis-capacitor-') as tmp:
    root = Path(tmp)
    # dx=(.001,.002,.003); C=epsilon_added*area/length.
    capacitance = (EPS0 * 2e-3 * 3e-3 / 1e-3,
                   EPS0 * 1e-3 * 3e-3 / 2e-3,
                   EPS0 * 1e-3 * 2e-3 / 3e-3)
    bulk = run(root, 'bulk', 2*EPS0)
    compare(bulk, run(root, 'capacitor', EPS0, capacitance))
    compare(bulk, run(root, 'zeros', 2*EPS0, (0, 0, 0)))
    compare(bulk, run(root, 'disabled', 2*EPS0, capacitance, enabled=False))
    for i, axis in enumerate('xyz'):
        directional = tuple(capacitance[j] if j == i else 0 for j in range(3))
        compare(run(root, axis+'bulk', 2*EPS0, component=axis),
                run(root, axis+'capacitor', EPS0, directional, component=axis))
        other = 'xyz'[(i+1) % 3]
        compare(run(root, axis+'vacuum', EPS0, component=other),
                run(root, axis+'isolated', EPS0, directional, component=other))
    compare(run(root, 'wavebulk', 2*EPS0, component='x', wave=True),
            run(root, 'wavecap', EPS0, capacitance, component='x', wave=True))
    compare(run(root, 'inductorbulk', 2*EPS0, inductor=True),
            run(root, 'inductorcap', EPS0, capacitance, inductor=True))
    compare(run(root, 'rlcbulk', 2*EPS0, inductor=True, resistor=True),
            run(root, 'rlccap', EPS0, capacitance, inductor=True, resistor=True))
    # Independent check: uniform E decays by two centered ADI half-steps.
    q = (4*2*EPS0 - 1e-12) / (4*2*EPS0 + 1e-12)
    for old, new in zip(bulk, bulk[1:]):
        assert math.isclose(new[2] / old[2], q**4, rel_tol=2e-12)
    run(root, 'negative', EPS0, (-1, 0, 0), fail=True)
print('PASS: bulk equivalence, zero/disabled capacitors, directional isolation, wave, inductor/RLC, analytic damping, negative input')
