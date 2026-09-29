#!/usr/bin/env python3
"""CPU/GPU executable integration tests for centered ADI Josephson coupling.

Usage: python3 analysis.py /path/to/artemis.ex [--workdir /tmp/jj-tests]
The executable's working directories and logs are retained for review.
"""
import argparse
import math
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument('executable', type=Path)
parser.add_argument('--workdir', type=Path)
args = parser.parse_args()
exe = args.executable.resolve()
root = args.workdir or Path(tempfile.mkdtemp(prefix='artemis-jj-'))
root.mkdir(parents=True, exist_ok=True)
EPS = 8.8541878128e-12
QE = 1.602176634e-19
HBAR = 1.054571817e-34
DX = (1e-3, 2e-3, 3e-3)
OMEGA = 1e10
BASE = '''
geometry.dims = 3
geometry.prob_lo = 0 0 0
geometry.prob_hi = .008 .016 .024
amr.n_cell = 8 8 8
amr.max_level = 0
amr.max_grid_size = 8
amr.blocking_factor = 8
boundary.field_lo = periodic periodic periodic
boundary.field_hi = periodic periodic periodic
warpx.verbose = 0
warpx.use_filter = 0
warpx.do_particle_cfl_guards = 0
algo.em_solver_medium = macroscopic
algo.time_stepping_scheme = adi
macroscopic.mu = 1.25663706212e-6
warpx.E_ext_grid_init_style = parse_E_ext_grid_function
warpx.reduced_diags_names = probe energy
energy.type = FieldEnergy
energy.intervals = 1
energy.path = diags/
probe.type = RawEFieldReduction
probe.reduction_type = integral
probe.integration_type = volume
probe.reduced_function(x,y,z) = "1/(.008*.016*.024)"
probe.intervals = 1
probe.path = diags/
particles.nspecies = 0
'''


def run(name, axis=0, dt=1e-11, steps=32, phase=1., omega=OMEGA,
        sigma=0., rc=False, enabled=True, e0=0., extra='', fail=None,
        checkpoint=0, restart=None):
    folder = root / name
    folder.mkdir(exist_ok=True)
    eta = 2*QE/HBAR*DX[axis]
    area = DX[(axis+1)%3]*DX[(axis+2)%3]
    eps = EPS * (2 if rc else 1)
    ic = omega**2 * eps * area / eta
    text = BASE + f'\nmax_step = {steps}\nwarpx.const_dt = {dt:.17g}\n'
    text += f'macroscopic.epsilon = {EPS:.17g}\nmacroscopic.sigma = {sigma:.17g}\n'
    if enabled:
        text += f'algo.use_josephson_junction = 1\njosephson.initial_phase = {phase:.17g}\n'
        text += 'josephson.newton_rtol = 1e-12\njosephson.newton_atol = 1e-15\n'
        for i, a in enumerate('xyz'):
            text += f'josephson.Ic_{a}_function(x,y,z) = "{ic if i == axis else 0:.17g}"\n'
    for i, a in enumerate('xyz'):
        text += f'warpx.E{a}_external_grid_function(x,y,z) = "{e0 if i == axis else 0:.17g}"\n'
    if rc:
        text += 'warpx.use_lumped_capacitor = 1\nwarpx.use_lumped_resistor = 1\n'
        for i, a in enumerate('xyz'):
            cap = EPS*area/DX[axis] if i == axis else 0.
            # Shunt damping rate sigma_R / eps_eff = 0.1 omega.
            res = DX[axis]/(area * eps * 0.1*OMEGA) if i == axis else 0.
            text += f'macroscopic.lumped_capacitor_{a}_function(x,y,z) = "{cap:.17g}"\n'
            text += f'macroscopic.lumped_resistor_{a}_function(x,y,z) = "{res:.17g}"\n'
    if checkpoint:
        text += f'''diagnostics.diags_names = chk plot
chk.diag_type = Full
chk.format = checkpoint
chk.intervals = {checkpoint}
chk.file_prefix = chk
plot.diag_type = Full
plot.format = plotfile
plot.intervals = {checkpoint}
plot.file_prefix = plt
plot.fields_to_plot = Ex Ey Ez josephson_phi_x josephson_phi_y josephson_phi_z josephson_Ic_x
'''
    if restart:
        text += f'amr.restart = {restart}\n'
    text += extra
    (folder/'inputs').write_text(text)
    result = subprocess.run([str(exe), 'inputs'], cwd=folder, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=180)
    (folder/'run.log').write_text(result.stdout)
    if fail:
        assert result.returncode != 0 and fail in result.stdout, result.stdout[-4000:]
        return None
    assert result.returncode == 0, f'{folder}:\n{result.stdout[-5000:]}'
    rows = [[float(v) for v in line.split()] for line in
            (folder/'diags/probe.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
    return rows


def reference(phase, time, damping=0.):
    # Independent RK4 reference in dimensionless time tau=omega*t,
    # u=eta*E/omega: phi'=u, u'=-sin(phi)-damping*u.
    p, u = phase, 0.
    n = 20000
    h = OMEGA*time/n
    def f(p, u):
        return u, -math.sin(p)-damping*u
    for _ in range(n):
        a,b = f(p,u); c,d = f(p+h*a/2,u+h*b/2)
        e,g = f(p+h*c/2,u+h*d/2); j,k = f(p+h*e,u+h*g)
        p += h*(a+2*c+2*e+j)/6
        u += h*(b+2*d+2*g+k)/6
    return p,u


def value(rows, axis):
    return rows[-1][2+axis]


# Exact recovery of RC when the junction is absent, even with nonzero stored phase.
a = run('zero-ic', omega=0., rc=True, e0=.001)
b = run('disabled', enabled=False, rc=True, e0=.001)
assert all(math.isclose(x,y,rel_tol=2e-12,abs_tol=1e-15)
           for ra,rb in zip(a,b) for x,y in zip(ra,rb))
print('PASS zero-Ic RC recovery', flush=True)

# Nonlinear oscillation, independent reference, second-order refinement; all axes.
for axis in range(3):
    eta = 2*QE/HBAR*DX[axis]
    _, ref = reference(1.8, 32e-11)
    coarse = run(f'nonlinear-{axis}', axis=axis, phase=1.8)
    fine = run(f'fine-{axis}', axis=axis, dt=5e-12, steps=64, phase=1.8)
    ec = abs(value(coarse,axis)*eta/OMEGA-ref)
    ef = abs(value(fine,axis)*eta/OMEGA-ref)
    assert ef < ec/3.5 and ef < .002, (axis,ec,ef)
    for row in fine:
        assert max(abs(row[2+i]) for i in range(3) if i != axis) < 1e-14
print('PASS nonlinear reference, second-order refinement, directional isolation', flush=True)

eta = 2*QE/HBAR*DX[0]
# Linearized plasma at omega*h=2.5, beyond symplectic-Euler stability limit.
large = run('large-step', dt=5e-10, steps=8, phase=1e-5)
expected = -1e-5*OMEGA/eta*math.sin(4*8*math.atan(OMEGA*5e-10/4))
assert math.isclose(value(large,0),expected,rel_tol=2e-6,abs_tol=1e-14)
print('PASS large-step linearized plasma', flush=True)

rc = run('rcsj', dt=5e-12, steps=64, phase=1., rc=True)
_, ref = reference(1.,32e-11,damping=.1)
assert abs(value(rc,0)*eta/OMEGA-ref) < .001
print('PASS RCSJ against independent ODE', flush=True)

full = run('checkpoint', steps=16, phase=1., checkpoint=8)
chk = root/'checkpoint/chk000008'
assert (chk/'Level_0/jj_phi_x_H').exists(), list((root/'checkpoint').iterdir())
restart = run('restart', steps=16, phase=1., restart=chk.resolve())
assert all(math.isclose(x,y,rel_tol=2e-11,abs_tol=1e-14)
           for x,y in zip(full[-1],restart[-1]))
print('PASS phase checkpoint/restart and plot diagnostics', flush=True)

# A spatially varying junction drives the Maxwell curl and pencil coupling.
spatial = 'josephson.Ic_x_function(x,y,z) = "1.748434719095732e-9*(1+0.4*cos(2*3.141592653589793*y/.016))"\n'
s1 = run('spatial', steps=8, extra=spatial)
s2 = run('spatial-split', steps=8, extra=spatial+'amr.max_grid_size = 4\namr.blocking_factor = 4\n')
assert all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-13)
           for ra,rb in zip(s1,s2) for x,y in zip(ra,rb))
def energies(name):
    return [[float(x) for x in line.split()] for line in
            (root/name/'diags/energy.txt').read_text().splitlines()
            if line.strip() and not line.startswith('#')]
assert all(math.isclose(x,y,rel_tol=1e-9,abs_tol=1e-28)
           for ra,rb in zip(energies('spatial'),energies('spatial-split'))
           for x,y in zip(ra,rb))
assert energies('spatial')[-1][-1] > 0., 'Spatial junction should generate magnetic energy'
print('PASS spatial junction and multiple-box decomposition', flush=True)

run('negative', extra='josephson.Ic_x_function(x,y,z) = "-1"\n', fail='finite and nonnegative')
run('nonconvergence', phase=1.8, dt=1e-10,
    extra='josephson.max_iterations = 1\n', fail='Josephson Newton')
print(f'PASS invalid input and nonlinear failure; logs: {root}', flush=True)
