"""Matched-grid backend regression; tiny grids test parity, not convergence."""
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest
from scipy.constants import c, e, epsilon_0, m_e

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('alpha,nu', [(0.3, 0.), (3., 0.), (3., 1e7)])
def test_metal_matches_cpu(tmp_path, alpha, nu):
    cpu = ROOT/'tmp/gaussian_target/release/meteor-fdtd'
    metal = ROOT/'tmp/metal-fdtd'
    if not cpu.exists() or not metal.exists():
        pytest.skip('Build both native backends before running Metal parity tests')
    n = 32
    dx = .1
    axis = (np.arange(n)-n/2)*dx
    critical = epsilon_0*m_e*(2*np.pi*50e6)**2/e**2
    density = tmp_path/'density.bin'
    with density.open('wb') as stream:
        for q in range(3):
            coords = [axis + (dx/2 if j == q else 0) for j in range(3)]
            x, y, z = np.meshgrid(*coords, indexing='ij')
            r2 = x*x+y*y+z*z
            ne = critical*alpha**2*np.exp(-2*r2/.125**2)
            ne[r2 > .5**2] = 0
            stream.write(ne.astype('<f8').tobytes())
    results = []
    for backend in [cpu, metal]:
        output = tmp_path/(backend.name+'.json')
        config = dict(n=n, dx_m=dx, pml_cells=4, frequency_hz=50e6,
                      collision_rate_s1=nu, magnetic_field_t=[0., 0., 0.],
                      polarization='linear', support_radius_m=.5, periods=4.,
                      density_file=str(density), output_file=str(output),
                      threads=2, field_frames=0)
        path = tmp_path/'config.json'
        path.write_text(json.dumps(config))
        subprocess.run([str(backend), str(path)], check=True, capture_output=True)
        results.append(json.loads(output.read_text()))
    difference = 10*np.log10(results[1]['rcs_m2'][-1]/results[0]['rcs_m2'][-1])
    assert abs(difference) < .01, difference
    a = np.asarray(results[0]['radiation_amplitude'][-1])
    b = np.asarray(results[1]['radiation_amplitude'][-1])
    assert np.linalg.norm(a-b)/np.linalg.norm(a) < .002
