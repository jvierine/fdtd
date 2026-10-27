"""Sparse Figure 8 Gaussian-plasma check, Marshall & Close (2015), Eq. 9.

Targets are approximate readings of the simulated locus, not archived samples.
The unmodified native solver is used without magnetic field or collisions.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/marshall_gaussian_50MHz_20261027'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backend', choices=['cpu', 'metal'], default='cpu')
    parser.add_argument('--kr-p', nargs=3, type=float, default=[.42, .70, 1.87])
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CARGO_TARGET_DIR=str(ROOT / 'tmp/gaussian_target'),
               MPICC='/opt/homebrew/bin/mpicc')
    if args.backend == 'cpu':
        subprocess.run(['cargo', 'build', '--release', '--locked', '--jobs', '2',
                        '--bin', 'meteor-fdtd'], cwd=ROOT, env=env, check=True)
        executable = ROOT / 'tmp/gaussian_target/release/meteor-fdtd'
    else:
        executable = ROOT / 'tmp/metal-fdtd'
        subprocess.run(['xcrun', 'swiftc', '-O', str(ROOT/'examples/metal_fdtd.swift'),
                        '-o', str(executable), '-framework', 'Metal', '-framework', 'Foundation'], check=True)
    frequency = 50e6
    wavelength = c / frequency
    critical = epsilon_0 * m_e * (2*np.pi*frequency)**2 / e**2
    for target, kr, alpha in zip([-10., 0., 10.], args.kr_p, [3., 3., 6.]):
        rp = kr*wavelength/(2*np.pi)
        r0 = rp/np.sqrt(np.log(alpha))
        dx = min(wavelength/100., r0/12.)
        support = 4*r0
        n = int(np.ceil(2*(support/dx+40)/16)*16)
        name = f'target{target:+g}_n{n}' + ('_metal' if args.backend == 'metal' else '')
        name += f'_kr{kr:g}'
        output = OUT/(name+'.h5')
        if output.exists():
            continue
        density_path = OUT/(name+'.density.bin')
        axis = (np.arange(n)-n/2)*dx
        center = None
        # Exact Eq. 9 at the three staggered electric/current edge locations.
        with density_path.open('wb') as stream:
            for q in range(3):
                coordinates = [axis + (dx/2 if d == q else 0.) for d in range(3)]
                y, z = np.meshgrid(coordinates[1], coordinates[2], indexing='ij')
                for x in coordinates[0]:
                    radius2 = x*x+y*y+z*z
                    density = critical*alpha**2*np.exp(-2*radius2/r0**2)
                    density[radius2 > support**2] = 0.
                    stream.write(density.astype('<f8').tobytes())
                if q == 2:
                    x, z = np.meshgrid(coordinates[0], coordinates[2], indexing='ij')
                    center = critical*alpha**2*np.exp(-2*(x*x+z*z)/r0**2)
        config = dict(n=n, dx_m=dx, pml_cells=20, frequency_hz=frequency,
                      collision_rate_s1=0., magnetic_field_t=[0., 0., 0.],
                      polarization='linear', support_radius_m=support, periods=6.,
                      density_file=str(density_path), output_file=str(OUT/(name+'.json')),
                      threads=4, field_frames=0)
        path = OUT/(name+'.config.json')
        path.write_text(json.dumps(config))
        print('RUN', name, 'r0', r0, 'rp', rp, 'alpha', alpha, 'dx', dx, flush=True)
        subprocess.run([str(executable), str(path)],
                       env=env, check=True)
        result = json.loads(Path(config['output_file']).read_text())
        value = 10*np.log10(result['rcs_m2'][-1])
        with h5py.File(output, 'w') as h:
            h.attrs.update(reference_doi='10.1002/2015JA021238', target_dbsm=target,
                           target_status='Approximate visual reading of Figure 8 simulated locus',
                           r0_m=r0, contour_radius_m=rp, kr_p=kr, peak_wp_over_omega=alpha,
                           computed_dbsm=value, backend=args.backend, status='Pilot, not convergence certified',
                           profile='omega_p=omega_p0*exp(-r^2/r0^2); ne=ne0*exp(-2*r^2/r0^2)',
                           source_program=Path(__file__).read_text(),
                           rust_sha256=hashlib.sha256((ROOT/'src/main.rs').read_bytes()).hexdigest())
            if args.backend == 'metal':
                h.attrs['metal_source'] = (ROOT/'examples/metal_fdtd.swift').read_text()
            h['config_json'] = json.dumps(config)
            h['density_coordinate_m'] = axis
            h['density_y0_slice_m3'] = center
            for key, item in result.items():
                h[key] = item
        density_path.unlink()
        print('SAVED', name, 'RCS', value, 'target', target, flush=True)


if __name__ == '__main__':
    main()
