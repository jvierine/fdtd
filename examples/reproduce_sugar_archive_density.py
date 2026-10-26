"""Drive native Rust FDTD with Sugar's exact archived dimensionless density cube.

The archived cube is pi*D. Its finite extent is preserved, with zero density
outside it. The candidate archived omega_p multiplier is unchanged.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import h5py
import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator

import reproduce_sugar_fdtd as runner


def main():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--density-archive', type=Path, required=True)
    args, remaining = parser.parse_known_args()
    print('Loading original density cube', args.density_archive, flush=True)
    digest = hashlib.md5()
    with args.density_archive.open('rb') as source:
        for block in iter(lambda: source.read(8*1024*1024), b''):
            digest.update(block)
    if digest.hexdigest() != '15fde3f1eb52ae30a922ef2955472462':
        raise ValueError('Archived density checksum mismatch')
    cube = loadmat(args.density_archive)['np']
    axis = (np.arange(512)-255.5)*.005
    interpolate = RegularGridInterpolator((axis, axis, axis), cube,
                                         bounds_error=False, fill_value=0.)

    def make_density(path, n, dx, pml, peak_shape, amplitude, support_override=None):
        support = (n/2-pml-8)*dx
        if support < np.max(np.abs(axis)):
            raise ValueError('Box cannot contain original density cube')
        density = np.memmap(path, dtype='<f8', mode='w+', shape=(3, n, n, n))
        density[:] = 0.
        grid = (np.arange(n)-n/2)*dx
        maximum = 0.
        for component in range(3):
            coords = [grid+(dx/2 if q == component else 0) for q in range(3)]
            yy, zz = np.meshgrid(coords[1], coords[2], indexing='ij')
            for i, x in enumerate(coords[0]):
                if x < axis[0] or x > axis[-1]:
                    continue
                points = np.stack([np.full_like(yy, x), yy, zz], axis=-1)
                value = amplitude/np.pi*interpolate(points)
                density[component, i] = value
                maximum = max(maximum, float(np.max(value)))
            density.flush()
        sample = np.asarray(density[:, :, n//2, :]).copy()
        del density
        return support, maximum, grid, sample

    runner.make_density = make_density
    sys.argv = [sys.argv[0]]+remaining
    runner.main()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument('--output', type=Path, default=runner.ROOT/'outputs/sugar_reproduction')
    output_args, _ = output_parser.parse_known_args(remaining)
    for path in output_args.output.glob('wp*.h5'):
        with h5py.File(path, 'r+') as h:
            h.attrs['density_source'] = 'Sugar archived np cube, trilinear interpolation, zero outside'
            h.attrs['density_archive_md5'] = digest.hexdigest()
            h.attrs['density_driver_source'] = Path(__file__).read_text()


if __name__ == '__main__':
    main()
