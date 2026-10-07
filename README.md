# Meteor plasma FDTD

A native Rust solver for three-dimensional electromagnetic scattering from a
collisional, magnetized plasma. It uses a fixed Yee grid, CPML absorbing
boundaries, MPI decomposition along x and Rayon threads within each rank.
Python prepares densities, launches the solver and saves HDF5 and plots.

## Run the meteor example

Requires Rust (2024 edition), an MPI installation with `mpicc` and `mpiexec`,
and the Python packages in `requirements.txt`. Use matching MPI compiler and
launcher installations; `MPICC` and `MPIEXEC` can select them explicitly.

```bash
conda run -n base python -m pip install -r requirements.txt
cargo build --release --locked
conda run -n base python examples/maarsy_rcs.py
```

The example uses 53.5 MHz, a capped and tapered Dimant–Oppenheim density,
a 0.6 m support radius, 3.125 cm cells, electron damping of 1e7 s^-1 and
a magnetic field of (-50, 0, 0) microtesla. It launches four MPI ranks with
two threads each. Adjust these with `--ranks` and `--threads`.

Results go to `outputs/meteor.h5`, with a density slice and angular RCS in
PNG and PDF. The HDF5 file includes parameters, source code and source hashes.
The accompanying LaTeX snippet has a `\showrcsscriptsfalse` toggle to hide
script provenance; it is shown by default.

```bash
conda run -n base python examples/maarsy_rcs.py --resolution 64 --output outputs/finer.h5
conda run -n base python examples/maarsy_rcs.py --help
```

`--resolution` is cells per metre. The default is a short demonstration;
check mesh spacing, integration time, boundary separation, density cap and
tail cutoff before interpreting a meteor RCS. Density normalization and
damping are supplied parameters, not inferred from meteor velocity or altitude.

The native solver reads a JSON configuration and little-endian density array;
the Python example creates these temporary inputs. It evolves the scattered
fields and electron current, then integrates the current in the far field.
The reported RCS is 4π times the differential scattering cross-section.
The angular plot is a scattering pattern for one incident direction; use
`--aspect-deg` to rotate the meteor density relative to that direction.

## Tests

```bash
cargo test --locked
conda run -n base python -m pytest -q
```

Classical unmagnetized sphere benchmarks compare native FDTD with exact Mie
and Rayleigh scattering on grids specified as fractions of the wavelength:

```bash
PYTHONPATH=examples conda run -n base python tests/validate_sphere_scattering.py
```

This full convergence benchmark takes substantially longer than the example.
It writes HDF5, figures and a LaTeX caption with script provenance to `outputs/`.

## References

- Marshall & Close (2015), plasma FDTD and meteor head echoes: https://doi.org/10.1002/2015JA021238
- Dimant & Oppenheim (2017), analytic meteor density: https://doi.org/10.1002/2017JA023963
- Sugar et al. (2021), meteor head plasma scattering: https://doi.org/10.1029/2021JA029171

The density implementation documents its relationship to MetAblate in
`examples/dimant_oppenheim.py`. This repository contains the solver and example;
generated simulations and parameter sweeps are excluded from Git.
