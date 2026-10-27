# Metal Plasma FDTD Pilot

`metal_fdtd.swift` ports the unmagnetized update in `src/main.rs` to Apple Metal.
The validated Rust solver is unchanged. Both executables accept the same JSON
configuration and three component-major float64 Yee-edge density arrays.

Build:

```sh
xcrun swiftc -O examples/metal_fdtd.swift -o tmp/metal-fdtd -framework Metal -framework Foundation
```

Run the three Gaussian-plasma probes:

```sh
conda run -n base python examples/check_marshall_gaussian.py --backend metal
```

Run matched-grid CPU/GPU regression tests after building both binaries:

```sh
conda run -n base python -m pytest -q tests/test_metal_gaussian.py
```

The GPU stores fields, current, CPML memory, and time-Fourier integrals in
float32. Host input conversion, incident normalization, and final inter-group
reduction use float64. Fast math is disabled. Unsupported magnetized, circularly
polarized, and field-frame requests are rejected. Only the monostatic backscatter
angle is returned; this is not a full angular-output replacement for the Rust
solver. Electron collisions use the same constant damping parameter as Rust.

The small regression grids test backend agreement, including a nonzero collision
rate. They do not test spatial convergence or physical agreement with a paper.
The Gaussian driver samples Marshall & Close (2015), equation 9:

    omega_p(r) = omega_p0 exp(-r^2/r0^2)
    ne(r) = ne0 exp(-2r^2/r0^2)
    rp = r0 sqrt(log(omega_p0/omega))

The targets are approximate visual readings of Figure 8, not author-provided
individual simulation inputs. Zero collisions/B are a controlled pilot, not
an exact reproduction of every input in that figure.
