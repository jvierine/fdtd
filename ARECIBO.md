# HF reflection from the Arecibo sporadic-E layer

The self-contained two-way model setup is at
https://github.com/jvierine/arecibo-hf-conversion. It pins this field library and
https://github.com/jvierine/electron-fluid, exchanging E and current at every
Yee timestep. That repository includes the calibrated profile and collision
inputs. The examples below preserve the original coordinated-workspace
workflow; their archived observation fixtures are not bundled in this field
library. Their tests skip explicitly when those fixtures are absent.

This example calculates the 5.33 MHz electric field in the smooth, calibrated
2018-05-04 17:03:50 UTC electron-density profile. It uses all three field and
current components, the measured-profile geomagnetic prior, and electron–ion
plus MSIS-based electron–neutral momentum loss at an assumed 300 K.

The stratified solver is an extension of this repository's Yee/cold-current
scheme. Both executables use the same trapezoidal electron-current kernel.
Horizontal dependence is represented by exp(i kx x); altitude is resolved with
a graded Yee grid. A tangential current sheet launches the HF wave from below.
An independently computed homogeneous launch-medium reference fixes the upward
incident flux. Absorbing buffers are separate from physical collision loss.

The continuous-wave state is initialized by solving the **discrete FDTD
recurrence**, then evolved and recorded by Rust. This avoids waiting thousands
of cycles for a cold resonant response to settle. It is a stationary reflection
calculation, not a simulation of transmitter switch-on or measured growth time.

## Run

From the parent Arecibo workspace, after compiling this repository:

```sh
cargo build --release --locked --manifest-path hf_fdtd/Cargo.toml --bin stratified
conda run -n base python hf_fdtd/examples/arecibo_layer.py --scan --dz 1
conda run -n base python hf_fdtd/examples/arecibo_layer.py --run --graded --dz 0.001953125 --angle 16.75 --cycles 2
```

Inputs are the existing calibrated profile and neutral prior. Scientific outputs
are HDF5; temporary JSON and binary arrays are only native-program transport.

## Angle and polarization

The broad scan covers vacuum-equivalent angles -40 to +40 degrees and incidence
planes at 0, 45 and 90 degrees to the horizontal magnetic field. A refined scan
finds about **16.75 degrees** in the magnetic meridian. At each angle the complex
two-component source polarization is optimized for equal upward incident flux.
The field is elliptically polarized.

Here angle means asin(kx c / omega). Refraction changes the local propagation
angle; the source may excite two magnetoionic modes. This is a favorable
**conditional configuration**, not an identified transmitter direction.

The objective is the integral of |Ez| squared within 30 m of the cold vertical
resonance. This is a forcing proxy, not a calculation of Langmuir-wave conversion
efficiency. No global optimization over arbitrary 3-D layers is claimed.

## Recorded field and coupling

`arecibo_hf_field.h5` contains the component-specific altitude grids, complex
Ex/Ey/Ez and H phasors, actual Rust-recorded E(t,z) and its spatial quadrature,
frequency, kx, background density, collisions, incident reference and provenance.
Fields are normalized to **1 W/m² upward incident flux**. Scale fields by the
square root of the desired flux. This is a normalization, not an inferred event
power. The complex convention is Re{E(z) exp(i kx x - i omega t)}.

`replay_arecibo_field.HFField` evaluates the field at arbitrary times and
positions, preserving phase. Its default is the transverse electromagnetic
field. `acceleration` supplies -e E/m_e to the momentum equation.

The total cold field includes its self-consistent longitudinal charge-response
field. Do not add that field as a new external source while also recomputing it
with Poisson's equation. The saved discrete Helmholtz split gives transverse
external forcing and longitudinal cold response separately. Exact use of this
split requires the same horizontal Fourier wavenumber in continuity and Poisson;
a strictly vertical fluid model is an additional approximation.

This is one-way coupling: the warm density response does not modify the imposed
HF field. A fully coupled warm Maxwell/fluid solve is needed to predict conversion
efficiency. Cold resonant enhancement is not itself a propagating warm Langmuir
wave; thermal dispersion and Landau damping are absent here.

## Checks

- Vacuum and uniform-plasma propagation against analytic dispersion.
- No longitudinal field at normal incidence when B is zero.
- Native evolution agrees with the discrete harmonic solution.
- Scan, grid, peak-field and absorber-extent checks are saved as HDF5.
- Cold peak convergence requires millimetre cells near resonance; this is numerical
  resolution, not measurement resolution. The density observations have 300 m gates.

The 8 km domain contains the complete modeled layer and its reflection region.
The approximately 44 m scalar Airy scale and the sub-metre cold resonant peak are
separate lengths. Neither is a measured vertical source size.

Delivered-case checks: halving the finest cell from 1.953125 mm to 0.9765625 mm
changes the peak field by 0.19% and the integrated objective by 0.00048%. Lost
Poynting flux agrees with integrated collisional heating to 0.0104%. The Rust
recording spans two complete HF cycles (127,992 steps) and agrees with the
discrete harmonic field to 7e-12 relative maximum field error.

For a 0.5 mV/m incident complex-vector field norm, for example:

```python
from replay_arecibo_field import HFField
hf = HFField("output/hf_fdtd/arecibo_hf_field.h5",
             incident_field_norm_V_m=0.5e-3)
a_em = hf.acceleration(z_m=117942.0, time_s=1e-7, x_m=0.0)
```

The returned vector is ordered x, y, z in m/s². Add it to the electron momentum
equation alongside the warm model's pressure, magnetic, collisional and
self-consistent electrostatic terms.
