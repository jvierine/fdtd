"""Script the native Rust 3-D cold-plasma FDTD solver; save HDF5 and figures.

Marshall & Close (2015): doi:10.1002/2015JA021238.
Density application: Sugar et al. (2021): doi:10.1029/2021JA029171.
All numerical time stepping, CPML and radiation integrals are in src/main.rs.
"""
import argparse
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import h5py
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from scipy.interpolate import RegularGridInterpolator

from dimant_oppenheim import electron_density
from sphere_scattering import mie_cross_sections

ROOT = Path(__file__).resolve().parents[1]


def mpi_tools():
    """Match the compiler wrapper and launcher, avoiding mixed MPI installs."""
    native_path = os.pathsep.join(p for p in os.environ["PATH"].split(os.pathsep)
                                 if not p.startswith(sys.prefix))
    compiler = os.environ.get("MPICC") or shutil.which("mpicc", path=native_path) or shutil.which("mpicc")
    if not compiler:
        raise RuntimeError("Install MPI (mpicc/mpiexec) before building the Rust solver")
    launcher = os.environ.get("MPIEXEC") or str(Path(compiler).parent/"mpiexec")
    return compiler, launcher


@lru_cache(maxsize=4)
def dimant_interpolator(mean_free_path, n_star, radius):
    """Reuse the axisymmetric table across viewing angles and meshes."""
    radial = np.geomspace(radius * 1e-5, radius, 160)
    mu = np.linspace(-1, 1, 81)
    table = np.array([[electron_density(R*u/mean_free_path,
                      R*np.sqrt(max(0, 1-u*u))/mean_free_path)
                      for u in mu] for R in radial]) * n_star
    if not np.all(np.isfinite(table)) or np.any(table < -1e-8*n_star):
        raise ValueError("Nonfinite or significantly negative DO density")
    # The paper approximation has tiny negative far-upstream tails (below
    # 1e-9*n_star). Clip these unphysical values only in the FDTD input.
    # log1p admits exact zero and behaves like log at meaningful densities.
    scale = 1e-10*n_star
    interpolate = RegularGridInterpolator((np.log(radial), mu), np.log1p(np.maximum(table, 0)/scale))
    return radial, scale, interpolate


def dimant_grid(mean_free_path, n_star, density_cap, radius, aspect_deg, resolution):
    """Capped, tapered DO density averaged over volumes centred on Yee edges.

    Component q lives at vertex coordinates plus dx/2 along axis q.
    Eight Gauss points avoid the point-source singularity. The outer 20%
    of the radius is tapered to zero: an explicitly finite head model.
    """
    radial, scale, interpolate = dimant_interpolator(mean_free_path, n_star, radius)
    m = int(np.ceil(radius*resolution))
    dx = 1/resolution
    axis = (np.arange(2*m+1)-m)*dx
    ne = np.zeros((3, 2*m+1, 2*m+1, 2*m+1))
    offset = dx/np.sqrt(12)
    angle = np.deg2rad(aspect_deg)
    for component in range(3):
        coordinates = [axis + (dx/2 if q == component else 0) for q in range(3)]
        X, Y, Z = np.meshgrid(*coordinates, indexing="ij")
        for ox in [-offset, offset]:
            for oy in [-offset, offset]:
                for oz in [-offset, offset]:
                    R = np.sqrt((X+ox)**2+(Y+oy)**2+(Z+oz)**2)
                    behind = (X+ox)*np.cos(angle)+(Z+oz)*np.sin(angle)
                    points = np.stack([np.log(np.clip(R, radial[0], radius)),
                                       np.clip(behind/R, -1, 1)], axis=-1)
                    value = np.minimum(scale*np.expm1(interpolate(points)), density_cap)
                    taper = .5*(1+np.cos(np.pi*np.clip((R/radius-.8)/.2, 0, 1)))
                    ne[component] += value*taper/8
    return axis, ne


def mie_backscatter(radius, frequency_hz, permittivity):
    """Exact homogeneous, nonmagnetic sphere monostatic RCS (m^2)."""
    return mie_cross_sections(radius, frequency_hz, permittivity)["backscatter_m2"]


def solve_rcs(model="dimant", radius=6., frequency_hz=53.5e6, resolution=16,
              polarization="lhcp", periods=12, pml_wavelengths=.5,
              mean_free_path=.1, n_star=1e12, density_cap=1e14,
              aspect_deg=0., sphere_density=1e13, collision_rate=1e6,
              magnetic_field=(-50e-6, 0, 0), threads=2, gap_wavelengths=.5,
              mpi_ranks=4, field_frames=0):
    """Prepare Yee-edge density, invoke Rust, return numerical products.

    Temporary little-endian f64 density + JSON are interchange only.
    Persistent scientific products are HDF5. No Python field evolution.
    """
    positive = [radius, frequency_hz, resolution, periods, pml_wavelengths,
                mean_free_path, density_cap, gap_wavelengths]
    nonnegative = [n_star, sphere_density, collision_rate]
    if not np.all(np.isfinite(positive + nonnegative + list(magnetic_field))) or min(positive) <= 0 or min(nonnegative) < 0 or threads < 1 or mpi_ranks < 1:
        raise ValueError("Finite positive scales and threads; nonnegative densities/collisions required")
    if model not in ["dimant", "sphere", "vacuum"] or polarization not in ["lhcp", "rhcp", "linear"]:
        raise ValueError("Unknown model or polarization")
    dx, wavelength = 1/resolution, c/frequency_hz
    pml_cells = max(4, int(np.ceil(pml_wavelengths*wavelength/dx)))
    m = int(np.ceil(radius/dx))
    half = m + int(np.ceil(gap_wavelengths*wavelength/dx)) + pml_cells + 2
    n = 2*half
    axis = (np.arange(2*m+1)-m)*dx
    if model == "dimant" and n_star > 0:
        axis, ne = dimant_grid(mean_free_path, n_star, density_cap, radius, aspect_deg, resolution)
    elif model == "sphere":
        # Sample the sharp sphere at the actual Yee electric-field locations.
        # Staircase geometry is resolved through the sphere convergence test.
        ne = np.zeros((3, 2*m+1, 2*m+1, 2*m+1))
        for component in range(3):
            coordinates = [axis + (dx/2 if q == component else 0) for q in range(3)]
            X, Y, Z = np.meshgrid(*coordinates, indexing="ij")
            ne[component] = sphere_density*(X*X+Y*Y+Z*Z < radius**2)
    else:
        ne = np.zeros((3, 2*m+1, 2*m+1, 2*m+1))
    density = np.zeros((3, n, n, n), dtype="<f8")
    density[:, half-m:half+m+1, half-m:half+m+1, half-m:half+m+1] = ne
    compiler, launcher = mpi_tools()
    build_env = dict(os.environ, MPICC=compiler)
    subprocess.run(["cargo", "build", "--release", "--manifest-path", str(ROOT/"Cargo.toml")], check=True, env=build_env)
    with tempfile.TemporaryDirectory(prefix="meteor-fdtd-") as directory:
        directory = Path(directory)
        density.tofile(directory/"density.bin")
        config = dict(n=n, dx_m=dx, pml_cells=pml_cells, frequency_hz=frequency_hz,
                      collision_rate_s1=collision_rate, magnetic_field_t=list(magnetic_field),
                      polarization=polarization, support_radius_m=radius, periods=periods,
                      density_file=str(directory/"density.bin"), output_file=str(directory/"result.json"),
                      threads=int(threads), field_frames=int(field_frames),
                      field_file=str(directory/"ez.bin"))
        (directory/"config.json").write_text(json.dumps(config))
        command = [str(ROOT/"target/release/meteor-fdtd"), str(directory/"config.json")]
        if mpi_ranks > 1:
            command = [launcher, "--bind-to", "none", "-n", str(mpi_ranks)] + command
        subprocess.run(command, check=True)
        result = json.loads((directory/"result.json").read_text())
        if field_frames:
            result["field_ez_scattered_v_m"] = np.fromfile(directory/"ez.bin", dtype="<f4").reshape(-1, n, n)
            result["field_x_m"] = (np.arange(n)-n/2)*dx
            result["field_z_m"] = (np.arange(n)+.5-n/2)*dx
    result["density_axis_m"], result["electron_density_m3"] = axis, ne
    result["density_offsets_m"] = np.eye(3)*dx/2
    return result


def save_result(output, result, parameters):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(output, "w") as data:
        for name, value in parameters.items():
            data.attrs[name] = str(value) if isinstance(value, Path) else value
        for name, value in result.items():
            if name == "field_ez_scattered_v_m":
                data.create_dataset(name, data=value, compression="gzip", shuffle=True, chunks=(1, value.shape[1], value.shape[2]))
                data[name].attrs["units"] = "V/m for unit incident peak vector amplitude"
            else:
                data[name] = value
        data.attrs["solver"] = "native Rust Yee/CPML, edge electron ADE, trapezoidal Lorentz solve"
        data.attrs["phasor"] = "exp(-i omega t); incident propagation +x"
        data.attrs["script"] = Path(__file__).name
        data.attrs["rustc_version"] = subprocess.check_output(["rustc", "--version"], text=True).strip()
        for name in ["examples/maarsy_rcs.py", "examples/dimant_oppenheim.py", "examples/sphere_scattering.py", "src/main.rs", "Cargo.lock"]:
            data.attrs[name+"_sha256"] = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            data.create_dataset("source/"+name.replace("/", "_"), data=(ROOT/name).read_text())
        data.attrs["density_taper"] = "Dimant: cosine taper over outer 20 percent of radius, capped before averaging"
        data.attrs["density_interpolation"] = "log1p(ne/(1e-10*n_star)); tiny negative paper-approximation tails clipped to zero"
        data.attrs["status"] = "finite-head demonstration; verify mesh, runtime, PML, cap and tail for the chosen meteor"


def plot_result(output, result, frequency_mhz, model, hide_provenance=False):
    output = Path(output)
    axis, ne = np.asarray(result["density_axis_m"]), np.asarray(result["electron_density_m3"])
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(1, 2, figsize=(7., 3.8))
    if ne.max() > 0:
        im = axes[0].pcolormesh(axis, axis+result["dx_m"]/2, ne[2, :, len(axis)//2, :].T,
             norm=LogNorm(ne.max()*1e-5, ne.max()), shading="auto")
        fig.colorbar(im, ax=axes[0], label=r"$n_e$ (m$^{-3}$)")
        axes[0].set(xlabel="x (m)", ylabel="z (m)", aspect="equal")
    else:
        axes[0].text(.5, .5, "vacuum", ha="center", transform=axes[0].transAxes)
        axes[0].axis("off")
    axes[1].plot(np.rad2deg(result["angle_rad"]), 10*np.log10(np.maximum(result["rcs_m2"], 1e-100)))
    axes[1].set(xlabel="Scattering angle (degrees)", ylabel="RCS (dBsm)", xlim=(0, 180))
    fig.suptitle(f"Rust FDTD: {frequency_mhz:g} MHz, {model}")
    fig.tight_layout(rect=(0, .08, 1, 1))
    if not hide_provenance:
        fig.text(.5, .02, "Scripts: examples/maarsy_rcs.py; src/main.rs", ha="center", fontsize=10)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".tex").write_text(
        "\\newif\\ifshowrcsscripts\n\\showrcsscriptstrue\n"
        "\\begin{figure}\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{output.with_suffix('.png').as_posix()}}}\n"
        "\\caption{Finite meteor head density and three-dimensional FDTD radar cross section "
        f"at {frequency_mhz:g} MHz. "
        "\\ifshowrcsscripts Scripts: \\texttt{examples/maarsy\\_rcs.py}; \\texttt{src/main.rs}.\\fi}\n\\end{figure}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=["dimant", "sphere", "vacuum"], default="dimant")
    parser.add_argument("--frequency-mhz", type=float, default=53.5)
    parser.add_argument("--resolution", type=float, default=32, help="Yee cells per metre")
    parser.add_argument("--periods", type=float, default=2, help="Radar periods after the Gaussian pulse")
    parser.add_argument("--pml-wavelengths", type=float, default=.05)
    parser.add_argument("--gap-wavelengths", type=float, default=.05, help="Vacuum gap between density and PML")
    parser.add_argument("--mean-free-path", type=float, default=.0750188, help="metres")
    parser.add_argument("--n-star", type=float, default=1e14, help="Physical DO normalization, m^-3")
    parser.add_argument("--density-cap", type=float, default=1e16, help="Finite core cap, m^-3")
    parser.add_argument("--radius", type=float, default=.6, help="Finite DO cutoff or sphere radius, metres")
    parser.add_argument("--aspect-deg", type=float, default=0, help="Behind-meteor axis angle to +x")
    parser.add_argument("--collision-rate", type=float, default=1e7, help="Total electron damping, s^-1")
    parser.add_argument("--b-microtesla", type=float, nargs=3, default=[-50, 0, 0])
    parser.add_argument("--sphere-density", type=float, default=1e13)
    parser.add_argument("--polarization", choices=["lhcp", "rhcp", "linear"], default="lhcp")
    parser.add_argument("--threads", type=int, default=2, help="Threads per MPI rank")
    parser.add_argument("--ranks", type=int, default=4, help="MPI subdomains")
    parser.add_argument("--field-frames", type=int, default=0, help="Record Ez at y=0 for animation")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/meteor.h5")
    parser.add_argument("--hide-provenance", action="store_true")
    args = parser.parse_args()
    result = solve_rcs(args.model, args.radius, args.frequency_mhz*1e6, args.resolution,
                       args.polarization, args.periods, args.pml_wavelengths,
                       args.mean_free_path, args.n_star, args.density_cap, args.aspect_deg,
                       args.sphere_density, args.collision_rate, np.array(args.b_microtesla)*1e-6,
                       args.threads, args.gap_wavelengths, args.ranks, args.field_frames)
    save_result(args.output, result, vars(args))
    plot_result(args.output, result, args.frequency_mhz, args.model, args.hide_provenance)
    total = result["rcs_m2"][-1]
    print(f"Monostatic: {total:.8g} m^2 ({10*np.log10(total) if total > 0 else -np.inf:.3f} dBsm)")
    print(f"RHCP/LHCP receive: {result['backscatter_rhcp_lhcp_m2']}; {result['elapsed_s']:.2f} s in Rust")
    print(args.output)


if __name__ == "__main__":
    main()
