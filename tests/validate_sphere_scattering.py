"""Rust FDTD sphere benchmarks against classical Mie and Rayleigh solutions.

PYTHONPATH=examples conda run -n base python tests/validate_sphere_scattering.py
No magnetic field. Linear Ez incidence, scattering in the xz plane.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e

from maarsy_rcs import ROOT, save_result, solve_rcs
from sphere_scattering import mie_cross_sections, rayleigh_cross_sections


def plot_tests(output):
    """Render stored results, preserving the numerical run's source record."""
    plt.rcParams.update({"font.size": 10})
    fig, axes = plt.subplots(3, 2, figsize=(7, 8.5))
    with h5py.File(output, "r+") as data:
        for i, key in enumerate(["rayleigh", "mie_underdense", "mie_overdense"]):
            group = data[key]
            angles = group["angle_rad"][:]
            cells = group["cells_per_wavelength"][:]
            for level, curve in zip(cells, group["fdtd_rcs_m2"][:]):
                axes[i,0].plot(np.rad2deg(angles), curve, "o", ms=3,
                               label=rf"FDTD $\lambda/{level}$")
            if key == "rayleigh":
                axes[i,0].plot(np.rad2deg(angles), group["rayleigh/backscatter_m2"][()]*np.cos(angles)**2,
                               "--", label="Rayleigh")
            axes[i,0].plot(np.rad2deg(angles), group["mie/parallel_rcs_m2"][:], "k-", label="Exact Mie")
            name = key.replace("_", " ").capitalize()
            axes[i,0].set(title=f"{name}: ka={group.attrs['size_parameter']:.3g}",
                          xlabel="Scattering angle (degrees)", ylabel="RCS (m²)", xlim=(0, 180))
            axes[i,0].legend(fontsize=10)
            spacing = group["dx_over_wavelength"][:]
            axes[i,1].plot(spacing, 100*group["backscatter_relative_error"][:], "o-")
            axes[i,1].set_xscale("log", base=2)
            axes[i,1].set_xticks(spacing, [f"1/{level}" for level in cells])
            axes[i,1].axhline(0, color="gray", ls=":")
            axes[i,1].set(xlabel=r"Grid spacing $\Delta x/\lambda$", ylabel="Backscatter error vs Mie (%)")
        source = "source/figure_script"
        if source in data:
            del data[source]
        data[source] = Path(__file__).read_text()
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.suptitle("Homogeneous sphere benchmarks · B=0 · linear polarization")
    fig.text(.5, .02, "Scripts: validate_sphere_scattering.py;\nsphere_scattering.py; src/main.rs",
             ha="center", fontsize=10)
    fig.savefig(output.with_suffix(".png"), dpi=180)
    fig.savefig(output.with_suffix(".pdf"))
    plt.close(fig)
    output.with_suffix(".tex").write_text(
        "\\newif\\ifshowspheretestscripts\n\\showspheretestscriptstrue\n"
        "\\begin{figure}\n\\centering\n"
        f"\\includegraphics[width=\\textwidth]{{{output.with_suffix('.png').as_posix()}}}\n"
        "\\caption{Classical sphere tests: exact Mie and Rayleigh angular radar cross-sections "
        "compared with native Rust FDTD, and mesh errors in monostatic RCS. "
        "\\ifshowspheretestscripts Scripts: \\texttt{validate\\_sphere\\_scattering.py}; "
        "\\texttt{sphere\\_scattering.py}; \\texttt{src/main.rs}.\\fi}\n\\end{figure}\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranks", type=int, default=4)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--jobs", type=int, default=1, help="Independent runs at once; workers = jobs*ranks*threads")
    parser.add_argument("--cells-per-wavelength", type=int, nargs="+",
                        help="Override mesh levels for all cases; spacing is lambda/N")
    parser.add_argument("--periods", type=float, default=24)
    parser.add_argument("--plot-only", action="store_true", help="Render existing HDF5 without running FDTD")
    parser.add_argument("--output", type=Path, default=ROOT/"outputs/sphere_tests.h5")
    args = parser.parse_args()
    if args.plot_only:
        plot_tests(args.output)
        return
    if args.jobs < 1 or (args.cells_per_wavelength is not None and min(args.cells_per_wavelength) < 1):
        parser.error("Jobs and cells per wavelength must be positive")
    # Mie cases have ka=1; Rayleigh has ka=0.140 at the MAARSY frequency.
    cases = [("Rayleigh", .125, 53.5e6, .5, [200, 400, 800]),
             ("Mie underdense", .5, c/np.pi, .5, [50, 100, 200]),
             ("Mie overdense", .5, c/np.pi, 2., [50, 100, 200, 400])]
    # Python only orchestrates independent native runs. Each has its own MPI
    # communicator and temporary directory; HDF5 writes stay on this thread.
    pool = ThreadPoolExecutor(max_workers=args.jobs)
    runs = {}
    for name, radius, frequency, multiple, levels in cases:
        for cells in sorted(set(args.cells_per_wavelength or levels)):
            omega = 2*np.pi*frequency
            parameters = dict(model="sphere", radius=radius, frequency_hz=frequency,
                sphere_density=multiple*epsilon_0*m_e*omega**2/e**2,
                magnetic_field=[0., 0., 0.], polarization="linear",
                collision_rate=.03*omega, resolution=cells/(c/frequency), periods=args.periods,
                mpi_ranks=args.ranks, threads=args.threads, pml_wavelengths=.05, gap_wavelengths=.05)
            runs[name, cells] = (parameters, pool.submit(solve_rcs, **parameters))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    passed = True
    with h5py.File(args.output, "w") as data:
        data.attrs["script"] = Path(__file__).name
        data.attrs["rcs_definition"] = "4*pi*d(scattering cross section)/d(solid angle)"
        for source in ["tests/validate_sphere_scattering.py", "examples/sphere_scattering.py"]:
            data["source/"+source] = (ROOT/source).read_text()
        data.attrs["mesh_definition"] = "dx = vacuum wavelength / cells_per_wavelength"
        for name, radius, frequency, multiple, levels in cases:
            cells_per_wavelength = sorted(set(args.cells_per_wavelength or levels))
            wavelength = c/frequency
            resolutions = np.asarray(cells_per_wavelength)/wavelength
            omega = 2*np.pi*frequency
            collision = .03*omega
            density = multiple*epsilon_0*m_e*omega**2/e**2
            eps = 1-multiple/(1+.03j)
            angles = np.linspace(0, np.pi, 37)
            exact = mie_cross_sections(radius, frequency, eps, angles)
            rayleigh = rayleigh_cross_sections(radius, frequency, eps)
            group = data.create_group(name.lower().replace(" ", "_"))
            for key, value in dict(radius_m=radius, frequency_hz=frequency,
                    size_parameter=omega*radius/c, permittivity=eps,
                    collision_rate_s1=collision, sphere_density_m3=density,
                    vacuum_wavelength_m=wavelength).items():
                group.attrs[key] = value
            for prefix, reference in [("mie", exact), ("rayleigh", rayleigh)]:
                for key, value in reference.items():
                    group[prefix+"/"+key] = value
            group["angle_rad"] = angles
            curves, errors = [], []
            for cells in cells_per_wavelength:
                parameters, future = runs[name, cells]
                result = future.result()
                np.testing.assert_allclose(result["dx_m"]/wavelength, 1/cells, rtol=1e-14)
                output = args.output.with_name(f"{args.output.stem}_{group.name[1:]}_lambda{cells}.h5")
                save_result(output, result, dict(parameters, cells_per_wavelength=cells,
                                               dx_over_wavelength=1/cells, vacuum_wavelength_m=wavelength))
                curves.append(result["rcs_m2"])
                error = result["rcs_m2"][-1]/exact["backscatter_m2"]-1
                errors.append(error)
                print(f"{name}, dx=lambda/{cells} ({result['dx_m']:g} m): FDTD {result['rcs_m2'][-1]:.8g}, "
                      f"Mie {exact['backscatter_m2']:.8g} m^2, error {100*error:+.2f}%", flush=True)
            curves = np.asarray(curves)
            # Absolute angular error normalized by peak: well defined at dipole nulls.
            angular_error = np.max(abs(curves[-1]-exact["parallel_rcs_m2"]))/max(exact["parallel_rcs_m2"])
            rayleigh_error = exact["backscatter_m2"]/rayleigh["backscatter_m2"]-1
            case_passed = abs(errors[-1]) < .08 and angular_error < .08
            if name == "Rayleigh":
                case_passed &= abs(rayleigh_error) < .03
                case_passed &= abs(curves[-1,-1]/rayleigh["backscatter_m2"]-1) < .10
            group["resolution_per_m"] = resolutions
            group["cells_per_wavelength"] = cells_per_wavelength
            group["dx_over_wavelength"] = 1/np.asarray(cells_per_wavelength)
            group["dx_m"] = wavelength/np.asarray(cells_per_wavelength)
            group["fdtd_rcs_m2"] = curves
            group["backscatter_relative_error"] = errors
            group.attrs["peak_normalized_angular_error"] = angular_error
            group.attrs["mie_rayleigh_relative_difference"] = rayleigh_error
            group.attrs["passed"] = bool(case_passed)
            passed &= bool(case_passed)
            print(f"  Angular error {100*angular_error:.2f}%; Mie/Rayleigh difference "
                  f"{100*rayleigh_error:+.2f}%; {'PASS' if case_passed else 'FAIL'}", flush=True)
        data.attrs["passed"] = passed
        data.attrs["periods"] = args.periods
    pool.shutdown(wait=True)
    plot_tests(args.output)
    if not passed:
        raise SystemExit("Sphere benchmark failed; inspect mesh/runtime errors in HDF5")
    print("All classical sphere benchmarks passed.")


if __name__ == "__main__":
    main()
