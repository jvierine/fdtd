"""Classical nonmagnetic sphere in vacuum, exp(-i omega t); SI units.

Bohren & Huffman (1983), Ch. 4-5. Mie and Rayleigh normalizations:
https://miepython.readthedocs.io/en/latest/07_algorithm.html
https://miepython.readthedocs.io/en/latest/_modules/miepython/rayleigh.html
No scattering-library dependency; SciPy supplies spherical Bessel functions.
"""
import numpy as np
from scipy.constants import c
from scipy.special import spherical_jn, spherical_yn


def _inputs(radius, frequency_hz, permittivity):
    if (not np.all(np.isfinite([radius, frequency_hz, permittivity]))
            or radius <= 0 or frequency_hz <= 0):
        raise ValueError("Finite permittivity, positive radius and frequency required")
    return 2*np.pi*frequency_hz/c, complex(permittivity)


def rayleigh_cross_sections(radius, frequency_hz, permittivity):
    """Electric-dipole limit: total scattering and monostatic RCS, m^2.

    Requires ka << 1 and |sqrt(epsilon)| ka << 1, away from epsilon=-2.
    Backscatter RCS = 4*pi*d(sigma)/d(Omega) at 180 degrees.
    """
    k, eps = _inputs(radius, frequency_hz, permittivity)
    strength = k**4*radius**6*abs((eps-1)/(eps+2))**2
    return dict(scattering_m2=8*np.pi/3*strength, backscatter_m2=4*np.pi*strength)


def mie_cross_sections(radius, frequency_hz, permittivity, angle_rad=None):
    """Exact Mie total scattering, monostatic RCS, and optional angular RCS.

    parallel_rcs_m2 and perpendicular_rcs_m2 describe linear incident E
    parallel/perpendicular to the scattering plane. Angular RCS is
    4*pi*d(sigma)/d(Omega), not the integrated scattering cross-section.
    """
    k, eps = _inputs(radius, frequency_hz, permittivity)
    x, m = k*radius, np.sqrt(eps)
    n = np.arange(1, int(np.ceil(x+4*x**(1/3)+12))+1)
    psi = x*spherical_jn(n, x)
    dpsi = spherical_jn(n, x)+x*spherical_jn(n, x, derivative=True)
    xi = psi+1j*x*spherical_yn(n, x)
    dxi = dpsi+1j*(spherical_yn(n, x)+x*spherical_yn(n, x, derivative=True))
    if eps == 0:
        # Analytic m -> 0 limit, avoiding 0/0 in the regular coefficients.
        a = psi/xi
        b = (x*dpsi-(n+1)*psi)/(x*dxi-(n+1)*xi)
    else:
        pm = m*x*spherical_jn(n, m*x)
        dpm = spherical_jn(n, m*x)+m*x*spherical_jn(n, m*x, derivative=True)
        a = (m*pm*dpsi-psi*dpm)/(m*pm*dxi-xi*dpm)
        b = (pm*dpsi-m*psi*dpm)/(pm*dxi-m*xi*dpm)
    result = dict(
        scattering_m2=float(2*np.pi/k**2*np.sum((2*n+1)*(abs(a)**2+abs(b)**2))),
        backscatter_m2=float(np.pi/k**2*abs(np.sum((2*n+1)*(-1.)**n*(a-b)))**2))
    if angle_rad is not None:
        mu = np.cos(np.asarray(angle_rad))
        previous, current = np.zeros_like(mu), np.ones_like(mu)
        s1, s2 = np.zeros_like(mu, dtype=complex), np.zeros_like(mu, dtype=complex)
        for order, an, bn in zip(n, a, b):
            tau = order*mu*current-(order+1)*previous
            factor = (2*order+1)/(order*(order+1))
            s1 += factor*(an*current+bn*tau)
            s2 += factor*(an*tau+bn*current)
            previous, current = current, ((2*order+1)*mu*current-(order+1)*previous)/order
        result["parallel_rcs_m2"] = 4*np.pi/k**2*abs(s2)**2
        result["perpendicular_rcs_m2"] = 4*np.pi/k**2*abs(s1)**2
    return result
