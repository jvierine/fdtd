"""Analytic reference checks; native FDTD tests: validate_sphere_scattering.py."""
import numpy as np
import pytest
from scipy.constants import c

from sphere_scattering import mie_cross_sections, rayleigh_cross_sections


@pytest.mark.parametrize("eps,scattering,backscatter", [
    # Independent miepython 3.3.0 efficiencies_mx at ka=1, times pi*a^2.
    # Its absorbing-index convention uses conjugate(sqrt(eps)). No dependency.
    (2.25, .16893725688328814, .1465445454249897),
    (.5+.015j, .046330905469584134, .043827285265813656),
    (-1+.06j, 1.0893037490988353, .8796344065348501),
])
def test_mie_independent_reference_and_angular_normalization(eps, scattering, backscatter):
    mu, weights = np.polynomial.legendre.leggauss(32)
    result = mie_cross_sections(.5, c/np.pi, eps, np.arccos(mu))
    assert result["scattering_m2"] == pytest.approx(scattering, rel=1e-10)
    assert result["backscatter_m2"] == pytest.approx(backscatter, rel=1e-10)
    integrated = np.sum(weights*(result["parallel_rcs_m2"]+result["perpendicular_rcs_m2"]))/4
    assert integrated == pytest.approx(scattering, rel=1e-12)
    backward = mie_cross_sections(.5, c/np.pi, eps, [np.pi])
    for key in ["parallel_rcs_m2", "perpendicular_rcs_m2"]:
        assert backward[key][0] == pytest.approx(backscatter, rel=1e-12)


@pytest.mark.parametrize("eps", [2.25, .5+.015j, -1+.06j])
def test_mie_rayleigh_limit(eps):
    mie = mie_cross_sections(.001, 53.5e6, eps)
    rayleigh = rayleigh_cross_sections(.001, 53.5e6, eps)
    for key in rayleigh:
        assert mie[key] == pytest.approx(rayleigh[key], rel=1e-5, abs=0)
    assert rayleigh["backscatter_m2"]/rayleigh["scattering_m2"] == pytest.approx(1.5)


def test_rayleigh_radius_and_frequency_scaling():
    reference = rayleigh_cross_sections(.1, 53.5e6, 2.25)["backscatter_m2"]
    assert rayleigh_cross_sections(.2, 53.5e6, 2.25)["backscatter_m2"] == pytest.approx(64*reference)
    assert rayleigh_cross_sections(.1, 107e6, 2.25)["backscatter_m2"] == pytest.approx(16*reference)


def test_vacuum_and_zero_permittivity_limit():
    for function in [mie_cross_sections, rayleigh_cross_sections]:
        assert function(.5, 53.5e6, 1.)["backscatter_m2"] == pytest.approx(0, abs=1e-30)
        assert function(.5, 53.5e6, 1.)["scattering_m2"] == pytest.approx(0, abs=1e-30)
    zero = mie_cross_sections(.5, c/np.pi, 0)
    small = mie_cross_sections(.5, c/np.pi, 1e-8)
    for key in zero:
        assert zero[key] == pytest.approx(small[key], rel=1e-7)


@pytest.mark.parametrize("parameters", [(0, 53.5e6, 2.25), (.5, -1, 2.25), (.5, 53.5e6, np.nan)])
def test_invalid_scales(parameters):
    for function in [mie_cross_sections, rayleigh_cross_sections]:
        with pytest.raises(ValueError):
            function(*parameters)
