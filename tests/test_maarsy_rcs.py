"""Fast analytic checks; full native-engine checks live in validate_sphere_scattering.py."""
import numpy as np
import pytest
from scipy.constants import c

from maarsy_rcs import mie_backscatter, solve_rcs


def test_mie_matches_rayleigh_limit_and_vacuum():
    frequency, radius, eps = 53.5e6, .001, 2.25
    k = 2*np.pi*frequency/c
    rayleigh = 4*np.pi*k**4*radius**6*abs((eps-1)/(eps+2))**2
    assert mie_backscatter(radius, frequency, eps) == pytest.approx(rayleigh, rel=1e-5)
    assert mie_backscatter(radius, frequency, 1.) == pytest.approx(0, abs=1e-30)


def test_complex_mie_reference():
    # Independently checked with miepython 3.3.0 efficiencies_mx * pi*a^2.
    assert mie_backscatter(.5, 53.5e6, -4+.014j) == pytest.approx(5.878043601976701, rel=1e-9)


@pytest.mark.parametrize("parameters", [dict(resolution=0), dict(collision_rate=-1),
                         dict(magnetic_field=[0, np.nan, 0]), dict(model="unknown")])
def test_invalid_inputs_rejected_before_launch(parameters):
    with pytest.raises(ValueError):
        solve_rcs(**parameters)
