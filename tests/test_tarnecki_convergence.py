import numpy as np

from converge_tarnecki import geometry, window


def test_boxes_and_padding_are_in_wavelength_units():
    assert geometry(256,50.) == (25,1.52)
    assert geometry(384,50.) == (25,2.8)
    assert geometry(512,50.) == (25,4.08)
    pml, support=geometry(768,75.)
    assert pml/75. >= .5
    assert support >= 4.08


def test_window_is_smoothly_small_before_hard_vacuum():
    support=1.52;dx=.02
    x=np.linspace(0.,support,1000)
    w=window(x,x,support,dx,4)
    assert w[0] == 1.
    assert w[-1] == 0.
    assert np.all(np.diff(w)<=0.)
    assert window(support-2*dx,support-2*dx,support,dx,4)<1e-6


def test_larger_plasma_retains_inner_density():
    x=np.array([.001,.01,.1])
    small=window(x,x,1.52,.02,4)
    large=window(x,x,4.08,.02,4)
    assert np.all(large>=small)
    np.testing.assert_allclose(small[:2],1.,rtol=1e-7)
    np.testing.assert_allclose(large[:2],1.,rtol=1e-7)


def test_fixed_plasma_is_independent_of_the_vacuum_box():
    _,small_limit=geometry(256,50.)
    _,large_limit=geometry(384,50.)
    assert small_limit < large_limit
    small_x=(np.arange(256)-128)*.02
    large_x=(np.arange(384)-192)*.02
    np.testing.assert_array_equal(small_x,large_x[64:320])
    np.testing.assert_array_equal(window(abs(small_x),abs(small_x),small_limit,.02,4),
        window(abs(large_x),abs(large_x),small_limit,.02,4)[64:320])
