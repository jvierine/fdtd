"""Physical and recorded-field checks for the Arecibo stratified extension."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'examples'))
import pytest
if not (Path(__file__).resolve().parents[2]/"web/fluid/profiles.json").exists():
    pytest.skip("Archived Arecibo workspace fixture required; portable case is in arecibo-hf-conversion", allow_module_level=True)
from arecibo_layer import *
from replay_arecibo_field import HFField
import pytest

@pytest.mark.parametrize('vacuum',[True,False])
def test_analytic_dispersion_and_absorber(vacuum):
    sim=setup(2.,20.,reference=True,vacuum=vacuum,magnetic=False);z=sim['z'];field=sim['initial'][1,:,1]
    use=(z>114950)&(z<115250);dt=sim['dt'];wd=2*np.sin(sim['omega']*dt/2)/dt;ch=np.cos(sim['omega']*dt/2)
    i=np.argmin(abs(z-115100));wp=sim['wp'][1,i];nu=sim['nu'][1,i];kx=sim['kx']
    kz=2*np.arcsin(np.sqrt(wd*wd+1j*wd*wp*wp/(ch*nu-1j*wd)-kx*kx)) / 2
    basis=np.array([np.exp(1j*kz*(z[use]-115100)),np.exp(-1j*kz*(z[use]-115100))]).T
    fit=np.linalg.lstsq(basis,field[use],rcond=None)[0]
    assert np.linalg.norm(basis@fit-field[use])/np.linalg.norm(field[use])<1e-8
    assert abs(fit[1]/fit[0])<1e-4

def test_no_unmagnetized_normal_longitudinal_drive():
    s=setup(2.,0.,magnetic=False)
    assert np.max(abs(s['initial'][2]))<1e-12*np.max(abs(s['initial'][:2]))

def test_native_recording_phase_and_helmholtz_split():
    path=OUT/'arecibo_hf_field.h5'
    if not path.exists(): pytest.skip('Run the Arecibo field recording first')
    with h5py.File(path) as h:
        E=h['E_phasor_V_m'][:];t=h['time_s'][:];frames=h['E_time_V_m'][:]
        predicted=np.real(E[None,:,:]*np.exp(-1j*W*t[:,None,None]))
        assert np.max(abs(frames-predicted))/np.max(abs(E))<1e-8
        assert np.max(abs(E-h['E_transverse_phasor_V_m'][:]-h['E_longitudinal_phasor_V_m'][:]))<1e-9
        assert h.attrs['transverse_divergence_relative']<1e-8
        assert h.attrs['power_balance_relative_error']<2e-4
        # Unchanged saved collisional cold equations were actually advanced.
        report=json.loads(h.attrs['native_verification'])
        assert report['finite'] and report['steps']>100000 and report['relative_max_e_error']<1e-8
        z=h['z_E_components_m'][:];indices=np.array([1,len(z[0])//2,len(z[0])-2]);time=t[5]
        sampler=HFField(path,kind='total')
        for q in range(3):
            got=sampler.electric(z[q,indices],time)[q]
            assert np.allclose(got,frames[5,q,indices],rtol=1e-7,atol=1e-7)
    full=HFField(path);quarter=HFField(path,.25)
    assert np.allclose(quarter.acceleration(117942.,0),.5*full.acceleration(117942.,0))

def test_graded_peak_and_integral_convergence():
    if not (OUT/'final_convergence.h5').exists(): pytest.skip('Archived convergence runs required')
    with h5py.File(OUT/'final_convergence.h5') as h:
        a=h['delivered'].attrs;b=h['twice_finer'].attrs
        assert abs(a['score']/b['score']-1)<1e-4
        assert abs(a['peak_Ez_V_m']/b['peak_Ez_V_m']-1)<.003
