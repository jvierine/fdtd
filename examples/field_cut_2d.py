"""Reconstruct an x-z cut of the recorded single-kx FDTD solution.

Instantaneous real-vector magnitude, not complex phasor norm. No second solve.
Source provenance stays in scientific metadata; figures contain physical labels.
"""
from pathlib import Path
import hashlib
import numpy as np
import h5py
ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT/'output/hf_fdtd/arecibo_hf_field.h5'
DEST=ROOT/'output/hf_fdtd/field_cut_2d.h5'
ZR=117941.5946713
with h5py.File(SOURCE) as src,h5py.File(DEST,'w') as dst:
    E=src['E_phasor_V_m'][:];zs=src['z_E_components_m'][:]
    kx=float(src['kx_rad_m'][()]);frequency=float(src.attrs['frequency_hz'])
    j=np.argmin(abs(src['z_nodes_m'][:]-115100.));normalization=np.linalg.norm(src['incident_reference_E_V_m'][:,j])
    dst.attrs['source_sha256']=hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    dst.attrs['generator']=str(Path(__file__).relative_to(ROOT))
    dst.attrs['quantity']='sqrt(sum(Re{E_q(z) exp(i*kx*x-i*omega*t)}^2)) / incident complex-vector norm'
    dst.attrs['time_s']=0.;dst.attrs['kx_rad_m']=kx;dst.attrs['frequency_hz']=frequency
    dst.attrs['incident_reference_norm_V_m']=normalization
    dst.attrs['geometry']='Single conserved horizontal Fourier wavenumber; plane-wave reconstruction in a horizontally uniform measured-profile layer'
    for name,xe,ze in [
        ('overview',np.linspace(-600,600,602),np.unique(np.r_[np.arange(117400,118502,2),np.linspace(ZR-3,ZR+3,302)])),
        ('resonance',np.linspace(-300,300,602),np.linspace(ZR-3,ZR+3,602))]:
        x=(xe[:-1]+xe[1:])/2;z=(ze[:-1]+ze[1:])/2
        p=np.array([np.interp(z,zq,v.real)+1j*np.interp(z,zq,v.imag) for zq,v in zip(zs,E)])/normalization
        field=np.real(p[:,:,None]*np.exp(1j*kx*x[None,None,:]))
        magnitude=np.sqrt(np.sum(field**2,axis=0));rms=np.sqrt(np.sum(abs(p)**2,axis=0)/2)
        assert np.isfinite(magnitude).all() and magnitude.min()>=0
        # Instantaneous vector norm cannot exceed its complex-vector norm.
        assert np.all(magnitude <= np.linalg.norm(p,axis=0)[:,None]*(1+1e-12))
        # Independent phase average verifies that this is a field, not intensity.
        phases=np.arange(128)*2*np.pi/128
        average=np.mean(np.sum(np.real(p[:,:,None]*np.exp(1j*phases))**2,axis=0),axis=1)
        assert np.allclose(average,rms**2,rtol=1e-12,atol=1e-14)
        g=dst.create_group(name);g['horizontal_edges_m']=xe;g['height_edges_km']=ze/1000
        g['height_relative_edges_m']=ze-ZR
        g.create_dataset('instantaneous_magnitude_over_incident',data=magnitude,compression='gzip')
        g['rms_magnitude_over_incident']=rms
        for q,component in enumerate(['x','y','z']):
            g.create_dataset(f'instantaneous_E{component}_over_incident',data=field[q],compression='gzip')
        print(name,magnitude.shape,'peak',magnitude.max())
