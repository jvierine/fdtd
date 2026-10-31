"""Load the recorded HF field for an electron momentum equation.

E_T is the one-way electromagnetic forcing for a Fourier-x warm Poisson model.
Use -e*E_T/m_e in dv/dt; self-consistent E_L must be calculated by that model.
A vertical-only Poisson solver is an approximation, not this exact split.
Total cold E is also available for diagnostics or prescribed-field test particles.
"""
import numpy as np
import h5py
from scipy.constants import e,m_e

class HFField:
    def __init__(self,path,incident_vertical_flux_W_m2=1.,kind='transverse',incident_field_norm_V_m=None):
        if kind not in ['transverse','total','longitudinal']:raise ValueError(kind)
        if incident_vertical_flux_W_m2<0:raise ValueError('Flux must be nonnegative')
        name='E_phasor_V_m' if kind=='total' else f'E_{kind}_phasor_V_m'
        with h5py.File(path) as h:
            self.z=h['z_E_components_m'][:];self.phasor=h[name][:]
            self.omega=2*np.pi*h.attrs['frequency_hz'];self.kx=float(h['kx_rad_m'][()])
            self.scale=np.sqrt(incident_vertical_flux_W_m2/h.attrs['incident_vertical_flux_W_m2'])
            if incident_field_norm_V_m is not None:
                if incident_field_norm_V_m<0:raise ValueError('Field norm must be nonnegative')
                j=np.argmin(abs(h['z_nodes_m'][:]-115100.))
                self.scale=incident_field_norm_V_m/np.linalg.norm(h['incident_reference_E_V_m'][:,j])
    def electric(self,z_m,time_s,x_m=0.):
        z,t,x=np.broadcast_arrays(z_m,time_s,x_m)
        if z.min()<self.z[:,0].max() or z.max()>self.z[:,-1].min():raise ValueError('Requested altitude outside recorded field')
        p=np.array([np.interp(z.ravel(),zs,a.real)+1j*np.interp(z.ravel(),zs,a.imag) for zs,a in zip(self.z,self.phasor)])
        return (self.scale*p*np.exp(1j*(self.kx*x.ravel()-self.omega*t.ravel()))).real.reshape((3,)+z.shape)
    def acceleration(self,z_m,time_s,x_m=0.):
        return -e/m_e*self.electric(z_m,time_s,x_m)
