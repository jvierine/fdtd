"""Independent weak-scattering RCS check of the archived-cube Maxwell run."""
import argparse
from pathlib import Path
import h5py
import numpy as np
from scipy.constants import c, e, epsilon_0, m_e
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--density-archive', type=Path, required=True)
    p.add_argument('--fdtd-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args=p.parse_args()
    cube=loadmat(args.density_archive)['np']
    axis=(np.arange(512)-255.5)*.005
    interpolate=RegularGridInterpolator((axis,axis,axis),cube,bounds_error=False,fill_value=0.)
    with h5py.File(args.fdtd_result) as h:
        n=int(h['n'][()]); dx=float(h['dx_m'][()]); f=float(h['frequency_hz'][()])
        nstar=float(h.attrs['n_star_m3']); fdtd=float(h.attrs['computed_rcs_m2'])
    grid=(np.arange(n)-n/2)*dx
    coordinates=[grid,grid,grid+dx/2]
    coordinates=[v[(v>=axis[0])&(v<=axis[-1])] for v in coordinates]
    x,y,z=np.meshgrid(*coordinates,indexing='ij')
    ne=nstar/np.pi*interpolate(np.stack([x,y,z],axis=-1))
    k=2*np.pi*f/c
    integral=np.sum(ne*np.exp(2j*k*x))*dx**3
    re=e**2/(4*np.pi*epsilon_0*m_e*c**2)
    # Radar cross section is 4*pi times the differential scattering cross section.
    born=4*np.pi*re**2*abs(integral)**2
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(args.output,'w') as h:
        h.attrs['source_program']=Path(__file__).read_text()
        h.attrs['approximation']='First Born, zero magnetic field, transverse linear polarization'
        h['born_rcs_m2']=born;h['fdtd_rcs_m2']=fdtd
        h['difference_db']=10*np.log10(fdtd/born)
        h['density_fourier_integral']=integral
    print('Born',10*np.log10(born),'FDTD',10*np.log10(fdtd),
          'difference dB',10*np.log10(fdtd/born),flush=True)


if __name__=='__main__':
    main()
