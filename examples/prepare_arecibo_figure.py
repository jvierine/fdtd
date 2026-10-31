"""Prepare field-only figure arrays from the FDTD HDF5 (no new physical solve)."""
from pathlib import Path
import numpy as np,h5py
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'output/hf_fdtd'
with h5py.File(OUT/'arecibo_hf_field.h5') as h:
 z=h['z_nodes_m'][:];ze=h['z_E_components_m'][:];E=h['E_phasor_V_m'][:];ET=h['E_transverse_phasor_V_m'][:]
 ref=h['incident_reference_E_V_m'][:];i=np.argmin(abs(z-115100));norm=np.linalg.norm(ref[:,i]);zr=117941.5946713
 with h5py.File(OUT/'field_plot.h5','w') as f:
  f.attrs['incident_field_norm_V_m']=norm
  f.attrs['angle_deg']=h.attrs['angle_deg'];f.attrs['normalization']='complex vector norm of the incident reference field at 115100 m'
  for name,zgrid in [('wide',np.linspace(117400,118600,2401)),('fine',np.linspace(zr-3,zr+3,2401))]:
   f[f'{name}/z_km']=zgrid/1000;f[f'{name}/z_relative_m']=zgrid-zr
   f[f'{name}/density']=np.interp(zgrid,z,h['electron_density_m3'][0])/1e11
   for q,comp in enumerate(['x','y','z']):
    for label,array in [('total',E),('transverse',ET)]:
     v=np.interp(zgrid,ze[q],array[q].real)+1j*np.interp(zgrid,ze[q],array[q].imag)
     f[f'{name}/{label}_{comp}']=abs(v)/norm
  # Native recorded total Ex map, sampled by component position; x=0.
  times=h['time_s'][:];samples=h['E_time_V_m'][:,0,:]
  zz=np.linspace(117500,118500,501)
  f['time_ns']=(times-times[0])*1e9;f['map_height_km']=zz/1000
  f['Ex_time_over_incident']=np.array([np.interp(zz,ze[0],row)/norm for row in samples]).T
  f.attrs['native_error']=__import__('json').loads(h.attrs['native_verification'])['relative_max_e_error']
print('Saved field plot arrays')
