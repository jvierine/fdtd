"""Convergence and analytic wave tests for the stratified FDTD extension."""
from arecibo_layer import *

def main():
    results=[]
    for az in [0.,10.,20.]:
        for angle in np.arange(11.,20.,1.):
            r=optimize(.5,float(angle),az);results.append(r);print('REFINE',r,flush=True)
    best=max(results,key=lambda x:x['score']);print('BEST',best,flush=True)
    with h5py.File(OUT/'angle_refinement.h5','w') as h:
        for key in best:h[key]=[r[key] for r in results]
    conv=[]
    for dz in [1.,.5,.25,.125]:
        r=optimize(dz,best['angle_deg'],best['azimuth_deg']);conv.append(r);print('GRID',r,flush=True)
    for buffer in [900.,1200.]:
        r=optimize(.25,best['angle_deg'],best['azimuth_deg'],buffer);r['buffer_m']=buffer;print('BUFFER',r,flush=True)
        with h5py.File(OUT/'boundary_checks.h5','a') as h:
            g=h.require_group(str(int(buffer)))
            for k,v in r.items():g.attrs[k]=v
    with h5py.File(OUT/'grid_convergence.h5','w') as h:
        for key in best:h[key]=[r[key] for r in conv]
    checks={}
    for vacuum in [True,False]:
        sim=setup(2.,20.,reference=True,vacuum=vacuum,magnetic=False)
        z=sim['z'];q=1;field=sim['initial'][q,:,1];use=(z>114950)&(z<115250)
        dt=sim['dt'];om=sim['omega'];wd=2*np.sin(om*dt/2)/dt;ch=np.cos(om*dt/2)
        j=np.argmin(abs(z-115100));wp=sim['wp'][q,j];nu=sim['nu'][q,j];kx=sim['kx']
        kz=2*np.arcsin(2/2*np.sqrt(wd**2+1j*wd*wp**2/(ch*nu-1j*wd)-kx**2))/2
        basis=np.array([np.exp(1j*kz*(z[use]-115100)),np.exp(-1j*kz*(z[use]-115100))]).T
        fit=np.linalg.lstsq(basis,field[use],rcond=None)[0]
        residual=np.linalg.norm(basis@fit-field[use])/np.linalg.norm(field[use]);back=abs(fit[1]/fit[0])
        key='vacuum' if vacuum else 'uniform_plasma'
        checks[key+'_analytic_residual']=residual;checks[key+'_boundary_reflection_amplitude']=back
        print('ANALYTIC',key,residual,back,flush=True)
        assert residual<1e-5 and back<.01
    s=setup(2,0,magnetic=False);normal_ratio=np.max(abs(s['initial'][2]))/np.max(abs(s['initial'][:2]))
    assert normal_ratio<1e-10;checks['unmagnetized_normal_Ez_ratio']=normal_ratio
    with h5py.File(OUT/'analytic_checks.h5','w') as h:
        for k,v in checks.items():h.attrs[k]=v
    # Final production run is a separate explicit CLI operation; validation
    # never overwrites the delivered field record.
if __name__=='__main__':main()
