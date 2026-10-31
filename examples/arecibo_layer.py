"""Stratified extension of jvierine/fdtd, CW discrete steady initialization.
Scientific outputs HDF5. Binary/JSON files are temporary native transport only.
The sparse initializer solves the exact Yee + trapezoidal-current recurrence;
Rust subsequently advances and records it. No claim of simulated turn-on.
"""
from pathlib import Path
import argparse, json, subprocess, hashlib, time
import numpy as np
import h5py
from scipy.constants import c, epsilon_0, e, m_e, k as kb
from scipy.interpolate import CubicHermiteSpline
from scipy.sparse import bmat, diags, eye, coo_matrix
from scipy.sparse.linalg import splu
ROOT=Path(__file__).resolve().parents[2]
REPO=Path(__file__).resolve().parents[1]
OUT=ROOT/'output/hf_fdtd'; OUT.mkdir(exist_ok=True,parents=True)
F=5.33e6; W=2*np.pi*F; EPOCH=1525453430
P=json.loads((ROOT/'web/fluid/profiles.json').read_text())[0]
PROFILE=CubicHermiteSpline(P['z'],P['ne'],P['dne'])

def rates(z,ne):
    tev=kb*300/e; ncm=np.maximum(ne*1e-6,1e-30)
    ei=2.91e-6*ncm*(23-np.log(np.sqrt(ncm)*tev**-1.5))/tev**1.5
    p=P['neutrals']; n2,o2,o=[np.exp(np.interp(z,p['z'],np.log(p[name]))) for name in ['N2_m3','O2_m3','O_m3']]
    t=300.; en=2.33e-17*n2*(1-1.21e-4*t)*t+1.82e-16*o2*(1+.036*np.sqrt(t))*np.sqrt(t)+8.9e-17*o*(1+5.7e-4*t)*np.sqrt(t)
    return ei+en

def setup(dz=1.,angle=0.,azimuth=0.,reference=False,buffer=600.,vacuum=False,magnetic=True,graded=False):
    lo=114000.-(buffer-600);hi=122000.+(buffer-600)
    if graded:
        points=[117941.595]
        for direction,bound in [(1,hi),(-1,lo)]:
            v=117941.595;side=[]
            while (v-bound)*direction<0:
                step=min(2.,dz+.025*max(abs(v-117941.595)-3.,0))
                v+=direction*step;side.append(v)
            points.extend(side)
        z=np.array(sorted(points));lo=z[0];hi=z[-1]+2.
    else:z=np.arange(lo,hi,dz)
    n=len(z);dzv=np.diff(np.r_[z,hi]);dual=(dzv+np.roll(dzv,1))/2;zf=z+dzv/2
    positions=np.array([z,z,zf]);clipped=np.clip(positions,115300,120800)
    ne=PROFILE(clipped)
    if reference: ne[:]=PROFILE(115300.)
    nu=rates(clipped,ne)
    if vacuum: ne[:]=0.;nu[:]=0.
    wp=np.sqrt(np.maximum(ne,0)*e*e/(epsilon_0*m_e))/c
    nu/=c
    sph=np.array([z,z,zf,zf,zf,z])
    depth=np.maximum((lo+buffer-sph)/buffer,(sph-(hi-buffer))/buffer).clip(0,1)
    sponge=(40/buffer)*depth**3
    # Integer number of steps per pump period; CFL < .45.
    spp=int(np.ceil(c/F/(.45*dz)));dt=c/F/spp
    om=W/c; wd=2*np.sin(om*dt/2)/dt;ch=np.cos(om*dt/2)
    kx=om*np.sin(np.deg2rad(angle));kx_d=kx;cx=1.
    tilt=np.deg2rad(P['field_angle_deg']);az=np.deg2rad(azimuth)
    gyro=e*P['magnetic_t']/m_e/c*np.array([np.sin(tilt)*np.cos(az),np.sin(tilt)*np.sin(az),-np.cos(tilt)])
    if not magnetic:gyro[:]=0
    i=np.arange(n);shift=coo_matrix((np.ones(n),(i,(i+1)%n)),shape=(n,n)).tocsr();I=eye(n,format='csr');Z=0*I
    df=diags(1/dzv)@(shift-I);db=diags(1/dual)@(I-shift.T);av=(I+shift)/2;ab=diags(1/dual)@av.T@diags(dzv)
    cf=bmat([[Z,-df,Z],[df,Z,-1j*kx_d*I],[Z,1j*kx_d*I,Z]],format='csr')
    cb=bmat([[Z,-db,Z],[db,Z,-1j*kx_d*I],[Z,1j*kx_d*I,Z]],format='csr')
    gx,gy,gz=gyro
    G=bmat([[Z,-gz*cx*I,gy*cx*ab],[gz*cx*I,Z,-gx*ab],[-gy*cx*av,gx*av,Z]],format='csr')
    we=diags(wp.ravel());Le=diags((-1j*wd+ch*sponge[:3]).ravel());Lh=diags((-1j*wd+ch*sponge[3:]).ravel())
    Lj=diags((-1j*wd+ch*nu).ravel())-ch*G
    mat=bmat([[Le,-cb,we],[cf,Lh,None],[-we,None,Lj]],format='csc')
    source=np.zeros((9*n,2),complex);j=np.argmin(abs(z-114850.));source[j,0]=1/dual[j];source[n+j,1]=1/dual[j]
    # Node ordering preserves the narrow band of the stratified operator.
    perm=np.arange(9*n).reshape(9,n).T.ravel();mat=mat[perm,:][:,perm].tocsc()
    lu=splu(mat,permc_spec='NATURAL');tmp=lu.solve(source[perm]);sol=np.empty_like(tmp);sol[perm]=tmp
    sol=sol.reshape(9,n,2)
    return dict(z=z,dz=dz,dz_cells=dzv,dt=dt,omega=om,kx=kx,gyro=gyro,wp=wp,nu=nu,sponge=sponge,initial=sol,source=source.reshape(9,n,2),spp=spp,ne=ne)

def flux_matrix(state,z,height=115100.):
    j=np.argmin(abs(z-height));ex,ey=state[0,j],state[1,j]
    hx=(state[3,j]+state[3,j-1])/2;hy=(state[4,j]+state[4,j-1])/2
    Q=np.outer(hy.conj(),ex)-np.outer(hx.conj(),ey)
    return (Q+Q.conj().T)*epsilon_0*c/4

def optimize(dz,angle,azimuth,buffer=600.,return_state=False,graded=False):
    sim=setup(dz,angle,azimuth,buffer=buffer,graded=graded);ref=setup(dz,angle,azimuth,reference=True,buffer=buffer,graded=graded)
    power=flux_matrix(ref['initial'],ref['z'])
    if np.linalg.eigvalsh(power).min()<=0:raise ValueError('Incident reference is not upward propagating')
    use=abs(sim['z']+sim['dz_cells']/2-117941.595)<30
    ez=sim['initial'][2,use];metric=ez.conj().T@(ez*sim['dz_cells'][use,None])
    L=np.linalg.cholesky(power); Li=np.linalg.inv(L)
    vals,vec=np.linalg.eigh(Li@metric@Li.conj().T);v=Li.conj().T@vec[:,-1]
    # Fix arbitrary polarization phase by making source Ex positive real.
    v*=np.exp(-1j*np.angle(v[np.argmax(abs(v))]))
    state=np.einsum('qnp,p->qn',sim['initial'],v)
    inc=np.einsum('qnp,p->qn',ref['initial'],v)
    peak=float(abs(state[2,use]).max());score=float(vals[-1])
    result=dict(angle_deg=angle,azimuth_deg=azimuth,dz_m=dz,score=score,peak_Ez_V_m=peak,
        source_ey_over_ex_real=float((v[1]/v[0]).real),source_ey_over_ex_imag=float((v[1]/v[0]).imag))
    if return_state:return result,sim,state,inc,v
    return result

def save_run(dz,angle,azimuth,buffer=600.,cycles=4,graded=False):
    result,sim,state,inc,v=optimize(dz,angle,azimuth,buffer,True,graded);n=len(sim['z'])
    raw=OUT/'field_frames.bin';cfg=OUT/'native_input.json';native={}
    for key in ['dz','dz_cells','dt','omega','kx','gyro','wp','nu','sponge']:
        val=sim[key];native[key]=val.tolist() if isinstance(val,np.ndarray) else val
    source=np.einsum('qnp,p->qn',sim['source'],v)[:3]
    as_pair=lambda a:np.stack([a.real,a.imag],axis=-1).tolist()
    native.update(n=n,initial=as_pair(state),source=as_pair(source),steps=cycles*sim['spp'],
        record_start=(cycles-2)*sim['spp'],record_stride=max(1,sim['spp']//24),output=str(raw))
    cfg.write_text(json.dumps(native,separators=(',',':')))
    run=subprocess.run([str(REPO/'target/release/stratified'),str(cfg)],check=True,capture_output=True,text=True)
    report=json.loads(run.stdout); print('Native:',report,flush=True)
    rows=np.fromfile(raw,dtype='<f8').reshape(report['frames'],1+6*n);record=rows[:,1:].reshape(-1,3,n,2)
    path=OUT/'arecibo_hf_field.h5'
    with h5py.File(path,'w') as h:
        for key,value in result.items():h.attrs[key]=value
        h.attrs['frequency_hz']=F;h.attrs['epoch_unix_s']=EPOCH;h.attrs['electron_temperature_K']=300
        h.attrs['incident_vertical_flux_W_m2']=1.;h.attrs['phasor_convention']='Re{E(z) exp(i kx x - i omega t)}'
        h.attrs['angle_definition']='vacuum-equivalent asin(kx c / omega), not local ray angle'
        h.attrs['azimuth_definition']='B horizontal projection relative to incidence x-z plane; Bz downward'
        h.attrs['model']='magnetized collisional cold-plasma Yee FDTD; CW discrete harmonic initialization; no thermal pressure or Landau damping'
        h.attrs['native_verification']=json.dumps(report);h.attrs['repository']='https://github.com/jvierine/fdtd'
        h.attrs['base_commit']=subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
        h.attrs['profile_sha256']=P['provenance']['sha256'];h.attrs['profile_note']='calibrated smooth 300-m-sampled measurement; density is not measured on the FDTD mesh'
        h.attrs['field_warning']='Total self-consistent cold E; do not add its longitudinal response again as external drive while independently solving the same Poisson response'
        h.attrs['source_definition']='tangential harmonic current sheet at 114850 m; normalized using homogeneous launch-medium reference'
        h['z_nodes_m']=sim['z'];h['z_E_components_m']=sim['z'][None,:]+np.array([0,0,.5])[:,None]*sim['dz_cells'][None,:]
        h['electron_density_m3']=sim['ne'];h['collision_rate_s1']=sim['nu']*c;h['gyro_rad_s']=sim['gyro']*c
        h['kx_rad_m']=sim['kx'];h['dt_s']=sim['dt']/c;h['dz_cells_m']=sim['dz_cells']
        h['E_phasor_V_m']=state[:3];h['H_phasor_A_m']=state[3:6]*epsilon_0*c
        h['current_scaled_phasor_V_m']=state[6:9];h['current_phasor_A_m2']=epsilon_0*c*sim['wp']*state[6:9]
        h['magnetic_field_T']=sim['gyro']*m_e*c/e
        widths=sim['dz_cells'];dual=(widths+np.roll(widths,1))/2
        hc=(state[3:5]+np.roll(state[3:5],1,axis=1))/2
        flux=epsilon_0*c/2*np.real(state[0]*hc[1].conj()-state[1]*hc[0].conj())
        ia=np.argmin(abs(sim['z']-115400));ib=np.argmin(abs(sim['z']-120500))
        mask=(sim['z']>=sim['z'][ia])&(sim['z']<=sim['z'][ib]);weights=np.array([dual,dual,widths])
        heating=epsilon_0/2*np.sum(sim['nu']*c*abs(state[6:9])**2*weights*mask)
        h['vertical_poynting_flux_W_m2']=flux
        h.attrs['power_balance_relative_error']=abs((flux[ia]-flux[ib])/heating-1)
        h.attrs['modeled_absorbed_flux_W_m2']=heating

        h['incident_reference_E_V_m']=inc[:3];h['source_polarization']=v
        h['time_s']=rows[:,0];h.create_dataset('E_time_V_m',data=record[...,0],compression='gzip')
        h.create_dataset('E_time_quadrature_V_m',data=record[...,1],compression='gzip')
        h['momentum_acceleration_phasor_m_s2']=-e/m_e*state[:3]
        # Reconstruct density from Gauss law; half-cell charge, same Yee stencil.
        ex,ey,ez=state[:3];div=1j*sim['kx']*ex+(ez-np.roll(ez,1))/((sim['dz_cells']+np.roll(sim['dz_cells'],1))/2)
        h['cold_density_phasor_m3']=-epsilon_0/e*div
        # Discrete Helmholtz split: keep transverse EM forcing separate from
        # longitudinal self-field before coupling to a warm Poisson solver.
        nn=len(ex);ii=np.arange(nn);S=coo_matrix((np.ones(nn),(ii,(ii+1)%nn)),shape=(nn,nn)).tocsr();I=eye(nn,format='csr')
        widths=sim['dz_cells'];dual=(widths+np.roll(widths,1))/2
        df=diags(1/widths)@(S-I);db=diags(1/dual)@(I-S.T)
        phi=splu((db@df-sim['kx']**2*I).astype(complex).tocsc()).solve(-div)
        longitudinal=np.array([-1j*sim['kx']*phi,0*phi,-df@phi]);transverse=state[:3]-longitudinal
        h['E_transverse_phasor_V_m']=transverse;h['E_longitudinal_phasor_V_m']=longitudinal
        h['external_acceleration_phasor_m_s2']=-e/m_e*transverse
        h.attrs['coupling']='Use E_transverse as imposed EM force with a matching Fourier-x warm-fluid/Poisson solve; total cold E already includes a longitudinal self-field. One-way coupling holds the HF field fixed.'
        h.attrs['transverse_divergence_relative']=float(np.linalg.norm(1j*sim['kx']*transverse[0]+db@transverse[2])/np.linalg.norm(div))
        h['E_phasor_real_V_m']=state[:3].real;h['E_phasor_imag_V_m']=state[:3].imag
        h['E_magnitude_V_m']=np.abs(state[:3]);h['E_transverse_magnitude_V_m']=np.abs(transverse)
        h['incident_reference_magnitude_V_m']=np.abs(inc[:3])

        diff=subprocess.check_output(['git','-C',str(REPO),'diff'],text=True)
        h.attrs['modified_source_sha256']=hashlib.sha256(''.join(p.read_text() for p in [REPO/'src/bin/stratified.rs',REPO/'src/lib.rs',Path(__file__)]).encode()).hexdigest()
        h.create_dataset('provenance/native_source',data=(REPO/'src/bin/stratified.rs').read_text())
        h.create_dataset('provenance/shared_kernel',data=(REPO/'src/lib.rs').read_text())
        h.create_dataset('provenance/preparation_source',data=Path(__file__).read_text())
    cfg.unlink();raw.unlink();print('Saved',path,flush=True)
    return result

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--scan',action='store_true');ap.add_argument('--run',action='store_true');ap.add_argument('--graded',action='store_true');ap.add_argument('--cycles',type=int,default=2);ap.add_argument('--dz',type=float,default=1);ap.add_argument('--angle',type=float,default=0);ap.add_argument('--azimuth',type=float,default=0);ap.add_argument('--buffer',type=float,default=600);args=ap.parse_args()
    if args.scan:
        results=[]
        for az in [0.,45.,90.]:
            for angle in np.arange(-40,41,5):
                r=optimize(args.dz,float(angle),az);results.append(r);print(json.dumps(r),flush=True)
        with h5py.File(OUT/'angle_scan.h5','w') as h:
            for key in results[0]:h[key]=[r[key] for r in results]
            h.attrs['metric']='integral |Ez|^2 dz within +/-30 m of cold vertical resonance, per 1 W/m2 upward incident flux; optimized complex two-component source polarization'
        print('BEST',max(results,key=lambda r:r['score']),flush=True)
    elif args.run:save_run(args.dz,args.angle,args.azimuth,args.buffer,cycles=args.cycles,graded=args.graded)
    else:print(optimize(args.dz,args.angle,args.azimuth,graded=args.graded),flush=True)
if __name__=='__main__':main()
