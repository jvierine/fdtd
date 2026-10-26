"""Run the native Rust Maxwell solver against Sugar's Figure 8 archive.

Pilot cases are not convergence-certified. Archive wp normalization and the
physical neutral-production normalization are recorded separately; never fit
an RCS scale factor. Run field evolution on avaruus, not the laptop.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import urllib.request
import zipfile
import h5py
import numpy as np
from scipy.constants import c,epsilon_0,m_e,e
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator
from dimant_oppenheim import electron_density
from maarsy_rcs import mpi_tools

ROOT=Path(__file__).resolve().parents[1]
FREQUENCY=38.45e6
U,VT,NA,BETA=40000.,951.,1e19,.7055413586
COLLISION=5.61e-19*40**(-.8)
LENGTH=VT/(NA*COLLISION*U)


def dimant_interpolator(cutoff):
    radial=np.geomspace(cutoff*1e-5,cutoff,160)
    mu=np.linspace(-1,1,81)
    table=np.array([[electron_density(r*u,r*np.sqrt(max(0,1-u*u)))
                     for u in mu] for r in radial])
    if not np.all(np.isfinite(table)) or np.min(table)<-1e-8:
        raise ValueError('Invalid normalized D-O density')
    scale=1e-10
    return radial,scale,RegularGridInterpolator(
        (np.log(radial),mu),np.log1p(np.maximum(table,0)/scale))


def archive_cases(directory):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/'RCS_Data.zip'
    if not path.exists():
        urllib.request.urlretrieve('https://zenodo.org/api/records/4514583/files/RCS_Data.zip/content',path)
    assert hashlib.md5(path.read_bytes()).hexdigest()=='0f77d3247f135be91aa2a6dd5db00526'
    vp=np.sqrt(1/(4*np.pi*1e-7*8.845e-12))
    rows=[]
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            if not name.startswith('RCS_Data/type3_freq38450000'):
                continue
            value=loadmat(io.BytesIO(archive.read(name)),simplify_cells=True)['sig']
            if 'k' not in value:
                continue
            k=np.atleast_1d(value['k'])
            j=int(np.argmin(abs(k*vp/(2*np.pi)-FREQUENCY)))
            if abs(k[j]*vp/(2*np.pi)/FREQUENCY-1)>.01:
                continue
            wpmax=float(name.split('wpmax_')[1].split('rcs.mat')[0])
            factor=float(value['wpfactor'])
            rate=factor**2/1.397739812665341e10*5e13
            rows.append(dict(wpmax_label=wpmax,wpfactor=factor,archive_ablation_rate_particles_s=rate,
                archive_rcs_m2=float(np.atleast_1d(value['sig'])[j]),
                archive_rp_m=float(np.atleast_1d(value['rp'])[j]),archive_file=name))
    return {r['wpmax_label']:r for r in rows}


def make_density(path,n,dx,pml,peak_shape,amplitude):
    """Analytic D-O sampled at native Yee edges; dimensional coordinates /lambda_T."""
    support=(n/2-pml-8)*dx
    cutoff=np.sqrt(3)*support/LENGTH
    radial,scale,interpolate=dimant_interpolator(cutoff)
    density=np.memmap(path,dtype='<f8',mode='w+',shape=(3,n,n,n))
    axis=(np.arange(n)-n/2)*dx
    max_ne=0.
    for component in range(3):
        coords=[axis+(dx/2 if q==component else 0) for q in range(3)]
        yy,zz=np.meshgrid(coords[1],coords[2],indexing='ij')
        for i in range(n):
            xx=np.full_like(yy,coords[0][i])
            radius=np.sqrt(xx**2+yy**2+zz**2)
            mu=np.divide(xx,radius,out=np.zeros_like(radius),where=radius>0)
            points=np.stack([np.log(np.clip(radius/LENGTH,radial[0],radial[-1])),mu],axis=-1)
            shape=scale*np.expm1(interpolate(points))
            shape=np.minimum(np.maximum(shape,0),peak_shape)
            # The native solver requires vacuum in the CPML. Retain a fixed
            # physical inner region and test the outer taper by varying box size.
            distance=np.maximum.reduce([abs(xx),abs(yy),abs(zz)])
            taper=.5*(1+np.cos(np.pi*np.clip((distance-(support-2*dx))/(2*dx),0,1)))
            value=amplitude*shape*taper
            density[component,i]=value
            max_ne=max(max_ne,float(np.max(value)))
        density.flush()
        print('Density component',component,'prepared',flush=True)
    sample=np.asarray(density[:,:,n//2,:]).copy()
    del density
    return support,max_ne,axis,sample


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--cells-per-wavelength',type=float,default=50.)
    p.add_argument('--wpmax',type=float,nargs='+',default=[1.])
    p.add_argument('--normalization',choices=['archive_wp','physical_rate'],default='archive_wp')
    p.add_argument('--ranks',type=int,default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--periods',type=float,default=2.)
    p.add_argument('--collision-rate',type=float,default=0.)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/sugar_reproduction')
    args=p.parse_args()
    if args.n%2 or args.n<64:
        p.error('Use an even grid size >=64')
    args.output.mkdir(parents=True,exist_ok=True)
    cases=archive_cases(args.output/'source')
    compiler,launcher=mpi_tools()
    subprocess.run(['cargo','build','--release','--locked','--bin','meteor-fdtd',
                    '--manifest-path',str(ROOT/'Cargo.toml')],env=dict(os.environ,MPICC=compiler),check=True)
    dx=c/FREQUENCY/args.cells_per_wavelength
    for key in args.wpmax:
        case=cases[key]
        name=f"wp{key:g}_n{args.n}_cpw{args.cells_per_wavelength:g}_{args.normalization}_T{args.periods:g}_nu{args.collision_rate:g}"
        output=args.output/(name+'.h5')
        if output.exists():
            print('Already saved',output,flush=True)
            continue
        if args.normalization=='archive_wp':
            # Archived np = pi*D, independently checked at physical grid points.
            # Candidate omega_p = wpfactor*sqrt(np); this must be tested against
            # the archived contour and RCS, not assumed validated by the run.
            amplitude=np.pi*case['wpfactor']**2*epsilon_0*m_e/e**2
        else:
            source=case['archive_ablation_rate_particles_s']/(4*np.sqrt(2*np.pi)*VT)
            amplitude=8*np.pi*source*NA*(1+3.8e-26/4.7e-26)*(COLLISION*BETA/(4*np.pi))/(np.sqrt(3)*LENGTH)
        density_path=args.output/(name+'.density.bin')
        support,maximum,axis,slice_=make_density(density_path,args.n,dx,12,98.34240942139034/np.pi,amplitude)
        result_path=args.output/(name+'.json')
        config=dict(n=args.n,dx_m=dx,pml_cells=12,frequency_hz=FREQUENCY,
            collision_rate_s1=args.collision_rate,magnetic_field_t=[0.,0.,0.],polarization='linear',
            support_radius_m=support,periods=args.periods,density_file=str(density_path.resolve()),
            output_file=str(result_path.resolve()),threads=args.threads,field_frames=0)
        config_path=args.output/(name+'.config.json')
        config_path.write_text(json.dumps(config))
        command=[launcher,'--bind-to','none','-n',str(args.ranks),str(ROOT/'target/release/meteor-fdtd'),str(config_path)]
        print('RUN',name,'nstar',amplitude,'lambda_T',LENGTH,'max_ne',maximum,flush=True)
        subprocess.run(command,check=True)
        result=json.loads(result_path.read_text())
        value=float(np.asarray(result['rcs_m2'])[-1])
        with h5py.File(output,'w') as h:
            h.attrs.update(**case,normalization=args.normalization,n_star_m3=amplitude,
                lambda_T_m=LENGTH,velocity_m_s=U,thermal_speed_m_s=VT,neutral_number_m3=NA,
                collision_cross_section_m2=COLLISION,peak_shape_cap=98.34240942139034/np.pi,
                status='native Rust FDTD pilot; convergence and archive normalization not yet certified',
                source_program=Path(__file__).read_text(),native_rust_source=(ROOT/'src/main.rs').read_text(),
                native_rust_sha256=hashlib.sha256((ROOT/'src/main.rs').read_bytes()).hexdigest(),
                max_ne_m3=maximum,computed_rcs_m2=value,
                difference_from_archive_db=10*np.log10(value/case['archive_rcs_m2']))
            for k,v in result.items():
                h[k]=v
            h['density_coordinate_m']=axis
            h['density_y0_slice_m3']=slice_
            h['config_json']=json.dumps(config)
        density_path.unlink()
        print('SAVED',output,'RCS dBsm',10*np.log10(value),'archive',10*np.log10(case['archive_rcs_m2']),flush=True)


if __name__=='__main__':
    main()
