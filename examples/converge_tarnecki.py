"""Controlled D-O domain convergence without changing the validated Rust solver.

Two strong-source probes, not a repeated parameter sweep. Smooth windows
regularize the unbounded D-O tail. Domain stability must survive a window,
vacuum-padding, duration and mesh check before acceptance.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

import h5py
import numpy as np
from scipy.constants import c

import reproduce_tarnecki_sparse as sparse

ROOT=sparse.ROOT
SOURCE=Path(__file__).read_text()


def window(radius, distance, support, dx, power):
    smooth=np.exp(-16*(radius/support)**power)
    edge=.5*(1+np.cos(np.pi*np.clip((distance-(support-2*dx))/(2*dx),0.,1.)))
    return smooth*edge


def geometry(n, cpw):
    pml=int(np.ceil(.5*cpw))
    gap=int(np.ceil(.5*cpw))
    return pml,(n/2-pml-gap-2)/cpw


def density(path, n, dx, length, nstar, support, power, shape):
    radial,scale,interpolate=shape
    if np.sqrt(3)*support/length>radial[-1] or dx/(2*length)<radial[0]:
        raise ValueError('D-O table does not cover sampled coordinates')
    data=np.memmap(path,dtype='<f8',mode='w+',shape=(3,n,n,n))
    axis=(np.arange(n)-n/2)*dx
    maximum=0.
    for component in range(3):
        coords=[axis+(dx/2 if q==component else 0.) for q in range(3)]
        yy,zz=np.meshgrid(coords[1],coords[2],indexing='ij')
        for i,x in enumerate(coords[0]):
            if abs(x)>=support:
                data[component,i]=0.
                continue
            xx=np.full_like(yy,x)
            radius=np.sqrt(xx**2+yy**2+zz**2)
            distance=np.maximum.reduce([abs(xx),abs(yy),abs(zz)])
            mu=xx/radius
            points=np.stack([np.log(np.clip(radius/length,radial[0],radial[-1])),mu],axis=-1)
            value=nstar*scale*np.expm1(interpolate(points))*window(radius,distance,support,dx,power)
            data[component,i]=value
            maximum=max(maximum,float(value.max()))
        data.flush()
        print('Density component',component,'prepared',flush=True)
    sample=np.array(data[:,:,n//2,:])
    del data
    return axis,sample,maximum


def save_progress(args, records, conclusions, status):
    path=args.output/'convergence.partial.h5'
    with h5py.File(path,'w') as h:
        h.attrs.update(source_program=SOURCE,status=status,
            tolerance_db=args.tolerance_db,velocity_km_s=float(sparse.VELOCITY[1]),
            source_m_inv=float(sparse.SIZES[3]),
            neutral_number_density_m3=args.neutral_density,
            result_definition='Monostatic RCS; no empirical scaling to the archived table')
        h['conclusions_json']=json.dumps(conclusions,sort_keys=True)
        for i,row in enumerate(records):
            g=h.create_group(f'case_{i:03d}')
            for key,value in row.items():g.attrs[key]=value
    path.replace(args.output/'convergence.h5')


def run_case(args, radar, n, cpw, power, periods, support_lambda, phase, shape, launcher, rust_source):
    frequency=sparse.RADARS[radar]
    wavelength=c/frequency
    pml,available=geometry(n,cpw)
    if support_lambda>available+1e-12 or support_lambda<=0:
        raise ValueError('Plasma does not fit with half-wave PML and half-wave vacuum gap')
    name=f'{radar}_{phase}_n{n}_cpw{cpw:g}_p{power}_T{periods:g}_R{support_lambda:.6g}'
    output=args.output/(name+'.h5')
    metadata=dict(radar=radar,phase=phase,n=n,cells_per_wavelength=cpw,
        window_power=power,periods=periods,support_lambda=support_lambda,
        side_lambda=n/cpw,source_m_inv=float(sparse.SIZES[3]),
        velocity_km_s=float(sparse.VELOCITY[1]))
    if output.exists():
        with h5py.File(output) as h:
            return dict(metadata,rcs_dbsm=float(h.attrs['rcs_dbsm']),file=str(output))
    length,nstar,collision,beta=sparse.plasma_parameters(sparse.VELOCITY[1],sparse.SIZES[3],args.neutral_density)
    dx=wavelength/cpw
    raw=args.output/'active_density.bin'
    axis,slice_,maximum=density(raw,n,dx,length,nstar,support_lambda*wavelength,power,shape)
    ranks,threads=args.ranks,args.threads
    if args.wait_pid and not Path(f'/proc/{args.wait_pid}').exists():
        ranks,threads=args.idle_ranks,args.idle_threads
    result_path=args.output/'active_result.json'
    config=dict(n=n,dx_m=dx,pml_cells=pml,frequency_hz=frequency,
        collision_rate_s1=0.,magnetic_field_t=[0.,0.,0.],polarization='linear',
        support_radius_m=support_lambda*wavelength,periods=periods,
        density_file=str(raw.resolve()),output_file=str(result_path.resolve()),
        threads=threads,field_frames=0)
    config_path=args.output/'active_config.json'
    config_path.write_text(json.dumps(config))
    print('RUN',name,'side lambda',n/cpw,'plasma half-width m',config['support_radius_m'],flush=True)
    subprocess.run([launcher,'--bind-to','none','-n',str(ranks),
        str(ROOT/'target/release/meteor-fdtd'),str(config_path.resolve())],check=True)
    result=json.loads(result_path.read_text())
    dbsm=float(10*np.log10(result['rcs_m2'][-1]))
    temporary=output.with_suffix('.partial.h5')
    with h5py.File(temporary,'w') as h:
        h.attrs.update(**metadata,rcs_dbsm=dbsm,source_program=SOURCE,
            native_rust_source=rust_source,
            native_rust_sha256=hashlib.sha256(rust_source.encode()).hexdigest(),
            lambda_T_m=length,n_star_m3=nstar,collision_cross_section_m2=collision,beta=beta,
            neutral_number_density_m3=args.neutral_density,max_ne_m3=maximum,
            window='exp[-16*(r/R)^p], zeroed smoothly at the outer cube edge',
            atmosphere='MSIS2.1 Sanya 2024-04-22 18 UTC; original Tarnecki atmosphere unavailable')
        for key,value in result.items():h[key]=value
        h['config_json']=json.dumps(config)
        h['density_coordinate_m']=axis;h['density_y0_slice_m3']=slice_
    temporary.replace(output)
    raw.unlink()
    print('SAVED',name,'RCS dBsm',dbsm,flush=True)
    return dict(metadata,rcs_dbsm=dbsm,file=str(output))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/tarnecki_convergence_v1_20261027')
    p.add_argument('--sizes',type=int,nargs='+',default=[256,384,512])
    p.add_argument('--ranks',type=int,default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--idle-ranks',type=int,default=8)
    p.add_argument('--idle-threads',type=int,default=12)
    p.add_argument('--wait-pid',type=int)
    p.add_argument('--neutral-density',type=float,default=1.247715306362518e19)
    p.add_argument('--tolerance-db',type=float,default=.5)
    args=p.parse_args()
    if len(args.sizes)<3 or any(n%2 or n<256 for n in args.sizes) or sorted(set(args.sizes))!=args.sizes:
        p.error('At least three increasing even box sizes >=256 required')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=vars(args).copy();manifest['output']=str(args.output.resolve())
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Changed settings require a new output directory')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    compiler,launcher=sparse.driver.mpi_tools()
    subprocess.run(['cargo','build','--release','--locked','--bin','meteor-fdtd'],
        cwd=ROOT,env=dict(os.environ,MPICC=compiler),check=True)
    shape=sparse.shared_shape()
    rust_source=(ROOT/'src/main.rs').read_text()
    records=[];conclusions={}
    cases={radar:[] for radar in sparse.RADARS}
    # Interleave radars so both have a new result before the largest boxes run.
    for n in args.sizes:
        for radar in sparse.RADARS:
            support=geometry(n,50.)[1]
            row=run_case(args,radar,n,50.,4,4.,support,'extent',shape,launcher,rust_source)
            records.append(row);cases[radar].append(row)
            changes=np.diff([r['rcs_dbsm'] for r in cases[radar]])
            conclusions[radar]=dict(extent_changes_db=changes.tolist(),
                domain_passed=bool(len(changes)>=2 and np.all(abs(changes[-2:])<=args.tolerance_db)),
                fully_verified=False)
            save_progress(args,records,conclusions,'Domain convergence in progress; no table accepted')
    for radar,rows in cases.items():
        if not conclusions[radar]['domain_passed']:
            conclusions[radar]['reason']='Domain test failed; do not run another parameter sweep'
            continue
        final=rows[-1];n=final['n'];support=final['support_lambda'];base=final['rcs_dbsm']
        # Larger vacuum box around identical plasma, then window, duration, mesh.
        vacuum=run_case(args,radar,n+128,50.,4,4.,support,'vacuum',shape,launcher,rust_source)
        alternate=run_case(args,radar,n,50.,2,4.,support,'window',shape,launcher,rust_source)
        duration=run_case(args,radar,n,50.,4,8.,support,'duration',shape,launcher,rust_source)
        fine=run_case(args,radar,3*n//2,75.,4,4.,support,'mesh',shape,launcher,rust_source)
        records.extend([vacuum,alternate,duration,fine])
        differences={key:row['rcs_dbsm']-base for key,row in
            [('vacuum',vacuum),('window',alternate),('duration',duration),('mesh',fine)]}
        conclusions[radar].update(check_changes_db=differences,
            fully_verified=bool(abs(differences['vacuum'])<=.1 and
                abs(differences['duration'])<=.1 and
                abs(differences['window'])<=args.tolerance_db and
                abs(differences['mesh'])<=args.tolerance_db))
        save_progress(args,records,conclusions,'Additional convergence checks in progress')
    passed=all(row['fully_verified'] for row in conclusions.values())
    save_progress(args,records,conclusions,'Convergence passed for strong-source probes only' if passed else
        'NOT CONVERGED: inspect differences before extending the table')
    print('CONCLUSIONS',json.dumps(conclusions,indent=2),flush=True)


if __name__=='__main__':
    main()
