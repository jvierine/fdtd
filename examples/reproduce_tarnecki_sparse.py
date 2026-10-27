"""Sparse 4x4 native-Maxwell comparison with Tarnecki's 100-km lookup.

Only 32 cases: two frequencies, four velocities, four source strengths.
The original atmosphere and FDTD input generator are not archived. This
uses the analytic D-O shape and explicitly recorded neutral density.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
import tarfile
import urllib.request

import h5py
import numpy as np
from scipy.constants import c
from scipy.interpolate import RegularGridInterpolator
from scipy.io import loadmat

import reproduce_sugar_fdtd as driver
from dimant_oppenheim import electron_density

ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path(__file__).read_text()
VELOCITY_INDEX=np.array([0,8,16,24])
SIZE_INDEX=np.array([0,10,19,29])
VELOCITY=np.linspace(20.,70.,25)[VELOCITY_INDEX]
SIZES=(np.logspace(21.,23.,30)*(5e-5)**2)[SIZE_INDEX]
RADARS={'maarsy':53.5e6,'eiscat':929e6}


def plasma_parameters(speed_km_s, size_m_inv, na):
    relative=speed_km_s-.951
    collision=5.61e-19*relative**(-.8)
    beta=.933*(relative-8.86)**2*relative**(-1.94)
    length=951./(na*collision*speed_km_s*1000.)
    nstar=8*np.pi*size_m_inv*na*(1+3.8e-26/4.7e-26)*(collision*beta/(4*np.pi))/(np.sqrt(3)*length)
    return length,nstar,collision,beta


def shared_shape():
    # One dimensionless table for every case; coordinates are physical / lambda_T.
    radius=np.geomspace(1e-5,2000.,240)
    mu=np.linspace(-1.,1.,81)
    table=np.array([[electron_density(r*u,r*np.sqrt(max(0.,1-u*u)))
                     for u in mu] for r in radius])
    if not np.all(np.isfinite(table)) or table.min() < -1e-8:
        raise ValueError('Invalid D-O interpolation table')
    scale=1e-10
    interp=RegularGridInterpolator((np.log(radius),mu),np.log1p(np.maximum(table,0.)/scale))
    return radius,scale,interp


def reference(output):
    path=output/'lookup-tables.tar.gz'
    if not path.exists():
        urllib.request.urlretrieve('https://zenodo.org/api/records/4723667/files/lookup-tables.tar.gz/content',path)
    if hashlib.md5(path.read_bytes()).hexdigest()!='21a7372e69644b10a7a82a829b71b619':
        raise ValueError('Tarnecki archive checksum mismatch')
    with tarfile.open(path) as archive:
        return {name:loadmat(io.BytesIO(archive.extractfile(
            f'lookup-tables/table_{name}_100.mat').read()))['table_alt'] for name in RADARS}


def save_summary(output, tables, args):
    temporary=output/'lookup.partial.h5'
    with h5py.File(temporary,'w') as h:
        h.attrs.update(source_program=SOURCE,altitude_km=100.,
            neutral_number_density_m3=args.neutral_density,
            atmosphere='MSIS2.1 Sanya 2024-04-22 18 UTC; original atmosphere unavailable',
            status='Sparse independent Maxwell calculation; not convergence-certified',
            density_prescription='Analytic Dimant-Oppenheim; no arbitrary peak-density cap',
            reference_doi='10.1029/2021JA029525',source_doi='10.5281/zenodo.4723667',
            n=args.n,cells_per_wavelength=args.cells_per_wavelength,periods=args.periods)
        h['velocity_km_s']=VELOCITY
        h['log10_n0_rM2_m_inv']=np.log10(SIZES)
        h['reference_velocity_km_s']=np.linspace(20.,70.,25)
        h['reference_log10_n0_rM2_m_inv']=np.log10(np.logspace(21.,23.,30)*(5e-5)**2)
        for radar,frequency in RADARS.items():
            g=h.create_group(radar);g.attrs['frequency_hz']=frequency
            measured=np.full((4,4),np.nan)
            for j in range(4):
                for i in range(4):
                    path=output/f'{radar}_v{i}_s{j}.h5'
                    if path.exists():
                        with h5py.File(path) as result:
                            measured[j,i]=10*np.log10(result.attrs['computed_rcs_m2'])
            published=tables[radar][np.ix_(SIZE_INDEX,VELOCITY_INDEX)]
            g['native_fdtd_dbsm']=measured
            g['tarnecki_fdtd_dbsm']=published
            g['reference_full_dbsm']=tables[radar]
            g['difference_db']=measured-published
            g['completed']=np.isfinite(measured)
    temporary.replace(output/'lookup.h5')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n',type=int,default=128)
    p.add_argument('--cells-per-wavelength',type=float,default=50.)
    p.add_argument('--periods',type=float,default=4.)
    p.add_argument('--neutral-density',type=float,default=1.247715306362518e19)
    p.add_argument('--ranks',type=int,default=4)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--output',type=Path,default=ROOT/'outputs/tarnecki_sparse4x4_20261027')
    args=p.parse_args()
    if args.n < 64 or args.n%2 or min(args.cells_per_wavelength,args.periods,args.neutral_density,args.ranks,args.threads)<=0:
        p.error('Positive scales and even n>=64 required')
    args.output.mkdir(parents=True,exist_ok=True)
    manifest=vars(args).copy();manifest['output']=str(args.output.resolve())
    manifest_path=args.output/'manifest.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text())!=manifest:
        raise ValueError('Use a new output directory for changed numerical settings')
    manifest_path.write_text(json.dumps(manifest,indent=2))
    tables=reference(args.output)
    save_summary(args.output,tables,args)
    compiler,launcher=driver.mpi_tools()
    subprocess.run(['cargo','build','--release','--locked','--bin','meteor-fdtd'],
                   cwd=ROOT,env=dict(os.environ,MPICC=compiler),check=True)
    rust_source=(ROOT/'src/main.rs').read_text()
    shape=shared_shape()
    driver.dimant_interpolator=lambda cutoff: shape
    for i,speed in enumerate(VELOCITY):
        length,_,collision,beta=plasma_parameters(speed,1.,args.neutral_density)
        driver.LENGTH=length
        for radar,frequency in RADARS.items():
            if all((args.output/f'{radar}_v{i}_s{j}.h5').exists() for j in range(4)):
                continue
            dx=c/frequency/args.cells_per_wavelength
            cutoff=np.sqrt(3)*(args.n/2-20)*dx/length
            if cutoff>shape[0][-1] or dx/(2*length)<shape[0][0]:
                raise ValueError('Interpolation table does not cover Yee-grid samples')
            base_path=args.output/f'{radar}_v{i}_unit.bin'
            support,maximum,axis,slice_=driver.make_density(base_path,args.n,dx,12,np.inf,1.)
            base=np.memmap(base_path,dtype='<f8',mode='r',shape=(3,args.n,args.n,args.n))
            for j,size in enumerate(SIZES):
                output=args.output/f'{radar}_v{i}_s{j}.h5'
                if output.exists():
                    continue
                _,nstar,_,_=plasma_parameters(speed,size,args.neutral_density)
                density_path=args.output/'active_density.bin'
                density=np.memmap(density_path,dtype='<f8',mode='w+',shape=base.shape)
                np.multiply(base,nstar,out=density);density.flush();del density
                result_path=args.output/'active_result.json'
                config=dict(n=args.n,dx_m=dx,pml_cells=12,frequency_hz=frequency,
                    collision_rate_s1=0.,magnetic_field_t=[0.,0.,0.],polarization='linear',
                    support_radius_m=support,periods=args.periods,
                    density_file=str(density_path.resolve()),output_file=str(result_path.resolve()),
                    threads=args.threads,field_frames=0)
                config_path=args.output/'active_config.json'
                config_path.write_text(json.dumps(config))
                print(f'RUN {radar} U={speed:g} N={size:g} lambda_T={length:g} nstar={nstar:g}',flush=True)
                subprocess.run([launcher,'--bind-to','none','-n',str(args.ranks),
                    str(ROOT/'target/release/meteor-fdtd'),str(config_path.resolve())],check=True)
                result=json.loads(result_path.read_text())
                value=float(result['rcs_m2'][-1])
                with h5py.File(output,'w') as h:
                    h.attrs.update(radar=radar,velocity_km_s=speed,source_m_inv=size,
                        lambda_T_m=length,n_star_m3=nstar,neutral_number_density_m3=args.neutral_density,
                        collision_cross_section_m2=collision,beta=beta,computed_rcs_m2=value,
                        source_program=SOURCE,native_rust_source=rust_source,
                        native_rust_sha256=hashlib.sha256(rust_source.encode()).hexdigest(),
                        reference_dbsm=tables[radar][SIZE_INDEX[j],VELOCITY_INDEX[i]],
                        max_ne_m3=maximum*nstar,altitude_km=100.)
                    for key,data in result.items(): h[key]=data
                    h['config_json']=json.dumps(config)
                    h['density_coordinate_m']=axis
                    h['density_y0_slice_m3']=slice_*nstar
                density_path.unlink()
                save_summary(args.output,tables,args)
                print('SAVED',output,'RCS dBsm',10*np.log10(value),flush=True)
            del base
            base_path.unlink()
    print('COMPLETE: 32 native FDTD cases; no dense parameter sweep.',flush=True)


if __name__=='__main__':
    main()
