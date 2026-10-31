//! Horizontal Fourier reduction of Yee curls and the shared cold-current update.
//! exp(i*kx*x-i*omega*t), z up. E_z/J_z and H_x/H_y at half z cells.
//! CW initialization solves this same recurrence; not a transmitter turn-on.
use meteor_fdtd::{current_rhs,current_next};
use meteor_fdtd::stratified::{advance_magnetic,advance_electric};
use serde::{Deserialize,Serialize};
use std::{fs,io::Write};
type Z=[f64;2];
#[inline(always)] fn add(a:Z,b:Z)->Z{[a[0]+b[0],a[1]+b[1]]}
#[inline(always)] fn sub(a:Z,b:Z)->Z{[a[0]-b[0],a[1]-b[1]]}
#[inline(always)] fn mul(a:Z,b:f64)->Z{[a[0]*b,a[1]*b]}
#[inline(always)] fn phase(a:Z,p:f64)->Z{let(s,c)=p.sin_cos();[a[0]*c-a[1]*s,a[0]*s+a[1]*c]}
#[derive(Deserialize)] struct Input{n:usize,dz:f64,dz_cells:Vec<f64>,dt:f64,omega:f64,kx:f64,gyro:[f64;3],wp:Vec<Vec<f64>>,nu:Vec<Vec<f64>>,sponge:Vec<Vec<f64>>,initial:Vec<Vec<Z>>,source:Vec<Vec<Z>>,steps:usize,record_stride:usize,record_start:usize,output:String}
#[derive(Serialize)] struct Report{steps:usize,dt_s:f64,frames:usize,relative_max_e_error:f64,finite:bool}
fn gyro(j:&[Vec<Z>],q:usize,i:usize,n:usize,g:[f64;3],cx:f64,dz:&[f64])->Z{
 let im=(i+n-1)%n;let ip=(i+1)%n;
 let avgz=mul(add(mul(j[2][i],dz[i]),mul(j[2][im],dz[im])),1./(dz[i]+dz[im]));
 match q{0=>sub(mul(avgz,g[1]*cx),mul(j[1][i],g[2]*cx)),
 1=>sub(mul(j[0][i],g[2]*cx),mul(avgz,g[0])),
 _=>sub(mul(add(j[1][i],j[1][ip]),0.5*g[0]),mul(add(j[0][i],j[0][ip]),0.5*g[1]*cx))}
}
fn main()->Result<(),Box<dyn std::error::Error>>{
 let cfg:Input=serde_json::from_slice(&fs::read(std::env::args().nth(1).ok_or("config")?)?)?;
 let n=cfg.n;let dt=cfg.dt;let a=cfg.omega*dt;let k=cfg.kx;let cx=1.;
 assert!(n>4 && cfg.dz>0. && dt>0. && dt/cfg.dz<0.5);
 let mut e=cfg.initial[0..3].to_vec();let mut h=cfg.initial[3..6].to_vec();let mut j=cfg.initial[6..9].to_vec();
 for f in h.iter_mut().chain(j.iter_mut()){for v in f{*v=phase(*v,a/2.);}}
 let mut rhs=j.clone();let mut next=j.clone();let mut file=fs::File::create(&cfg.output)?;let mut frames=0;
 for step in 0..cfg.steps{
  advance_magnetic(&e,&mut h,&cfg.dz_cells,k,dt,&cfg.sponge);
  for q in 0..3{for i in 0..n{let g=gyro(&j,q,i,n,cfg.gyro,cx,&cfg.dz_cells);let d=0.5*dt*cfg.nu[q][i];
   for r in 0..2{rhs[q][i][r]=current_rhs(j[q][i][r],cfg.wp[q][i]*e[q][i][r],g[r],dt,d);}
  }}
  let contraction=0.5*dt*cfg.gyro.iter().map(|v|v*v).sum::<f64>().sqrt();
  let iterations=if contraction>0. {(1e-14_f64.ln()/contraction.ln()).ceil().max(2.) as usize}else{1};
  for _ in 0..iterations{for q in 0..3{for i in 0..n{let g=gyro(&j,q,i,n,cfg.gyro,cx,&cfg.dz_cells);let d=0.5*dt*cfg.nu[q][i];
   for r in 0..2{next[q][i][r]=current_next(rhs[q][i][r],g[r],dt,d);}
  }}std::mem::swap(&mut j,&mut next);}
  advance_electric(&mut e,&h,&j,&cfg.wp,&cfg.dz_cells,k,dt,&cfg.sponge,&cfg.source,-a*(step as f64+0.5));
  if step>=cfg.record_start && (step-cfg.record_start)%cfg.record_stride==0{
   file.write_all(&((step+1) as f64*dt/299792458.).to_le_bytes())?;
   for q in 0..3{for v in &e[q]{for val in v{file.write_all(&val.to_le_bytes())?;}}}frames+=1;
  }
 }
 let mut error=0f64;let mut scale=0f64;let mut finite=true;
 for q in 0..3{for i in 0..n{let reference=phase(cfg.initial[q][i],-a*cfg.steps as f64);
  for r in 0..2{finite &= e[q][i][r].is_finite();error=error.max((e[q][i][r]-reference[r]).abs());scale=scale.max(reference[r].abs());}}}
 let report=Report{steps:cfg.steps,dt_s:dt/299792458.,frames,relative_max_e_error:error/scale,finite};
 println!("{}",serde_json::to_string(&report)?);if !finite || error/scale>1e-5{return Err("FDTD harmonic consistency failed".into())}Ok(())
}
