// Unmagnetized Yee/CPML port of src/main.rs. Shared input densities and units.
// GPU uses float32; reductions and incident normalization use host float64.
import Foundation
import Metal

struct Params {
    var n: UInt32; var count: UInt32; var step: UInt32; var pad: UInt32 = 0
    var dx: Float; var dt: Float; var k: Float; var sigma: Float
    var launch: Float; var damping: Float; var reserved0: Float = 0; var reserved1: Float = 0
}

let source = """
#include <metal_stdlib>
using namespace metal;
struct Params { uint n,count,step,pad; float dx,dt,k,sigma,launch,damping,r0,r1; };
kernel void curl_step(device float *out [[buffer(0)]], device const float *input [[buffer(1)]],
 device float *mem [[buffer(2)]], device const float2 *pml [[buffer(3)]],
 device const float *wp [[buffer(4)]], device const float *p [[buffer(5)]],
 constant Params &s [[buffer(6)]], constant uint &forward [[buffer(7)]], uint id [[thread_position_in_grid]]) {
 if(id>=s.count) return;
 uint n=s.n, m=s.count; uint3 pos=uint3(id/(n*n),(id/n)%n,id%n);
 if(any(pos==uint3(0)) || any(pos==uint3(n-1))) return;
 uint strides[3]={n*n,n,1};
 for(uint q=0;q<3;q++) {
   uint a=(q+1)%3,b=(q+2)%3;
   float da=forward ? (input[b*m+id+strides[a]]-input[b*m+id])/s.dx
                    : (input[b*m+id]-input[b*m+id-strides[a]])/s.dx;
   float db=forward ? (input[a*m+id+strides[b]]-input[a*m+id])/s.dx
                    : (input[a*m+id]-input[a*m+id-strides[b]])/s.dx;
   float2 ca=pml[pos[a]],cb=pml[pos[b]];
   if(ca.x!=0) { uint j=(2*q)*m+id; mem[j]=ca.y*mem[j]+ca.x*da; da+=mem[j]; }
   if(cb.x!=0) { uint j=(2*q+1)*m+id; mem[j]=cb.y*mem[j]+cb.x*db; db+=mem[j]; }
   out[q*m+id]+=(forward ? -s.dt : s.dt)*(da-db);
   if(!forward) out[q*m+id]-=s.dt*wp[q*m+id]*p[q*m+id];
 }
}
kernel void current_step(device const float *electric [[buffer(0)]], device const float *wp [[buffer(1)]],
 device float *p [[buffer(2)]],device float *fr [[buffer(3)]],device float *fi [[buffer(4)]],
 constant Params &s [[buffer(5)]],uint id [[thread_position_in_grid]]) {
 if(id>=s.count) return;
 float tau=float(s.step)*s.dt;
 float x=(float(id/(s.n*s.n))-float(s.n)/2)*s.dx;
 float u=tau-x-s.launch;
 float incident=exp(-0.5f*(u/s.sigma)*(u/s.sigma))*cos(s.k*u);
 float phase=s.k*(tau+0.5f*s.dt);
 for(uint q=0;q<3;q++) {
   uint j=q*s.count+id; float w=wp[j];
   if(w==0) continue;
   p[j]=((1-s.damping)*p[j]+s.dt*w*(electric[j]+(q==2 ? incident : 0)))/(1+s.damping);
   float value=s.dt*w*p[j]; fr[j]+=value*cos(phase); fi[j]+=value*sin(phase);
 }
}
kernel void reduce_back(device const float *fr [[buffer(0)]],device const float *fi [[buffer(1)]],
 device float4 *out [[buffer(2)]],constant Params &s [[buffer(3)]],
 uint id [[thread_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
 uint group [[threadgroup_position_in_grid]]) {
 threadgroup float4 tmp[256]; float4 v=0;
 if(id<s.count) {
   float x=(float(id/(s.n*s.n))-float(s.n)/2)*s.dx;
   float cs=cos(s.k*x),sn=sin(s.k*x);
   uint y=s.count+id,z=2*s.count+id;
   v=float4(fr[y]*cs-fi[y]*sn,fr[y]*sn+fi[y]*cs,fr[z]*cs-fi[z]*sn,fr[z]*sn+fi[z]*cs);
 }
 tmp[tid]=v; threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint stride=128;stride>0;stride/=2) {
   if(tid<stride) tmp[tid]+=tmp[tid+stride];
   threadgroup_barrier(mem_flags::mem_threadgroup);
 }
 if(tid==0) out[group]=tmp[0];
}
"""

func run() throws {
    guard CommandLine.arguments.count == 2 else { fatalError("Usage: metal-fdtd config.json") }
    let data = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[1]))
    let config = try JSONSerialization.jsonObject(with: data) as! [String: Any]
    func number(_ key: String) -> Double { (config[key] as! NSNumber).doubleValue }
    guard number("collision_rate_s1") >= 0, config["polarization"] as? String == "linear",
          (config["magnetic_field_t"] as! [Double]).allSatisfy({$0 == 0}),
          number("field_frames") == 0 else { fatalError("Metal backend supports linear B=0, no field frames only") }
    guard let device=MTLCreateSystemDefaultDevice(), let queue=device.makeCommandQueue() else { fatalError("No Metal device") }
    let options=MTLCompileOptions(); options.mathMode = .safe
    let library=try device.makeLibrary(source: source, options: options)
    let curl=try device.makeComputePipelineState(function:library.makeFunction(name:"curl_step")!)
    let current=try device.makeComputePipelineState(function:library.makeFunction(name:"current_step")!)
    let reduce=try device.makeComputePipelineState(function:library.makeFunction(name:"reduce_back")!)
    let n=Int(number("n")),m=n*n*n,dx=number("dx_m"),frequency=number("frequency_hz")
    let c=299792458.0, qe=1.602176634e-19,eps=8.8541878188e-12,me=9.1093837139e-31
    let k=2*Double.pi*frequency/c, wavelength=c/frequency,support=number("support_radius_m")
    func buffer(_ count:Int)->MTLBuffer {
        guard let b=device.makeBuffer(length:count*4,options:.storageModeShared) else { fatalError("Metal buffer allocation failed") }
        memset(b.contents(),0,b.length);return b
    }
    let electric=buffer(3*m),magnetic=buffer(3*m),epml=buffer(6*m),hpml=buffer(6*m)
    let wp=buffer(3*m),p=buffer(3*m),fr=buffer(3*m),fi=buffer(3*m)
    let density=try Data(contentsOf:URL(fileURLWithPath:config["density_file"] as! String),options:.mappedIfSafe)
    guard density.count == 24*m else { fatalError("Invalid density length") }
    var maxWP=0.0,active=0
    let w=wp.contents().bindMemory(to:Float.self,capacity:3*m)
    let pmlCount=Int(number("pml_cells"))
    try density.withUnsafeBytes { raw in
        let values=raw.bindMemory(to:Double.self)
        for j in 0..<3*m {
            let ne=values[j]
            if !ne.isFinite || ne<0 { throw NSError(domain:"Invalid density",code:1) }
            if ne>0 {
                let id=j%m, pos=[id/(n*n),(id/n)%n,id%n]
                guard pos.allSatisfy({$0>pmlCount && $0<n-pmlCount-2}) else { fatalError("Density overlaps PML") }
                let omega=sqrt(ne*qe*qe/(eps*me*c*c));w[j]=Float(omega);maxWP=max(maxWP,omega);active+=1
            }
        }
    }
    let dt=min(0.5*dx,0.7/maxWP), launch=6*wavelength+support
    let steps=Int(ceil((launch+6*wavelength+support+number("periods")*wavelength)/dt))
    func pml(_ shift:Double)->MTLBuffer {
        let b=buffer(2*n),v=b.contents().bindMemory(to:Float.self,capacity:2*n)
        for i in 0..<n {
            let x=Double(i)+shift,depth=min(1,max(0,max(Double(pmlCount)-x,x-Double(n-pmlCount))/Double(pmlCount)))
            if depth==0 {v[2*i]=0;v[2*i+1]=1;continue}
            let sigma = -4*log(1e-8)/(2*Double(pmlCount)*dx)*pow(depth,3)
            let alpha=0.05*k*(1-depth), b=exp(-(sigma+alpha)*dt)
            v[2*i]=Float(sigma*(b-1)/(sigma+alpha));v[2*i+1]=Float(b)
        }
        return b
    }
    let intPML=pml(0),halfPML=pml(0.5)
    var params=Params(n:UInt32(n),count:UInt32(m),step:0,dx:Float(dx),dt:Float(dt),k:Float(k),sigma:Float(wavelength),launch:Float(launch),damping:Float(0.5*dt*number("collision_rate_s1")/c))
    let started=Date();var incidentRe=0.0,incidentIm=0.0
    print("Metal \(device.name): \(n)^3 cells, \(active) current edges, \(steps) steps");fflush(stdout)
    for start in stride(from:0,to:steps,by:32) {
        let command=queue.makeCommandBuffer()!
        for step in start..<min(start+32,steps) {
            params.step=UInt32(step)
            let tau=Double(step)*dt,u=tau-launch
            let ref=exp(-0.5*pow(u/wavelength,2))*cos(k*u)
            incidentRe+=dt*ref*cos(k*tau);incidentIm+=dt*ref*sin(k*tau)
            func encodeCurl(_ forward:UInt32) {
                let enc=command.makeComputeCommandEncoder()!;enc.setComputePipelineState(curl)
                let buffers=forward==1 ? [magnetic,electric,hpml,halfPML,wp,p] : [electric,magnetic,epml,intPML,wp,p]
                for (j,b) in buffers.enumerated(){enc.setBuffer(b,offset:0,index:j)}
                enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:6)
                var direction=forward;enc.setBytes(&direction,length:4,index:7)
                enc.dispatchThreads(MTLSize(width:m,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1));enc.endEncoding()
            }
            encodeCurl(1)
            let enc=command.makeComputeCommandEncoder()!;enc.setComputePipelineState(current)
            for (j,b) in [electric,wp,p,fr,fi].enumerated(){enc.setBuffer(b,offset:0,index:j)}
            enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:5)
            enc.dispatchThreads(MTLSize(width:m,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1));enc.endEncoding()
            encodeCurl(0)
        }
        command.commit();command.waitUntilCompleted()
        if let error=command.error {throw error}
        if start%320==0 { print("\(min(start+32,steps))/\(steps) (\(Date().timeIntervalSince(started)) s)");fflush(stdout) }
    }
    let groups=(m+255)/256,out=buffer(groups*4),command=queue.makeCommandBuffer()!,enc=command.makeComputeCommandEncoder()!
    enc.setComputePipelineState(reduce)
    for (j,b) in [fr,fi,out].enumerated(){enc.setBuffer(b,offset:0,index:j)}
    enc.setBytes(&params,length:MemoryLayout<Params>.stride,index:3)
    enc.dispatchThreadgroups(MTLSize(width:groups,height:1,depth:1),threadsPerThreadgroup:MTLSize(width:256,height:1,depth:1));enc.endEncoding()
    command.commit();command.waitUntilCompleted();if let error=command.error {throw error}
    let values=out.contents().bindMemory(to:Float.self,capacity:groups*4)
    var sums=[Double](repeating:0,count:4)
    for g in 0..<groups {for q in 0..<4 {sums[q]+=Double(values[4*g+q])*pow(dx,3)}}
    let intensity=incidentRe*incidentRe+incidentIm*incidentIm
    let rcs=k*k/(4*Double.pi*intensity)*sums.reduce(0){$0+$1*$1}
    guard rcs.isFinite && rcs>0 else {fatalError("Invalid Metal RCS")}
    let result:[String:Any] = ["backend":"Metal float32 unmagnetized pilot", "device":device.name,
        "n":n,"dx_m":dx,"dt_s":dt/c,"steps":steps,"elapsed_s":Date().timeIntervalSince(started),
        "angle_rad":[Double.pi],"rcs_m2":[rcs],"incident_fourier":[[0.0,0.0],[0.0,0.0],[incidentRe,incidentIm]],
        "radiation_amplitude":[[[0.0,0.0],[sums[0],sums[1]],[sums[2],sums[3]]]],
        "collision_rate_s1":number("collision_rate_s1"),"frequency_hz":frequency]
    try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:URL(fileURLWithPath:config["output_file"] as! String))
    print("RCS \(10*log10(rcs)) dBsm")
}
do { try run() } catch { fputs("\(error)\n",stderr);exit(1) }
