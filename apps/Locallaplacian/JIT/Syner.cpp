#include "Syner.h"
#include <iostream>
void Syner::Schedule(){
	target = Halide::get_host_target();
	std::vector<Halide::Target::Feature> gpuFeatures;
	gpuFeatures.push_back(Halide::Target::OpenCL);
#ifdef DEBUG
	gpuFeatures.push_back(Halide::Target::Debug);
#endif // DEBUG
	target.set_features(gpuFeatures); 
	gpuLL->Schedule(target);
	cpuLL->Schedule();		
}
Halide::Buffer Syner::Realize(int x,int y){
	Halide::Buffer buf(Halide::UInt(16),x,y,3);
	Halide::Buffer CPUbuf(Halide::UInt(16),x,CPUWorkload,3,0,buf.raw_buffer()->host);
	CPUbuf.raw_buffer()->stride[2]=x*y;
	Halide::Buffer GPUbuf(Halide::UInt(16),x,y-CPUWorkload,3,0,buf.raw_buffer()->host+(x*CPUWorkload*buf.raw_buffer()->elem_size*1));
//	Halide::Buffer GPUbuf(Halide::UInt(16),x,y,3,0,buf.raw_buffer()->host);
	GPUbuf.raw_buffer()->stride[2]=x*y;
//	GPUbuf.set_min(0,0,0);
//	GPUbuf.set_
//	std::cout<<i<<std::endl;
//	gpuLL->Realize(GPUbuf);
	pthread_t thread;
	struct PthreadArgs args;
	args.GPU=gpuLL;
	args.buf=GPUbuf;
	pthread_create(&thread, NULL,GPUThread, &args);
	cpuLL->Realize(CPUbuf);
	
	void* unused = NULL;
	pthread_join(thread, &unused);
	return buf;
}
