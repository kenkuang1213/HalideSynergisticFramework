#ifndef SYNER_H_
#define SYNER_H_
#include "CPULL.h"
#include "Halide.h"
struct PthreadArgs{
	Halide::Buffer buf;
	CPULL* GPU;
};
class Syner {
public:
	Syner(){}
	Syner(Halide::Image<uint16_t> buf,int iLevels,int iAlpha,int iBeta):inputSizeX(buf.width()),inputSizeY(buf.height()),
	cpuBuf(Halide::UInt(16),buf.width(),CPUWorkload,3,0,buf.raw_buffer()->host),
	gpuBuf(Halide::UInt(16),buf.width(),inputSizeY-CPUWorkload,3,0,buf.raw_buffer()->host+(inputSizeX*CPUWorkload*buf.raw_buffer()->elem_size*1))
	{
//		Halide::Buffer cpuBuf(Halide::UInt(16),buf.width(),CPUWorkload,3,0,buf.raw_buffer()->host);
		cpuBuf.raw_buffer()->stride[2]=inputSizeX*inputSizeY;
	
//		Halide::Buffer gpuBuf(Halide::UInt(16),buf.width(),inputSizeY-CPUWorkload,3,0,buf.raw_buffer()->host+(inputSizeX*CPUWorkload*buf.raw_buffer()->elem_size*1));
		gpuBuf.raw_buffer()->stride[2]=inputSizeX*inputSizeY;
		Halide::Image<uint16_t> cpuImg(cpuBuf);
		Halide::Image<uint16_t> gpuImg(gpuBuf);
		cpuLL=new CPULL(cpuImg,iLevels,iAlpha,iBeta);
		gpuLL=new CPULL(gpuImg,iLevels,iAlpha,iBeta);
		cpuLL->Algorithm();
		gpuLL->Algorithm();
	}
	void Schedule();
	Halide::Buffer Realize(int x,int y);
private:
	Halide::Target target;
	
	int inputSizeX;
	int inputSizeY;
	CPULL *cpuLL;
	CPULL *gpuLL;
	Halide::Buffer cpuBuf;
	Halide::Buffer gpuBuf;
	static const int CPUWorkload=100;
	static void* GPUThread(void* arguments){
		struct PthreadArgs* args=(struct PthreadArgs*)arguments;
		args->GPU->Realize(args->buf);
//		gpuLL->Realize(args->buf);
//		gpuFunc.realize(buf);
//		buf.copy_to_host();	
	  return NULL;
	}

};



#endif
