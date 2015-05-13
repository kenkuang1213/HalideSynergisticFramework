#ifndef PIPELINEJITINTERFACE_H_
#define PIPELINEJITINTERFACE_H_
#include "Halide.h"
template<typename T> 
class PipelineJITInterface {
public:
//	virtual void Algorithm() =0;
	virtual void CPUSchedule () =0;
	virtual void GPUSchedule () =0;
//	virtual Halide::Image<uint8_t> Realize(int x,int y,int z)=0;
//	virtual buffer_t Realize(int x,int y)=0;
//	virtual buffer_t Realize(int x)=0;
	virtual ~PipelineJITInterface(){};
	Halide::Image< T > input;
private:


};
#endif // PIPELINEJITINTERFACE_H_
