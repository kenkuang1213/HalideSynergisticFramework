#ifndef JITINTERFACE_H_
#define JITINTERFACE_H_
#include "Halide.h"
class JITInterface{
public :
	Halide::Buffer *input;
	JITInterface(Halide::Buffer &buf):Buffer(buf){}
	virtual void Algorithm()=0;
	virtual void Schedule()=0;
	virtual Halide::Buffer Realize(int x,int y)=0;
	virtual Halide::Buffer Realize(int x,int y,Halide::Buffer &buf,int cpuHeight)=0;

};

#endif // JITINTERFACE_H_
