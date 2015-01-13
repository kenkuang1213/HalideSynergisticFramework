#ifndef JITINTERFACE_H_
#define JITINTERFACE_H_
#include "Halide.h"
class JITInterface{
public :
	Halide::ImageParam  input;
	JITInterface(){}
	JITInterface(Halide::Buffer buf,Halide::Type t,int d):input(t,d){
		input.set(buf);
	}
	virtual void Algorithm()=0;
	virtual void Schedule()=0;
	virtual Halide::Buffer Realize(int x,int y)=0;
	virtual Halide::Buffer Realize(int x,int y,Halide::Buffer &buf,int cpuHeight)=0;

};

#endif // JITINTERFACE_H_
