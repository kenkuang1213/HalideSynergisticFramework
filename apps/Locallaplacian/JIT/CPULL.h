#ifndef CPULL_H_
#define CPULL_H_
#include "JitInterface.h"
class CPULL : public JITInterface{
public :
	CPULL(){}
	CPULL(Halide::Image<uint16_t> buf,int iLevels,float iAlpha,float iBeta):JITInterface(buf,Halide::UInt(16),3){
		levels=iLevels;
		alpha=iAlpha/(iLevels-1);
		beta=iBeta;
	}

	void Algorithm();
	void Schedule();
	void Schedule(Halide::Target target);
	Halide::Buffer Realize(int x,int y);
	Halide::Buffer Realize(Halide::Buffer buf);
	Halide::Buffer Realize(int x,int y,Halide::Buffer &buf,int cpuHeight);
private :
	static const int J = 8;
	Halide::Expr levels;
    Halide::Expr alpha, beta;
	Halide::Var x,y,c, k;
	Halide::Func remap,floating,clamped,gray,gPyramid[J],lPyramid[J],outLPyramid[J],outGPyramid[J],color,inGPyramid[J],output;
	Halide::Func downsample(Halide::Func f);
	Halide::Func upsample(Halide::Func f);
};
#endif // CPULL_H_
