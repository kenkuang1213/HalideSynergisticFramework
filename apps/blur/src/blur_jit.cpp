#include "Halide.h"

#include "blurJITCG.h"
#include <cstdio>

using namespace Halide;
Image<uint16_t> Cblur(Image<uint16_t> in) {
    Image<uint16_t> tmp(in.width()-8, in.height());
    Image<uint16_t> out(in.width()-8, in.height()-2);

//    begin_timing;

    for (int y = 0; y < tmp.height(); y++)
        for (int x = 0; x < tmp.width(); x++)
            tmp(x, y) = (in(x, y) + in(x+1, y) + in(x+2, y))/3;

    for (int y = 0; y < out.height(); y++)
        for (int x = 0; x < out.width(); x++)
            out(x, y) = (tmp(x, y) + tmp(x, y+1) + tmp(x, y+2))/3;

//    end_timing;

    return out;
}

int main(){
	Func blur_x("blur_x"), blur_y("blur_y");
    Image<uint16_t> input(6408, 4802);

    for (int y = 0; y < input.height(); y++) {
        for (int x = 0; x < input.width(); x++) {
            input(x, y) = rand() & 0xfff;
        }
    }
//	
//    Var x("x"), y("y"), xi("xi"), yi("yi");
//    
//    // The algorithm
//    blur_x(x, y) = (input(x, y) + input(x+1, y) + input(x+2, y))/3;
//    blur_y(x, y) = (blur_x(x, y) + blur_x(x, y+1) + blur_x(x, y+2))/3;
//    
//    // How to schedule it
//    Target target= get_target_from_environment();
////    target.set_feature(Target::OpenCL);
//    if(target.has_gpu_feature()){
//		printf("It's gpu schedule\n");
//		blur_y.compute_root().gpu_tile(x,y,16,16);
//		blur_x.compute_at(blur_y,Var::gpu_threads());
//	}
//	else{
//		printf("It's need GPU feature\n");
//		return -1;
//	}
	blurJITCG<uint16_t> *blurCPU=new blurJITCG<uint16_t> (input);
	blurCPU->CPUAlgorithm();
	blurCPU->CPUSchedule();
	blurCPU->test_CPUperformance(6400,4800);
	
	blurJITCG<uint16_t> *blurGPU=new blurJITCG<uint16_t>(input);
	blurGPU->GPUAlgorithm();
	blurGPU->GPUSchedule();
	blurGPU->test_GPUperformance(6400,4800);
	
	
	blurJITCG<uint16_t> *blurCGPU=new  blurJITCG<uint16_t>(input);
	blurCGPU->CPUAlgorithm(0,6400,0,2400);
	blurCGPU->GPUAlgorithm(0,6400,2401,4800);
	blurCGPU->CPUSchedule();
	blurCGPU->GPUSchedule();
	blurCGPU->test_CGperformance(6400,4800);
	Image<uint16_t> out=blurCPU->Realize("CPU",6400,4800);
	Image<uint16_t> outGPU=blurGPU->Realize("GPU",6400,4800);
	Image<uint16_t> outCGPU=blurCGPU->Realize("CG",6400,4800);
	Image<uint16_t> blurry = Cblur(input);
    for (int y = 0 ;y < input.height() - 4; y++) {
        for (int x = 0; x < input.width() -8; x++) {
            if ( blurry(x, y) != out(x, y)||out(x, y)!=outGPU(x,y)||outGPU(x,y)!=outCGPU(x,y))
                printf("difference at (%d,%d): %d %d %d %d\n", x, y, blurry(x, y),  out(x, y),outGPU(x, y),outCGPU(x, y));
        }
    }

}
