#include "CPULL.h"
#include "Syner.h"
#include "Halide.h"
using namespace Halide;
#include "image_io.h"
#include <sys/time.h>
#include <iostream>
using namespace std;
int main(){
	Image<uint16_t> input=load<uint16_t>("rgb.png");
	
	
	Buffer cpubuf(UInt(16),input.width(),input.height(),3,0,input.raw_buffer()->host);
	cpubuf.raw_buffer()->stride[2]=input.raw_buffer()->stride[2];
	
	Image<uint16_t> cpuImg(cpubuf);
	cout<<cpuImg.width()<<" "<<cpuImg.height()<<endl;
	CPULL cpuLL(cpuImg,8,1,1);
	cpuLL.Algorithm();
	cpuLL.Schedule();
	
	Target target = get_host_target();
	std::vector<Target::Feature> gpuFeatures;
	gpuFeatures.push_back(Target::OpenCL);
#ifdef DEBUG
	gpuFeatures.push_back(Target::Debug);
#endif // DEBUG
	target.set_features(gpuFeatures); 
	
	
	
	
	
	CPULL gpuLL(input,8,1,1);
	gpuLL.Algorithm();
	gpuLL.Schedule(target);
//	
	Syner syner(input,8,1,1);
	syner.Schedule();

	timeval t1, t2;
	unsigned int bestT = 0xffffffff;
    for (int i = 0; i < 5; i++) {
      gettimeofday(&t1, NULL);
	  cpuLL.Realize(cpuImg.width(), cpuImg.height());
      gettimeofday(&t2, NULL);
      unsigned int t = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
      if (t < bestT) bestT = t;
    }
    printf("CPU Best Time:%u\n", bestT);
    
//    
    bestT = 0xffffffff;
	for (int i = 0; i < 5; i++) {
      gettimeofday(&t1, NULL);
	  gpuLL.Realize(input.width(), input.height());
      gettimeofday(&t2, NULL);
      unsigned int t = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
      if (t < bestT) bestT = t;
    }
    printf("GPU Best Time:%u\n", bestT);
    
	bestT = 0xffffffff;
	for (int i = 0; i < 5; i++) {
      gettimeofday(&t1, NULL);
	  syner.Realize(input.width(), input.height());
      gettimeofday(&t2, NULL);
      unsigned int t = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
      if (t < bestT) bestT = t;
    }
    printf("CPU+GPU Best Time:%u\n", bestT);
    
	Image<uint16_t> outCPU(cpuLL.Realize(input.width(), input.height()));
	Image<uint16_t> outGPU(gpuLL.Realize(input.width(), input.height()));
	Image<uint16_t> outCG(syner.Realize(input.width(), input.height()));
	
	save_png(outCPU,"outCPU.png");
	save_png(outGPU,"outGPU.png");
	save_png(outCG,"outCG.png");
}
