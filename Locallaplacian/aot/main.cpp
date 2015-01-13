#include "static_image.h"
#include "image_io.h"
#include "local_laplacian_cpu.h"
#include "fusion.h"
#include <iostream>
#include <sys/time.h>
#include <iomanip>


#define CPU 1
#define GPU 2
#define Fus 3
using namespace std;
inline void testPerformance(int type,int levels,float alpha,float beta,buffer_t* input) {
	Fusion<uint16_t> fusion(levels,alpha,beta);
	timeval t1, t2;
	unsigned int bestT = 0xffffffff;
	unsigned int worstT=0;
	switch (type) {
	case CPU:
		fusion.realizeCPU((buffer_t*)(input));
		for (int i = 0; i < 100; i++) {
			gettimeofday(&t1, NULL);
			Buffer buf=fusion.realizeCPU((buffer_t*)(input));
			gettimeofday(&t2, NULL);
			unsigned int t = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
			if (t < bestT) bestT = t;
			if (t > worstT) worstT = t;
		}
		cout<<setw(15)<<"Best CPU: "<<setw(10)<<bestT<<setw(15)<<" Worst CPU: "<<setw(10)<<worstT<<endl;
		break;
	case GPU:
		fusion.realizeGPU((buffer_t*)(input));
		for (int i = 0; i < 5; i++) {
			gettimeofday(&t1, NULL);
			fusion.realizeGPU((buffer_t*)(input));
			gettimeofday(&t2, NULL);
			unsigned int t = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
			if (t < bestT) bestT = t;
			if (t > worstT) worstT = t;
		}
		cout<<setw(15)<<"Best GPU: "<<setw(10)<<bestT<<setw(15)<<" Worst GPU: "<<setw(10)<<worstT<<endl;
		break;
	case Fus:
		fusion.realize((buffer_t*)(input),100);
		for (int i = 0; i < 5; i++) {
			gettimeofday(&t1, NULL);
			fusion.realize((buffer_t*)(input),200);
			gettimeofday(&t2, NULL);
			unsigned int t = (t2.tv_sec - t1.tv_sec) * 1000000 + (t2.tv_usec - t1.tv_usec);
			if (t < bestT) bestT = t;
			if (t > worstT) worstT = t;
		}
		cout<<setw(15)<<"Best Fusion: "<<setw(10)<<bestT<<setw(15)<<" Worst Fusion: "<<setw(10)<<worstT<<endl;
		break;
	}
}


int main(int argc,char** argv) {
	if (argc < 6) {
		printf("Usage: ./process input.png levels alpha beta output.png\n"
		       "e.g.: ./process input.png 8 1 1 output.png\n");
		return 0;
	}

	Image<uint16_t> input = load<uint16_t>(argv[1]);
	int levels = atoi(argv[2]);
	float alpha = atof(argv[3]), beta = atof(argv[4]);

//	Image<uint16_t> output(input.width(), input.height(), 3);
//	local_laplacian_cpu(levels, alpha/(levels-1), beta, input, output);

////
	testPerformance(CPU,levels,alpha,beta,(buffer_t*)(input));
	testPerformance(GPU,levels,alpha,beta,(buffer_t*)(input));
	testPerformance(Fus,levels,alpha,beta,(buffer_t*)(input));



	Fusion<uint16_t> fusion(levels,alpha,beta);
	Buffer buf=fusion.realize((buffer_t*)(input),100);
	Buffer bufg=fusion.realizeGPU((buffer_t*)(input));
	Image<uint16_t> output(buf.buffer,buf.ptr);
	Image<uint16_t> outputg(bufg.buffer,bufg.ptr);
	save(output, argv[5]);
	save(outputg, "outGPU.png");
}
