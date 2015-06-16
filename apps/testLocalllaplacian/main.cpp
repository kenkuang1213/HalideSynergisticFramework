#include <iostream>
#include "static_image.h"
#include "image_io.h"
#include "local_laplacian_cpu.h"
#include "local_laplacian_gpu.h"
#include "StaticDispatch.h"
#include "clock.h"
#include <limits>
#include <iomanip>
#include "testPerformance.h"

#ifndef DBL_MAX
#define DBL_MAX 1.79769e+308
#endif



using namespace std;
using namespace Fusion::Static;


int main(int argc,char** argv) {
    if (argc < 6)
    {
        printf("Usage: ./process input.png levels alpha beta output.png\n"
               "e.g.: ./process input.png 8 1 1 output.png\n");
        return 0;
    }
    int workload=50;
    if(argc==7)
        workload=atoi(argv[6]);
    Image<uint16_t> input = load<uint16_t>(argv[1]);
    Image<uint16_t> output(input.width(),input.height(),input.channels());
    cout<<"Image Size : "<<input.width()<<" X "<<input.height()<<" X "<<input.channels()<<endl;
    int levels = atoi(argv[2]);
    float alpha = atof(argv[3]), beta = atof(argv[4]);

    Fusion::Test::testStaticPerformance(CPU,local_laplacian_cpu,input,output,levels,alpha/(levels-1),beta);
    Fusion::Test::testStaticPerformance(GPU,local_laplacian_gpu,input,output,levels,alpha/(levels-1),beta);

    Fusion::Test::testStaticPerformance(local_laplacian_cpu,local_laplacian_gpu,input,output,workload,levels,alpha/(levels-1),beta);
    Fusion::Test::testDynamicPerformance(local_laplacian_cpu,local_laplacian_gpu,input,output,levels,alpha/(levels-1),beta);

    Fusion::Test::testSizePerformance(CPU,local_laplacian_cpu,input,output,workload,levels,alpha/(levels-1),beta);
    Fusion::Test::testSizePerformance(GPU,local_laplacian_gpu,input,output,workload,levels,alpha/(levels-1),beta);

    return 0;

}
