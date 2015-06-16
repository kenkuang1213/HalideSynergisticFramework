#include <iostream>
#include "static_image.h"
#include "image_io.h"
#include "bilateral_grid_gpu.h"
#include "bilateral_grid_cpu.h"
#include "StaticDispatch.h"
#include "clock.h"
#include <limits>
#include <iomanip>

#ifndef DBL_MAX
#define DBL_MAX 1.79769e+308
#endif

#include "testPerformance.h"
using namespace std;
using namespace Fusion::Static;

int main(int argc,char** argv)
{
    if (argc < 3)
    {
        printf("Usage: ./process input.png r_sigma output.png\n"
               "e.g.: ./process input.png 0.1 output.png\n");
        return 0;
    }
    int workload=50;
    if(argc>=5){
        workload=atoi(argv[4]);
    }

    Image<float> input = load<float>(argv[1]);
    Image<float> output(input.width(),input.height(),input.channels());
    cout<<"Image Size : "<<input.width()<<" X "<<input.height()<<" X "<<input.channels()<<endl;
    float r_sigma = atof(argv[2]);




    Fusion::Test::testStaticPerformance(CPU,bilateral_grid_cpu,input,output,r_sigma);
    Fusion::Test::testStaticPerformance(GPU,bilateral_grid_gpu,input,output,r_sigma);

    Fusion::Test::testStaticPerformance(bilateral_grid_cpu,bilateral_grid_gpu,input,output,workload,r_sigma);
    Fusion::Test::testDynamicPerformance(bilateral_grid_cpu,bilateral_grid_gpu,input,output,r_sigma);

    Fusion::Test::testSizePerformance(CPU,bilateral_grid_cpu,input,output,workload,r_sigma);
    Fusion::Test::testSizePerformance(GPU,bilateral_grid_gpu,input,output,workload,r_sigma);

    StaticDispatch fusion(input,output);
    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,workload,r_sigma);
//    save(output, argv[3]);

    return 0;

}
