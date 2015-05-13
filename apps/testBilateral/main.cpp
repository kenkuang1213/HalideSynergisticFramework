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
#define CPU 1
#define GPU 2
#define Fus 3

using namespace std;
using namespace Fusion::Static;
inline void testPerformance(int type,float r_sigma,buffer_t* input,buffer_t* output) {
    StaticDispatch<float> fusion(bilateral_grid_cpu,bilateral_grid_gpu,input,output);

    double bestT =DBL_MAX;
    double worstT=0;
    switch (type) {
    case CPU:
        fusion.realizeCPU(r_sigma);
        for (int i = 0; i < 5; i++) {
            double t1=current_time();
            fusion.realizeCPU(r_sigma);
            double t2=current_time();
            double t=t2-t1;
            if (t < bestT) bestT = t;
            if (t > worstT) worstT = t;
        }
        cout<<setw(15)<<"Best CPU: "<<setw(10)<<bestT<<setw(15)<<" Worst CPU: "<<setw(10)<<worstT<<endl;
        break;
    case GPU:
        fusion.realizeGPU(r_sigma);
        for (int i = 0; i < 5; i++) {
            double t1=current_time();
            fusion.realizeGPU(r_sigma);
            double t2=current_time();
            double t=t2-t1;
            if (t < bestT) bestT = t;
            if (t > worstT) worstT = t;
        }
        cout<<setw(15)<<"Best GPU: "<<setw(10)<<bestT<<setw(15)<<" Worst GPU: "<<setw(10)<<worstT<<endl;
        break;
    case Fus:
        fusion.realize(r_sigma,200);
        for (int i = 0; i < 5; i++) {
            double t1=current_time();
            fusion.realizeWithStealing(r_sigma,1450);
            double t2=current_time();
            double t=t2-t1;
            if (t < bestT) bestT = t;
            if (t > worstT) worstT = t;
        }
        cout<<setw(15)<<"Best Fusion: "<<setw(10)<<bestT<<setw(15)<<" Worst Fusion: "<<setw(10)<<worstT<<endl;
        break;
    }
}


int main(int argc,char** argv) {
    if (argc < 3)
    {
        printf("Usage: ./process input.png r_sigma output.png\n"
               "e.g.: ./process input.png 0.1 output.png\n");
        return 0;
    }
    Image<float> input = load<float>(argv[1]);
    Image<float> output(input.width(),input.height(),input.channels());
    cout<<"Image Size : "<<input.width()<<" X "<<input.height()<<" X "<<input.channels()<<endl;
    float r_sigma = atof(argv[2]);

    StaticDispatch<float> fusion(bilateral_grid_cpu,bilateral_grid_gpu,input,output);



    testPerformance(CPU,r_sigma,(buffer_t*)(input),output);
    testPerformance(GPU,r_sigma,(buffer_t*)(input),output);
    testPerformance(Fus,r_sigma,input,output);

    fusion.realize(r_sigma,360);
    save(output, argv[3]);

    return 0;

}
