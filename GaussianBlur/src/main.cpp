#include <static_image.h>
#include <image_io.h>
#include "gaussinBlur_cpu.h"
#include "gaussinBlur_gpu.h"
#include "halide_gray.h"
#include <fusion.h>
#include <iostream>

#include <iomanip>
#include "clock.h"
#ifndef DBL_MAX
#define DBL_MAX 1.79769e+308
#endif
#define CPU 1
#define GPU 2
#define Fus 3
inline void testPerformance(int type,buffer_t* input,buffer_t* output)
{
    Fusion::Fusions<> fusion(gaussinBlur_cpu,gaussinBlur_gpu,input,output);

    double bestT =DBL_MAX;
    double worstT=0;
    switch (type)
    {
    case CPU:
        fusion.realizeCPU();
        for (int i = 0; i < 5; i++)
        {
            double t1=current_time();
            fusion.realizeCPU();
            double t2=current_time();
            double t=t2-t1;
            if (t < bestT) bestT = t;
            if (t > worstT) worstT = t;
        }
        cout<<setw(15)<<"Best CPU: "<<setw(10)<<bestT<<setw(15)<<" Worst CPU: "<<setw(10)<<worstT<<endl;
        break;
    case GPU:
        fusion.realizeGPU();
        for (int i = 0; i < 5; i++)
        {
            double t1=current_time();
            fusion.realizeGPU();
            double t2=current_time();
            double t=t2-t1;
            if (t < bestT) bestT = t;
            if (t > worstT) worstT = t;
        }
        cout<<setw(15)<<"Best GPU: "<<setw(10)<<bestT<<setw(15)<<" Worst GPU: "<<setw(10)<<worstT<<endl;
        break;
    case Fus:
        fusion.realize(1309);
        for (int i = 0; i < 5; i++)
        {
            double t1=current_time();
            fusion.realize(1515);
            double t2=current_time();
            double t=t2-t1;
            if (t < bestT) bestT = t;
            if (t > worstT) worstT = t;
        }
        cout<<setw(15)<<"Best Fusion: "<<setw(10)<<bestT<<setw(15)<<" Worst Fusion: "<<setw(10)<<worstT<<endl;
        break;
    }
}
int main(int argc,char** argv)
{
    char *filename="rgb.png";
    if(argc>1)
        filename=argv[1];

    Image<uint16_t>input=load<uint16_t>(filename);

    Image<uint16_t> gray(input.width(),input.height());
//     Image<uint16_t> gray2(input.width(),input.height());
    Image<uint16_t> output(input.width(),input.height());
    halide_gray((buffer_t*)input,gray);
    Fusion::Fusions<> fusion(gaussinBlur_cpu,gaussinBlur_gpu,gray,output);
    testPerformance(CPU,gray,output);
    testPerformance(GPU,gray,output);
    testPerformance(Fus,gray,output);
//    gray.copy_to_host();
    fusion.testSize();
    for(int i=0; i<100; i++)
    {
        double t1=current_time();
        fusion.realize();
        double t2=current_time();
        cout<<"execution time : "<<t2-t1<<"\n";
    }
    cout<<endl;

//    gaussinBlur_gpu(gray,output);
//    halide_copy_to_host(NULL,output);
//    save_png(output,"out.png");



    return 0;
}
