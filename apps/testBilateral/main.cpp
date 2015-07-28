#include <iostream>
#include "static_image.h"
#include "image_io.h"
#include "bilateral_grid_gpu.h"
#include "bilateral_grid_cpu.h"
#include "StaticDispatch.h"
#include "clock.h"
#include <limits>
#include <iomanip>
#ifdef ANDROID
#include <android/log.h>
#endif // ANDROID
#ifndef DBL_MAX
#define DBL_MAX 1.79769e+308
#endif
#include <unistd.h>
#include "testPerformance.h"
using namespace std;
using namespace Fusion::Static;
using namespace Fusion::Dynamic;
void testDynamic(buffer_t* input,buffer_t* output,float r_sigma);
void testStatic(buffer_t* input,buffer_t* output,float r_sigma,int workload);
void testGPU(buffer_t* input,buffer_t* output,float r_sigma);
int main(int argc,char** argv) {
    if (argc < 3) {
        printf("Usage: ./process input.png r_sigma output.png\n"
               "e.g.: ./process input.png 0.1 output.png\n");
        return 0;
    }
    int workload=50;
    if(argc>=5) {
        workload=atoi(argv[4]);
    }

    Image<float> input = load<float>(argv[1]);
    Image<float> output(input.width(),input.height(),input.channels());
    cout<<"Image Size : "<<input.width()<<" X "<<input.height()<<" X "<<input.channels()<<endl;
    float r_sigma = atof(argv[2]);



    int bufferWidth=((buffer_t*)output)->extent[1]*workload/100;
    buffer_t *tmpBuffer=Fusion::Internal::divBuffer(input,0,bufferWidth);
#ifndef ANDROID
    char opt;
    while((opt=getopt(argc,argv,"sdgnh"))!=-1) {
        switch(opt) {
        case 'n':
            Fusion::Test::testStaticPerformance(CPU,bilateral_grid_cpu,input,output,r_sigma);
            Fusion::Test::testStaticPerformance(GPU,bilateral_grid_gpu,input,output,r_sigma);

            Fusion::Test::testStaticPerformance(bilateral_grid_cpu,bilateral_grid_gpu,input,output,workload,r_sigma);
            Fusion::Test::testDynamicPerformance(bilateral_grid_cpu,bilateral_grid_gpu,input,output,r_sigma);

            Fusion::Test::testSizePerformance(CPU,bilateral_grid_cpu,input,output,workload,r_sigma);
            Fusion::Test::testSizePerformance(GPU,bilateral_grid_gpu,input,output,workload,r_sigma);
            break;
        case 'g':
#endif // ANDROID
            testGPU(input,tmpBuffer,r_sigma);
#ifndef ANDROID
            break;
        case 's':
            testStatic(input,output,r_sigma,workload);
            save_png(output,"test.png");
            break;
        case 'd':
            testDynamic(input,output,r_sigma);
            break;
        case 'h':
            cout<<"Usage:   "<<"run"<<" [-option] [argument]"<<endl;
            cout<<"option:  "<<"-h  show help information"<<endl;
            cout<<"         "<<"-s  test Static dispatch"<<endl;
            cout<<"         "<<"-d  test Dynamic dispatch"<<endl;
            cout<<"         "<<"-g  test gpuonly"<<endl;
            break;

        }


    }
#endif // ANDROID
//    StaticDispatch fusion(input,output);
//       DynamicDispatch fusion(input,output);
//    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,r_sigma);
//    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,workload,r_sigma);
//    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,50,r_sigma);
//    fusion.realize(bilateral_grid_gpu,r_sigma);
//    fusion.realize(bilateral_grid_gpu,r_sigma);
//    input.set_host_dirty(true);
//#ifdef ANDROID
//    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
//#else
//    cout<<"sec"<<endl;
//#endif // ANDROID
//    double t1=current_time();
//        fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,r_sigma);
//    fusion.realize(bilateral_grid_gpu,r_sigma);
//    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,workload,r_sigma);
//    cout<<current_time()-t1<<endl;
//#ifdef ANDROID
//    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
//#else
//    cout<<"sec"<<endl;
//#endif // ANDROID


//    save(output, argv[3]);

    return 0;

}
void testDynamic(buffer_t* input,buffer_t* output,float r_sigma) {
    DynamicDispatch dd(input,output);
    input->host_dirty=true;
    dd.realize(bilateral_grid_cpu,bilateral_grid_gpu,r_sigma);
    input->host_dirty=true;
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
    dd.realize(bilateral_grid_cpu,bilateral_grid_gpu,r_sigma);
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
}
void testGPU(buffer_t* input,buffer_t* output,float r_sigma) {
    StaticDispatch fusion(input,output);
    fusion.realize(bilateral_grid_gpu,r_sigma);
    input->host_dirty=true;
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
    fusion.realize(bilateral_grid_gpu,r_sigma);
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
}
void testStatic(buffer_t* input,buffer_t* output,float r_sigma,int workload) {
    StaticDispatch fusion(input,output);
    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,workload,r_sigma);
    input->host_dirty=true;
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
    fusion.realize(bilateral_grid_cpu,bilateral_grid_gpu,workload,r_sigma);
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
}
