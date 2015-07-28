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
#include <unistd.h>
#ifdef ANDROID
#include <android/log.h>
#endif // ANDROID

using namespace std;
using namespace Fusion::Static;
using namespace Fusion::Dynamic;
void testDynamic(buffer_t* input,buffer_t* output,int levels,float alpha,float beta);
void testStatic(buffer_t* input,buffer_t* output,int levels,float alpha,float beta,int workload);
void testGPU(buffer_t* input,buffer_t* output,int levels,float alpha,float beta);

int main(int argc,char** argv) {
    if (argc < 6) {
        printf("Usage: ./process input.png levels alpha beta output.png\n"
               "e.g.: ./process input.png 8 1 1 output.png\n");
        return 0;
    }
    int workload=50;
    if(argc>=7)
        workload=atoi(argv[6]);
    int levels = atoi(argv[2]);
    float alpha = atof(argv[3]), beta = atof(argv[4]);
    Image<uint16_t> input = load<uint16_t>(argv[1]);
    Image<uint16_t> output(input.width(),input.height(),input.channels());
    cout<<"Image Size : "<<input.width()<<" X "<<input.height()<<" X "<<input.channels()<<endl;
    int bufferWidth=((buffer_t*)output)->extent[1]*workload/100;
    buffer_t *tmpBuffer=Fusion::Internal::divBuffer(input,0,bufferWidth);
#ifndef ANDROID

    char opt;
    while((opt=getopt(argc,argv,"sdgnh"))!=-1) {
        switch(opt) {
        case 'n':
            Fusion::Test::testStaticPerformance(CPU,local_laplacian_cpu,input,output,levels,alpha/(levels-1),beta);
            Fusion::Test::testStaticPerformance(GPU,local_laplacian_gpu,input,output,levels,alpha/(levels-1),beta);

            Fusion::Test::testStaticPerformance(local_laplacian_cpu,local_laplacian_gpu,input,output,workload,levels,alpha/(levels-1),beta);
            Fusion::Test::testDynamicPerformance(local_laplacian_cpu,local_laplacian_gpu,input,output,levels,alpha/(levels-1),beta);
//
            Fusion::Test::testSizePerformance(CPU,local_laplacian_cpu,input,output,workload,levels,alpha/(levels-1),beta);

            Fusion::Test::testSizePerformance(GPU,local_laplacian_gpu,input,output,workload,levels,alpha/(levels-1),beta);

            break;
        case 'g':
#endif // ANDROID
            if(workload!=100)
                testGPU(input,tmpBuffer,levels,alpha/(levels-1),beta);
            else
                testGPU(input,output,levels,alpha/(levels-1),beta);
#ifndef ANDROID
            break;
        case 's':
            testStatic(input,output,levels,alpha/(levels-1),beta,workload);
            save_png(output,"test.png");
            break;
        case 'd':
            testDynamic(input,output,levels,alpha/(levels-1),beta);
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

//    Image<uint16_t> input = load<uint16_t>(argv[1]);
//    Image<uint16_t> output(input.width(),input.height(),input.channels());
//    cout<<"Image Size : "<<input.width()<<" X "<<input.height()<<" X "<<input.channels()<<endl;

//
//    Fusion::Test::testStaticPerformance(CPU,local_laplacian_cpu,input,output,levels,alpha/(levels-1),beta);
//    Fusion::Test::testStaticPerformance(GPU,local_laplacian_gpu,input,output,levels,alpha/(levels-1),beta);
//
//    Fusion::Test::testStaticPerformance(local_laplacian_cpu,local_laplacian_gpu,input,output,workload,levels,alpha/(levels-1),beta);
//    Fusion::Test::testDynamicPerformance(local_laplacian_cpu,local_laplacian_gpu,input,output,levels,alpha/(levels-1),beta);
//
//    Fusion::Test::testSizePerformance(CPU,local_laplacian_cpu,input,output,workload,levels,alpha/(levels-1),beta);
//    Fusion::Test::testSizePerformance(GPU,local_laplacian_gpu,input,output,workload,levels,alpha/(levels-1),beta);

    return 0;

}
void testDynamic(buffer_t* input,buffer_t* output,int levels,float alpha,float beta) {
    DynamicDispatch dd(input,output);
    input->host_dirty=true;
    dd.realize(local_laplacian_cpu,local_laplacian_gpu,levels,alpha/(levels-1),beta);
    input->host_dirty=true;
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
    dd.realize(local_laplacian_cpu,local_laplacian_gpu,levels,alpha/(levels-1),beta);
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
}
void testGPU(buffer_t* input,buffer_t* output,int levels,float alpha,float beta) {
    StaticDispatch fusion(input,output);
    fusion.realize(local_laplacian_gpu,levels,alpha/(levels-1),beta);
    input->host_dirty=true;
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
    fusion.realize(local_laplacian_gpu,levels,alpha/(levels-1),beta);
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
}
void testStatic(buffer_t* input,buffer_t* output,int levels,float alpha,float beta,int workload) {
    StaticDispatch fusion(input,output);
    fusion.realize(local_laplacian_cpu,local_laplacian_gpu,workload,levels,alpha/(levels-1),beta);
    input->host_dirty=true;
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
    fusion.realize(local_laplacian_cpu,local_laplacian_gpu,workload,levels,alpha/(levels-1),beta);
#ifdef ANDROID
    __android_log_print(ANDROID_LOG_INFO,"halide","sec\n");
#else
    cout<<"sec"<<endl;
#endif // ANDROID
}


