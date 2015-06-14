#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdint.h>

typedef struct buffer_t
{
    uint64_t dev;
    uint8_t* host;
    int32_t extent[4];
    int32_t stride[4];
    int32_t min[4];
    int32_t elem_size;
    bool host_dirty;
    bool dev_dirty;
} buffer_t;
#endif
#ifndef WORKLOAD_THRESHOULD
#define WORKLOAD_THRESHOULD 1.0
#endif // WORKLOAD_THRESHOULD

#ifdef ANDROID
#define fusion_printf(...) __android_log_print(ANDROID_LOG_DEBUG,"fusion_debug",__VA_ARGS__)
#else
#define fusion_printf(...) printf("%s\n",__VA_ARGS__ );
#endif

#ifndef FUSION_H
#define FUSION_H

#include <cmath>
#include <iostream>
#include <utility>
#include <thread>
#include <assert.h>
#include <string>
#include <mutex>
#include "fusion_info.h"
#include "internal.h"
#include "clock.h"
#include "HalideRuntime.h"
#include "HalideRuntimeOpenCL.h"
using namespace std;

#ifdef ANDROID
#include <string>
#include <sstream>
template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}
#endif

extern "C" int halide_copy_to_host(void* user_context, buffer_t* buf);
namespace Fusion
{

#ifdef DEBUG
static double exe_time_cpu,exe_time_gpu;
#endif



//
///**
//    TODO:find out a way to interrupt cpu work thread
//
//*/
//template<typename ...Args>
//void workThread(Args ...args,function<int(Args...,buffer_t*,buffer_t*)>  func,buffer_t* input ,buffer_t *output,status table[],int offset,mutex* table_mutex)
//{
//    int end=output->extent[1];
//    int start=offset*9;
//    bool bBreak=false;
//    for(int i=9; i>=0; i--)
//    {
//        table_mutex->lock();
//        if(table[i]!=idle)
//            bBreak=true;
//        else
//            table[i]=computing;
//        table_mutex->unlock();
//        if(bBreak)
//            break;
//        buffer_t* buf=Internal::divBuffer(output,start,end);
//        func(forward<Args>(args)...,input,buf);
//
//        table_mutex->lock();
//        if(table[i]==finished)
//            bBreak=true;
//        else
//            table[i]=writing;
//        table_mutex->unlock();
//        if(bBreak)
//            break;
//        halide_copy_to_host(NULL,buf);
//        delete buf;
//        table_mutex->lock();
//        table[i]=finished;
//        table_mutex->unlock();
//        end=offset*i;
//        start=offset*(i-1);
//#ifdef DEBUG
//        fusion_printf("GPU Workload %d0%",10-i);
//#endif
//    }
//}
//
//template<typename ...Args>
//void gpuThreadWithTime(Args ...args,function<int(Args...,buffer_t*,buffer_t*)>  func,buffer_t* input ,buffer_t *output,double* time)
//{
//    double t1=current_time();
//    func(forward<Args>(args)...,input,output);
//    halide_copy_to_host(NULL,output);
//    double t2=current_time();
//    *time=(t2-t1);
//}
//
//
//template<typename ...Args>
//class Fusions
//{
//public:
//    Fusions() {}
//    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc)
//    {
//        input=NULL;
//        initFunc();
//        output=NULL;
//
//    }
//    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input)
//    {
//        output=NULL;
//        initFunc();
//
//
//    }
//    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input,buffer_t* _output) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input),output(_output)
//    {
//        initFunc();
//
//    }
//    int autoRealize(Args...);
//    int realizeCPU(Args...);
//    int realizeGPU(Args...);
//    int realizeCPU(Args...,buffer_t* _output);
//    int realizeGPU(Args...,buffer_t* _output);
//    int testSize(Args...);
//    int realizeWithStealing (Args... args,int s);
//    void setInput(buffer_t* _input)
//    {
//        input=_input;
//    }
//    void setOutput(buffer_t* _output)
//    {
//        output=_output;
//    }
//
//    int realize(Args...,int s);
//    int realize(Args...);
//    mutex *table_mutex;
//private:
//    function<int(Args...,buffer_t*,buffer_t*)> cpuFunc,gpuFunc;
//
//    inline void initFunc()
//    {
//        for(int i=0; i<2; i++)
//        {
//            for(int j=0; j<2; j++)
//            {
//                CPUTime[i][j]=-1.0;
//                GPUTime[i][j]=-1.0;
//            }
//            bFuncOK[i]=false;
//        }
//        CPUWorkload=-1;
//        bCPU=0;
//        bGPU=0;
//        table_mutex=new mutex();
//
//    }
//    buffer_t *input,*output;
//
//    int CPUWorkload;
//
//    buffer_t* createBuffers(int x,int y,int z,int w,int s);
//
//    double CPUTime[2][2],GPUTime[2][2],cpufuncFactor[2],gpufuncFactor[2];
//    bool bFuncOK[2];
//    int bCPU,bGPU;
//};
//
//template <typename ...Args>
//int Fusions<Args...>::realizeCPU(Args... args,buffer_t* _output)
//{
//    cpuFunc(forward<Args>(args)...,args...,input,_output);
//    return 0;
//}
//
//template <typename ...Args>
//int Fusions<Args...>::realizeGPU(Args... args,buffer_t* _output)
//{
//    gpuFunc(forward<Args>(args)...,input,_output);
//    halide_copy_to_host(NULL,output);
//    return 0;
//}
//
//template <typename ...Args>
//int Fusions<Args...>::realizeGPU(Args... args)
//{
//#ifdef DEBUG
//    double t1=current_time();
//#endif
//
//    gpuFunc(forward<Args>(args)...,input,output);
//
//    halide_copy_to_host(NULL,output);
//#ifdef DEBUG
//    double t2=current_time();
//    exe_time_cpu=t2-t1;
//    double fps=1000/exe_time_cpu;
//
//    fusion_printf("GPU  (FPS) : %d %d\n",exe_time_cpu,fps);
//#endif
//
//    return 0;
//}
//
//template <typename ...Args>
//int Fusions<Args...>::realizeCPU(Args... args)
//{
//
//#ifdef DEBUG
//    double t1=current_time();
//#endif
//
//    cpuFunc(forward<Args>(args)...,input,output);
//
//#ifdef DEBUG
//    double t2=current_time();
//    exe_time_cpu=t2-t1;
//    double fps=1000/exe_time_cpu;
//    string str="CPU  (FPS) : "+to_string(exe_time_cpu)+"   ( "+to_string(fps)+ " ) ";
//    fusion_printf("CPU  (FPS) : %d (%d)\n",exe_time_cpu,fps);
//#endif
//
//    return 0;
//}
//
//
//template <typename ...Args>
//int Fusions<Args...>::realize(Args... args,int s)
//{
//    if(output==NULL)
//        return -1;
//    buffer_t* cpuBuf=Internal::divBuffer(output,0,s);
//    buffer_t* gpuBuf=Internal::divBuffer(output,s,input->extent[1]);
//    thread gputThread(gpuThread<args...>,forward<Args>(args)...,gpuFunc,input,gpuBuf);
//#ifdef DEBUG
//    double t1=current_time();
//#endif
//    cpuFunc(forward<Args>(args)...,input,cpuBuf);
//#ifdef DEBUG
//    double t2=current_time();
//    exe_time_cpu=t2-t1;
//#endif
//    gputThread.join();
//#ifdef DEBUG
//    double fps=1000/max(exe_time_gpu,exe_time_cpu);
//
//    fusion_printf("CPU V.S GPU (FPS) : %.f %.f (%.f)\n",exe_time_cpu,exe_time_gpu,fps);
//#endif
//    return 0;
//
//
//}
//
//
//template <typename ...Args>
//int Fusions<Args...>::realizeWithStealing (Args... args,int s)
//{
//    #ifdef DEBUG
//    double t1=current_time();
//    #endif
//    if(output==NULL)
//        return -1;
//    buffer_t* cpuBuf=Internal::divBuffer(output,0,s);
//    buffer_t* gpuBuf=Internal::divBuffer(output,s,input->extent[1]);
//    status table[10]= {idle};
//    int offset=floor(cpuBuf->extent[1]/10);
//    thread gputThread(gpuStealing<args...>,forward<Args>(args)...,gpuFunc,input,cpuBuf,gpuBuf,table,offset,table_mutex);
//
//    bool bBreak=false;
//    for(int i=0; i<=9; i++)
//    {
//        table_mutex->lock();
//        if(table[i]>idle)
//        {
//            bBreak=true;
//        }
//        table[i]=computing;
//        table_mutex->unlock();
//        if(bBreak)
//            break;
//        buffer_t* buf=Internal::divBuffer(output,i*offset,(i+1)*offset);
//
//        cpuFunc(args...,input,buf);
//        delete buf;
//        table_mutex->lock();
//        table[i]=finished;
//        table_mutex->unlock();
//    }
//    #ifdef DEBUG
//    double t2=current_time();
//    exe_time_cpu=t2-t1;
//    #endif
//    gputThread.join();
//    #ifdef DEBUG
//    double fps=1000/max(exe_time_gpu,exe_time_cpu);
//
//    fusion_printf("CPU V.S GPU (FPS) : %.f %.f (%.f)\n",exe_time_cpu,exe_time_gpu,fps);
//    #endif
//    return 0;
//
//
//}
//
//template <typename ...Args>
//int Fusions<Args...>::realize(Args... args)
//{
//#ifdef DEBUG
//    string str;//use string variaty one string to show in android log
//#endif
//
//    if(output==NULL)
//        return -1;
//    double   t1,t2;
//    if(CPUTime[0][0]==-1)
//    {
//        CPUTime[0][0]=output->extent[1];
//        t1=current_time();
//        realizeCPU(forward<Args>(args)...,output);
//        t2=current_time();
//        CPUTime[0][1]=t2-t1;
//        return 0;
//    }
//    if(GPUTime[0][0]==-1)
//    {
//        GPUTime[0][0]=output->extent[1];
//        t1=current_time();
//        realizeGPU(forward<Args>(args)...,output);
//        t2=current_time();
//        GPUTime[0][1]=t2-t1;
//        return 1;
//    }
//    if(CPUTime[1][0]==-1)
//    {
//        buffer_t* cpuBuf=Internal::divBuffer(output,0,output->extent[1]/2);
//        CPUTime[1][0]=floor(output->extent[1]/2);
//        t1=current_time();
//        realizeCPU(forward<Args>(args)...,cpuBuf);
//        t2=current_time();
//        CPUTime[1][1]=t2-t1;
//
//        buffer_t* gpuBuf=Internal::divBuffer(output,(output->extent[1]/2)+1,output->extent[1]);
//        GPUTime[1][0]=floor(output->extent[1]/2);
//        t1=current_time();
//        realizeGPU(forward<Args>(args)...,gpuBuf);
//        t2=current_time();
//        GPUTime[1][1]=t2-t1;
//        return 2;
//    }
//    if(bGPU>500)
//    {
//
//        fusion_printf("Use GPU only\n");
//        realizeGPU(forward<Args>(args)...);
//        return 3;
//    }
//    if(bCPU>500)
//    {
//        fusion_printf("Use CPU only\n");
//        realizeCPU(forward<Args>(args)...);
//        return 4;
//    }
//
//    //Solve CPU Workload
////    if(CPUWorkload==-1)
////    {
//    double tmpCPUTime[2][2],tmpGPUTime[2][2];
//    for(int i=0; i<2; i++)
//    {
//        for (int j=0; j<2; j++)
//        {
//            tmpCPUTime[i][j]=CPUTime[i][j];
//            tmpGPUTime[i][j]=GPUTime[i][j];
//        }
//    }
//
//    cpufuncFactor[0]=((CPUTime[0][1]-CPUTime[1][1]))/(CPUTime[0][0]-CPUTime[1][0]);
//
//    gpufuncFactor[0]=((GPUTime[0][1]-GPUTime[1][1]))/(GPUTime[0][0]-GPUTime[1][0]);
//
//
//
//    cpufuncFactor[1]=CPUTime[0][1]-(CPUTime[0][0]*cpufuncFactor[0]);
//    gpufuncFactor[1]=GPUTime[0][1]-(GPUTime[0][0]*gpufuncFactor[0]);
//
//    assert(cpufuncFactor[0]+gpufuncFactor[0]!=0);
//
//    int workload=(output->extent[1]*gpufuncFactor[0]+gpufuncFactor[1]-cpufuncFactor[1])/(cpufuncFactor[0]+gpufuncFactor[0]);
//    float exe_time=cpufuncFactor[0]*CPUWorkload+gpufuncFactor[1];
//
//
//    if(workload<=0)
//    {
//
//        //reset CPU and GPU time
//        fusion_printf("Use GPU Only\n");
//        bGPU++;
//        for(int i=0; i<2; i++)
//        {
//            for (int j=0; j<2; j ++)
//            {
//                CPUTime[i][j]=tmpCPUTime[i][j];
//                GPUTime[i][j]=tmpGPUTime[i][j];
//            }
//        }
//        realizeGPU(forward<Args>(args)...,output);
//        return 33;
//    }
//
//    else if(workload>output->extent[1])
//    {
//        bCPU++;
//        fusion_printf("Use CPU Only");
//        for(int i=0; i<2; i++)
//        {
//            for (int j=0; j<2; j++)
//            {
//                CPUTime[i][j]=-1;
//                GPUTime[i][j]=-1;
//            }
//        }
//        realizeCPU(forward<Args>(args)...,output);
//        return workload;
//
//    }
//    CPUWorkload=workload;
//
//
//    buffer_t* cpuBuf=Internal::divBuffer(output,0,CPUWorkload);
//    buffer_t* gpuBuf=Internal::divBuffer(output,CPUWorkload,input->extent[1]);
//    double _gputime;
//    thread gputThread(gpuThreadWithTime<args...>,gpuFunc,input,gpuBuf,&_gputime);
//
//    t1=current_time();
//    realizeCPU(forward<Args>(args)...,cpuBuf);
//    t2=current_time();
//    double _cputime=t2-t1;
//
//
//    gputThread.join();
//
//#ifdef DEBUG
//    double fps=1000/max(_gputime,_cputime);
//
//    fusion_printf("CPU V.S GPU (FPS) : %d %d (%d)\n",_cputime,_gputime,fps);
//#endif
//
//
////    if(CPUWorkload>CPUTime[1][0])
////    {
////        CPUTime[0][0]=CPUWorkload;
////        CPUTime[0][1]=_cputime;
////        if((output->extent[1]-CPUWorkload)>GPUTime[1][0])
////        {
////            GPUTime[0][0]=output->extent[1]-CPUWorkload;
////            GPUTime[0][1]=_gputime;
////        }
////        else
////        {
////            GPUTime[1][0]=output->extent[1]-CPUWorkload;
////            GPUTime[1][1]=_gputime;
////        }
////    }
////    else
////    {
////        CPUTime[1][0]=CPUWorkload;
////        CPUTime[1][1]=_cputime;
////        if((output->extent[1]-CPUWorkload)>GPUTime[1][0])
////        {
////            GPUTime[0][0]=output->extent[1]-CPUWorkload;
////            GPUTime[0][1]=_gputime;
////        }
////        else
////        {
////            GPUTime[1][0]=output->extent[1]-CPUWorkload;
////            GPUTime[1][1]=_gputime;
////        }
////    }
//    CPUTime[1][0]=CPUWorkload;
//    CPUTime[1][1]=_cputime;
//    GPUTime[1][0]=output->extent[1]-CPUWorkload;
//    GPUTime[1][1]=_gputime;
//
//
//    return CPUWorkload;
//
//
//}
//
//
//#define cleanBufferDev(buffer) \
//buffer->dev=0; \
//buffer->dev_dirty=false; \
//buffer->host_dirty=false;
//
//
//
//template <typename ...Args>
//int Fusions<Args...>::testSize(Args... args)
//{
//    double cputime[2],gputime[2];
//    buffer_t* halfBuf=Internal::divBuffer(output,0,(output->extent[1]/2));
////    warm up
//
//    gpuFunc(forward<Args>(args)...,input,output);
//    cleanBufferDev(output);
//    gpuFunc(forward<Args>(args)...,input,halfBuf);
//    cleanBufferDev(halfBuf);
//    double t1, t2,t3;
//
//
//    //    calculate gpu time with full size
//
//    t1=current_time();
//    gpuFunc(forward<Args>(args)...,input,output);
//    t2=current_time();
//    halide_copy_to_host(NULL,output);
//    t3=current_time();
//
//    gputime[0]=t3-t1;
////    gpuCopyTime[0]=t3-t2;
//    cleanBufferDev(output);
//
////    calculate gpu time with half size
//
//    t1=current_time();
//    gpuFunc(forward<Args>(args)...,input,halfBuf);
//    t2=current_time();
//
//    halide_copy_to_host(NULL,halfBuf);
//
//    t3=current_time();
//    gputime[1]=t3-t1;
////    gpuCopyTime[1]=t3-t2;
//    cleanBufferDev(halfBuf);
//
//    //warm up
//    cpuFunc(forward<Args>(args)...,input,output);
//
//
//    //    calculate cpu time with full size
//
//    t1=current_time();
//    cpuFunc(forward<Args>(args)...,input,output);
//    t2=current_time();
//    cputime[0]=double(t2-t1);
//
////    calculate cpu time with half size
//
//    t1=current_time();
//    cpuFunc(forward<Args>(args)...,input,halfBuf);
//    t2=current_time();
//    cputime[1]=double(t2-t1);
//
//
//
//#ifdef DEBUG
//    cout<<"cpu execution Time :"<<cputime[0]<<" "<<cputime[1]<<endl;
//    cout<<"gpu execution time :"<<gputime[0]<<" "<< gputime[1]<<endl;
//#endif // DEBUG
//    //Solve the equation
//    double a1,a2,b1,b2,x1,x2,a3,b3;
//    a1=((cputime[0]-cputime[1])*2)/input->extent[1];
//    a2=((gputime[0]-gputime[1])*2)/input->extent[1];
//
////    a3=((gpuCopyTime[0]-gpuCopyTime[1])*2)/input->extent[1];
//
//    b1=cputime[0]-(input->extent[1]*a1);
//    b2=gputime[0]-(input->extent[1]*a2);
//#ifdef DEBUG
//    fusion_printf("CPU Function %d %d\n",a1,b1);
//    fusion_printf("GPU Function %d %d\n",a2,b2);
//#endif // DEBUG
//    assert(a1+a2!=0);
//    x1=(input->extent[1]*a2+b2-b1)/(a1+a2);
//    float exe_time=a1*x1+b1;
//    x2=(exe_time-b2)/a2;
//    float copy_time=a3*x2+b3;
//    if(copy_time<0)
//        copy_time=0;
////    float copy_time=a3*x2+b3;
//    fusion_printf("estimate execution time %d\n",exe_time);
//    if(x1<=0.0)
//    {
//        fusion_printf("The GPU workload should be 100%\n");
//        return 0;
//    }
//    if(x2<=0.0)
//    {
//        fusion_printf("The CPU workload should be 100%\n");
//        return 0;
//    }
//    float improve=((min(cputime[0],gputime[0])/(exe_time))-1.0)*100;
//    if(improve<=WORKLOAD_THRESHOULD)
//    {
//        if(cputime[0]<gputime[0])
//        {
//            cout<<"The CPU workload should be 100%";
//        }
//        else if(cputime[0]>gputime[0])
//        {
//
//            cout<<"The GPU workload should be 100%";
//        }
//        else
//            cout<<"You can choose any device you want";
//        cout<<" , because the improvement is less then " <<WORKLOAD_THRESHOULD<<" % "<<endl;
//        return 0;
//    }
//    cout<<"The CPU workload should be "<<x1<<endl;
//    cout<<"By the assignment,We can save "<<improve<<"% execution time"<<endl;
//    return x1;
//}
//template <typename ...Args>
//int Fusions<Args...>::autoRealize(Args... args)
//{
//    status table[10]= {idle};
//    int offset=floor(input->extent[1]/10);
//    table_mutex->unlock();
//    thread gpuWorkThread(workThread<args...>,gpuFunc,input,output,table,offset,table_mutex);
//    bool bBreak=false;
//    for(int i=0; i<=9; i++)
//    {
//        table_mutex->lock();
//        if(table[i]>idle)
//        {
//            bBreak=true;
//        }
//        table[i]=computing;
//        table_mutex->unlock();
//        if(bBreak)
//            break;
//        buffer_t* buf=Internal::divBuffer(output,i*offset,(i+1)*offset);
//
//        cpuFunc(args...,input,buf);
//        delete buf;
//        table_mutex->lock();
//        table[i]=finished;
//        table_mutex->unlock();
//    }
//    gpuWorkThread.join();
//
//}

}
#endif // FUSION_H

