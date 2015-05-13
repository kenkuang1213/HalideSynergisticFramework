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


#ifndef FUSION_H
#define FUSION_H


#include <cmath>
#include <iostream>
#include <utility>
#include <thread>
#include <assert.h>
#include "clock.h"
using namespace std;

extern "C" int halide_copy_to_host(void* user_context, buffer_t* buf);
namespace Fusion
{
template<typename ...Args>
void gpuThread(Args ...args,function<int(Args...,buffer_t*,buffer_t*)>  func,buffer_t* input ,buffer_t *output)
{
    func(forward<Args>(args)...,input,output);
    halide_copy_to_host(NULL,output);

}

template<typename ...Args>
void gpuThreadWithTime(Args ...args,function<int(Args...,buffer_t*,buffer_t*)>  func,buffer_t* input ,buffer_t *output,double* time)
{
    double t1=current_time();
    func(forward<Args>(args)...,input,output);
    halide_copy_to_host(NULL,output);
    double t2=current_time();
    *time=(t2-t1);
}


template<typename ...Args>
class Fusions
{
public:
    Fusions() {}
    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc)
    {
        input=NULL;
        initFunc();
        output=NULL;
        CPUWorkload=-1;
        bCPU=0;
        bGPU=0;
    }
    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input)
    {
        output=NULL;
        initFunc();
        bitOfInput=input->elem_size;
        CPUWorkload=-1;
        bCPU=0;
        bGPU=0;
    }
    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input,buffer_t* _output) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input),output(_output)
    {
        initFunc();
        bitOfInput=input->elem_size;
        CPUWorkload=-1;
        bCPU=0;
        bGPU=0;
    }

    int realizeCPU(Args...);
    int realizeGPU(Args...);
    int realizeCPU(Args...,buffer_t* _output);
    int realizeGPU(Args...,buffer_t* _output);
    int testSize(Args...);
    void setInput(buffer_t* _input)
    {
        input=_input;
    }
    void setOutput(buffer_t* _output)
    {
        output=_output;
    }

    int realize(Args...,int s);
    int realize(Args...);
private:
    function<int(Args...,buffer_t*,buffer_t*)> cpuFunc,gpuFunc;
    inline void initFunc()
    {
        for(int i=0; i<2; i++)
        {
            for(int j=0; j<2; j++)
            {
                CPUTime[i][j]=-1.0;
                GPUTime[i][j]=-1.0;
            }
            bFuncOK[i]=false;
        }
    }
    buffer_t *input,*output;
    int bitOfInput;
    int CPUWorkload;
    uint8_t* createBuffer_t(int x,int y,int z,int w,buffer_t* buf);
    buffer_t* createBuffers(int x,int y,int z,int w,int s);
    buffer_t* divBuffer(buffer_t *buf,int start,int nend) ;
    double CPUTime[2][2],GPUTime[2][2],cpufuncFactor[2],gpufuncFactor[2];
    bool bFuncOK[2];
    int bCPU,bGPU;


};
template<typename ...Args>
uint8_t* Fusions<Args...>::createBuffer_t(int x,int y,int z,int w,buffer_t* buf)
{
    buf->extent[0] = x;
    buf->extent[1] = y;
    buf->extent[2] = z;
    buf->extent[3] = w;
    buf->stride[0] = 1;
    buf->stride[1] = x;
    buf->stride[2] = x*y;
    buf->stride[3] = x*y*z;
    buf->elem_size = bitOfInput;
    size_t size = 1;
    if (x) size *= x;
    if (y) size *= y;
    if (z) size *= z;
    if (w) size *= w;
    uint8_t* ptr = new uint8_t[bitOfInput*size + 40];
    buf->host = ptr;
    buf->dev = 0;
    buf->host_dirty = false;
    buf->dev_dirty = false;
    buf->min[0] = 0;
    buf->min[1] = 0;
    buf->min[2] = 0;
    buf->min[3] = 0;
    while ((size_t)buf->host & 0x1f) buf->host++; // Memory Alias
    return ptr;
}

//Return A split buffer_t pointer
template <typename ...Args>
buffer_t* Fusions<Args...>::divBuffer(buffer_t *buf,int start,int nend)
{
    buffer_t *buff= new buffer_t();
    buff->extent[0] = buf->extent[0];
    buff->extent[1] = nend-start;
    if(buff->extent[2])
        buff->extent[2] = buf->extent[2];
    if(buff->extent[3])
        buff->extent[3] = buf->extent[3];
    buff->stride[0] = buf->stride[0];
    buff->stride[1] = buf->stride[1];
    if(buff->extent[2])
        buff->stride[2] = buf->stride[2];
    if(buff->extent[3])
        buff->stride[3] = buf->stride[3];
    buff->elem_size = buf->elem_size;
    buff->host_dirty = true;
    buff->dev_dirty = false;
    buff->dev = 0;
    if(start==0)
    {
        buff->host = buf->host;
        return buff;
    }
    int offset=buf->extent[0]*start*buf->elem_size*1;
    buff->min[1]=start;
    buff->host= buf->host+offset;
    return buff;
}


template <typename ...Args>
int Fusions<Args...>::realizeCPU(Args... args,buffer_t* _output)
{

//    input->host_dirty=true;
//    if(_output==NULL)
//        return -1;
    cpuFunc(forward<Args>(args)...,input,_output);
    return 0;
}

template <typename ...Args>
int Fusions<Args...>::realizeGPU(Args... args,buffer_t* _output)
{

//    input->host_dirty=true;
//    if(_output==NULL)
//        return -1;
    gpuFunc(forward<Args>(args)...,input,_output);
    halide_copy_to_host(NULL,output);
    return 0;
}

template <typename ...Args>
int Fusions<Args...>::realizeGPU(Args... args)
{

//    input->host_dirty=true;
//    if(output==NULL)
//        return -1;
    gpuFunc(forward<Args>(args)...,input,output);

    halide_copy_to_host(NULL,output);
    return 0;
}

template <typename ...Args>
int Fusions<Args...>::realizeCPU(Args... args)
{

//    input->host_dirty=true;
//    if(output==NULL)
//        return -1;
    gpuFunc(forward<Args>(args)...,input,output);


    return 0;
}


template <typename ...Args>
int Fusions<Args...>::realize(Args... args,int s)
{
    if(output==NULL)
        return -1;
    buffer_t* cpuBuf=divBuffer(output,0,s);
    buffer_t* gpuBuf=divBuffer(output,s,input->extent[1]);
    thread gputThread(gpuThread<args...>,gpuFunc,input,gpuBuf);

    cpuFunc(forward<Args>(args)...,input,cpuBuf);


    gputThread.join();
    return 0;
//    halide_copy_to_host(NULL,gpuBuf);

}

template <typename ...Args>
int Fusions<Args...>::realize(Args... args)
{
    if(output==NULL)
        return -1;
    double   t1,t2;
    if(CPUTime[0][0]==-1)
    {
        CPUTime[0][0]=output->extent[1];
        t1=current_time();
        realizeCPU(forward<Args>(args)...);
        t2=current_time();
        CPUTime[0][1]=t2-t1;
        return 0;
    }
    if(GPUTime[0][0]==-1)
    {
        GPUTime[0][0]=output->extent[1];
        t1=current_time();
        realizeGPU(forward<Args>(args)...);
        t2=current_time();
        GPUTime[0][1]=t2-t1;
        return 0;
    }
    if(CPUTime[1][0]==-1)
    {
        buffer_t* cpuBuf=divBuffer(output,0,output->extent[1]/2);
        CPUTime[1][0]=floor(output->extent[1]/2);
        t1=current_time();
        realizeCPU(forward<Args>(args)...,cpuBuf);
        t2=current_time();
        CPUTime[1][1]=t2-t1;

        buffer_t* gpuBuf=divBuffer(output,(output->extent[1]/2)+1,output->extent[1]);
        GPUTime[1][0]=floor(output->extent[1]/2);
        t1=current_time();
        realizeGPU(forward<Args>(args)...,gpuBuf);
        t2=current_time();
        GPUTime[1][1]=t2-t1;
        return 0;
    }
    if(bGPU>10)
    {
        cout<<"Use GPU only"<<endl;
        gpuFunc(forward<Args>(args)...,input,output);
        return 0;
    }
    if(bCPU>10)
    {
        cout<<"Use CPU only"<<endl;
        cpuFunc(forward<Args>(args)...,input,output);
        return 0;
    }

    //Solve CPU Workload
//    if(CPUWorkload==-1)
//    {
    double tmpCPUTime[2][2],tmpGPUTime[2][2];
    for(int i=0; i<2; i++)
    {
        for (int j=0; j<2; j++)
        {
            tmpCPUTime[i][j]=CPUTime[i][j];
            tmpGPUTime[i][j]=GPUTime[i][j];
        }
    }
#ifdef DEBUG
    cout<<"CPU exectuion time: "<<CPUTime[0][1] <<" "<<CPUTime[1][1]<<" "<<endl;
    cout<<"GPU exectuion time: "<<GPUTime[0][1] <<" "<<GPUTime[1][1]<<" "<<endl;
#endif // DEBUG
    cpufuncFactor[0]=((CPUTime[0][1]-CPUTime[1][1]))/(CPUTime[0][0]-CPUTime[1][0]);

    gpufuncFactor[0]=((GPUTime[0][1]-GPUTime[1][1]))/(GPUTime[0][0]-GPUTime[1][0]);

//    a3=((gpuCopyTime[0]-gpuCopyTime[1])*2)/input->extent[1];

    cpufuncFactor[1]=CPUTime[0][1]-(CPUTime[0][0]*cpufuncFactor[0]);
    gpufuncFactor[1]=GPUTime[0][1]-(GPUTime[0][0]*gpufuncFactor[0]);

    assert(cpufuncFactor[0]+gpufuncFactor[0]!=0);
#ifdef DEBUG
    cout<<"CPU Function"<<cpufuncFactor[0]<<" "<<cpufuncFactor[1]<<endl;
    cout<<"GPU Function"<<gpufuncFactor[0]<<" "<<gpufuncFactor[1]<<endl;
#endif // DEBUG
    int workload=(output->extent[1]*gpufuncFactor[0]+gpufuncFactor[1]-cpufuncFactor[1])/(cpufuncFactor[0]+gpufuncFactor[0]);
//          x1=(input->extent[1]*a2+b2-b1)/(a1+a2);
    float exe_time=cpufuncFactor[0]*CPUWorkload+gpufuncFactor[1];


    if(workload<=0)
    {

        //reset CPU and GPU time
        cout<<"Use GPU Only"<<endl;
        bGPU++;
        for(int i=0; i<2; i++)
        {
            for (int j=0; j<2; j ++)
            {
                CPUTime[i][j]=tmpCPUTime[i][j];
                GPUTime[i][j]=tmpGPUTime[i][j];
            }
        }
        gpuFunc(forward(args)...,input,output);
        return 0;
    }

    else if(workload>output->extent[1])
    {
        bCPU++;
        cout<<"Use CPU Only"<<endl;
        for(int i=0; i<2; i++)
        {
            for (int j=0; j<2; j++)
            {
                CPUTime[i][j]=tmpCPUTime[i][j];
                GPUTime[i][j]=tmpGPUTime[i][j];
            }
        }
        cpuFunc(forward(args)...,input,output);
        return 0;

    }
    CPUWorkload=workload;
    cout<<"Workload "<<CPUWorkload<<endl;
//        #ifdef DEBUG
//    cout<<"\r CPU Workload:"<<workload<<endl;
//        #endif // DEBUG
//        realize(forward<Args>(args)...,CPUWorkload);

    buffer_t* cpuBuf=divBuffer(output,0,CPUWorkload);
    buffer_t* gpuBuf=divBuffer(output,CPUWorkload+1,input->extent[1]);
    double _gputime;
//        CPUWorkload=workload;
    thread gputThread(gpuThreadWithTime<args...>,gpuFunc,input,gpuBuf,&_gputime);

    t1=current_time();
    cpuFunc(forward<Args>(args)...,input,cpuBuf);
    t2=current_time();
    double _cputime=t2-t1;


    gputThread.join();
//    if(CPUWorkload>CPUTime[1][0])
//    {
//        CPUTime[0][0]=CPUWorkload;
//        CPUTime[0][1]=_cputime;
//        if((output->extent[1]-CPUWorkload)>GPUTime[1][0])
//        {
//            GPUTime[0][0]=output->extent[1]-CPUWorkload;
//            GPUTime[0][1]=_gputime;
//        }
//        else
//        {
//            GPUTime[1][0]=output->extent[1]-CPUWorkload;
//            GPUTime[1][1]=_gputime;
//        }
//    }
//    else
//    {
//        CPUTime[1][0]=CPUWorkload;
//        CPUTime[1][1]=_cputime;
//        if((output->extent[1]-CPUWorkload)>GPUTime[1][0])
//        {
//            GPUTime[0][0]=output->extent[1]-CPUWorkload;
//            GPUTime[0][1]=_gputime;
//        }
//        else
//        {
//            GPUTime[1][0]=output->extent[1]-CPUWorkload;
//            GPUTime[1][1]=_gputime;
//        }
//    }
    CPUTime[1][0]=CPUWorkload;
    CPUTime[1][1]=_cputime;
    GPUTime[1][0]=output->extent[1]-CPUWorkload;
    GPUTime[1][1]=_gputime;


    return CPUWorkload;


}


#define cleanBufferDev(buffer) \
buffer->dev=0; \
buffer->dev_dirty=false; \
buffer->host_dirty=false;
template <typename ...Args>
int Fusions<Args...>::testSize(Args... args)
{
    double cputime[2],gputime[2];
    buffer_t* halfBuf=divBuffer(output,0,(output->extent[1]/2));
//    warm up

    gpuFunc(forward<Args>(args)...,input,output);
    cleanBufferDev(output);
    gpuFunc(forward<Args>(args)...,input,halfBuf);
    cleanBufferDev(halfBuf);
    double t1, t2,t3;


    //    calculate gpu time with full size

    t1=current_time();
    gpuFunc(forward<Args>(args)...,input,output);
    t2=current_time();
    halide_copy_to_host(NULL,output);
    t3=current_time();

    gputime[0]=t3-t1;
//    gpuCopyTime[0]=t3-t2;
    cleanBufferDev(output);

//    calculate gpu time with half size

    t1=current_time();
    gpuFunc(forward<Args>(args)...,input,halfBuf);
    t2=current_time();

    halide_copy_to_host(NULL,halfBuf);

    t3=current_time();
    gputime[1]=t3-t1;
//    gpuCopyTime[1]=t3-t2;
    cleanBufferDev(halfBuf);

    //warm up
    cpuFunc(forward<Args>(args)...,input,output);


    //    calculate cpu time with full size

    t1=current_time();
    cpuFunc(forward<Args>(args)...,input,output);
    t2=current_time();
    cputime[0]=double(t2-t1);

//    calculate cpu time with half size

    t1=current_time();
    cpuFunc(forward<Args>(args)...,input,halfBuf);
    t2=current_time();
    cputime[1]=double(t2-t1);



#ifdef DEBUG
    cout<<"cpu execution Time :"<<cputime[0]<<" "<<cputime[1]<<endl;
    cout<<"gpu execution time :"<<gputime[0]<<" "<< gputime[1]<<endl;
#endif // DEBUG
    //Solve the equation

    double a1,a2,b1,b2,x1,x2,a3,b3;

    a1=((cputime[0]-cputime[1])*2)/input->extent[1];

    a2=((gputime[0]-gputime[1])*2)/input->extent[1];

//    a3=((gpuCopyTime[0]-gpuCopyTime[1])*2)/input->extent[1];

    b1=cputime[0]-(input->extent[1]*a1);
    b2=gputime[0]-(input->extent[1]*a2);
#ifdef DEBUG
    cout<<"CPU Function"<<a1<<" "<<b1<<endl;
    cout<<"GPU Function"<<a2<<" "<<b2<<endl;
#endif // DEBUG
    assert(a1+a2!=0);

    x1=(input->extent[1]*a2+b2-b1)/(a1+a2);
    float exe_time=a1*x1+b1;

    x2=(exe_time-b2)/a2;
    float copy_time=a3*x2+b3;
    if(copy_time<0)
        copy_time=0;
//    float copy_time=a3*x2+b3;
    cout<<"estimate execution time "<<exe_time<<endl;
    if(x1<=0.0)
    {
        cout<<"The GPU workload should be 100%"<<endl;
        return 0;
    }
    if(x2<=0.0)
    {
        cout<<"The CPU workload should be 100%"<<endl;
        return 0;
    }
    float improve=((min(cputime[0],gputime[0])/(exe_time))-1.0)*100;
    if(improve<=WORKLOAD_THRESHOULD)
    {
        if(cputime[0]<gputime[0])
        {
            cout<<"The CPU workload should be 100%";
        }
        else if(cputime[0]>gputime[0])
        {

            cout<<"The GPU workload should be 100%";
        }
        else
            cout<<"You can choose any device you want";
        cout<<" , because the improvement is less then " <<WORKLOAD_THRESHOULD<<" % "<<endl;
        return 0;
    }

    cout<<"The CPU workload should be "<<x1<<endl;

    cout<<"By the assignment,We can save "<<improve<<"% execution time"<<endl;

    return x1;



}
}
#endif // FUSION_H
