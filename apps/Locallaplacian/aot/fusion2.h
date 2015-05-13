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


#ifndef FUSION2_H
#define FUSION2_H



#include <iostream>
#include <utility>
#include <thread>
#include "clock.h"
using namespace std;
template<typename ...Args>
void gputhread(function<int(Args...,buffer_t*,buffer_t*)> func, Args ...args,buffer_t* input ,buffer_t *output)
{
    func(forward<Args>(args)...,input,output);


}
extern "C" int halide_copy_to_host(void* user_context, buffer_t* buf);
template<typename ...Args>
class Fusions
{
public:
    Fusions() {}
    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc)
    {
        input=NULL;
    }
    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input)
    {
        bitOfInput=input->elem_size;
    }
    Fusions(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input,buffer_t* _output) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input),output(_output)
    {
        bitOfInput=input->elem_size;
    }
    void realizeCPU(Args...);
    void realizeGPU(Args...);
    int testSize(Args...);
    void setInput(buffer_t* _input)
    {
        input=_input;
    }
    void setOutput(buffer_t* _output)
    {
        output=_output;
    }

    void realize(Args...,int s);

private:
    function<int(Args...,buffer_t*,buffer_t*)> cpuFunc,gpuFunc;
    buffer_t *input,*output;
    int bitOfInput;
    uint8_t* createBuffer_t(int x,int y,int z,int w,buffer_t* buf);
    buffer_t* createBuffers(int x,int y,int z,int w,int s);
    buffer_t* divBuffer(buffer_t *buf,int start,int nend) ;


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
    buff->extent[2] = buf->extent[2];
    buff->extent[3] = buf->extent[3];
    buff->stride[0] = buf->stride[0];
    buff->stride[1] = buf->stride[1];
    buff->stride[2] = buf->stride[2];
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
void Fusions<Args...>::realizeCPU(Args... args)
{

    input->host_dirty=true;
    cpuFunc(forward<Args>(args)...,input,output);

}
template <typename ...Args>
void Fusions<Args...>::realizeGPU(Args... args)
{

    input->host_dirty=true;
    gpuFunc(forward<Args>(args)...,input,output);
    halide_copy_to_host(NULL,output);
}

template <typename ...Args>
void Fusions<Args...>::realize(Args... args,int s)
{
    buffer_t* cpuBuf=divBuffer(output,0,s);
    buffer_t* gpuBuf=divBuffer(output,s,input->extent[1]);
    thread gputThread(gpuFunc,forward<Args>(args)...,input,gpuBuf);
    cpuFunc(forward<Args>(args)...,input,cpuBuf);


    gputThread.join();
    halide_copy_to_host(NULL,gpuBuf);

}
#define cleanBufferDev(buffer) \
buffer->dev=0; \
buffer->dev_dirty=false; \
buffer->host_dirty=false;
template <typename ...Args>
int Fusions<Args...>::testSize(Args... args)
{
    double cputime[2],gputime[2];
//    double gpuCopyTime[2];
    buffer_t* halfBuf=divBuffer(output,0,(output->extent[1]/2));
//    warm up
    cpuFunc(forward<Args>(args)...,input,output);
    gpuFunc(forward<Args>(args)...,input,output);
    cleanBufferDev(output);
    gpuFunc(forward<Args>(args)...,input,halfBuf);
    cleanBufferDev(halfBuf);
    double t1, t2,t3;


    //    calculate gpu time with full size

    t1=current_time();
    gpuFunc(args...,input,output);
//    t2=current_time();
    halide_copy_to_host(NULL,output);
    t3=current_time();

    gputime[0]=t3-t1;
//    gpuCopyTime[0]=t3-t2;
    cleanBufferDev(output);

//    calculate gpu time with half size

    t1=current_time();
    gpuFunc(args...,input,halfBuf);


    halide_copy_to_host(NULL,halfBuf);

    t3=current_time();
    gputime[1]=t3-t1;
//    gpuCopyTime[1]=t3-t2;
    cleanBufferDev(halfBuf);

    //    calculate cpu time with full size

    t1=current_time();
    cpuFunc(args...,input,output);
    t2=current_time();
    cputime[0]=double(t2-t1);

//    calculate cpu time with half size

    t1=current_time();
    cpuFunc(args...,input,halfBuf);
    t2=current_time();
    cputime[1]=double(t2-t1);




    cout<<"cpu execution Time :"<<cputime[0]<<endl;
    cout<<"gpu execution time "<<gputime[0]<<" "<< gputime[1]<<endl;
    //Solve the equation

    double a1,a2,b1,b2,x1,x2,a3,b3;

    a1=((cputime[0]-cputime[1])*2)/input->extent[1];

    a2=((gputime[0]-gputime[1])*2)/input->extent[1];

//    a3=((gpuCopyTime[0]-gpuCopyTime[1])*2)/input->extent[1];

    b1=cputime[0]-(input->extent[1]*a1);
    b2=gputime[0]-(input->extent[1]*a2);
//    b3=gpuCopyTime[0]-(input->extent[1]*a3);
//    cout<<"a3: "<<a3<<" b3: "<<b3<<endl;
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
#endif // FUSION_H
