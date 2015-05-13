#ifndef DYNAMICDISPATCH_H
#define DYNAMICDISPATCH_H

#include "fusion.h"
namespace Fusion{
namespace Dynamic{


/**
    TODO:find out a way to interrupt cpu work thread

*/
template<typename ...Args>
void workThread(Args ...args,function<int(Args...,buffer_t*,buffer_t*)>  func,buffer_t* input ,buffer_t *output,status table[],int offset,mutex* table_mutex)
{
    int end=output->extent[1];
    int start=offset*9;
    bool bBreak=false;
    for(int i=9; i>=0; i--)
    {
        table_mutex->lock();
        if(table[i]!=idle)
            bBreak=true;
        else
            table[i]=computing;
        table_mutex->unlock();
        if(bBreak)
            break;
        buffer_t* buf=Internal::divBuffer(output,start,end);
        func(forward<Args>(args)...,input,buf);

        table_mutex->lock();
        if(table[i]==finished)
            bBreak=true;
        else
            table[i]=writing;
        table_mutex->unlock();
        if(bBreak)
            break;
        halide_copy_to_host(NULL,buf);
        delete buf;
        table_mutex->lock();
        table[i]=finished;
        table_mutex->unlock();
        end=offset*i;
        start=offset*(i-1);
#ifdef DEBUG
        fusion_printf("GPU Workload %d0%",10-i);
#endif
    }
}


template<typename ...Args>
class DynamicDispatch
{
public:
    DynamicDispatch() {}
    DynamicDispatch(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc)
    {
        input=NULL;
        initFunc();
        output=NULL;
        #if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
        #endif

    }
    DynamicDispatch(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input)
    {
        output=NULL;
        initFunc();
        #if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
        #endif

    }
    DynamicDispatch(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input,buffer_t* _output) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input),output(_output)
    {
        initFunc();
        #if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
        #endif

    }
    int Realize(Args...);
    int realizeCPU(Args...);
    int realizeGPU(Args...);
    int realizeCPU(Args...,buffer_t* _output);
    int realizeGPU(Args...,buffer_t* _output);


    void setInput(buffer_t* _input)
    {
        input=_input;
    }
    void setOutput(buffer_t* _output)
    {
        output=_output;
    }


    mutex *table_mutex;
private:
    function<int(Args...,buffer_t*,buffer_t*)> cpuFunc,gpuFunc;

    buffer_t *input,*output;

    int CPUWorkload;



};

template <typename ...Args>
int DynamicDispatch<Args...>::realizeCPU(Args... args,buffer_t* _output)
{
    cpuFunc(forward<Args>(args)...,args...,input,_output);
    return 0;
}

template <typename ...Args>
int DynamicDispatch<Args...>::realizeGPU(Args... args,buffer_t* _output)
{
    gpuFunc(forward<Args>(args)...,input,_output);
    halide_copy_to_host(NULL,output);
    return 0;
}

template <typename ...Args>
int DynamicDispatch<Args...>::realizeGPU(Args... args)
{
#ifdef DEBUG
    double t1=current_time();
#endif

    gpuFunc(forward<Args>(args)...,input,output);

    halide_copy_to_host(NULL,output);
#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
    double fps=1000/exe_time_cpu;

    fusion_printf("GPU  (FPS) : %d %d\n",exe_time_cpu,fps);
#endif

    return 0;
}

template <typename ...Args>
int DynamicDispatch<Args...>::realizeCPU(Args... args)
{

#ifdef DEBUG
    double t1=current_time();
#endif

    cpuFunc(forward<Args>(args)...,input,output);

#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
    double fps=1000/exe_time_cpu;
    string str="CPU  (FPS) : "+to_string(exe_time_cpu)+"   ( "+to_string(fps)+ " ) ";
    fusion_printf("CPU  (FPS) : %d (%d)\n",exe_time_cpu,fps);
#endif

    return 0;
}




template <typename ...Args>
int DynamicDispatch<Args...>::Realize(Args... args)
{
    status table[10]= {idle};
    int offset=floor(input->extent[1]/10);
    table_mutex->unlock();
    thread gpuWorkThread(workThread<args...>,gpuFunc,input,output,table,offset,table_mutex);
    bool bBreak=false;
    for(int i=0; i<=9; i++)
    {
        table_mutex->lock();
        if(table[i]>idle)
        {
            bBreak=true;
        }
        table[i]=computing;
        table_mutex->unlock();
        if(bBreak)
            break;
        buffer_t* buf=Internal::divBuffer(output,i*offset,(i+1)*offset);

        cpuFunc(args...,input,buf);
        delete buf;
        table_mutex->lock();
        table[i]=finished;
        table_mutex->unlock();
    }
    gpuWorkThread.join();

}
}
}
#endif
