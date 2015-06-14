#ifndef DYNAMICDISPATCH_H
#define DYNAMICDISPATCH_H

#include "fusion.h"
namespace Fusion{
namespace Dynamic{


/**
    TODO:find out a way to interrupt cpu work thread

*/
template<typename Func,typename ...Args>
void workThread(Func func,buffer_t* input ,buffer_t *output,status table[],int offset,mutex* table_mutex,Args ...args)
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



class DynamicDispatch
{
public:
    DynamicDispatch() {}
    DynamicDispatch(buffer_t* _input, buffer_t* _output):input(_input),output(_output)
    {


        #if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
        #endif

    }
    template<typename Func,typename ...Args>
    int realize(Func fucn,Args...);
    template<typename Func,typename ...Args>
    int realize(Func cpuFunc,Func gpuFunc,Args...);


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


    buffer_t *input,*output;

    int CPUWorkload;



};

template <typename Func,typename ...Args>
int DynamicDispatch<Args...>::realize(Func func,Args... args)
{
    func(forward<Args>(args)...,input,_utput);
    halide_copy_to_host(NULL,output);
    return 0;
}


template <typename Func,typename ...Args>
int DynamicDispatch::realize(Func cpuFunc,Func gpuFunc,Args... args)
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
