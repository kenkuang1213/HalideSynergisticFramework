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

    int i;
    for(i=9; i>=0; i--)
    {
        table_mutex->lock();
        if(table[i]!=idle)
            bBreak=true;
        else
            table[i]=computing;
        table_mutex->unlock();
        if(bBreak){
            i--;
            break;
        }
        buffer_t* buf=Internal::divBuffer(output,start,end);
        func(forward<Args>(args)...,input,buf);
        table_mutex->lock();
        if(table[i]==finished)
            bBreak=true;
        else
            table[i]=computing;
        table_mutex->unlock();
        if(bBreak){
//            halide_device_release(NULL,halide_opencl_device_interface);
            i--;
            break;

        }

        halide_device_sync(NULL,buf);
        table_mutex->lock();
        if(table[i]==finished){

            bBreak=true;
        }
        else
            table[i]=writing;
        table_mutex->unlock();
        if(bBreak){
            i--;
//            halide_device_release(NULL,halide_opencl_device_interface());
            break;
        }
        halide_copy_to_host(NULL,buf);
        delete buf;
        table_mutex->lock();
        table[i]=finished;
        table_mutex->unlock();
        end=offset*i;
        start=offset*(i-1);

    }
    #ifdef DEBUG
        fusion_printf("GPU Workload %d\n",9-i);
#endif
}



class DynamicDispatch
{
public:
    DynamicDispatch(buffer_t* _input, buffer_t* _output):input(_input),output(_output)
    {

        table_mutex=new mutex();
        #if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
        #endif

    }
    template<typename Func,typename ...Args>
    int realize(Func fucn,Args...);

    /** \brief this will realize synergistically
     *
     * \param Func CPU Function
     * \param Func GPU fucntion
     * \param Args... arguments for Halide function
     * \return
     *
     */
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
int DynamicDispatch::realize(Func func,Args... args)
{

    func(forward<Args>(args)...,input,output);
    halide_copy_to_host(NULL,output);
    return 0;
}


template <typename Func,typename... Args>
int DynamicDispatch::realize(Func cpuFunc,Func gpuFunc,Args... args)
{
    status table[10]= {idle};
    int offset=floor(input->extent[1]/10);
    table_mutex->unlock();
    thread gpuWorkThread(workThread<Func,Args...>,std::forward<Func>(gpuFunc),std::forward<buffer_t*>(input),
                         std::forward<buffer_t*>(output),table,offset,table_mutex,std::forward<Args>(args)...);
    bool bBreak=false;
    int i;
    for(i=0; i<=9; i++)
    {
        table_mutex->lock();
        if(table[i]>idle)
        {
            i--;
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
#ifdef DEBUG
    fusion_printf("The CPU Workload is %d\n",i);

#endif // DEBUG
    return 0;
}


}
}
#endif
