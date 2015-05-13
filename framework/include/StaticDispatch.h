#ifndef STATICDISPATCH_H
#define STATICDISPATCH_H
#include "clock.h"
#include "fusion.h"
#include "HalideRuntime.h"
#include "HalideRuntimeOpenCL.h"
namespace Fusion {
namespace Static {



/** \brief This Function is a thread function for Static execution way with Stealing.
 *
 * \param args Args arguments for Halide function.
 * \param func function<int(Args...,buffer_t*,buffer_t*)> Halide function
 * \return null
 *
 *
 * \details  This Function is a thread function for Static execution way with Stealing.
 * GPU will execute the target workload first.After that ,GPU will check the table,
 * if there are still blocks which have not been execute, GPU will help to execute the block.
 */
template<typename Function,typename ...Args>
void gpuStealing(Function  func,buffer_t* input ,buffer_t *cpuBuf,buffer_t *gpuBuf,status table[],int offset,mutex* table_mutex,Args ...args ) {
#ifdef DEBUG
    double t1=current_time();
#endif
    func(forward<Args>(args)...,input,gpuBuf);
    halide_copy_to_host(NULL,gpuBuf);

    if(table[9]==idle) {

        int end=cpuBuf->extent[1];
        int start=offset*9;
        bool bBreak=false;
        for(int i=9; i>=0; i--) {
            table_mutex->lock();
            if(table[i]!=idle)
                bBreak=true;
            else
                table[i]=computing;
            table_mutex->unlock();
            if(bBreak)
                break;
            buffer_t* buf=Internal::divBuffer(cpuBuf,start,end);
            func(forward<Args>(args)...,input,buf);

            table_mutex->lock();
            table[i]=writing;
            table_mutex->unlock();
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
#ifdef DEBUG
    double t2=current_time();
    exe_time_gpu=t2-t1;
#endif
}


/** \brief  This Function is a thread for Static execution way WITHOUT stealing.
*
* \param Args arguments for Halide function
* \param function<int(Args...,buffer_t*,buffer_t*)> function of Halide
* \param buffer_t* input
* \param buffer_t* output
* \return void
*
*/
template<typename Function,typename ...Args>
void gpuThread(Function  func,buffer_t* input ,buffer_t *output,Args ...args) {
#ifdef DEBUG
    double t1=current_time();
#endif
    func(forward<Args>(args)...,input,output);
    halide_copy_to_host(NULL,output);
#ifdef DEBUG
    double t2=current_time();
    exe_time_gpu=t2-t1;
#endif
}

class GPUThread {
public:
    GPUThread(buffer_t* _input,buffer_t* _output):input(_input),output(_output) {}
    template <typename Function,typename... Args>
    void run(Function func,Args... args) {
        thr=new thread(gpuThread<Function, Args...>,
                       std::forward<Function>(func),
                        std::forward<buffer_t*>(input),  //Halide Function
                       std::forward<buffer_t*>(output),std::forward<Args>(args)...
                       );  //Halide Function Argument
    }
    template <typename Function,typename... Args>
    void run(Function func,buffer_t* cpuBuf,buffer_t* gpuBuf,status table[],int offset,mutex* table_mutex,Args... args) {
        thr=new thread(gpuStealing<Function, Args...>,
                       std::forward<Function>(func), std::forward<buffer_t*>(input),  //Halide Function
                       std::forward<buffer_t*>(cpuBuf), std::forward<buffer_t*>(gpuBuf),table,offset,table_mutex,std::forward<Args>(args)...);  //Halide Function Argument

    }
    void join() {
        thr->join();
    }
private:
    buffer_t* input,*output;
    std::thread *thr;
};


/** \class StaticDispatch
 *  \brief static way
 *
 *
 */

template <typename ...Args>
class StaticDispatch {
public:
    StaticDispatch() {}
    StaticDispatch(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc) {
        input=NULL;
        output=NULL;
#if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
#endif
    }
    StaticDispatch(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input) {
        output=NULL;
#if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
#endif
    }
    StaticDispatch(function<int(Args...,buffer_t*,buffer_t*)>  _cpuFunc,function<int(Args...,buffer_t*,buffer_t*)>  _gpuFunc,buffer_t* _input,buffer_t* _output) :cpuFunc(_cpuFunc),gpuFunc(_gpuFunc),input(_input),output(_output) {
#if COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
#endif
    }
    int realizeCPU(Args...);/**< Using CPU only */
    int realizeGPU(Args...);/**< Using GPU only */
    int realizeCPU(Args...,buffer_t* _output);/**< Using CPU only with specific output buffer  */
    int realizeGPU(Args...,buffer_t* _output);/**< Using GPU only with specific output buffer */
    /** \brief realize With CPU stealing
     *
     * \param Args arguments for Halide function
     * \param int CPU workload
     * \return int
     *
     */
    int realizeWithStealing (Args... args,int s);
    void setInput(buffer_t* _input) {
        input=_input;
    }
    void setOutput(buffer_t* _output) {
        output=_output;
    }

    int realize(Args...,int s);/**< Realize With Specific Workload */
    int realize(Args...,int s,int kernalSize);/**< Realize With Specific Workload with specific kernel size */
    mutex *table_mutex;
private:
    function<int(Args...,buffer_t*,buffer_t*)> cpuFunc,gpuFunc;
    buffer_t *input,*output;
    int CPUWorkload;
    buffer_t* createBuffers(int x,int y,int z,int w,int s);
};

template <typename ...Args>
int StaticDispatch<Args...>::realizeCPU(Args... args,buffer_t* _output) {
    cpuFunc(forward<Args>(args)...,args...,input,_output);
    return 0;
}

template <typename ...Args>
int StaticDispatch<Args...>::realizeGPU(Args... args,buffer_t* _output) {
    gpuFunc(forward<Args>(args)...,input,_output);
    halide_copy_to_host(NULL,output);
    return 0;
}

template <typename ...Args>
int StaticDispatch<Args...>::realizeGPU(Args... args) {
#ifdef DEBUG
    double t1=current_time();
#endif

    gpuFunc(forward<Args>(args)...,input,output);

    halide_copy_to_host(NULL,output);
#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
    double fps=1000.0/exe_time_cpu;

    fusion_printf("GPU  (FPS) : %.f %.f\n",exe_time_cpu,fps);
#endif

    return 0;
}

template <typename ...Args>
int StaticDispatch<Args...>::realizeCPU(Args... args) {

#ifdef DEBUG
    double t1=current_time();
#endif

    cpuFunc(forward<Args>(args)...,input,output);

#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
    double fps=1000/exe_time_cpu;

    fusion_printf("CPU  (FPS) : %.f (%.f)\n",exe_time_cpu,fps);
#endif

    return 0;
}


template <typename ...Args>
int StaticDispatch<Args...>::realize(Args... args,int s) {
    if(output==NULL)
        return -1;
    buffer_t* cpuBuf=Internal::divBuffer(output,0,s);
    buffer_t* gpuBuf=Internal::divBuffer(output,s,input->extent[1]);

    GPUThread gThread(input,gpuBuf);
    gThread.run(gpuFunc,forward<Args>(args)...);
#ifdef DEBUG
    double t1=current_time();
#endif
//    cpuFunc(forward<Args>(args)...,input,cpuBuf);
#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
#endif
    gThread.join();
#ifdef DEBUG
    double fps=1000/max(exe_time_gpu,exe_time_cpu);

    fusion_printf("CPU V.S GPU (FPS) : %.f %.f (%.f)\n",exe_time_cpu,exe_time_gpu,fps);
#endif
    return 0;
}


template <typename ...Args>
int StaticDispatch<Args...>::realize(Args... args,int s,int kernelSize) {
    if(output==NULL)
        return -1;
    buffer_t* cpuBuf=Internal::divBuffer(output,0,s);
    buffer_t* gpuInput=Internal::divBuffer(input,s-kernelSize,input->extent[1]);
    buffer_t* gpuBuf=Internal::divBuffer(output,s,input->extent[1]);
#ifdef DEBUG
    fusion_printf("GPU Copy Size %d",input->extent[1]-s+kernelSize);
#endif
    thread gThread(gpuThread<args...>,forward<Args>(args)...,gpuFunc,input,gpuBuf);


#ifdef DEBUG
    double t1=current_time();
#endif
    cpuFunc(forward<Args>(args)...,input,cpuBuf);
#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
#endif
    gThread.join();
    delete cpuBuf;
    delete gpuBuf;
    delete gpuInput;
#ifdef DEBUG
    double fps=1000/max(exe_time_gpu,exe_time_cpu);

    fusion_printf("CPU V.S GPU (FPS) : %.f %.f (%.f)\n",exe_time_cpu,exe_time_gpu,fps);
#endif
    return 0;
}


template <typename ...Args>
int StaticDispatch<Args...>::realizeWithStealing (Args... args,int s) {
#ifdef DEBUG
    double t1=current_time();
#endif
    if(output==NULL)
        return -1;
    buffer_t* cpuBuf=Internal::divBuffer(output,0,s);
    buffer_t* gpuBuf=Internal::divBuffer(output,s,input->extent[1]);
    status table[10]= {idle};
    int offset=floor(cpuBuf->extent[1]/10);

    table_mutex=new mutex;
    GPUThread gThread(input,gpuBuf);
    gThread.run(gpuFunc,cpuBuf,gpuBuf,table,offset,table_mutex,forward<Args>(args)...);

    bool bBreak=false;
    for(int i=0; i<=9; i++) {
        table_mutex->lock();
        if(table[i]>idle) {
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

#ifdef DEBUG
    double t2=current_time();
    exe_time_cpu=t2-t1;
#endif
    gThread.join();
#ifdef DEBUG
    double fps=1000/max(exe_time_gpu,exe_time_cpu);
    fusion_printf("CPU V.S GPU (FPS) : %.f %.f (%.f)\n",exe_time_cpu,exe_time_gpu,fps);
#endif
    delete cpuBuf;
    delete gpuBuf;
    return 0;
}

}
}
#endif
