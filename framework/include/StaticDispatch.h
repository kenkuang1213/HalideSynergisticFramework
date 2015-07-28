#ifndef STATICDISPATCH_H
#define STATICDISPATCH_H
#include "clock.h"
#include "fusion.h"

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
    halide_device_sync(NULL,output);
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


class StaticDispatch {
public:
    StaticDispatch() {}
    StaticDispatch(buffer_t* _input,buffer_t* _output):input(_input),output(_output) {
#ifdef COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
#endif
    }
    StaticDispatch(buffer_t* _input):input(_input) {
        output=NULL;
#ifdef COMPILING_FOR_OPENCL
        halide_opencl_set_device_type("gpu");
#endif
    }
    template<typename Func,typename ...Args>
    int realize(Func func,Args...);/**< Using only One Device */

    template<typename Func,typename ...Args>
    int realize(buffer_t* _output,Func func,Args...);/**< Using only One Device with specific output buffer */

    /** \brief realize on both CPU and GPU with specific workload
     *
     * \param Func CPU Function
     * \param Func GPU Function
     * \param int Workload assign to CPU from 1 to 99 percentage
     * \param Args.. argument for Halide Function
     * \return int
     *
     */
    template<typename Func,typename ...Args>
    int realize(Func cFunc,Func gFUnc,int workload,Args... args);

    void setOutput(buffer_t* _output) {
        output=_output;
    }
    mutex *table_mutex;
private:
    buffer_t *input,*output;
    buffer_t* createBuffers(int x,int y,int z,int w,int s);
};

template <typename Func,typename ...Args>
int StaticDispatch::realize(Func func,Args... args) {
    func(forward<Args>(args)...,input,output);
    halide_device_sync(NULL,output);
    halide_copy_to_host(NULL,output); //because we can't sure with Device programer used ,so we will always call copy_to_host
    return 0;
}

template <typename Func,typename ...Args>
int StaticDispatch::realize(buffer_t* _output,Func func,Args... args) {
    func(forward<Args>(args)...,input,_output);//because we can't sure with Device programer used ,so we will always call copy_to_host
    halide_device_sync(NULL,_output);
    halide_copy_to_host(NULL,_output);
    return 0;
}

template <typename Func,typename ...Args>
int StaticDispatch::realize(Func cFunc,Func gFunc,int workload,Args... args) {
    if(output==NULL)
        return -1;
    if(workload<=0){
        gFunc(forward<Args>(args)...,input,output);
        return 0;
    }
    else if(workload>=100){
        cFunc(forward<Args>(args)...,input,output);
        return 0;
    }
    int bufferWidth=(output->extent[1]*workload)/100;
    buffer_t* cpuBuf=Internal::divBuffer(output,0,bufferWidth);
    buffer_t* gpuBuf=Internal::divBuffer(output,bufferWidth,input->extent[1]);

    GPUThread gThread(input,gpuBuf);
    gThread.run(gFunc,forward<Args>(args)...);
#ifdef DEBUG
    double t1=current_time();
#endif
    cFunc(forward<Args>(args)...,input,cpuBuf);
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


}
}
#endif
