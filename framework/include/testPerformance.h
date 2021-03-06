#ifndef DBL_MAX
#define DBL_MAX 1.79769e+308
#endif

#ifndef FUSION_TEST_H
#define FUSION_TEST_H
#include "fusion.h"
#include "DynamicDispatch.h"
#include "StaticDispatch.h"
#define CPU 1
#define GPU 2
#define Fus 3
#include "static_image.h"
#include "image_io.h"
namespace Fusion
{
namespace Test
{
template <typename Func,typename ...Args>
void testStaticPerformance(int type,Func func,buffer_t* input,buffer_t* output,Args ...args)
{
    Static::StaticDispatch fusion(input,output);
    double bestT =DBL_MAX;
    double worstT=0;
    fusion.realize(std::forward<Func>(func),std::forward<Args>(args)...);
//    input->host_dirty=1;
    for (int i = 0; i < 5; i++)
    {
        double t1=current_time();
        fusion.realize(std::forward<Func>(func),std::forward<Args>(args)...);
        double t2=current_time();
//        input->host_dirty=1;
        double t=t2-t1;
        if (t < bestT) bestT = t;
        if (t > worstT) worstT = t;
    }
    switch (type)
    {
    case CPU:

        cout<<setw(15)<<"Best CPU: "<<setw(10)<<bestT<<setw(15)<<" Worst CPU: "<<setw(10)<<worstT<<endl;
        break;
    case GPU:
        cout<<setw(15)<<"Best GPU: "<<setw(10)<<bestT<<setw(15)<<" Worst GPU: "<<setw(10)<<worstT<<endl;
        break;
    }
}
template <typename Func,typename ...Args>
void testStaticPerformance(Func cFunc,Func gFunc,buffer_t* input,buffer_t* output,int workload,Args ...args)
{
    Static::StaticDispatch fusion(input,output);
    double bestT =DBL_MAX;
    double worstT=0;
   fusion.realize(std::forward<Func>(cFunc),std::forward<Func>(gFunc),std::forward<int>(workload),std::forward<Args>(args)...);
   input->host_dirty=1;
    for (int i = 0; i < 5; i++)
    {
        double t1=current_time();
        fusion.realize(std::forward<Func>(cFunc),std::forward<Func>(gFunc),std::forward<int>(workload),std::forward<Args>(args)...);
        double t2=current_time();
         input->host_dirty=1;
        double t=t2-t1;
        input->host_dirty=1;
        if (t < bestT) bestT = t;
        if (t > worstT) worstT = t;
    }
    cout<<setw(15)<<"Best Fusion with "<<workload<<"% : "<<setw(10)<<bestT<<setw(15)<<" Worst Fusion: "<<setw(10)<<worstT<<endl;

}
template <typename Func,typename ...Args>
void testDynamicPerformance(Func cFunc,Func gFunc,buffer_t* input,buffer_t* output,Args ...args)
{
    Dynamic::DynamicDispatch fusion(input,output);
    double bestT =DBL_MAX;
    double worstT=0;
    fusion.realize(std::forward<Func>(cFunc),std::forward<Func>(gFunc),std::forward<Args>(args)...);
    input->host_dirty=1;
    for (int i = 0; i < 5; i++)
    {
        double t1=current_time();
        fusion.realize(std::forward<Func>(cFunc),std::forward<Func>(gFunc),std::forward<Args>(args)...);
        double t2=current_time();
        input->host_dirty=1;
        double t=t2-t1;
        if (t < bestT) bestT = t;
        if (t > worstT) worstT = t;
    }
    cout<<setw(15)<<"Best Dynamic Fusion: "<<setw(10)<<bestT<<setw(15)<<" Worst Fusion: "<<setw(10)<<worstT<<endl;

}
template <typename Func,typename ...Args>
void testSizePerformance(int type,Func func,buffer_t* input,buffer_t* output,int workload,Args ...args)
{
    int bufferWidth=output->extent[1]*workload/100;
    buffer_t *tmp=Fusion::Internal::divBuffer(output,0,bufferWidth);

    Static::StaticDispatch fusion(input,tmp);

    double bestT =DBL_MAX;
    double worstT=0;

    fusion.realize(std::forward<Func>(func),std::forward<Args>(args)...);
    input->host_dirty=1;
    for (int i = 0; i < 5; i++)
    {
        double t1=current_time();
        fusion.realize(std::forward<Func>(func),std::forward<Args>(args)...);
        double t2=current_time();
        input->host_dirty=1;
        double t=t2-t1;

        if (t < bestT) bestT = t;
        if (t > worstT) worstT = t;
    }
    switch (type)
    {
    case CPU:

        cout<<setw(15)<<"Best CPU with "<<workload<<"% : "<<setw(10)<<bestT<<setw(15)<<" Worst CPU: "<<setw(10)<<worstT<<endl;
        break;
    case GPU:
        cout<<setw(15)<<"Best GPU with "<<workload<<"% : "<<setw(10)<<bestT<<setw(15)<<" Worst GPU: "<<setw(10)<<worstT<<endl;
        break;


    }
//    delete tmp;

}

}
}
#endif // FUSION_TEST_H
