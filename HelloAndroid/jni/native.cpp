#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/native_window_jni.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include "gaussinBlur_cpu.h"
#include "gaussinBlur_gpu.h"
#include "sobel_cpu.h"
#include "sobel_gpu.h"
#include <HalideRuntime.h>
#include "StaticDispatch.h"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,"halide_native",__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,"halide_native",__VA_ARGS__)
#define  LOGA(...)  __android_log_print(ANDROID_LOG_DEBUG,"fusion_analysis",__VA_ARGS__)
#define DEBUG 1

extern "C" void halide_set_error_handler(int (*handler)(void *user_context, const char *));
extern "C" int halide_host_cpu_count();
extern "C" int64_t halide_current_time_ns();
extern "C" int halide_copy_to_host(void *, buffer_t *);
extern "C" int halide_copy_to_dev(void *, buffer_t *);
extern "C" int halide_dev_malloc(void *, buffer_t *);
extern "C" int halide_dev_free(void *, buffer_t *);

int handler(void *, const char *msg) {
    LOGE("%s", msg);
}

extern "C" {
JNIEXPORT void JNICALL Java_com_example_hellohalide_CameraPreview_processFrame(
    JNIEnv *env, jobject obj, jbyteArray jSrc, jint j_w, jint j_h, jint j_workload,jintArray jreturnData,jobject surf) {


    const int w = j_w, h = j_h;
    
    int *returnData=(int*)env->GetIntArrayElements(jreturnData,NULL);
    int workload=j_workload;
    static int count=0;
    static int lastworkload=0;
    halide_set_error_handler(handler);

    unsigned char *src = (unsigned char *)env->GetByteArrayElements(jSrc, NULL);
    if (!src) {
        LOGD("src is null\n");
        return;
    }

    ANativeWindow *win = ANativeWindow_fromSurface(env, surf);
    ANativeWindow_acquire(win);

    static bool first_call = true;
    static unsigned counter = 0;
    static unsigned times[16];
    if (first_call) {
        LOGD("According to Halide, host system has %d cpus\n", halide_host_cpu_count());
        LOGD("Resetting buffer format");
        ANativeWindow_setBuffersGeometry(win, w, h, 0);
        first_call = false;
        for (int t = 0; t < 16; t++) times[t] = 0;
    }

    ANativeWindow_Buffer buf;
    ARect rect = {0, 0, w, h};

    if (int err = ANativeWindow_lock(win, &buf, NULL)) {
        LOGD("ANativeWindow_lock failed with error code %d\n", err);
        return;
    }

    uint8_t *dst = (uint8_t *)buf.bits;

    // If we're using opencl, use the gpu backend for it.
    halide_set_ocl_device_type("gpu");

    // Make these static so that we can reuse device allocations across frames.
    static buffer_t srcBuf = {0};
    static buffer_t dstBuf = {0};

    static buffer_t srcUBuf = {0};
    static buffer_t dstUBuf = {0};

    static buffer_t srcVBuf = {0};
    static buffer_t dstVBuf = {0};


    if (dst) {
        srcBuf.host = (uint8_t *)src;
        srcBuf.host_dirty = true;
        srcBuf.extent[0] = w;
        srcBuf.extent[1] = h;
        srcBuf.extent[2] = 0;
        srcBuf.extent[3] = 0;
        srcBuf.stride[0] = 1;
        srcBuf.stride[1] = w;
        srcBuf.min[0] = 0;
        srcBuf.min[1] = 0;
        srcBuf.elem_size = 1;

        srcUBuf.host = (uint8_t *)src+w*h;
        srcUBuf.host_dirty = true;
        srcUBuf.extent[0] = w/2;
        srcUBuf.extent[1] = h/2;
        srcUBuf.extent[2] = 0;
        srcUBuf.extent[3] = 0;
        srcUBuf.stride[0] = 1;
        srcUBuf.stride[1] = w/2;
        srcUBuf.min[0] = 0;
        srcUBuf.min[1] = 0;
        srcUBuf.elem_size = 1;


        srcVBuf.host = (uint8_t *)src+w*h+w*h/4;
        srcVBuf.host_dirty = true;
        srcVBuf.extent[0] = w/2;
        srcVBuf.extent[1] = h/2;
        srcVBuf.extent[2] = 0;
        srcVBuf.extent[3] = 0;
        srcVBuf.stride[0] = 1;
        srcVBuf.stride[1] = w/2;
        srcVBuf.min[0] = 0;
        srcVBuf.min[1] = 0;
        srcVBuf.elem_size = 1;
      

        dstBuf.host = dst;
        dstBuf.extent[0] = w;
        dstBuf.extent[1] = h;
        dstBuf.extent[2] = 0;
        dstBuf.extent[3] = 0;
        dstBuf.stride[0] = 1;
        dstBuf.stride[1] = w;
        dstBuf.min[0] = 0;
        dstBuf.min[1] = 0;
        dstBuf.elem_size = 1;

        dstUBuf.host = dst+w*h;
        dstUBuf.host_dirty = true;
        dstUBuf.extent[0] = w/2;
        dstUBuf.extent[1] = h/2;
        dstUBuf.extent[2] = 0;
        dstUBuf.extent[3] = 0;
        dstUBuf.stride[0] = 2;
        dstUBuf.stride[1] = w;
        dstUBuf.min[0] = 0;
        dstUBuf.min[1] = 0;
        dstUBuf.elem_size = 1;

      
        dstVBuf.host = dst+w*h+1;
        dstVBuf.host_dirty = true;
        dstVBuf.extent[0] = w/2;
        dstVBuf.extent[1] = h/2;
        dstVBuf.extent[2] = 0;
        dstVBuf.extent[3] = 0;
        dstVBuf.stride[0] = 2;
        dstVBuf.stride[1] = w;
        dstVBuf.min[0] = 0;
        dstVBuf.min[1] = 0;
        dstVBuf.elem_size = 1;


        int cpuworkload=0;
        // Just copy over chrominance untouched
        memcpy(dst + w*h, src + w*h, (w*h)/2);
        // memcpy(dst , src , (w*h)+(w*h)/2);
        // static Fusion::Fusions<> fusion(gaussinBlur_cpu,gaussinBlur_gpu,&srcBuf,&dstBuf);
        static Fusion::Static::StaticDispatch<> fusionSobel(sobel_cpu,sobel_gpu,&srcBuf,&dstBuf);
        // static Fusion::Fusions<> fusionU(gaussinBlur_cpu,gaussinBlur_gpu,&srcUBuf,&dstUBuf);
        // static Fusion::Fusions<> fusionV(gaussinBlur_cpu,gaussinBlur_gpu,&srcVBuf,&dstVBuf);
        int64_t t1 = halide_current_time_ns();
        if(workload==100){
          fusionSobel.realizeCPU();
             // fusionU.realizeCPU();
             // fusionV.realizeCPU();
        }
        else if(workload==0){
            fusionSobel.realizeGPU();
             // fusionU.realizeGPU();
             // fusionV.realizeGPU();
            // Use Halide GPU Function rather then fusion
            // gaussinBlur_gpu(&srcBuf,&dstBuf);
            //  halide_copy_to_host(NULL, &dstBuf);
        }
        else if(workload==-1){
            int cpuworkload=1200;
            // int tmp;
            fusionSobel.realizeWithStealing(cpuworkload);
            // returnData[1]=((double)tmp/(double)h)*(double)100;
            // LOGD("Wordload: %d percentage" ,returnData[1]);
        }
        else {
            int cpuworkload=h*workload/100;
            // LOGD("CPU Wordload: %d" ,cpuworkload);
            fusionSobel.realize(cpuworkload);
            // int UVcpuworkload=h*workload/200;
            // fusionU.realize(UVcpuworkload);
            //  fusionV.realize(UVcpuworkload);
        }
        // gaussinBlur_cpu(&srcBuf, &dstBuf);
        // fusion.realizeGPU();
        // for(int i=0;i<(w*h)/2;i++){
        //         dst[w*h+i]=125;
        // }
        // if (dstBuf.dev) {
        //     halide_copy_to_host(NULL, &dstBuf);
        // }

        int64_t t2 = halide_current_time_ns();
        unsigned elapsed_us = (t2 - t1)/1000;
        double fps=1000000.0/elapsed_us;

        times[counter & 15] = elapsed_us;
        counter++;
        unsigned min = times[0];
        for (int i = 1; i < 16; i++) {
            if (times[i] < min) min = times[i];
        }

        returnData[0]=(int)fps;
        LOGD("FPS: %d" ,(int)fps);
        if(count==0)
            LOGA("Wordload : %d",workload);
        if(count>50&&count<=550)
            LOGA("FPS = %d",(int)fps);
        if(lastworkload==workload)
            count ++;
        else{
            lastworkload=workload;
            count=0;
        }


        
    }

    ANativeWindow_unlockAndPost(win);
    ANativeWindow_release(win);

    env->ReleaseByteArrayElements(jSrc, (jbyte *)src, 0);
}
}
