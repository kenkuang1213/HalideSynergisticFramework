#include <jni.h>
#include <android/log.h>
#include <android/bitmap.h>
#include <android/native_window_jni.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>

#include "gaussinBlur_cpu.h"
#include "gaussinBlur_gpu.h"
#include <HalideRuntime.h>
#include "fusion.h"
#define  LOGD(...)  __android_log_print(ANDROID_LOG_DEBUG,"halide_native",__VA_ARGS__)
#define  LOGE(...)  __android_log_print(ANDROID_LOG_ERROR,"halide_native",__VA_ARGS__)

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
    JNIEnv *env, jobject obj, jbyteArray jSrc, jint j_w, jint j_h, jint j_wordload,jobject surf) {

    const int w = j_w, h = j_h;
    int workload=j_wordload;
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
        int cpuworkload=0;
        // Just copy over chrominance untouched
        memcpy(dst + w*h, src + w*h, (w*h)/2);
        static Fusion::Fusions<> fusion(gaussinBlur_cpu,gaussinBlur_gpu,&srcBuf,&dstBuf);
        int64_t t1 = halide_current_time_ns();
        if(workload==100){
            fusion.realizeCPU();
        }
        else if(workload==0){
            fusion.realizeGPU();
        }
        else{
            int cpuworkload=h*workload/100;
            // LOGD("CPU Wordload: %d" ,cpuworkload);
            fusion.realize(cpuworkload);
        }
        // gaussinBlur_cpu(&srcBuf, &dstBuf);
        // fusion.realizeGPU();
        // for(int i=0;i<(w*h)/2;i++){
        //         dst[w*h+i]=125;
        // }
        if (dstBuf.dev) {
            halide_copy_to_host(NULL, &dstBuf);
        }

        int64_t t2 = halide_current_time_ns();
        unsigned elapsed_us = (t2 - t1)/1000;
	unsigned fps=1000000/elapsed_us;

        times[counter & 15] = elapsed_us;
        counter++;
        unsigned min = times[0];
        for (int i = 1; i < 16; i++) {
            if (times[i] < min) min = times[i];
        }
        // LOGD("Time taken: %d (%d)", elapsed_us, min);
        LOGD("FPS: %d" ,fps);
        
    }

    ANativeWindow_unlockAndPost(win);
    ANativeWindow_release(win);

    env->ReleaseByteArrayElements(jSrc, (jbyte *)src, 0);
}
}
