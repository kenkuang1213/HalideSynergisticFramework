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


#ifdef ANDROID
#define fusion_printf(...) __android_log_print(ANDROID_LOG_DEBUG,"fusion_debug",__VA_ARGS__)
#else
#define fusion_printf(...) printf(__VA_ARGS__ );
#endif

#ifndef FUSION_H
#define FUSION_H

#include <cmath>
#include <iostream>
#include <utility>
#include <thread>
#include <assert.h>
#include <string>
#include <mutex>
#include "fusion_info.h"
#include "internal.h"
#include "clock.h"
#include "HalideRuntime.h"
#include "HalideRuntimeOpenCL.h"
using namespace std;

#ifdef ANDROID
#include <string>
#include <sstream>
template <typename T>
std::string to_string(T value)
{
    std::ostringstream os ;
    os << value ;
    return os.str() ;
}
#endif

extern "C" int halide_copy_to_host(void* user_context, buffer_t* buf);
namespace Fusion
{

#ifdef DEBUG
static double exe_time_cpu,exe_time_gpu;
#endif



}
#endif // FUSION_H

