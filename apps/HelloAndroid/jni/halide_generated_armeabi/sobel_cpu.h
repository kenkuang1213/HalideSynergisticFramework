#ifndef HALIDE_sobel_cpu
#define HALIDE_sobel_cpu
#ifndef BUFFER_T_DEFINED
#define BUFFER_T_DEFINED
#include <stdint.h>
typedef struct buffer_t {
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
#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif
#ifdef __cplusplus
extern "C"
#endif
int sobel_cpu(buffer_t *_input, buffer_t *_output) HALIDE_FUNCTION_ATTRS;
#endif
