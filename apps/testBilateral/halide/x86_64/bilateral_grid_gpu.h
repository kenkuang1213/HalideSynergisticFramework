#ifndef HALIDE__bilateral_grid_gpu_h
#define HALIDE__bilateral_grid_gpu_h
#ifndef HALIDE_ATTRIBUTE_ALIGN
  #ifdef _MSC_VER
    #define HALIDE_ATTRIBUTE_ALIGN(x) __declspec(align(x))
  #else
    #define HALIDE_ATTRIBUTE_ALIGN(x) __attribute__((aligned(x)))
  #endif
#endif
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
    HALIDE_ATTRIBUTE_ALIGN(1) bool host_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) bool dev_dirty;
    HALIDE_ATTRIBUTE_ALIGN(1) uint8_t _padding[10 - sizeof(void *)];
} buffer_t;
#endif
struct halide_filter_metadata_t;
#ifndef HALIDE_FUNCTION_ATTRS
#define HALIDE_FUNCTION_ATTRS
#endif
#ifdef __cplusplus
extern "C" {
#endif
int bilateral_grid_gpu(const float _r_sigma, buffer_t *_p0_buffer, buffer_t *_bilateral_grid_buffer) HALIDE_FUNCTION_ATTRS;
int bilateral_grid_gpu_argv(void **args) HALIDE_FUNCTION_ATTRS;
extern const halide_filter_metadata_t bilateral_grid_gpu_metadata;
#ifdef __cplusplus
}  // extern "C"
#endif
#endif
