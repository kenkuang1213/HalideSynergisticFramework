LOCAL_PATH := $(call my-dir)

include $(CLEAR_VARS)

LOCAL_MODULE    := native
LOCAL_ARM_MODE  := arm
LOCAL_SRC_FILES := native.cpp
LOCAL_LDFLAGS   := -Ljni -lOpenCL
LOCAL_LDLIBS    := -lm -llog -landroid jni/halide_generated_$(TARGET_ARCH_ABI)/gaussinBlur_gpu.o   jni/halide_generated_$(TARGET_ARCH_ABI)/gaussinBlur_cpu.o  # -lllvm-a3xx
LOCAL_SHARED_LIBRARIES :=-lOpenCL
LOCAL_STATIC_LIBRARIES := android_native_app_glue
LOCAL_C_INCLUDES := $(HALIDE_PATH)/include $(LOCAL_PATH)/halide_generated_$(TARGET_ARCH_ABI)/
LOCAL_CPPFLAGS := -std=c++11  -Llibs -lOpenCL

include $(BUILD_SHARED_LIBRARY)

$(call import-module,android/native_app_glue)
