include $(HALIDE_PATH)/apps/support/Makefile.inc
TARGET?=host
DEBUG?=0
all:test
halide_blur: ../../ halide_blur.cpp
	$(CXX) halide_blur.cpp -I $(HALIDE_PATH)/include $(HALIDE_PATH)/bin/libHalide.so -o halide_blur -ldl -lpthread -lz $(LDFLAGS)

halide_blur.o: halide_blur
	HL_DEBUG_CODEGEN=$(DEBUG) ./halide_blur
# -O2 is faster than -O3 for this app (O3 unrolls too much)
test: test.cpp halide_blur.o
	$(CXX) $(OPENMP_FLAGS) -msse2 -Wall -O0 -I ../support/ test.cpp halide_blur.o -o test -lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS) -I $(HALIDE_PATH)/include $(HALIDE_PATH)/bin/libHalide.so
blur_jit:blur_jit.cpp  PipelineJITInterface.h blurJITCG.h
	 $(CXX)  blur_jit.cpp -I $(HALIDE_PATH)/include $(HALIDE_PATH)/bin/libHalide.so -o blur_jit -ldl -lpthread -lz $(LDFLAGS) $(OPENCL_LDFAGS)
clean:
	rm -f test halide_blur.o halide_blur
