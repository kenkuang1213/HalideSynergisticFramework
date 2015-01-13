export HALIDE_PATH=/home/ken/src/Halide
export LD_LIBRARY_PATH="/home/ken/src/Halide/bin:$LD_LIBRARY_PATH"
include $(HALIDE_PATH)/apps/support/Makefile.inc
OPENCL_LDFLAGS=-L/opt/AMDAPPSDK-2.9-1/lib/x86_64/ -lOpenCL
all: Debug

halide/local_laplacian: halide/local_laplacian.cpp
	$(CXX) halide/local_laplacian.cpp -g -I $(HALIDE_PATH)/include $(HALIDE_PATH)/$(LIB_HALIDE) -o halide/local_laplacian -lpthread -ldl -lz $(LDFLAGS)

local_laplacian_cpu.o: halide/local_laplacian
	HL_TARGET=host ./halide/local_laplacian
local_laplacian_gpu.o: halide/local_laplacian
	HL_TARGET=host-opencl ./halide/local_laplacian
Debug: main.cpp local_laplacian_gpu.o local_laplacian_cpu.o
	$(CXX)  -Wall -g -O2 main.cpp local_laplacian_gpu.o local_laplacian_cpu.o -o process -lpthread -ldl -std=c++11 $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)
release: main.cpp local_laplacian_gpu.o local_laplacian_cpu.o
	$(CXX)  -Wall -O3  main.cpp local_laplacian_gpu.o local_laplacian_cpu.o -o process -lpthread -ldl $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)
out.png: process
	./process rgb.png 8 1 1 out.png

clean:
	rm -f process local_laplacian_cpu.o local_laplacian_cpu.h local_laplacian_gpu.o local_laplacian_gpu.h local_laplacian
