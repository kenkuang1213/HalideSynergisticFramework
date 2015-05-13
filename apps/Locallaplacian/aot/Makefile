include $(HALIDE_PATH)/apps/support/Makefile.inc
OPENCL_LDFLAGS=-L/opt/AMDAPPSDK-2.9-1/lib/x86_64/ -lOpenCL
ANDROID_OPENCL_LDFALGS=-lOpenCL
ANDROID_CXX?=arm-linux-androideabi-g++
ANDROID_PNGFLAGS=-lpng
all: Debug

halide/local_laplacian: halide/local_laplacian.cpp
	$(CXX) halide/local_laplacian.cpp -g -I $(HALIDE_PATH)/include -L$(HALIDE_PATH)/bin -o halide/local_laplacian -lpthread -ldl -lz -lHalide $(LDFLAGS)

local_laplacian_cpu.o: halide/local_laplacian
	HL_TARGET=host ./halide/local_laplacian
local_laplacian_gpu.o: halide/local_laplacian
	HL_TARGET=host-opencl-debug ./halide/local_laplacian
Debug: main.cpp local_laplacian_gpu.o local_laplacian_cpu.o
	$(CXX)  -Wall -g -O2 main.cpp local_laplacian_gpu.o local_laplacian_cpu.o -o process -lpthread -ldl -std=c++11 $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)
release: main.cpp local_laplacian_gpu.o local_laplacian_cpu.o
	$(CXX)  -Wall -O3  main.cpp local_laplacian_gpu.o local_laplacian_cpu.o -o process -lpthread -ldl -std=c++11  $(PNGFLAGS) $(CUDA_LDFLAGS) $(OPENCL_LDFLAGS) $(OPENGL_LDFLAGS)
android/local_laplacian_cpu.o: halide/local_laplacian
	cd android;HL_TARGET=arm-32-android ../halide/local_laplacian
android/local_laplacian_gpu.o: halide/local_laplacian
	cd android;HL_TARGET=arm-32-android-opencl ../halide/local_laplacian

android: main.cpp android/local_laplacian_gpu.o android/local_laplacian_cpu.o
	$(ANDROID_CXX)  -Wall -O3  main.cpp android/local_laplacian_gpu.o android/local_laplacian_cpu.o -o process -I android -landroid -llog -ldl -std=c++11 $(ANDROID_PNGFLAGS)  $(ANDROID_OPENCL_LDFALGS)

android_debug: main.cpp halide/local_laplacian android/local_laplacian_cpu.o
	cd android;HL_TARGET=arm-32-android-opencl-debug ../halide/local_laplacian
	$(ANDROID_CXX)  -Wall -O3  main.cpp android/local_laplacian_gpu.o android/local_laplacian_cpu.o -o process -I android -landroid -llog -ldl -std=c++11 $(ANDROID_PNGFLAGS)  $(ANDROID_OPENCL_LDFALGS)

out.png: process
	./process rgb.png 8 1 1 out.png

clean:
	rm -f process local_laplacian_cpu.o local_laplacian_cpu.h local_laplacian_gpu.o local_laplacian_gpu.h local_laplacian android/local_laplacian_gpu.o android/local_laplacian_cpu.o android/local_laplacian_cpu.h android/local_laplacian_gpu.h
